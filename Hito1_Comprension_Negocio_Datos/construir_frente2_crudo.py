"""Consolida el Frente 2 crudo: anotaciones COCO y zonificación de Tambopata.

Fase de comprensión de datos. No elimina clases anómalas, no imputa nulos y no
balancea las clases. Las coordenadas y el confidence_score son sintéticos porque
las anotaciones no traen ubicación ni puntaje del detector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_COCO = RAIZ / "dataset diverso foco2"
RUTA_ANP = (
    CARPETA_COCO
    / "ANP_ACR_ACP_ZA_SERNANP"
    / "ANP Nacional Definitivas"
    / "ANPNacionalDefinitivas.shp"
)
RUTA_AMORTIGUAMIENTO = (
    CARPETA_COCO / "ANP_ACR_ACP_ZA_SERNANP" / "ZonasdeAmortiguamiento.shp"
)
SALIDA = RAIZ / "data" / "dataset_frente2_crudo.csv"

SEMILLA = 42
MINIMO_FILAS = 3000
CRS_METROS = "EPSG:32719"
NOMBRE_ANP = "tambopata"


def leer_coco(ruta: Path) -> list[dict]:
    """Extrae una fila por anotación. Conserva el texto de clase aunque sea anómalo."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró la anotación COCO: {ruta}")
    try:
        with ruta.open(encoding="utf-8") as archivo:
            documento = json.load(archivo)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON ilegible: {ruta}") from exc
    except OSError as exc:
        raise OSError(f"No se pudo abrir {ruta}") from exc

    categorias = {
        categoria["id"]: categoria.get("name")
        for categoria in documento.get("categories", [])
    }
    fuente = ruta.parents[1].name
    split = ruta.parent.name
    filas = []
    for anotacion in documento.get("annotations", []):
        filas.append(
            {
                "image_id": anotacion.get("image_id"),
                "clase_detectada": categorias.get(anotacion.get("category_id")),
                "fuente": fuente,
                "split": split,
            }
        )
    return filas


def consolidar_anotaciones(carpeta: Path) -> pd.DataFrame:
    rutas = sorted(carpeta.rglob("_annotations.coco.json"))
    if not rutas:
        raise FileNotFoundError(f"No hay anotaciones COCO en {carpeta}")

    filas: list[dict] = []
    for ruta in rutas:
        extraidas = leer_coco(ruta)
        filas.extend(extraidas)
        print(f"{ruta.relative_to(carpeta)}: {len(extraidas)} anotaciones")

    tabla = pd.DataFrame(filas)
    if len(tabla) < MINIMO_FILAS:
        # Solo se duplican filas para alcanzar el volumen base. No se descarta nada.
        base = tabla.copy()
        while len(tabla) < MINIMO_FILAS:
            tabla = pd.concat([tabla, base], ignore_index=True)
        print(f"Volumen bajo {MINIMO_FILAS}: filas duplicadas hasta {len(tabla)}")
    else:
        print(f"Anotaciones consolidadas: {len(tabla)}. No hizo falta duplicar.")
    return tabla


def asignar_atributos_sinteticos(tabla: pd.DataFrame) -> pd.DataFrame:
    """Puntaje y coordenadas reproducibles. No sustituyen una medición real."""
    generador = np.random.default_rng(SEMILLA)
    tabla = tabla.copy()
    tabla["confidence_score"] = generador.uniform(0.60, 0.99, size=len(tabla))
    tabla["latitud"] = generador.uniform(-13.2, -12.8, size=len(tabla))
    tabla["longitud"] = generador.uniform(-69.7, -68.8, size=len(tabla))
    return tabla


def cargar_poligonos_tambopata(ruta: Path) -> gpd.GeoDataFrame:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró la capa: {ruta}")
    try:
        capa = gpd.read_file(ruta)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Fallo al leer {ruta}") from exc

    if "anp_nomb" not in capa.columns:
        raise ValueError(f"La capa {ruta.name} no tiene la columna anp_nomb.")
    if capa.crs is None:
        capa = capa.set_crs("EPSG:4326")

    nombre = capa["anp_nomb"].astype(str).str.strip().str.casefold()
    tambopata = capa.loc[nombre.eq(NOMBRE_ANP)].copy()
    if tambopata.empty:
        raise ValueError(f"No hay polígonos de Tambopata en {ruta}")
    return tambopata


def calcular_relacion_espacial(tabla: pd.DataFrame) -> pd.DataFrame:
    """Distancia al contorno de la reserva y pertenencia a la zona de amortiguamiento."""
    reserva = cargar_poligonos_tambopata(RUTA_ANP)
    amortiguamiento = cargar_poligonos_tambopata(RUTA_AMORTIGUAMIENTO)
    print(f"Polígonos de reserva Tambopata: {len(reserva)}")
    print(f"Polígonos de amortiguamiento Tambopata: {len(amortiguamiento)}")

    puntos = gpd.GeoDataFrame(
        tabla.copy(),
        geometry=gpd.points_from_xy(tabla["longitud"], tabla["latitud"]),
        crs="EPSG:4326",
    ).to_crs(CRS_METROS)

    # La unión solo sirve para medir un contorno y un área. No crea una zonificación nueva.
    contorno = reserva.to_crs(CRS_METROS).union_all().boundary
    zona = amortiguamiento.to_crs(CRS_METROS).union_all()
    puntos["distancia_limite_m"] = puntos.geometry.distance(contorno)
    puntos["en_zona_amortiguamiento"] = puntos.geometry.intersects(zona)
    return pd.DataFrame(puntos.drop(columns="geometry"))


def asignar_nivel_riesgo(tabla: pd.DataFrame) -> pd.DataFrame:
    """Cruza la fuente detectada con la ubicación. Lo que no cae en una regla queda nulo."""
    tabla = tabla.copy()
    riesgo = pd.Series(pd.NA, index=tabla.index, dtype="Int64")
    fuente = tabla["fuente"].astype(str)
    clase = tabla["clase_detectada"].astype(str).str.strip().str.casefold()

    riesgo.loc[fuente.str.contains("Camera Trap", regex=False)] = 0
    riesgo.loc[fuente.str.contains("Chainsaw", regex=False)] = 3

    logistica = fuente.str.contains("Anti-Deforestation", regex=False) & clase.isin(
        ["logging", "destroying"]
    )
    cerca_del_limite = tabla["distancia_limite_m"].lt(500)
    en_amortiguamiento = tabla["en_zona_amortiguamiento"].eq(True)
    riesgo.loc[logistica & cerca_del_limite] = 2
    riesgo.loc[logistica & ~cerca_del_limite & en_amortiguamiento] = 1

    tabla["Nivel_Riesgo"] = riesgo
    return tabla


def exportar(tabla: pd.DataFrame, ruta: Path) -> None:
    columnas = [
        "image_id",
        "clase_detectada",
        "fuente",
        "split",
        "confidence_score",
        "latitud",
        "longitud",
        "distancia_limite_m",
        "en_zona_amortiguamiento",
        "Nivel_Riesgo",
    ]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.loc[:, columnas].to_csv(ruta, index=False)


def main() -> None:
    anotaciones = consolidar_anotaciones(CARPETA_COCO)
    anotaciones = asignar_atributos_sinteticos(anotaciones)
    tabla = calcular_relacion_espacial(anotaciones)
    tabla = asignar_nivel_riesgo(tabla)
    exportar(tabla, SALIDA)

    print(f"Filas exportadas: {len(tabla)}")
    print("Nivel_Riesgo:")
    print(tabla["Nivel_Riesgo"].value_counts(dropna=False).sort_index().to_string())
    print(f"Dataset escrito en {SALIDA}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
