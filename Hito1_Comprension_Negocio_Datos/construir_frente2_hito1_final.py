"""Consolida el Frente 2 tabular para el Hito 1.

Lee las anotaciones COCO, conserva la fauna de Tambopata y arma Nivel_Riesgo.
No borra el texto residual de Roboflow, no duplica filas y no balancea clases.
Las coordenadas y el confidence_score son sintéticos: los JSON no traen
ubicación ni puntaje del detector.
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
SALIDA = RAIZ / "data" / "dataset_frente2_hito1_final.csv"

SEMILLA = 42
CRS_GEOGRAFICO = "EPSG:4326"
CRS_METROS = "EPSG:32719"
NOMBRE_ANP = "tambopata"
OESTE, SUR, ESTE, NORTE = -69.7, -13.2, -68.8, -12.8

FAUNA_DESCARTADA = {
    "elephant",
    "lion",
    "female lion",
    "male lion",
    "leopard",
    "tiger",
    "impala",
    "warthog",
    "hyena",
    "jackal",
    "honey badger",
    "sable",
    "waterbuck",
    "wild dog",
    "buffalo",
    "panda",
    "rabbit",
    "mouse",
    "cat",
    "dog",
    "cow",
    "bird",
    "animal",
    "jaguar-leopard",
}
MAPA_FAUNA = {
    "monkey": "mono",
    "snake": "snake",
    "face of capibara": "capibara",
    "capibaras": "capibara",
    "frog": "rana",
    "jaguar": "jaguar",
    "otter": "lobo_de_rio",
    "red_macaw": "guacamayo",
}
MARCAS_CAMION = {
    "daewoo",
    "benz",
    "scania",
    "renault",
    "faw",
    "vovlo",
    "dima",
    "empower",
}
MAPA_AMENAZA = {
    "destroying": "vehiculo_maderero",
    "truck": "vehiculo_maderero",
    "chainsaw": "motosierra",
    "rifle": "arma_fuego",
    "handgun": "arma_fuego",
    "weapon": "arma_fuego",
    "weapons": "arma_fuego",
    "poaching": "trampa_caza",
    "poacher": "trampa_caza",
    "trap": "trampa_caza",
}
NO_CAMION = {"car", "bus", "bike", "motorcycle", "person"}
FAUNA_RIESGO_BAJO = {
    "mono",
    "snake",
    "capibara",
    "rana",
    "jaguar",
    "lobo_de_rio",
    "guacamayo",
    "tapir",
    "sajino",
}
COLUMNAS_SALIDA = [
    "image_id",
    "clase_detectada",
    "fuente",
    "confidence_score",
    "latitud",
    "longitud",
    "en_zona_intangible",
    "en_zona_amortiguamiento",
    "distancia_limite_m",
    "distancia_a_carretera_m",
    "distancia_a_rio_m",
    "Nivel_Riesgo",
]


def carpetas_fuente(raiz: Path) -> list[Path]:
    """Carpetas .coco y, si existe, drunk_truck aunque no lleve ese sufijo."""
    if not raiz.is_dir():
        raise FileNotFoundError(f"No se encontró la carpeta de datasets: {raiz}")
    encontradas = []
    for ruta in raiz.iterdir():
        if not ruta.is_dir():
            continue
        nombre = ruta.name.casefold()
        if ".coco" in nombre or "drunk" in nombre:
            encontradas.append(ruta)
    if not encontradas:
        raise FileNotFoundError(f"No hay carpetas COCO en {raiz}")
    return sorted(encontradas)


def leer_coco(ruta: Path, fuente: str) -> list[dict]:
    """Una fila por anotación. No guarda la partición train, valid o test."""
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
    filas = []
    for anotacion in documento.get("annotations", []):
        puntaje = anotacion.get("score", anotacion.get("confidence"))
        filas.append(
            {
                "image_id": anotacion.get("image_id"),
                "clase_detectada": categorias.get(anotacion.get("category_id")),
                "fuente": fuente,
                "confidence_score": puntaje,
            }
        )
    return filas


def consolidar_anotaciones(raiz: Path) -> pd.DataFrame:
    filas: list[dict] = []
    for carpeta in carpetas_fuente(raiz):
        rutas = sorted(carpeta.rglob("_annotations.coco.json"))
        if not rutas:
            print(f"{carpeta.name}: sin anotaciones COCO")
            continue
        extraidas = []
        for ruta in rutas:
            extraidas.extend(leer_coco(ruta, carpeta.name))
        filas.extend(extraidas)
        print(f"{carpeta.name}: {len(extraidas)} anotaciones")

    if not filas:
        raise ValueError(f"No hay anotaciones COCO en {raiz}")
    return pd.DataFrame(filas)


def clase_normalizada(valores: pd.Series) -> pd.Series:
    return valores.astype(str).str.strip().str.casefold()


def filtrar_fauna_ajena(tabla: pd.DataFrame) -> pd.DataFrame:
    """Quita fauna que no habita Tambopata. No reescribe esas especies."""
    clase = clase_normalizada(tabla["clase_detectada"])
    ajena = clase.isin(FAUNA_DESCARTADA)
    print(f"Filas de fauna ajena a Tambopata, descartadas: {int(ajena.sum())}")
    return tabla.loc[~ajena].reset_index(drop=True)


def mapear_clases(tabla: pd.DataFrame) -> pd.DataFrame:
    """Traduce fauna de la reserva y amenazas. El resto de etiquetas se queda."""
    tabla = tabla.copy()
    clase = clase_normalizada(tabla["clase_detectada"])
    fuente = tabla["fuente"].astype(str).str.casefold()
    mapeada = clase.copy()

    mapeada = mapeada.replace(MAPA_FAUNA)
    amenaza = clase.map(MAPA_AMENAZA)
    mapeada = mapeada.where(amenaza.isna(), amenaza)
    marcas = clase.isin(MARCAS_CAMION)
    mapeada.loc[marcas] = "vehiculo_maderero"

    camion = clase.str.contains("truck", regex=False) & ~clase.isin(NO_CAMION)
    mapeada.loc[camion] = "vehiculo_maderero"
    carpeta_drunk = fuente.str.contains("drunk", regex=False)
    etiqueta_camion = carpeta_drunk & ~clase.isin(NO_CAMION) & clase.str.contains("truck", regex=False)
    mapeada.loc[etiqueta_camion] = "vehiculo_maderero"

    tabla["clase_detectada"] = mapeada
    return tabla


def asignar_atributos_sinteticos(tabla: pd.DataFrame) -> pd.DataFrame:
    """Puntaje y coordenadas reproducibles dentro del bounding box de Tambopata."""
    generador = np.random.default_rng(SEMILLA)
    tabla = tabla.copy()
    puntaje = pd.to_numeric(tabla["confidence_score"], errors="coerce")
    faltan = puntaje.isna()
    if faltan.any():
        puntaje.loc[faltan] = generador.uniform(0.60, 0.99, size=int(faltan.sum()))
    tabla["confidence_score"] = puntaje
    tabla["latitud"] = generador.uniform(SUR, NORTE, size=len(tabla))
    tabla["longitud"] = generador.uniform(OESTE, ESTE, size=len(tabla))
    return tabla


def cargar_tambopata(ruta: Path) -> gpd.GeoDataFrame:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró la capa: {ruta}")
    try:
        capa = gpd.read_file(ruta)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Fallo al leer {ruta}") from exc

    if "anp_nomb" not in capa.columns:
        raise ValueError(f"La capa {ruta.name} no tiene la columna anp_nomb.")
    if capa.crs is None:
        capa = capa.set_crs(CRS_GEOGRAFICO)

    nombre = capa["anp_nomb"].astype(str).str.strip().str.casefold()
    tambopata = capa.loc[nombre.eq(NOMBRE_ANP)].copy()
    if tambopata.empty:
        raise ValueError(f"No hay polígonos de Tambopata en {ruta}")
    return tambopata.to_crs(CRS_METROS)


def descargar_red(tags: dict) -> gpd.GeoDataFrame | None:
    """Descarga una capa OSM. Si falla, el llamador deja las distancias en NaN."""
    import osmnx as ox

    if hasattr(ox, "settings"):
        ox.settings.requests_timeout = 90
    try:
        red = ox.features_from_bbox(bbox=(OESTE, SUR, ESTE, NORTE), tags=tags)
    except TypeError:
        red = ox.features_from_bbox(NORTE, SUR, ESTE, OESTE, tags)
    if red is None or red.empty:
        return None
    red = red[red.geometry.notna() & ~red.geometry.is_empty].copy()
    if red.empty:
        return None
    if red.crs is None:
        red = red.set_crs(CRS_GEOGRAFICO)
    return red.to_crs(CRS_METROS)


def distancia_a_red(puntos: gpd.GeoDataFrame, red: gpd.GeoDataFrame | None) -> pd.Series:
    if red is None or red.empty:
        return pd.Series(np.nan, index=puntos.index, dtype="float64")
    try:
        return puntos.geometry.distance(red.union_all())
    except Exception as exc:
        print(f"No se pudo medir la distancia a la red OSM: {exc}")
        return pd.Series(np.nan, index=puntos.index, dtype="float64")


def calcular_relacion_espacial(tabla: pd.DataFrame) -> pd.DataFrame:
    """Reproyecta puntos, reserva, amortiguamiento y OSM a UTM 19S antes de medir."""
    reserva = cargar_tambopata(RUTA_ANP)
    amortiguamiento = cargar_tambopata(RUTA_AMORTIGUAMIENTO)
    print(f"Polígonos de reserva Tambopata: {len(reserva)}")
    print(f"Polígonos de amortiguamiento Tambopata: {len(amortiguamiento)}")

    puntos = gpd.GeoDataFrame(
        tabla.copy(),
        geometry=gpd.points_from_xy(tabla["longitud"], tabla["latitud"]),
        crs=CRS_GEOGRAFICO,
    ).to_crs(CRS_METROS)

    intangible = reserva.union_all()
    zona = amortiguamiento.union_all()
    puntos["en_zona_intangible"] = puntos.geometry.intersects(intangible)
    puntos["en_zona_amortiguamiento"] = puntos.geometry.intersects(zona)
    puntos["distancia_limite_m"] = puntos.geometry.distance(intangible)

    try:
        rios = descargar_red({"waterway": True, "water": True})
        print(f"Geometrías de ríos OSM: {0 if rios is None else len(rios)}")
    except Exception as exc:
        print(f"Fallo al descargar ríos OSM. distancia_a_rio_m queda en NaN: {exc}")
        rios = None
    try:
        vias = descargar_red({"highway": True})
        print(f"Geometrías de vías OSM: {0 if vias is None else len(vias)}")
    except Exception as exc:
        print(f"Fallo al descargar vías OSM. distancia_a_carretera_m queda en NaN: {exc}")
        vias = None

    puntos["distancia_a_rio_m"] = distancia_a_red(puntos, rios)
    puntos["distancia_a_carretera_m"] = distancia_a_red(puntos, vias)
    return pd.DataFrame(puntos.drop(columns="geometry"))


def asignar_nivel_riesgo(tabla: pd.DataFrame) -> pd.DataFrame:
    """Reglas de supervisión débil. Si no cae en ninguna, el target queda nulo.

    Cuando un punto está en intangible y en amortiguamiento, manda la regla más alta.
    """
    tabla = tabla.copy()
    clase = tabla["clase_detectada"].astype(str)
    intangible = tabla["en_zona_intangible"].eq(True)
    amortiguamiento = tabla["en_zona_amortiguamiento"].eq(True)
    riesgo = pd.Series(pd.NA, index=tabla.index, dtype="Int64")

    riesgo.loc[clase.isin(FAUNA_RIESGO_BAJO)] = 0

    vehiculo = clase.eq("vehiculo_maderero")
    trampa = clase.eq("trampa_caza")
    riesgo.loc[vehiculo & amortiguamiento & ~intangible] = 1
    riesgo.loc[vehiculo & intangible] = 2
    riesgo.loc[trampa & amortiguamiento & ~intangible] = 2
    riesgo.loc[trampa & intangible] = 3
    riesgo.loc[clase.isin(["motosierra", "arma_fuego"])] = 3

    tabla["Nivel_Riesgo"] = riesgo
    return tabla


def exportar(tabla: pd.DataFrame, ruta: Path) -> None:
    faltan = [columna for columna in COLUMNAS_SALIDA if columna not in tabla.columns]
    if faltan:
        raise ValueError(f"Faltan columnas para exportar: {faltan}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.loc[:, COLUMNAS_SALIDA].to_csv(ruta, index=False)


def main() -> None:
    anotaciones = consolidar_anotaciones(CARPETA_COCO)
    print(f"Anotaciones leídas: {len(anotaciones)}")
    anotaciones = filtrar_fauna_ajena(anotaciones)
    anotaciones = mapear_clases(anotaciones)
    anotaciones = asignar_atributos_sinteticos(anotaciones)
    tabla = calcular_relacion_espacial(anotaciones)
    tabla = asignar_nivel_riesgo(tabla)
    exportar(tabla, SALIDA)

    print(f"Filas exportadas: {len(tabla)}")
    print("Nivel_Riesgo:")
    print(tabla["Nivel_Riesgo"].value_counts(dropna=False).sort_index().to_string())
    print("clase_detectada:")
    print(tabla["clase_detectada"].value_counts(dropna=False).to_string())
    print(f"Dataset escrito en {SALIDA}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
