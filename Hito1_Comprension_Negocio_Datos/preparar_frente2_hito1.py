"""Define el target del Frente 2 para el Hito 1.

Quita el texto basura de Roboflow, traduce la fauna africana a nombres
amazónicos y duplica las clases Moderado y Alto hasta una base mínima.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "data" / "dataset_frente2_crudo.csv"
SALIDA = RAIZ / "data" / "dataset_frente2_hito1.csv"

SEMILLA = 42
OBJETIVO_CLASE = 180
RUIDO_GRADOS = 0.0005
LARGO_MAXIMO = 40
PATRONES_RUIDO = (
    "roboflow",
    "collaborate",
    "export",
    "annotate",
    "dataset",
    "computer vision",
    "organize",
)
MAPA_FAUNA = {
    "Lion": "jaguar",
    "Female Lion": "jaguar",
    "Male Lion": "jaguar",
    "Leopard": "jaguar",
    "Elephant": "tapir",
    "Monkey": "mono",
    "Hyena": "fauna_silvestre",
    "Jackal": "fauna_silvestre",
    "Honey Badger": "fauna_silvestre",
    "Warthog": "fauna_silvestre",
    "Impala": "fauna_silvestre",
    "Sable": "fauna_silvestre",
    "Waterbuck": "fauna_silvestre",
    "Wild Dog": "fauna_silvestre",
}


def cargar_extraccion(ruta: Path) -> pd.DataFrame:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró la extracción del Frente 2: {ruta}")
    try:
        tabla = pd.read_csv(ruta)
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Fallo al leer {ruta}") from exc

    requeridas = {"clase_detectada", "latitud", "longitud", "Nivel_Riesgo"}
    faltan = requeridas.difference(tabla.columns)
    if faltan:
        raise ValueError(f"A la extracción le faltan columnas: {sorted(faltan)}")
    return tabla


def es_ruido(clase: object) -> bool:
    if pd.isna(clase):
        return False
    texto = str(clase).strip()
    if len(texto) > LARGO_MAXIMO:
        return True
    texto_bajo = texto.casefold()
    return any(patron in texto_bajo for patron in PATRONES_RUIDO)


def limpiar_y_mapear(tabla: pd.DataFrame) -> pd.DataFrame:
    clase = tabla["clase_detectada"]
    ruido = clase.map(es_ruido)
    print(f"Filas de ruido eliminadas: {int(ruido.sum())}")
    limpia = tabla.loc[~ruido].copy()
    limpia["clase_detectada"] = limpia["clase_detectada"].replace(MAPA_FAUNA)

    sin_target = limpia["Nivel_Riesgo"].isna()
    print(f"Filas sin Nivel_Riesgo eliminadas: {int(sin_target.sum())}")
    limpia = limpia.loc[~sin_target].copy()
    limpia["Nivel_Riesgo"] = limpia["Nivel_Riesgo"].astype("int64")
    return limpia.reset_index(drop=True)


def balancear_clase(tabla: pd.DataFrame, clase: int, generador: np.random.Generator) -> pd.DataFrame:
    """Conserva los originales y añade copias con un desplazamiento mínimo de coordenadas."""
    base = tabla.loc[tabla["Nivel_Riesgo"].eq(clase)]
    faltan = OBJETIVO_CLASE - len(base)
    if faltan <= 0:
        print(f"Clase {clase}: {len(base)} filas, no hizo falta duplicar.")
        return tabla

    elegidas = generador.choice(base.index.to_numpy(), size=faltan, replace=True)
    copias = tabla.loc[elegidas].copy()
    copias["latitud"] = copias["latitud"].to_numpy() + generador.normal(0, RUIDO_GRADOS, size=faltan)
    copias["longitud"] = copias["longitud"].to_numpy() + generador.normal(0, RUIDO_GRADOS, size=faltan)
    print(f"Clase {clase}: {len(base)} originales, {faltan} copias, total {OBJETIVO_CLASE}")
    return pd.concat([tabla, copias], ignore_index=True)


def exportar(tabla: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)


def main() -> None:
    extraccion = cargar_extraccion(ENTRADA)
    tabla = limpiar_y_mapear(extraccion)
    generador = np.random.default_rng(SEMILLA)
    for clase in (1, 2):
        tabla = balancear_clase(tabla, clase, generador)

    if tabla["Nivel_Riesgo"].isna().any():
        raise ValueError("El target Nivel_Riesgo todavía tiene nulos.")

    exportar(tabla, SALIDA)
    print(f"Filas exportadas: {len(tabla)}")
    print(tabla["Nivel_Riesgo"].value_counts().sort_index().to_string())
    print("Clases detectadas:")
    print(tabla["clase_detectada"].value_counts().to_string())
    print(f"Dataset escrito en {SALIDA}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
