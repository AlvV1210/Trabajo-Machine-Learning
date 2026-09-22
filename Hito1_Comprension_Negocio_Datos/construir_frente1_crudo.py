"""Consolida el Frente 1 crudo como panel semanal por cuadrante.

Fase de comprensión de datos. No imputa nulos, no rellena ceros y no calcula
rezagos ni medias móviles. El clima de Malinowsky se repite en toda la reserva
y las semanas-cuadrante sin alertas quedan en NaN.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import box

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_CLIMA = RAIZ / "dataset diverso foco1"
RUTAS_ALERTAS = (
    CARPETA_CLIMA / "Alertas_PNCB_shape_2025" / "PNCB_ATD_UTM_02_365_S.shp",
    CARPETA_CLIMA / "Alertas_PNCB_shape_2026" / "PNCB_ATD_UTM_01_243_S.shp",
)
RUTA_RESERVA = (
    RAIZ
    / "dataset diverso foco2"
    / "ANP_ACR_ACP_ZA_SERNANP"
    / "ANP Nacional Definitivas"
    / "ANPNacionalDefinitivas.shp"
)
SALIDA = RAIZ / "data" / "dataset_frente1_crudo.csv"

FRECUENCIA = "W-MON"
CRS_METROS = "EPSG:32719"
TAMANO_M = 5_000
NOMBRE_RESERVA = "tambopata"


def leer_csv_clima(ruta: Path) -> pd.DataFrame:
    """Lee un CSV SENAMHI. Las cinco primeras líneas son metadatos, no datos."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el CSV climático: {ruta}")

    ultimo_error: Exception | None = None
    for codificacion in ("utf-8", "latin-1"):
        try:
            tabla = pd.read_csv(ruta, skiprows=5, encoding=codificacion)
            break
        except UnicodeDecodeError as exc:
            ultimo_error = exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"No se pudo interpretar el CSV climático: {ruta}") from exc
    else:
        raise ValueError(f"Codificación no reconocida en {ruta}") from ultimo_error

    if tabla.shape[1] < 5:
        raise ValueError(f"El CSV {ruta.name} no trae las columnas climáticas esperadas.")

    tabla = tabla.iloc[:, :5].copy()
    tabla.columns = ["fecha_txt", "hora", "temperatura", "precipitacion", "humedad"]
    return tabla


def concatenar_clima(carpeta: Path) -> pd.DataFrame:
    """Une los CSV mensuales de Malinowsky sin rellenar los meses ausentes."""
    rutas = sorted(carpeta.glob("Tambopata-estacion-*.csv"))
    if not rutas:
        raise FileNotFoundError(f"No hay CSV de la estación en {carpeta}")

    partes = []
    for ruta in rutas:
        try:
            partes.append(leer_csv_clima(ruta))
        except (OSError, UnicodeError, ValueError) as exc:
            raise RuntimeError(f"Fallo al leer {ruta}") from exc

    clima = pd.concat(partes, ignore_index=True)
    marca = pd.to_datetime(
        clima["fecha_txt"].astype(str).str.strip()
        + " "
        + clima["hora"].astype(str).str.strip(),
        format="%Y/%m/%d %H:%M",
        errors="coerce",
    )
    invalidas = int(marca.isna().sum())
    if invalidas:
        print(f"Registros climáticos con fecha u hora ilegible, excluidos del índice: {invalidas}")

    clima = clima.loc[marca.notna()].copy()
    clima.index = pd.DatetimeIndex(marca.loc[marca.notna()], name="fecha_hora")
    for columna in ("temperatura", "precipitacion", "humedad"):
        # S/D y cualquier texto no numérico quedan en NaN. No se rellenan.
        clima[columna] = pd.to_numeric(clima[columna], errors="coerce")
    return clima.sort_index()


def resumir_por_semana(clima: pd.DataFrame) -> pd.DataFrame:
    """Promedio semanal de temperatura y humedad; suma de precipitación.

    Las semanas sin observaciones quedan en NaN. La suma usa min_count=1 para
    no convertir una semana vacía en cero.
    """
    semanal = pd.DataFrame(
        {
            "temperatura_media": clima["temperatura"].resample(FRECUENCIA).mean(),
            "humedad_media": clima["humedad"].resample(FRECUENCIA).mean(),
            "precipitacion_suma": clima["precipitacion"].resample(FRECUENCIA).sum(min_count=1),
        }
    )
    semanal.index = semanal.index.to_period(FRECUENCIA)
    semanal.index.name = "semana"
    return semanal


def cargar_reserva(ruta: Path) -> gpd.GeoDataFrame:
    """Carga el polígono de la Reserva Nacional Tambopata."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el polígono de la reserva: {ruta}")
    try:
        capa = gpd.read_file(ruta)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Fallo al leer {ruta}") from exc

    if "anp_nomb" not in capa.columns:
        raise ValueError(f"La capa {ruta.name} no tiene la columna anp_nomb.")
    if capa.crs is None:
        capa = capa.set_crs("EPSG:4326")

    nombre = capa["anp_nomb"].astype(str).str.strip().str.casefold()
    reserva = capa.loc[nombre.eq(NOMBRE_RESERVA)].copy()
    if reserva.empty:
        raise ValueError(f"No hay polígonos de Tambopata en {ruta}")
    return reserva.to_crs(CRS_METROS)


def construir_cuadricula(reserva: gpd.GeoDataFrame, tamano_m: int = TAMANO_M) -> gpd.GeoDataFrame:
    """Celdas de tamano_m por tamano_m metros que intersectan la reserva.

    El origen es el múltiplo inferior de tamano_m en EPSG:32719. Cada celda es
    el cuadrado completo; el polígono no recorta el borde.
    """
    if reserva.crs is None or reserva.crs.to_epsg() != 32719:
        raise ValueError("La cuadrícula exige el polígono de Tambopata en EPSG:32719.")

    geometria = reserva.union_all()
    minx, miny, maxx, maxy = reserva.total_bounds
    x = math.floor(minx / tamano_m) * tamano_m
    origen_y = math.floor(miny / tamano_m) * tamano_m
    celdas = []
    while x < maxx:
        y = origen_y
        while y < maxy:
            celda = box(x, y, x + tamano_m, y + tamano_m)
            if celda.intersects(geometria):
                celdas.append({"x": x, "y": y, "geometry": celda})
            y += tamano_m
        x += tamano_m

    if not celdas:
        raise ValueError("La cuadrícula no intersecta la reserva.")

    cuadricula = gpd.GeoDataFrame(celdas, crs=CRS_METROS)
    cuadricula = cuadricula.sort_values(["y", "x"]).reset_index(drop=True)
    cuadricula["cuadrante_id"] = [f"Q{indice + 1:03d}" for indice in range(len(cuadricula))]
    return cuadricula


def parsear_fechas_alerta(fechas: pd.Series) -> pd.Series:
    """Interpreta la fecha de la alerta como día/mes/año, sin inventar valores."""
    texto = fechas.astype(str).str.strip()
    texto = texto.mask(texto.str.lower().isin({"", "none", "nan", "nat"}))
    interpretadas = pd.to_datetime(texto, format="%d/%m/%Y", errors="coerce")
    pendiente = interpretadas.isna() & texto.notna()
    if pendiente.any():
        interpretadas.loc[pendiente] = pd.to_datetime(
            texto.loc[pendiente], dayfirst=True, errors="coerce"
        )
    return interpretadas


def leer_alertas_en_caja(ruta: Path, cuadricula: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Lee solo los puntos cuya caja intersecta la cuadrícula."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el shapefile de alertas: {ruta}")
    try:
        info = pyogrio.read_info(ruta)
        caja = gpd.GeoSeries([box(*cuadricula.total_bounds)], crs=cuadricula.crs).to_crs(info["crs"])
        minx, miny, maxx, maxy = caja.total_bounds
        puntos = gpd.read_file(ruta, bbox=(minx, miny, maxx, maxy))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Fallo al leer alertas en {ruta}") from exc

    if puntos.empty:
        return puntos
    if "fecha" not in puntos.columns:
        raise ValueError(f"El shapefile {ruta.name} no tiene columna fecha.")
    return puntos


def contar_alertas_por_cuadrante(
    rutas: tuple[Path, ...], cuadricula: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, int, int]:
    """Cuenta alertas por semana y cuadrante. Cada punto queda en un solo cuadrante."""
    capas = cuadricula[["cuadrante_id", "geometry"]]
    conteos = []
    fuera = 0
    ilegibles = 0

    for ruta in rutas:
        total = int(pyogrio.read_info(ruta)["features"])
        puntos = leer_alertas_en_caja(ruta, cuadricula)
        if puntos.empty:
            fuera += total
            print(f"{ruta.name}: {total} puntos, 0 dentro de un cuadrante")
            continue

        puntos = puntos.reset_index(drop=True)
        puntos["alerta_id"] = puntos.index
        puntos = puntos.to_crs("EPSG:32719")
        unidos = gpd.sjoin(puntos, capas, how="inner", predicate="intersects")
        unidos = unidos.sort_values("cuadrante_id").drop_duplicates("alerta_id")
        dentro = int(unidos["alerta_id"].nunique())
        fuera += total - dentro

        fechas = parsear_fechas_alerta(unidos["fecha"])
        ilegibles_archivo = int(fechas.isna().sum())
        ilegibles += ilegibles_archivo
        unidos = unidos.loc[fechas.notna()].copy()
        unidos["semana"] = fechas.loc[fechas.notna()].dt.to_period(FRECUENCIA)

        conteo = unidos.groupby(["semana", "cuadrante_id"], observed=True).size()
        conteos.append(conteo.rename("n_alertas"))
        print(
            f"{ruta.name}: {total} puntos, {dentro} en algún cuadrante, "
            f"{total - dentro} fuera, {ilegibles_archivo} fechas ilegibles"
        )

    if not conteos:
        vacio = pd.DataFrame(columns=["semana", "cuadrante_id", "n_alertas"])
        return vacio, fuera, ilegibles

    total_conteo = pd.concat(conteos).groupby(level=[0, 1]).sum().reset_index()
    return total_conteo, fuera, ilegibles


def construir_panel(clima_semanal: pd.DataFrame, conteo: pd.DataFrame, cuadrantes: pd.Series) -> pd.DataFrame:
    """Producto de semanas climáticas y cuadrantes. Sin alertas queda NaN, no cero."""
    panel = pd.MultiIndex.from_product(
        [clima_semanal.index, cuadrantes],
        names=["semana", "cuadrante_id"],
    ).to_frame(index=False)
    panel = panel.merge(clima_semanal.reset_index(), on="semana", how="left")
    panel = panel.merge(conteo, on=["semana", "cuadrante_id"], how="left")
    panel["n_alertas"] = panel["n_alertas"].astype("Float64")
    return panel


def exportar(tabla: pd.DataFrame, ruta: Path) -> None:
    columnas = [
        "semana",
        "cuadrante_id",
        "temperatura_media",
        "humedad_media",
        "precipitacion_suma",
        "n_alertas",
    ]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    salida = tabla.loc[:, columnas].copy()
    salida["semana"] = salida["semana"].astype(str)
    salida.to_csv(ruta, index=False)


def main() -> None:
    clima = concatenar_clima(CARPETA_CLIMA)
    print(f"Registros horarios unidos: {len(clima)}")
    print(f"Rango climático: {clima.index.min()} -> {clima.index.max()}")

    clima_semanal = resumir_por_semana(clima)
    reserva = cargar_reserva(RUTA_RESERVA).to_crs("EPSG:32719")
    cuadricula = construir_cuadricula(reserva)
    limites = cuadricula.geometry.bounds
    ancho = (limites["maxx"] - limites["minx"]).round(6)
    alto = (limites["maxy"] - limites["miny"]).round(6)
    if not (ancho.eq(TAMANO_M).all() and alto.eq(TAMANO_M).all()):
        raise ValueError("Hay celdas que no miden 5000 x 5000 m.")
    print(f"Cuadrantes que intersectan la reserva: {len(cuadricula)}")
    print(f"CRS de la cuadrícula: {cuadricula.crs.to_string()}")
    print(f"Lado de celda (m): {TAMANO_M} x {TAMANO_M}")

    conteo, fuera, ilegibles = contar_alertas_por_cuadrante(RUTAS_ALERTAS, cuadricula)
    print(f"Alertas fuera de todo cuadrante: {fuera}")
    print(f"Fechas de alerta ilegibles, fuera del conteo: {ilegibles}")

    panel = construir_panel(clima_semanal, conteo, cuadricula["cuadrante_id"])
    exportar(panel, SALIDA)

    print(f"Filas del panel: {len(panel)}")
    print(f"Semanas: {panel['semana'].nunique()}")
    print(f"Cuadrantes: {panel['cuadrante_id'].nunique()}")
    print(f"Filas sin alertas, dejadas en NaN: {int(panel['n_alertas'].isna().sum())}")
    print(f"Dataset escrito en {SALIDA}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise

