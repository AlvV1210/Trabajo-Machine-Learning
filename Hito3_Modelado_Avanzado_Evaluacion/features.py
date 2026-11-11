"""Ingeniería de características compartida por el Hito 2 (corregido) y el Hito 3.

Frente 1: panel semanal cuadrante x semana de ``dataset_frente1_hito1_final.csv``
(273 cuadrantes x 164 semanas = 44 772 filas). Todas las variables se construyen
SOLO con información disponible al cierre de la semana t (sin fuga temporal):
rezagos, ventanas móviles hacia atrás, medias expansivas y contexto estructural.

Objetivos:
    perdida_7d  = pérdida (ha) de la semana t+1            -> horizonte 7 días
    perdida_30d = pérdida (ha) acumulada en t+1 ... t+4     -> horizonte ~30 días
    evento_7d / evento_30d = 1 si la pérdida del horizonte es > 0

Frente 2: detecciones del módulo de visión (``dataset_frente2_hito1_final.csv``)
con taxonomía normalizada y regla de riesgo extendida (Weak Supervision).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FR1_PATH = DATA_DIR / "dataset_frente1_hito1_final.csv"
FR2_PATH = DATA_DIR / "dataset_frente2_hito1_final.csv"

CLIMA = ["temperatura_media", "humedad_media", "precipitacion_suma"]
ESTATICAS = [
    "cobertura_anp", "tasa_deforestacion_ha_anio", "perdida_biomasa_t", "mb_bosque",
    "mb_bosque_inundable", "mb_agropecuario", "mb_expansion_agro", "mb_mineria",
    "mb_expansion_mineria", "mb_infraestructura", "mb_agua", "densidad_poblacional",
]
CATEGORICAS = ["vocacion_zee", "causa_principal"]
VENTANAS = [4, 8, 12, 26, 52]

# Factores de carbono (Inventario Nacional Forestal, selva baja) usados en el Hito 1.
T_C_POR_HA = 138.9
CO2_POR_C = 44 / 12


# --------------------------------------------------------------------------- #
# Frente 1
# --------------------------------------------------------------------------- #
def cargar_frente1(path: Path = FR1_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["semana_inicio"] = pd.to_datetime(df["semana"].astype(str).str.split("/").str[0])
    df = df.sort_values(["cuadrante_id", "semana_inicio"]).reset_index(drop=True)
    # n_alertas nulo = semana sin alerta ATD (ausencia de evento, no sensor apagado).
    df["alerta_observada"] = df["n_alertas"].notna().astype(int)
    df["n_alertas"] = df["n_alertas"].fillna(0)
    df["perdida_ha"] = df["perdida_ha"].fillna(0)
    # Clima: hueco de medición -> interpolación temporal por cuadrante + ffill/bfill.
    for c in CLIMA:
        df[c] = df.groupby("cuadrante_id")[c].transform(
            lambda s: s.interpolate(limit_direction="both").ffill().bfill()
        )
    return df


def _rolling_group(s: pd.Series, grupos: pd.Series, w: int, fn: str) -> pd.Series:
    r = s.groupby(grupos).rolling(w, min_periods=1)
    return getattr(r, fn)().reset_index(level=0, drop=True).sort_index()


def construir_features_frente1(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    g = df["cuadrante_id"]
    gp = df.groupby("cuadrante_id")["perdida_ha"]

    # Objetivos (futuro) ------------------------------------------------------
    df["perdida_7d"] = gp.shift(-1)
    df["perdida_30d"] = sum(gp.shift(-k) for k in range(1, 5))
    df["evento_7d"] = (df["perdida_7d"] > 0).astype(float).where(df["perdida_7d"].notna())
    df["evento_30d"] = (df["perdida_30d"] > 0).astype(float).where(df["perdida_30d"].notna())

    # Historia de la propia celda -------------------------------------------
    for k in range(4):
        df[f"perdida_lag{k}"] = gp.shift(k)
    for w in VENTANAS:
        df[f"perdida_suma_{w}s"] = _rolling_group(df["perdida_ha"], g, w, "sum")
    df["perdida_max_12s"] = _rolling_group(df["perdida_ha"], g, 12, "max")
    evento = (df["perdida_ha"] > 0).astype(float)
    df["semanas_activas_12s"] = _rolling_group(evento, g, 12, "sum")
    df["semanas_activas_52s"] = _rolling_group(evento, g, 52, "sum")
    df["perdida_media_hist"] = gp.transform(lambda s: s.expanding().mean())
    df["perdida_ewm_4"] = gp.transform(lambda s: s.ewm(halflife=4).mean())
    df["perdida_ewm_12"] = gp.transform(lambda s: s.ewm(halflife=12).mean())
    bloque = evento.groupby(g).cumsum()
    df["semanas_desde_ultima"] = df.groupby([g, bloque]).cumcount()
    df.loc[bloque == 0, "semanas_desde_ultima"] = 200  # nunca ha tenido alerta

    # Contexto regional (presión total en Tambopata) --------------------------
    reg = df.groupby("semana_inicio")["perdida_ha"].sum()
    df["region_perdida_t"] = df["semana_inicio"].map(reg)
    df["region_perdida_4s"] = df["semana_inicio"].map(reg.rolling(4, min_periods=1).sum())
    df["region_perdida_12s"] = df["semana_inicio"].map(reg.rolling(12, min_periods=1).sum())
    celdas_act = df.assign(e=evento).groupby("semana_inicio")["e"].sum()
    df["region_celdas_activas_4s"] = df["semana_inicio"].map(celdas_act.rolling(4, min_periods=1).sum())
    df["cuota_celda_52s"] = df["perdida_suma_52s"] / (
        df["semana_inicio"].map(reg.rolling(52, min_periods=1).sum()) + 1e-6
    )

    # Clima acumulado (efecto no instantáneo) ---------------------------------
    for c in CLIMA:
        df[f"{c}_lag1"] = df.groupby("cuadrante_id")[c].shift(1)
        df[f"{c}_mm3"] = _rolling_group(df[c], g, 3, "mean")
    df["precip_suma_4s"] = _rolling_group(df["precipitacion_suma"], g, 4, "sum")
    df["precip_suma_12s"] = _rolling_group(df["precipitacion_suma"], g, 12, "sum")
    df["deficit_hidrico_4s"] = (df.groupby("semana_inicio")["precip_suma_4s"].transform("mean").mean()
                                - df["precip_suma_4s"])

    # Calendario ---------------------------------------------------------------
    mes = df["semana_inicio"].dt.month
    df["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    df["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    df["temporada_seca"] = mes.isin([5, 6, 7, 8, 9, 10]).astype(int)

    # Estructurales e interacciones -------------------------------------------
    df["presion_antropica"] = df["mb_mineria"] + df["mb_agropecuario"] + df["mb_infraestructura"]
    df["en_reserva"] = (df["cobertura_anp"] > 0).astype(int)
    df["actividad_x_mineria"] = df["perdida_suma_12s"] * df["mb_mineria"]
    for c in CATEGORICAS:
        df[c] = df[c].astype("category")

    features = (
        [f"perdida_lag{k}" for k in range(4)]
        + [f"perdida_suma_{w}s" for w in VENTANAS]
        + ["perdida_max_12s", "semanas_activas_12s", "semanas_activas_52s", "perdida_media_hist",
           "perdida_ewm_4", "perdida_ewm_12", "semanas_desde_ultima", "alerta_observada",
           "region_perdida_t", "region_perdida_4s", "region_perdida_12s", "region_celdas_activas_4s",
           "cuota_celda_52s"]
        + CLIMA + [f"{c}_lag1" for c in CLIMA] + [f"{c}_mm3" for c in CLIMA]
        + ["precip_suma_4s", "precip_suma_12s", "deficit_hidrico_4s", "mes_sin", "mes_cos", "temporada_seca"]
        + ESTATICAS + ["presion_antropica", "en_reserva", "actividad_x_mineria"]
        + CATEGORICAS
    )
    for c in features:
        if c not in CATEGORICAS:
            df[c] = df[c].astype(float).fillna(0)
    return df, features


# --------------------------------------------------------------------------- #
# Frente 2
# --------------------------------------------------------------------------- #
MAPEO_CLASES = {
    "cutting-tree": "motosierra",
    "chain_saw - v1 2022-07-05 3-21pm": "motosierra",
    "knife": "arma_blanca",
    "sword": "arma_blanca",
    "hunter": "cazador",
    "person": "persona",
    "car": "vehiculo_ligero",
    "motorcycle": "vehiculo_ligero",
    "bus": "vehiculo_ligero",
    "bike": "vehiculo_ligero",
}
FAUNA = {"mono", "capibara", "snake", "rana", "guacamayo", "lobo_de_rio", "jaguar"}
AMENAZAS = {"vehiculo_maderero", "vehiculo_ligero", "persona", "trampa_caza", "cazador",
            "arma_blanca", "motosierra", "arma_fuego"}
# (fuera de zonas, amortiguamiento, intangible). Coincide con el Hito 1 donde ya
# había regla y la extiende a las clases que quedaban sin nivel.
REGLA = {
    "vehiculo_maderero": (0, 1, 2),
    "vehiculo_ligero": (0, 1, 2),
    "persona": (0, 1, 2),
    "trampa_caza": (1, 2, 3),
    "cazador": (1, 2, 3),
    "arma_blanca": (1, 2, 3),
    "motosierra": (3, 3, 3),
    "arma_fuego": (3, 3, 3),
}
NIVELES = ["Bajo", "Moderado", "Alto", "Crítico"]
MOVILES = {"persona", "cazador", "vehiculo_ligero", "vehiculo_maderero"}  # actores que se desplazan
F2_NUM = ["confidence_score", "distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m",
          "indice_estres_hidrico", "densidad_telemetria", "log_dist_carretera", "log_dist_rio",
          "hora_sin", "hora_cos", "amenazas_7d_estacion"]
F2_BOOL = ["en_zona_intangible", "en_zona_amortiguamiento", "cerca_limite_1km", "cerca_carretera_5km",
           "cerca_rio_2km", "es_amenaza", "es_noche"]
F2_CAT = ["clase"]
FR2_REALISTA = DATA_DIR / "dataset_frente2_hito3_realista.csv"


def normalizar_frente2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    clase = df["clase_detectada"].astype(str).str.strip().str.lower().replace(MAPEO_CLASES)
    df["clase"] = clase
    # Las "clases" que son texto de exportación de Roboflow no son detecciones.
    df = df[df["clase"].isin(FAUNA | AMENAZAS)].copy()
    for b in ["en_zona_intangible", "en_zona_amortiguamiento"]:
        df[b] = df[b].astype(str).str.lower().isin(["true", "1"]).astype(int)
    return df


def asignar_riesgo(df: pd.DataFrame) -> pd.Series:
    """Regla de negocio (Weak Supervision).

    Nivel base = clase x zona (tabla REGLA, igual que en el Hito 1). Si el nivel base es
    al menos Moderado, sube un nivel por cada agravante (máximo Crítico):
      * actor móvil (persona, cazador, vehículo) detectado de noche (19:00-05:00);
      * reincidencia: 2 o más amenazas en la misma estación en los 7 días previos.
    Sin las columnas `es_noche` / `amenazas_7d_estacion` se aplica solo el nivel base.
    """
    zona = np.where(df["en_zona_intangible"] == 1, 2, np.where(df["en_zona_amortiguamiento"] == 1, 1, 0))
    nivel = pd.Series(0, index=df.index, dtype=int)
    for clase, niveles in REGLA.items():
        m = (df["clase"] == clase).to_numpy()
        nivel[m] = np.array(niveles)[zona[m]]
    noche = df["es_noche"].to_numpy() if "es_noche" in df else np.zeros(len(df), dtype=int)
    rein = df["amenazas_7d_estacion"].to_numpy() if "amenazas_7d_estacion" in df else np.zeros(len(df), dtype=int)
    agrava = ((noche == 1) & df["clase"].isin(MOVILES).to_numpy()).astype(int) + (rein >= 2).astype(int)
    return (nivel + np.where(nivel >= 1, agrava, 0)).clip(upper=3).astype(int)


def construir_features_frente2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["confidence_score", "distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    if "hora" not in df:
        df["hora"] = 12
    if "es_noche" not in df:
        df["es_noche"] = ((df["hora"] >= 19) | (df["hora"] < 5)).astype(int)
    if "amenazas_7d_estacion" not in df:
        df["amenazas_7d_estacion"] = 0
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["indice_estres_hidrico"] = df["distancia_a_rio_m"] / 1000.0
    df["densidad_telemetria"] = df["confidence_score"] / (df["distancia_limite_m"] + 1.0)
    df["log_dist_carretera"] = np.log1p(df["distancia_a_carretera_m"])
    df["log_dist_rio"] = np.log1p(df["distancia_a_rio_m"])
    df["cerca_limite_1km"] = (df["distancia_limite_m"] <= 1000).astype(int)
    df["cerca_carretera_5km"] = (df["distancia_a_carretera_m"] <= 5000).astype(int)
    df["cerca_rio_2km"] = (df["distancia_a_rio_m"] <= 2000).astype(int)
    df["es_amenaza"] = df["clase"].isin(AMENAZAS).astype(int)
    return df


def cargar_frente2(path: Path | None = None) -> pd.DataFrame:
    """Telemetría realista del Hito 3 (si existe) o, en su defecto, el dataset del Hito 1."""
    path = path or (FR2_REALISTA if FR2_REALISTA.exists() else FR2_PATH)
    df = pd.read_csv(path)
    if "clase" not in df.columns:
        df = normalizar_frente2(df)
    if "fecha_hora" in df.columns:
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df["Nivel_Riesgo_v2"] = df["Nivel_Riesgo"].astype(int) if "es_noche" in df.columns else asignar_riesgo(df)
    return construir_features_frente2(df).reset_index(drop=True)


CLASES_F2 = sorted(FAUNA | AMENAZAS)
F2_VARIABLES = F2_NUM + F2_BOOL + F2_CAT          # variables "crudas" (para interpretar)
F2_COLUMNAS = F2_NUM + F2_BOOL + [f"clase_{c}" for c in CLASES_F2]   # matriz que recibe el modelo


def matriz_frente2(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz numérica con columnas fijas (one-hot de `clase`). Es la misma en entrenamiento
    e inferencia, de modo que el modelo guardado se puede usar sin objetos auxiliares."""
    X = df[F2_NUM + F2_BOOL].astype(float).copy()
    for c in CLASES_F2:
        X[f"clase_{c}"] = (df["clase"].to_numpy() == c).astype(float)
    return X[F2_COLUMNAS]
