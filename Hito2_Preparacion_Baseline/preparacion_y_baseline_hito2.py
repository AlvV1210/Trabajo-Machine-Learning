"""Hito 2 - Preparación de datos (CRISP-DM fase 3) y baseline (fase 4 inicial).

BIODIVERSITY-GUARD Predict.

Frente 1 - Alerta temprana de deforestación (7 y 30 días)
    Entrada: data/dataset_frente1_hito1_final.csv (panel semana x cuadrante, 273 cuadrantes).
    Para cada cuadrante y semana t se predice, con información disponible al cierre de t:
      * evento_7d / evento_30d : ¿habrá pérdida de cobertura en las próximas 1 / 4 semanas?
      * perdida_7d / perdida_30d: hectáreas que se perderán en ese horizonte.
    El horizonte de 30 días se implementa como 4 semanas completas (28 días), porque el panel
    es semanal.

Frente 2 - Nivel de riesgo de incursión ilegal (Bajo, Moderado, Alto, Crítico)
    Entrada: data/dataset_frente2_hito1_final.csv.

Decisiones de rigor metodológico (evitan métricas infladas):
  * n_alertas NO es variable de entrada: perdida_ha = n_alertas x 0.09, sería fuga del target.
  * Las variables se calculan solo con el pasado (rezagos, sumas móviles) y la validación es
    temporal: se entrena con semanas antiguas y se evalúa en semanas posteriores, dejando un
    hueco (purga) igual al horizonte para que ningún target de entrenamiento mire al periodo de prueba.
  * Se excluyen las semanas posteriores al 31/08/2026: el shapefile de alertas PNCB 2026 cubre
    hasta el día 243 del año, así que las semanas siguientes valen cero por falta de datos y no
    por ausencia de deforestación.
  * El umbral de alerta se elige en una validación interna dentro del entrenamiento, nunca en test.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEMILLA = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

FR1_IN = DATA_DIR / "dataset_frente1_hito1_final.csv"
FR2_IN = DATA_DIR / "dataset_frente2_hito1_final.csv"
FR1_OUT = DATA_DIR / "dataset_frente1_hito2_preparado.csv"
FR2_OUT = DATA_DIR / "dataset_frente2_hito2_preparado.csv"

# Última semana completamente cubierta por las alertas ATD (PNCB_ATD_UTM_01_243_S = días 1-243 de 2026).
ULTIMA_SEMANA_CON_ALERTAS = pd.Timestamp("2026-08-25")
FRACCION_TRAIN = 0.70  # primeras 70 % semanas para entrenamiento, resto para prueba
RECALL_OBJETIVO = 0.80  # el umbral de alerta busca captar al menos 80 % de los eventos

# Conversión a biomasa y CO2 (mismos factores del diccionario de datos del Hito 1):
# 138.9 t C/ha (Inventario Nacional Forestal, selva baja), fracción de carbono 0.47, CO2/C = 44/12.
BIOMASA_T_POR_HA = 138.9 / 0.47
CO2_T_POR_HA = 138.9 * 44 / 12

HORIZONTES = {"7d": 1, "30d": 4}  # semanas hacia adelante
ETIQUETAS_RIESGO = ["Bajo", "Moderado", "Alto", "Crítico"]

VARIABLES_ESTATICAS_F1 = [
    "cobertura_anp",
    "tasa_deforestacion_ha_anio",
    "mb_bosque",
    "mb_bosque_inundable",
    "mb_agropecuario",
    "mb_expansion_agro",
    "mb_mineria",
    "mb_infraestructura",
    "mb_agua",
    "densidad_poblacional",
]
CATEGORICAS_F1 = ["vocacion_zee", "causa_principal"]


def crear_directorios() -> None:
    for carpeta in (DATA_DIR, REPORTS_DIR, MODELS_DIR):
        carpeta.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# FRENTE 1 - Preparación
# ---------------------------------------------------------------------------
def _semanas_desde_ultima_alerta(serie: pd.Series) -> pd.Series:
    """Semanas transcurridas desde la última semana con pérdida (> 0). 200 si nunca hubo."""
    valores = serie.to_numpy()
    salida = np.empty(len(valores))
    ultima = None
    for i, valor in enumerate(valores):
        if valor > 0:
            ultima = i
        salida[i] = (i - ultima) if ultima is not None else 200
    return pd.Series(np.minimum(salida, 200), index=serie.index)


def preparar_frente1() -> tuple[pd.DataFrame, list[str], dict]:
    df = pd.read_csv(FR1_IN)
    filas_iniciales = len(df)
    print(f"Frente 1: {filas_iniciales} filas iniciales ({FR1_IN.name})")

    df["fecha"] = pd.to_datetime(df["semana"].astype(str).str.split("/").str[0], errors="coerce")
    semanas_sin_cobertura = int(df.loc[df["fecha"] > ULTIMA_SEMANA_CON_ALERTAS, "semana"].nunique())
    df = df[df["fecha"] <= ULTIMA_SEMANA_CON_ALERTAS].copy()
    df = df.sort_values(["cuadrante_id", "fecha"]).reset_index(drop=True)

    # --- Limpieza -----------------------------------------------------------
    # n_alertas nulo = semana sin alerta ATD (ausencia de evento, no sensor apagado).
    df["n_alertas"] = pd.to_numeric(df["n_alertas"], errors="coerce").fillna(0)
    df["perdida_ha"] = pd.to_numeric(df["perdida_ha"], errors="coerce").fillna(0)

    # Clima: interpolación lineal en el tiempo por cuadrante y relleno adelante/atrás
    # (respeta la estacionalidad; una media global la destruiría).
    nulos_clima = {}
    for col in ["temperatura_media", "humedad_media", "precipitacion_suma"]:
        nulos_clima[col] = int(df[col].isna().sum())
        df[col] = df.groupby("cuadrante_id")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        df[col] = df[col].fillna(df[col].median())

    # --- Feature engineering (solo pasado y presente de la semana t) -----------
    g = df.groupby("cuadrante_id")["perdida_ha"]
    df["perdida_lag1"] = g.shift(1).fillna(0)
    df["perdida_lag2"] = g.shift(2).fillna(0)
    for ventana in (4, 8, 12, 26, 52):
        df[f"perdida_suma_{ventana}s"] = g.transform(lambda s, w=ventana: s.rolling(w, min_periods=1).sum())
        df[f"semanas_con_alerta_{ventana}s"] = g.transform(
            lambda s, w=ventana: (s > 0).rolling(w, min_periods=1).sum()
        )
    df["perdida_ewm_4s"] = g.transform(lambda s: s.ewm(halflife=4).mean())
    semanas_observadas = df.groupby("cuadrante_id").cumcount() + 1
    df["perdida_media_historica"] = g.cumsum() / semanas_observadas
    df["frecuencia_alerta_historica"] = (
        df.groupby("cuadrante_id")["perdida_ha"].transform(lambda s: (s > 0).cumsum()) / semanas_observadas
    )
    df["semanas_desde_ultima_alerta"] = g.transform(_semanas_desde_ultima_alerta)

    # Nota: se probaron variables de contexto de toda la reserva (pérdida total y cuadrantes activos
    # recientes). En la validación interna no mejoraron el AUC y el modelo las usaba para "memorizar"
    # el periodo temporal, así que se descartaron del baseline.

    # Clima acumulado: el efecto de una sequía se acumula (medias móviles y rezagos).
    gc = df.groupby("cuadrante_id")
    df["temperatura_lag1"] = gc["temperatura_media"].shift(1)
    df["precipitacion_lag1"] = gc["precipitacion_suma"].shift(1)
    for ventana in (4, 12):
        df[f"temperatura_media_{ventana}s"] = gc["temperatura_media"].transform(
            lambda s, w=ventana: s.rolling(w, min_periods=1).mean()
        )
        df[f"humedad_media_{ventana}s"] = gc["humedad_media"].transform(
            lambda s, w=ventana: s.rolling(w, min_periods=1).mean()
        )
        df[f"precipitacion_suma_{ventana}s"] = gc["precipitacion_suma"].transform(
            lambda s, w=ventana: s.rolling(w, min_periods=1).sum()
        )
    df["temperatura_lag1"] = df["temperatura_lag1"].fillna(df["temperatura_media"])
    df["precipitacion_lag1"] = df["precipitacion_lag1"].fillna(df["precipitacion_suma"])

    # Estacionalidad
    df["mes"] = df["fecha"].dt.month
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["temporada_seca"] = df["mes"].isin([5, 6, 7, 8, 9, 10]).astype(int)

    # Codificación one-hot de variables categóricas (se elimina la primera para evitar multicolinealidad)
    dummies = pd.get_dummies(df[CATEGORICAS_F1], prefix=["zee", "causa"], drop_first=True, dtype=int)
    dummies.columns = [c.replace(" ", "_").replace("/", "_") for c in dummies.columns]
    df = pd.concat([df, dummies], axis=1)

    # --- Targets futuros --------------------------------------------------------
    g = df.groupby("cuadrante_id")["perdida_ha"]
    df["perdida_7d"] = g.shift(-1)
    df["perdida_30d"] = sum(g.shift(-k) for k in range(1, 5))
    df["evento_7d"] = (df["perdida_7d"] > 0).astype(float).where(df["perdida_7d"].notna())
    df["evento_30d"] = (df["perdida_30d"] > 0).astype(float).where(df["perdida_30d"].notna())

    features = (
        [
            "perdida_ha",
            "perdida_lag1",
            "perdida_lag2",
            *[f"perdida_suma_{w}s" for w in (4, 8, 12, 26, 52)],
            *[f"semanas_con_alerta_{w}s" for w in (4, 8, 12, 26, 52)],
            "perdida_ewm_4s",
            "perdida_media_historica",
            "frecuencia_alerta_historica",
            "semanas_desde_ultima_alerta",
            "temperatura_lag1",
            "precipitacion_lag1",
            *[f"{v}_{w}s" for w in (4, 12) for v in ("temperatura_media", "humedad_media", "precipitacion_suma")],
            "mes_sin",
            "mes_cos",
            "temporada_seca",
        ]
        + VARIABLES_ESTATICAS_F1
        + list(dummies.columns)
    )

    columnas_salida = ["semana", "fecha", "cuadrante_id"] + features + [
        "perdida_7d",
        "perdida_30d",
        "evento_7d",
        "evento_30d",
    ]
    preparado = df[columnas_salida].copy()
    preparado.to_csv(FR1_OUT, index=False)

    resumen = {
        "archivo_entrada": FR1_IN.name,
        "filas_iniciales": int(filas_iniciales),
        "filas_preparadas": int(len(preparado)),
        "semanas": int(preparado["semana"].nunique()),
        "cuadrantes": int(preparado["cuadrante_id"].nunique()),
        "semanas_excluidas_sin_cobertura_satelital": semanas_sin_cobertura,
        "nulos_clima_imputados": nulos_clima,
        "n_features": len(features),
        "perdida_media_ha": float(df["perdida_ha"].mean()),
        "perdida_maxima_ha": float(df["perdida_ha"].max()),
        "porcentaje_semanas_con_perdida": float((df["perdida_ha"] > 0).mean() * 100),
        "variables_excluidas": {
            "n_alertas": "fuga: perdida_ha = n_alertas x 0.09",
            "perdida_biomasa_t, emisiones_co2_t": "transformaciones lineales de tasa_deforestacion_ha_anio (r = 1.0)",
            "mb_expansion_mineria": "prácticamente idéntica a mb_mineria (dif. máx. 0.005)",
        },
    }
    return preparado, features, resumen


# ---------------------------------------------------------------------------
# FRENTE 1 - Baseline (Random Forest) con validación temporal
# ---------------------------------------------------------------------------
def particion_temporal(df: pd.DataFrame, pasos: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Train: semanas anteriores al corte menos una purga de `pasos` semanas. Test: desde el corte."""
    fechas = np.sort(df["fecha"].unique())
    i_corte = int(len(fechas) * FRACCION_TRAIN)
    corte = pd.Timestamp(fechas[i_corte])
    limite_train = pd.Timestamp(fechas[i_corte - pasos])
    return df[df["fecha"] < limite_train], df[df["fecha"] >= corte], corte


def elegir_umbral(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Máximo F1 entre los umbrales que alcanzan el recall objetivo (si ninguno, máximo F1)."""
    precision, recall, umbrales = precision_recall_curve(y_true, prob)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    candidatos = np.where(recall >= RECALL_OBJETIVO)[0]
    indice = candidatos[np.argmax(f1[candidatos])] if len(candidatos) else int(np.argmax(f1))
    return float(umbrales[indice])


def captura_hectareas(fechas: pd.Series, puntaje: np.ndarray, perdida_real: pd.Series, fraccion: float) -> float:
    """% de las hectáreas realmente perdidas que caen en el top `fraccion` de cuadrantes priorizados cada semana."""
    tabla = pd.DataFrame({"fecha": fechas.values, "p": puntaje, "y": perdida_real.values})
    tabla["rango"] = tabla.groupby("fecha")["p"].rank(ascending=False, pct=True, method="first")
    return float(tabla.loc[tabla["rango"] <= fraccion, "y"].sum() / max(tabla["y"].sum(), 1e-9) * 100)


def nuevo_rf_clasificador() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=100,  # hojas grandes: el target es ruidoso y con pocos eventos
        max_features=0.2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEMILLA,
    )


def nuevo_rf_regresor() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=50,
        max_features=0.3,
        n_jobs=-1,
        random_state=SEMILLA,
    )


def evaluar_frente1(df: pd.DataFrame, features: list[str]) -> tuple[dict, dict]:
    metricas: dict = {}
    artefactos: dict = {}
    for nombre, pasos in HORIZONTES.items():
        y_evento, y_ha = f"evento_{nombre}", f"perdida_{nombre}"
        datos = df.dropna(subset=[y_ha])
        train, test, corte = particion_temporal(datos, pasos)

        # 1) Umbral elegido en validación interna (último 20 % de semanas del train, con purga)
        fechas_train = np.sort(train["fecha"].unique())
        i_val = int(len(fechas_train) * 0.8)
        sub_train = train[train["fecha"] < fechas_train[i_val - pasos]]
        validacion = train[train["fecha"] >= fechas_train[i_val]]
        modelo_val = nuevo_rf_clasificador().fit(sub_train[features], sub_train[y_evento])
        umbral = elegir_umbral(validacion[y_evento].to_numpy(), modelo_val.predict_proba(validacion[features])[:, 1])

        # 2) Modelo de riesgo (clasificación) entrenado con todo el train
        clf = nuevo_rf_clasificador().fit(train[features], train[y_evento])
        prob = clf.predict_proba(test[features])[:, 1]
        pred = (prob >= umbral).astype(int)
        y_true = test[y_evento].astype(int).to_numpy()

        # Líneas base ingenuas (referencia obligatoria para saber si el modelo aporta):
        #  * regla binaria: "alerta si el cuadrante tuvo pérdida en las últimas 4 semanas"
        #  * puntaje de persistencia: ordenar cuadrantes por la pérdida de las últimas 12 semanas
        regla = (test["perdida_suma_4s"] > 0).astype(int).to_numpy()
        puntaje_persistencia = (
            test["perdida_suma_12s"].to_numpy() + 1e-6 * test["perdida_media_historica"].to_numpy()
        )

        # 3) Modelo de magnitud (hectáreas)
        reg = nuevo_rf_regresor().fit(train[features], train[y_ha])
        ha_pred = np.clip(reg.predict(test[features]), 0, None)
        ha_real = test[y_ha]
        # Línea base ingenua: persistencia del ritmo de las últimas 12 semanas
        ha_naive = test["perdida_suma_12s"] / 12 * pasos

        por_cuadrante = pd.DataFrame({"q": test["cuadrante_id"], "y": ha_real, "p": ha_pred}).groupby("q").sum()

        metricas[nombre] = {
            "periodo_test_desde": str(corte.date()),
            "filas_train": int(len(train)),
            "filas_test": int(len(test)),
            "tasa_eventos_test": float(y_true.mean()),
            "riesgo": {
                "roc_auc": float(roc_auc_score(y_true, prob)),
                "pr_auc": float(average_precision_score(y_true, prob)),
                "umbral": umbral,
                "recall": float(recall_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred)),
                "accuracy": float(accuracy_score(y_true, pred)),
                "captura_ha_top10_pct": captura_hectareas(test["fecha"], prob, ha_real, 0.10),
                "captura_ha_top20_pct": captura_hectareas(test["fecha"], prob, ha_real, 0.20),
                "captura_ha_top30_pct": captura_hectareas(test["fecha"], prob, ha_real, 0.30),
                "ha_en_cuadrantes_alertados_pct": float(ha_real[pred == 1].sum() / max(ha_real.sum(), 1e-9) * 100),
                "naive_recall": float(recall_score(y_true, regla)),
                "naive_precision": float(precision_score(y_true, regla, zero_division=0)),
                "naive_f1": float(f1_score(y_true, regla)),
                "persistencia_roc_auc": float(roc_auc_score(y_true, puntaje_persistencia)),
                "persistencia_captura_ha_top20_pct": captura_hectareas(
                    test["fecha"], puntaje_persistencia, ha_real, 0.20
                ),
            },
            "magnitud": {
                "mae_ha": float(mean_absolute_error(ha_real, ha_pred)),
                "rmse_ha": float(np.sqrt(mean_squared_error(ha_real, ha_pred))),
                "r2_cuadrante_semana": float(r2_score(ha_real, ha_pred)),
                "naive_mae_ha": float(mean_absolute_error(ha_real, ha_naive)),
                "naive_rmse_ha": float(np.sqrt(mean_squared_error(ha_real, ha_naive))),
                "spearman_ranking_espacial": float(spearmanr(por_cuadrante["y"], por_cuadrante["p"]).statistic),
                "ha_reales_test": float(ha_real.sum()),
                "ha_predichas_test": float(ha_pred.sum()),
            },
        }
        artefactos[nombre] = {
            "clf": clf,
            "reg": reg,
            "umbral": umbral,
            "test": test[["semana", "fecha", "cuadrante_id"]].assign(
                prob=prob, alerta=pred, ha_real=ha_real.values, ha_pred=ha_pred, ha_naive=ha_naive.values,
                persistencia=puntaje_persistencia,
            ),
            "y_true": y_true,
            "prob": prob,
            "importancias": pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False),
        }
        r, m = metricas[nombre]["riesgo"], metricas[nombre]["magnitud"]
        print(
            f"  [{nombre}] AUC={r['roc_auc']:.3f} Recall={r['recall']:.3f} Precision={r['precision']:.3f} "
            f"F1={r['f1']:.3f} Captura top20%={r['captura_ha_top20_pct']:.1f}% (persistencia AUC={r['persistencia_roc_auc']:.3f}) | "
            f"MAE={m['mae_ha']:.3f} (naive {m['naive_mae_ha']:.3f}) RMSE={m['rmse_ha']:.3f} "
            f"(naive {m['naive_rmse_ha']:.3f}) Spearman espacial={m['spearman_ranking_espacial']:.3f}"
        )
    return metricas, artefactos


def ranking_ultima_semana(df: pd.DataFrame, features: list[str], artefactos: dict) -> pd.DataFrame:
    """Predicción operativa: riesgo a 7 y 30 días para cada cuadrante desde la última semana observada."""
    ultima = df[df["fecha"] == df["fecha"].max()].copy()
    salida = ultima[["semana", "cuadrante_id"]].copy()
    for nombre in HORIZONTES:
        art = artefactos[nombre]
        salida[f"prob_{nombre}"] = art["clf"].predict_proba(ultima[features])[:, 1]
        salida[f"alerta_{nombre}"] = (salida[f"prob_{nombre}"] >= art["umbral"]).astype(int)
        salida[f"ha_riesgo_{nombre}"] = np.clip(art["reg"].predict(ultima[features]), 0, None)
    salida["biomasa_riesgo_30d_t"] = salida["ha_riesgo_30d"] * BIOMASA_T_POR_HA
    salida["co2_riesgo_30d_t"] = salida["ha_riesgo_30d"] * CO2_T_POR_HA
    salida["nivel_alerta"] = pd.cut(
        salida["prob_30d"], bins=[-0.01, 0.25, 0.5, 0.75, 1.01], labels=["Baja", "Media", "Alta", "Muy alta"]
    ).astype(str)
    salida = salida.sort_values("prob_30d", ascending=False)
    salida.to_csv(REPORTS_DIR / "hito2_ranking_cuadrantes.csv", index=False)
    return salida


# ---------------------------------------------------------------------------
# FRENTE 2 - Preparación
# ---------------------------------------------------------------------------
NUMERICAS_F2 = [
    "confidence_score",
    "en_zona_intangible",
    "en_zona_amortiguamiento",
    "cerca_limite",
    "cerca_carretera",
    "cerca_rio",
    "distancia_limite_m",
    "distancia_a_carretera_m",
    "distancia_a_rio_m",
    "indice_estres_hidrico",
    "densidad_telemetria_anomala",
]
CATEGORICAS_F2 = ["clase_detectada", "fuente"]


def preparar_frente2() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(FR2_IN)
    filas_iniciales = len(df)
    print(f"Frente 2: {filas_iniciales} filas iniciales ({FR2_IN.name})")

    df["clase_detectada"] = df["clase_detectada"].fillna("desconocida")
    df["fuente"] = df["fuente"].fillna("desconocida")
    for col in ["confidence_score", "distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["en_zona_intangible"] = df["en_zona_intangible"].astype(bool).astype(int)
    df["en_zona_amortiguamiento"] = df["en_zona_amortiguamiento"].astype(bool).astype(int)
    df["cerca_limite"] = (df["distancia_limite_m"] <= 1000).astype(int)
    df["cerca_carretera"] = (df["distancia_a_carretera_m"] <= 5000).astype(int)
    df["cerca_rio"] = (df["distancia_a_rio_m"] <= 5000).astype(int)
    # Variables derivadas descritas en el informe
    df["indice_estres_hidrico"] = df["distancia_a_rio_m"] / 1000
    df["densidad_telemetria_anomala"] = df["confidence_score"] / (df["distancia_limite_m"] + 1)
    # Grupo = imagen dentro de su fuente: anotaciones de la misma imagen no se reparten entre train y test
    df["grupo_imagen"] = df["fuente"].astype(str) + "::" + df["image_id"].astype(str)

    sin_etiqueta = int(df["Nivel_Riesgo"].isna().sum())
    df = df.dropna(subset=["Nivel_Riesgo"]).copy()
    df["Nivel_Riesgo"] = df["Nivel_Riesgo"].astype(int)

    preparado = df[["grupo_imagen"] + NUMERICAS_F2 + CATEGORICAS_F2 + ["Nivel_Riesgo"]].copy()
    preparado.to_csv(FR2_OUT, index=False)

    resumen = {
        "archivo_entrada": FR2_IN.name,
        "filas_iniciales": int(filas_iniciales),
        "filas_sin_regla_de_riesgo_excluidas": sin_etiqueta,
        "filas_preparadas": int(len(preparado)),
        "clases": {ETIQUETAS_RIESGO[k]: int(v) for k, v in df["Nivel_Riesgo"].value_counts().sort_index().items()},
    }
    return preparado, resumen


def nuevo_pipeline_frente2() -> ImbPipeline:
    preprocesador = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAS_F2),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS_F2),
        ]
    )
    return ImbPipeline(
        [
            ("preprocesador", preprocesador),
            ("smote", SMOTE(random_state=SEMILLA)),  # solo se aplica sobre el train de cada fold
            (
                "modelo",
                RandomForestClassifier(
                    n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEMILLA
                ),
            ),
        ]
    )


def evaluar_frente2(df: pd.DataFrame) -> tuple[dict, dict]:
    X = df[NUMERICAS_F2 + CATEGORICAS_F2]
    y = df["Nivel_Riesgo"].to_numpy()
    grupos = df["grupo_imagen"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    folds = []
    y_oof = np.zeros_like(y)
    prob_oof = np.zeros((len(y), 4))
    for train_idx, test_idx in cv.split(X, y, grupos):
        modelo = nuevo_pipeline_frente2().fit(X.iloc[train_idx], y[train_idx])
        pred = modelo.predict(X.iloc[test_idx])
        y_oof[test_idx] = pred
        prob_oof[test_idx] = modelo.predict_proba(X.iloc[test_idx])
        folds.append(
            {
                "accuracy": accuracy_score(y[test_idx], pred),
                "f1_macro": f1_score(y[test_idx], pred, average="macro"),
                "recall_critico": recall_score(y[test_idx], pred, labels=[3], average="macro"),
            }
        )
    folds_df = pd.DataFrame(folds)
    reporte = classification_report(y, y_oof, target_names=ETIQUETAS_RIESGO, output_dict=True, zero_division=0)
    metricas = {
        "validacion": "StratifiedGroupKFold (5 folds, agrupado por imagen)",
        "accuracy": float(accuracy_score(y, y_oof)),
        "f1_macro": float(f1_score(y, y_oof, average="macro")),
        "f1_weighted": float(f1_score(y, y_oof, average="weighted")),
        "roc_auc_ovr_macro": float(roc_auc_score(y, prob_oof, multi_class="ovr", average="macro")),
        "recall_por_clase": {k: float(reporte[k]["recall"]) for k in ETIQUETAS_RIESGO},
        "precision_por_clase": {k: float(reporte[k]["precision"]) for k in ETIQUETAS_RIESGO},
        "f1_macro_std_folds": float(folds_df["f1_macro"].std()),
        "recall_critico_min_folds": float(folds_df["recall_critico"].min()),
        "classification_report": reporte,
    }
    modelo_final = nuevo_pipeline_frente2().fit(X, y)
    print(
        f"  Accuracy={metricas['accuracy']:.4f} F1 macro={metricas['f1_macro']:.4f} "
        f"Recall Crítico={metricas['recall_por_clase']['Crítico']:.4f} (CV 5 folds agrupada)"
    )
    return metricas, {"modelo": modelo_final, "y": y, "y_pred": y_oof}


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
COLOR_7D, COLOR_30D, COLOR_GRIS = "#2a78b8", "#c2551b", "#8a8f98"


def figuras_frente1(art: dict) -> None:
    # ROC
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for nombre, color in (("7d", COLOR_7D), ("30d", COLOR_30D)):
        fpr, tpr, _ = roc_curve(art[nombre]["y_true"], art[nombre]["prob"])
        auc = roc_auc_score(art[nombre]["y_true"], art[nombre]["prob"])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"Horizonte {nombre} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=COLOR_GRIS, lw=1, label="Azar")
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos (Recall)",
           title="Frente 1 - Curva ROC del riesgo de deforestación")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "hito2_f1_roc.png", dpi=200)
    plt.close(fig)

    # Curva de captura: % de cuadrantes patrullados vs % de hectáreas perdidas cubiertas
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fracciones = np.linspace(0.01, 1, 100)
    for nombre, color in (("7d", COLOR_7D), ("30d", COLOR_30D)):
        t = art[nombre]["test"]
        curva = [captura_hectareas(t["fecha"], t["prob"].to_numpy(), t["ha_real"], f) for f in fracciones]
        ax.plot(fracciones * 100, curva, color=color, lw=2, label=f"Modelo RF {nombre}")
    t = art["30d"]["test"]
    curva_p = [captura_hectareas(t["fecha"], t["persistencia"].to_numpy(), t["ha_real"], f) for f in fracciones]
    ax.plot(fracciones * 100, curva_p, color="#555", lw=1.2, ls="-.", label="Persistencia 12 sem. (30d)")
    ax.plot([0, 100], [0, 100], ls="--", color=COLOR_GRIS, lw=1, label="Patrullaje aleatorio")
    ax.axvline(20, color="#444", lw=0.8, ls=":")
    ax.set(xlabel="% de cuadrantes priorizados por semana", ylabel="% de hectáreas perdidas cubiertas",
           title="Frente 1 - Captura de deforestación al priorizar cuadrantes")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "hito2_f1_captura.png", dpi=200)
    plt.close(fig)

    # Serie temporal: pérdida total a 30 días real vs estimada en el periodo de prueba
    t = art["30d"]["test"].groupby("fecha")[["ha_real", "ha_pred", "ha_naive"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t.index, t["ha_real"], color="#333", lw=1.8, label="Real (próximos 28 días)")
    ax.plot(t.index, t["ha_pred"], color=COLOR_30D, lw=1.8, ls="--", label="Estimado por el modelo")
    ax.plot(t.index, t["ha_naive"], color=COLOR_GRIS, lw=1.2, ls=":", label="Persistencia 12 semanas")
    ax.set(ylabel="Hectáreas (toda la reserva)", title="Frente 1 - Hectáreas en riesgo a 30 días (periodo de prueba)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "hito2_regresion.png", dpi=200)
    plt.close(fig)

    # Importancia de variables
    imp = art["30d"]["importancias"].head(15).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(imp.index, imp.values, color=COLOR_30D)
    ax.set(title="Frente 1 - Variables más influyentes (riesgo 30 días)", xlabel="Importancia (Gini)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "hito2_f1_importancia.png", dpi=200)
    plt.close(fig)


def figura_frente2(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set(title="Frente 2 - Matriz de confusión (CV 5 folds agrupada)", xlabel="Predicción", ylabel="Real")
    ax.set_xticks(range(4), ETIQUETAS_RIESGO)
    ax.set_yticks(range(4), ETIQUETAS_RIESGO)
    umbral_texto = cm.max() / 2
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > umbral_texto else "black")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "hito2_clasificacion.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dashboard estático (HTML)
# ---------------------------------------------------------------------------
def guardar_dashboard(m1: dict, m2: dict, ranking: pd.DataFrame) -> None:
    r7, r30 = m1["7d"]["riesgo"], m1["30d"]["riesgo"]
    g7, g30 = m1["7d"]["magnitud"], m1["30d"]["magnitud"]
    filas = "".join(
        f"<tr><td>{f.cuadrante_id}</td><td>{f.prob_7d:.2f}</td><td>{f.prob_30d:.2f}</td>"
        f"<td>{f.ha_riesgo_30d:.2f}</td><td>{f.co2_riesgo_30d_t:,.0f}</td><td>{f.nivel_alerta}</td></tr>"
        for f in ranking.head(15).itertuples()
    )
    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hito 2 - Dashboard preliminar</title>
<style>
body{{font-family:Arial,sans-serif;background:#f4f7fb;color:#1f2937;margin:0;padding:24px}}
.container{{max-width:1100px;margin:auto}} h1,h2{{color:#0f172a}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin:24px 0}}
.card,.panel{{background:#fff;border-radius:10px;padding:18px;box-shadow:0 2px 10px rgba(0,0,0,.06)}}
.panel{{margin:20px 0}} .metric{{font-size:1.7rem;font-weight:bold;color:#1d4ed8;margin:6px 0}}
img{{width:100%;border-radius:8px;border:1px solid #dfe7f5;margin-top:12px}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #e5e7eb;padding:6px;text-align:left}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
</style></head><body><div class="container">
<h1>Hito 2 - Preparación y Baseline</h1>
<p>BIODIVERSITY-GUARD Predict - fases 3 y 4 (inicial) de CRISP-DM. Validación temporal sin fuga de datos:
entrenamiento con semanas previas y prueba desde {m1['30d']['periodo_test_desde']}.</p>
<div class="cards">
 <div class="card"><h3>F1 · Riesgo 30 días</h3><div class="metric">AUC {r30['roc_auc']:.3f}</div>
  Recall {r30['recall']:.3f} · Precisión {r30['precision']:.3f} · F1 {r30['f1']:.3f}<br><small>Referencia persistencia: AUC {r30['persistencia_roc_auc']:.3f}</small></div>
 <div class="card"><h3>F1 · Riesgo 7 días</h3><div class="metric">AUC {r7['roc_auc']:.3f}</div>
  Recall {r7['recall']:.3f} · Precisión {r7['precision']:.3f} · F1 {r7['f1']:.3f}<br><small>Referencia persistencia: AUC {r7['persistencia_roc_auc']:.3f}</small></div>
 <div class="card"><h3>F1 · Captura de hectáreas</h3><div class="metric">{r30['captura_ha_top20_pct']:.0f}%</div>
  de la pérdida a 30 días ocurre en el 20 % de cuadrantes priorizados (7 días: {r7['captura_ha_top20_pct']:.0f}%)</div>
 <div class="card"><h3>F2 · Nivel de riesgo</h3><div class="metric">F1 macro {m2['f1_macro']:.3f}</div>
  Recall Crítico {m2['recall_por_clase']['Crítico']:.3f} · Accuracy {m2['accuracy']:.3f}</div>
</div>
<div class="panel"><h2>Hectáreas estimadas (magnitud)</h2>
<p>30 días: MAE {g30['mae_ha']:.3f} ha (ingenuo {g30['naive_mae_ha']:.3f}) · RMSE {g30['rmse_ha']:.3f} ha (ingenuo {g30['naive_rmse_ha']:.3f}) ·
Spearman espacial {g30['spearman_ranking_espacial']:.3f}<br>
7 días: MAE {g7['mae_ha']:.3f} ha (ingenuo {g7['naive_mae_ha']:.3f}) · RMSE {g7['rmse_ha']:.3f} ha (ingenuo {g7['naive_rmse_ha']:.3f}) ·
Spearman espacial {g7['spearman_ranking_espacial']:.3f}</p></div>
<div class="panel"><h2>Cuadrantes priorizados (última semana observada)</h2>
<table><tr><th>Cuadrante</th><th>P(7 días)</th><th>P(30 días)</th><th>ha en riesgo 30 d</th><th>t CO₂ en riesgo</th><th>Alerta</th></tr>{filas}</table></div>
<div class="panel"><h2>Visualizaciones predictivas</h2><div class="grid2">
<img src="hito2_f1_roc.png" alt="ROC"><img src="hito2_f1_captura.png" alt="Captura"></div>
<img src="hito2_regresion.png" alt="Serie 30 días"><div class="grid2">
<img src="hito2_f1_importancia.png" alt="Importancias"><img src="hito2_clasificacion.png" alt="Matriz de confusión"></div></div>
</div></body></html>"""
    (REPORTS_DIR / "hito2_dashboard.html").write_text(html, encoding="utf-8")


def _limpiar(objeto):
    """Convierte tipos numpy a nativos para JSON."""
    if isinstance(objeto, dict):
        return {str(k): _limpiar(v) for k, v in objeto.items()}
    if isinstance(objeto, (list, tuple)):
        return [_limpiar(v) for v in objeto]
    if isinstance(objeto, (np.floating, np.integer)):
        return objeto.item()
    return objeto


def main() -> None:
    crear_directorios()

    print("=== FRENTE 1: preparación ===")
    fr1, features, fr1_resumen = preparar_frente1()
    print("=== FRENTE 1: baseline Random Forest (validación temporal) ===")
    m1, art1 = evaluar_frente1(fr1, features)
    ranking = ranking_ultima_semana(fr1, features, art1)
    figuras_frente1(art1)

    print("=== FRENTE 2: preparación ===")
    fr2, fr2_resumen = preparar_frente2()
    print("=== FRENTE 2: baseline Random Forest + SMOTE ===")
    m2, art2 = evaluar_frente2(fr2)
    figura_frente2(art2["y"], art2["y_pred"])

    joblib.dump(
        {"features": features, **{k: {"clf": v["clf"], "reg": v["reg"], "umbral": v["umbral"]} for k, v in art1.items()}},
        MODELS_DIR / "hito2_frente1_rf.joblib",
        compress=3,
    )
    joblib.dump(art2["modelo"], MODELS_DIR / "hito2_frente2_rf_smote.joblib", compress=3)

    guardar_dashboard(m1, m2, ranking)
    salida = {
        "frente_1": {
            "summary": fr1_resumen,
            "features": features,
            "metrics": m1,
            "top_features_30d": art1["30d"]["importancias"].head(15).round(4).to_dict(),
            "dataset_path": str(FR1_OUT.relative_to(ROOT)),
        },
        "frente_2": {"summary": fr2_resumen, "metrics": m2, "dataset_path": str(FR2_OUT.relative_to(ROOT))},
    }
    with (REPORTS_DIR / "hito2_resumen.json").open("w", encoding="utf-8") as f:
        json.dump(_limpiar(salida), f, indent=2, ensure_ascii=False)

    print(f"Resumen: {REPORTS_DIR / 'hito2_resumen.json'}")
    print(f"Dashboard: {REPORTS_DIR / 'hito2_dashboard.html'}")


if __name__ == "__main__":
    main()
