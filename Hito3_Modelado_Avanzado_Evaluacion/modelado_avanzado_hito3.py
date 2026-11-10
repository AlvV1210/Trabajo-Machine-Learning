"""Hito 3 - Modelado avanzado, ajuste de hiperparámetros, evaluación robusta.

CRISP-DM fase 4 (avanzada) y fase 5 para BIODIVERSITY-GUARD Predict.

Frente 1 (regresión / series temporales, 273 cuadrantes x 164 semanas):
    * Validación walk-forward (origen rodante): se reentrena cada 8 semanas con
      todo el pasado y se predice el bloque siguiente. El objetivo de una fila de
      entrenamiento solo se usa si su ventana futura ya terminó (sin fuga).
    * Comparativa: persistencia, árbol baseline (Hito 2), Random Forest, XGBoost,
      LightGBM y modelo Hurdle (LightGBM clasificador x LightGBM magnitud log).
    * Ajuste de hiperparámetros del Hurdle con búsqueda aleatoria + TimeSeriesSplit
      sobre el periodo de entrenamiento inicial.
Frente 2 (clasificación ordinal de riesgo, 13 993 detecciones):
    * StratifiedGroupKFold por imagen (las anotaciones de una misma foto nunca se
      reparten entre entrenamiento y prueba).
    * Comparativa: Regresión logística, Árbol, Random Forest, XGBoost, LightGBM;
      GridSearch del modelo elegido; prueba de estrés con ruido de sensor.

Salidas: models/*.joblib, reports/hito3/*.png, reports/hito3/metricas_hito3.json,
data/predicciones_frente1.csv, data/predicciones_frente2.csv.

Ejecución: python Hito3_Modelado_Avanzado_Evaluacion/modelado_avanzado_hito3.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, cohen_kappa_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    CATEGORICAS, CO2_POR_C, F2_BOOL, F2_CAT, F2_NUM, NIVELES, T_C_POR_HA, cargar_frente1,
    cargar_frente2, construir_features_frente1,
)

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REP = ROOT / "reports" / "hito3"
for p in (MODELS, REP):
    p.mkdir(parents=True, exist_ok=True)

PASO_SEMANAS = 8        # frecuencia de reentrenamiento en el walk-forward
INICIO_TEST = 0.60      # el 40 % final de las semanas se evalúa fuera de muestra
UMBRAL_TOP = 0.20       # fracción de cuadrantes que se pueden patrullar por semana
COLORES = {"bajo": "#2e7d32", "mod": "#f9a825", "alto": "#ef6c00", "crit": "#c62828", "azul": "#1f4e79"}

LGB_BASE = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================================================================== #
# FRENTE 1
# =========================================================================== #
class HurdleRegressor:
    """E[y] = P(y>0) * E[y | y>0].

    Primera etapa: LightGBM clasificador (¿habrá pérdida?).
    Segunda etapa: LightGBM sobre log1p(y) solo en filas con pérdida, con
    corrección de retransformación (smearing de Duan).
    Es el enfoque estándar para objetivos inflados en ceros (92.9 % de ceros).
    """

    def __init__(self, params_clf: dict | None = None, params_reg: dict | None = None):
        self.params_clf = {**LGB_BASE, **(params_clf or {})}
        self.params_reg = {**LGB_BASE, **(params_reg or {})}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HurdleRegressor":
        self.clf = lgb.LGBMClassifier(**self.params_clf).fit(X, (y > 0).astype(int))
        pos = y > 0
        self.reg = lgb.LGBMRegressor(**self.params_reg).fit(X[pos], np.log1p(y[pos]))
        resid = np.log1p(y[pos]) - self.reg.predict(X[pos])
        self.smear = float(np.mean(np.exp(resid)))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def predict_magnitud(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(np.exp(self.reg.predict(X)) * self.smear - 1, 0, None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X) * self.predict_magnitud(X)


def _num(X: pd.DataFrame) -> pd.DataFrame:
    """Versión numérica (one-hot) para modelos de sklearn/xgboost."""
    return pd.get_dummies(X, columns=[c for c in CATEGORICAS if c in X], dtype=float)


def modelos_frente1() -> dict:
    return {
        "Árbol baseline (Hito 2)": lambda: DecisionTreeRegressor(max_depth=6, random_state=SEED),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=150, min_samples_leaf=20, max_features=0.4,
                                                       max_samples=0.5, n_jobs=2, random_state=SEED),
        "XGBoost": lambda: xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
                                            colsample_bytree=0.7, min_child_weight=5, tree_method="hist",
                                            n_jobs=2, random_state=SEED),
        "LightGBM": lambda: lgb.LGBMRegressor(**LGB_BASE),
    }


def ajustar_hurdle(df: pd.DataFrame, F: list[str], tgt: str, h: int, semanas: np.ndarray) -> tuple[dict, dict, list]:
    """Búsqueda aleatoria de hiperparámetros con validación temporal (TimeSeriesSplit)
    restringida al periodo de entrenamiento inicial (antes del 60 % de las semanas)."""
    rng = np.random.default_rng(SEED)
    espacio = {
        "num_leaves": [15, 31, 63], "min_child_samples": [20, 50, 100], "learning_rate": [0.03, 0.05, 0.08],
        "n_estimators": [200, 300, 500], "colsample_bytree": [0.5, 0.7, 0.9], "reg_lambda": [0.0, 1.0, 5.0],
    }
    corte = int(len(semanas) * INICIO_TEST)
    folds = [int(corte * f) for f in (0.55, 0.70, 0.85)]
    D = df.dropna(subset=[tgt])
    resultados = []
    candidatos = [{}] + [{k: rng.choice(v).item() for k, v in espacio.items()} for _ in range(9)]
    for i, params in enumerate(candidatos):
        aps, maes = [], []
        for f in folds:
            tr = D[D["semana_inicio"] <= semanas[f - h]]
            va = D[D["semana_inicio"].isin(semanas[f:f + 8])]
            m = HurdleRegressor(params, params).fit(tr[F], tr[tgt])
            p = m.predict_proba(va[F])
            aps.append(average_precision_score(va[tgt] > 0, p))
            maes.append(mean_absolute_error(va[tgt], m.predict(va[F])))
        resultados.append({"id": i, **params, "PR_AUC": float(np.mean(aps)), "MAE": float(np.mean(maes))})
    tabla = pd.DataFrame(resultados)
    # Criterio: máximo PR-AUC (detectar eventos raros) desempatando por MAE.
    tabla["rank"] = tabla["PR_AUC"].rank(ascending=False) + tabla["MAE"].rank()
    mejor = tabla.sort_values("rank").iloc[0]
    params = {k: (int(mejor[k]) if k in ("num_leaves", "min_child_samples", "n_estimators") else float(mejor[k]))
              for k in espacio if not pd.isna(mejor.get(k, np.nan))}
    t = tabla.drop(columns="rank").round(4)
    return params, params, t.astype(object).where(t.notna(), None).to_dict("records")


def walk_forward(df: pd.DataFrame, F: list[str], tgt: str, h: int, semanas: np.ndarray, params: dict) -> pd.DataFrame:
    D = df.dropna(subset=[tgt])
    salidas = []
    for i in range(int(len(semanas) * INICIO_TEST), len(semanas) - h, PASO_SEMANAS):
        tr = D[D["semana_inicio"] <= semanas[i - h]]  # objetivo ya observado al momento i
        te = D[D["semana_inicio"].isin(semanas[i:i + PASO_SEMANAS])].copy()
        if te.empty:
            continue
        Xtr_n, Xte_n = _num(tr[F]), _num(te[F]).reindex(columns=_num(tr[F]).columns, fill_value=0)
        te["Persistencia"] = te["perdida_lag0"] if h == 1 else te["perdida_suma_4s"]
        for nombre, fabrica in modelos_frente1().items():
            m = fabrica()
            if nombre == "LightGBM":
                m.fit(tr[F], tr[tgt]); te[nombre] = np.clip(m.predict(te[F]), 0, None)
            else:
                m.fit(Xtr_n, tr[tgt]); te[nombre] = np.clip(m.predict(Xte_n), 0, None)
        hur = HurdleRegressor(params, params).fit(tr[F], tr[tgt])
        te["Hurdle LightGBM"] = hur.predict(te[F])
        te["prob_evento"] = hur.predict_proba(te[F])
        te["fold"] = i
        salidas.append(te)
        log(f"   {tgt}: bloque desde {pd.Timestamp(semanas[i]).date()} ({len(tr)} filas de entrenamiento)")
    return pd.concat(salidas)


def metricas_regresion(R: pd.DataFrame, tgt: str, modelo: str) -> dict:
    y, p = R[tgt], R[modelo]
    semanas_con = [g for _, g in R.groupby("semana_inicio") if g[tgt].sum() > 0]
    captura = np.mean([g.nlargest(int(len(g) * UMBRAL_TOP), modelo)[tgt].sum() / g[tgt].sum() for g in semanas_con])
    rho = np.nanmean([spearmanr(g[tgt], g[modelo]).correlation for g in semanas_con])
    mae_por_fold = R.groupby("fold").apply(lambda g: mean_absolute_error(g[tgt], g[modelo]))
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "MAE_std_folds": float(mae_por_fold.std()),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "R2_log": float(r2_score(np.log1p(y), np.log1p(p))),
        "Spearman_espacial": float(rho),
        "Captura_top20": float(captura),
    }


def metricas_deteccion(y: pd.Series, p: np.ndarray, umbral: float) -> dict:
    pred = (p >= umbral).astype(int)
    return {
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
        "tasa_base": float(y.mean()),
        "umbral": float(umbral),
        "Recall": float(recall_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred)),
    }


def umbral_recall(y: pd.Series, p: np.ndarray, objetivo: float = 0.80) -> float:
    """Mayor umbral que mantiene un recall >= objetivo (minimizar incursiones no vistas)."""
    for u in np.linspace(0.95, 0.01, 95):
        if recall_score(y, (p >= u).astype(int)) >= objetivo:
            return float(u)
    return 0.01


def graficos_frente1(res: dict, R7: pd.DataFrame, R30: pd.DataFrame, imp: pd.DataFrame) -> None:
    # 1. Comparativa de modelos (MAE y captura)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (h, key) in zip(axes, [("7 días", "perdida_7d"), ("30 días", "perdida_30d")]):
        t = pd.DataFrame(res[key]["comparativa"]).T
        orden = t.sort_values("MAE", ascending=False)
        cols = [COLORES["azul"] if n == "Hurdle LightGBM" else "#9aa9b8" for n in orden.index]
        ax.barh(orden.index, orden["MAE"], color=cols)
        for j, (v, c) in enumerate(zip(orden["MAE"], orden["Captura_top20"])):
            ax.text(v, j, f"  MAE {v:.3f} | captura {c:.0%}", va="center", fontsize=8.5)
        ax.set_title(f"Horizonte {h}: MAE walk-forward (ha)", fontsize=11)
        ax.set_xlim(0, orden["MAE"].max() * 1.7)
        ax.grid(axis="x", alpha=0.25)
    plt.tight_layout(); plt.savefig(REP / "f1_comparativa_modelos.png", dpi=170); plt.close()

    # 2. ROC + calibración del detector
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for R, ev, lab, c in [(R7, "evento_7d", "7 días", "#5b8db8"), (R30, "evento_30d", "30 días", COLORES["azul"])]:
        fpr, tpr, _ = roc_curve(R[ev], R["prob_evento"])
        axes[0].plot(fpr, tpr, color=c, lw=2, label=f"{lab} (AUC={roc_auc_score(R[ev], R['prob_evento']):.3f})")
        fr, mp = calibration_curve(R[ev], R["prob_evento"], n_bins=10, strategy="quantile")
        axes[1].plot(mp, fr, "o-", color=c, label=lab)
    axes[0].plot([0, 1], [0, 1], "--", color="grey"); axes[0].set_title("Curva ROC – detección de pérdida")
    axes[0].set_xlabel("Tasa de falsos positivos"); axes[0].set_ylabel("Tasa de verdaderos positivos"); axes[0].legend()
    axes[1].plot([0, 1], [0, 1], "--", color="grey"); axes[1].set_title("Calibración de la probabilidad")
    axes[1].set_xlabel("Probabilidad predicha"); axes[1].set_ylabel("Frecuencia observada"); axes[1].legend()
    for a in axes: a.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(REP / "f1_roc_calibracion.png", dpi=170); plt.close()

    # 3. Serie regional real vs predicha (30 días)
    serie = R30.groupby("semana_inicio")[["perdida_30d", "Hurdle LightGBM", "Persistencia"]].sum()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(serie.index, serie["perdida_30d"], color="black", lw=2, label="Real (ha en los 30 días siguientes)")
    ax.plot(serie.index, serie["Hurdle LightGBM"], color=COLORES["azul"], lw=2, ls="--", label="Hurdle LightGBM")
    ax.plot(serie.index, serie["Persistencia"], color="#bbbbbb", lw=1.4, label="Persistencia")
    ax.set_title("Frente 1 – pérdida total proyectada a 30 días en Tambopata (fuera de muestra)")
    ax.set_ylabel("hectáreas"); ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(REP / "f1_serie_30d.png", dpi=170); plt.close()

    # 4. Curva de captura (valor operativo)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    fr = np.linspace(0.02, 1, 50)
    for modelo, c in [("Hurdle LightGBM", COLORES["azul"]), ("Persistencia", "#999999")]:
        cap = [np.mean([g.nlargest(max(1, int(len(g) * f)), modelo)["perdida_30d"].sum() / g["perdida_30d"].sum()
                        for _, g in R30.groupby("semana_inicio") if g["perdida_30d"].sum() > 0]) for f in fr]
        ax.plot(fr * 100, np.array(cap) * 100, color=c, lw=2, label=modelo)
    ax.plot([0, 100], [0, 100], "--", color="grey", label="Patrullaje aleatorio")
    ax.axvline(20, color=COLORES["crit"], ls=":", lw=1)
    ax.set_xlabel("% de cuadrantes patrullados (ordenados por riesgo)")
    ax.set_ylabel("% de la pérdida real a 30 días cubierta")
    ax.set_title("Curva de captura de deforestación"); ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(REP / "f1_curva_captura.png", dpi=170); plt.close()

    # 5. Importancia de variables
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["variable"], top["importancia"], color=COLORES["azul"])
    ax.set_title("Frente 1 – importancia por permutación (caída de PR-AUC, 30 días)")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout(); plt.savefig(REP / "f1_importancia.png", dpi=170); plt.close()

    # 6. Errores por zona y causa
    R30 = R30.assign(error=R30["Hurdle LightGBM"] - R30["perdida_30d"])
    grupo = R30.groupby("causa_principal", observed=True).agg(MAE=("error", lambda e: e.abs().mean()),
                                                              sesgo=("error", "mean"))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(grupo))
    ax.bar(x - 0.2, grupo["MAE"], 0.4, label="MAE", color=COLORES["azul"])
    ax.bar(x + 0.2, grupo["sesgo"], 0.4, label="Sesgo (pred − real)", color=COLORES["alto"])
    ax.set_xticks(x, grupo.index); ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Análisis de errores a 30 días por causa principal"); ax.set_ylabel("ha"); ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(REP / "f1_errores_causa.png", dpi=170); plt.close()


def frente1() -> dict:
    log("FRENTE 1 · construyendo panel y variables")
    base = cargar_frente1()
    df, F = construir_features_frente1(base)
    semanas = np.sort(df["semana_inicio"].unique())
    res: dict = {"filas": int(len(df)), "cuadrantes": int(df["cuadrante_id"].nunique()),
                 "semanas": int(len(semanas)), "n_features": len(F), "features": F,
                 "porcentaje_ceros": float((df["perdida_ha"] == 0).mean())}
    finales = {}
    for tgt, ev, h in [("perdida_7d", "evento_7d", 1), ("perdida_30d", "evento_30d", 4)]:
        log(f"FRENTE 1 · ajuste de hiperparámetros ({tgt})")
        pc, pr, busqueda = ajustar_hurdle(df, F, tgt, h, semanas)
        log(f"FRENTE 1 · walk-forward ({tgt}) con {pc}")
        R = walk_forward(df, F, tgt, h, semanas, pc)
        comp = {m: metricas_regresion(R, tgt, m) for m in
                ["Persistencia", *modelos_frente1().keys(), "Hurdle LightGBM"]}
        u = umbral_recall(R[ev], R["prob_evento"], 0.80 if h == 4 else 0.70)
        det = metricas_deteccion(R[ev], R["prob_evento"], u)
        mejora = 1 - comp["Hurdle LightGBM"]["MAE"] / comp["Persistencia"]["MAE"]
        res[tgt] = {"busqueda_hiperparametros": busqueda, "mejores_parametros": pc, "comparativa": comp,
                    "deteccion": det, "mejora_MAE_vs_persistencia": float(mejora),
                    "filas_evaluadas": int(len(R)), "folds": int(R["fold"].nunique())}
        R.to_csv(DATA / f"walkforward_{tgt}.csv", index=False,
                 columns=["cuadrante_id", "semana_inicio", tgt, ev, "prob_evento", "Hurdle LightGBM", "Persistencia"])
        # Modelo final con todo el histórico etiquetado
        D = df.dropna(subset=[tgt])
        finales[tgt] = HurdleRegressor(pc, pr).fit(D[F], D[tgt])
        finales[tgt].umbral = u
        res[tgt]["R_df"] = R  # temporal, se elimina antes de guardar

    # Importancia por permutación sobre el último bloque fuera de muestra (30 días)
    log("FRENTE 1 · importancia por permutación")
    R30 = res["perdida_30d"]["R_df"]
    ult = R30[R30["fold"] == R30["fold"].max()]
    D = df.dropna(subset=["perdida_30d"])
    tr = D[D["semana_inicio"] <= semanas[int(ult["fold"].iloc[0]) - 4]]
    clf = lgb.LGBMClassifier(**{**LGB_BASE, **res["perdida_30d"]["mejores_parametros"]}).fit(tr[F], tr["evento_30d"])
    Xp = ult[F]
    pim = permutation_importance(clf, Xp, ult["evento_30d"], scoring="average_precision", n_repeats=5,
                                 random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"variable": F, "importancia": pim.importances_mean}).sort_values("importancia", ascending=False)
    res["importancia_permutacion"] = imp.head(20).round(5).to_dict("records")
    graficos_frente1(res, res["perdida_7d"]["R_df"], R30, imp)

    # Análisis de errores por zona
    R30z = R30.assign(zona=np.where(R30["cuadrante_id"].str[1:].astype(int) <= 167, "Reserva", "Amortiguamiento"))
    res["errores_por_zona"] = {z: metricas_regresion(g, "perdida_30d", "Hurdle LightGBM")
                               for z, g in R30z.groupby("zona")}
    for k in ("perdida_7d", "perdida_30d"):
        res[k].pop("R_df")

    # Pronóstico operativo para la última semana disponible (insumo del dashboard)
    ultima = df[df["semana_inicio"] == semanas[-1]].copy()
    pred = ultima[["cuadrante_id", "semana", "semana_inicio", "cobertura_anp", "causa_principal", "vocacion_zee",
                   "mb_mineria", "mb_agropecuario", "tasa_deforestacion_ha_anio", "perdida_suma_4s",
                   "perdida_suma_52s", "densidad_poblacional"]].copy()
    for tgt, sufijo in [("perdida_7d", "7d"), ("perdida_30d", "30d")]:
        m = finales[tgt]
        pred[f"prob_evento_{sufijo}"] = m.predict_proba(ultima[F])
        pred[f"ha_esperadas_{sufijo}"] = m.predict(ultima[F])
        pred[f"alerta_{sufijo}"] = (pred[f"prob_evento_{sufijo}"] >= m.umbral).astype(int)
    pred["co2_en_riesgo_30d_t"] = pred["ha_esperadas_30d"] * T_C_POR_HA * CO2_POR_C
    pred.to_csv(DATA / "predicciones_frente1.csv", index=False)
    joblib.dump({"modelos": finales, "features": F}, MODELS / "frente1_hurdle_lightgbm.joblib")
    log("FRENTE 1 · listo")
    return res


# =========================================================================== #
# FRENTE 2
# =========================================================================== #
def preprocesador_f2() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), F2_NUM),
        ("bool", "passthrough", F2_BOOL),
        ("cat", OneHotEncoder(handle_unknown="ignore"), F2_CAT),
    ])


def modelos_frente2() -> dict:
    return {
        "Regresión logística": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Árbol baseline (Hito 2)": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=2, random_state=SEED),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, tree_method="hist",
                                     n_jobs=2, random_state=SEED),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, verbose=-1, n_jobs=2),
    }


def metricas_clasificacion(y, pred, proba=None) -> dict:
    out = {
        "Accuracy": float(accuracy_score(y, pred)),
        "F1_macro": float(f1_score(y, pred, average="macro")),
        "F1_ponderado": float(f1_score(y, pred, average="weighted")),
        "Recall_Critico": float(recall_score(y, pred, labels=[3], average="macro")),
        "Recall_Alto": float(recall_score(y, pred, labels=[2], average="macro")),
        "Kappa_cuadratico": float(cohen_kappa_score(y, pred, weights="quadratic")),
    }
    if proba is not None:
        out["ROC_AUC_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr"))
    return out


def estresar(X: pd.DataFrame, tasa: float, rng: np.random.Generator) -> pd.DataFrame:
    """Simula fallas del módulo IA: error de GPS (±300 m) y confusión de clase
    entre categorías del mismo grupo (fauna↔fauna, amenaza↔amenaza)."""
    X = X.copy()
    fuera = X["en_zona_intangible"] == 0
    # Detecciones a menos de 300 m del límite pueden caer dentro de la zona por el error de GPS
    borde = fuera & (X["distancia_limite_m"] < 300) & (rng.random(len(X)) < 0.5)
    for c in ["distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m"]:
        ruido = rng.normal(0, 300, len(X))
        X[c] = np.clip(X[c] + ruido, 0, None)
    X.loc[~fuera, "distancia_limite_m"] = 0.0  # dentro de la reserva la distancia al límite es 0
    X.loc[borde, "en_zona_intangible"] = 1
    X.loc[borde, "distancia_limite_m"] = 0.0
    X["densidad_telemetria"] = X["confidence_score"] / (X["distancia_limite_m"] + 1)
    X["indice_estres_hidrico"] = X["distancia_a_rio_m"] / 1000
    X["log_dist_carretera"] = np.log1p(X["distancia_a_carretera_m"])
    X["log_dist_rio"] = np.log1p(X["distancia_a_rio_m"])
    cambiar = rng.random(len(X)) < tasa * (1.2 - X["confidence_score"].to_numpy())
    for grupo in [X["es_amenaza"] == 1, X["es_amenaza"] == 0]:
        idx = X.index[cambiar & grupo.to_numpy()]
        opciones = X.loc[grupo, "clase"].unique()
        X.loc[idx, "clase"] = rng.choice(opciones, len(idx))
    return X


def frente2() -> tuple[dict, pd.DataFrame]:
    log("FRENTE 2 · cargando detecciones normalizadas")
    df = cargar_frente2()
    y = df["Nivel_Riesgo_v2"]
    X = df[F2_NUM + F2_BOOL + F2_CAT]
    grupos = df["fuente"].astype(str) + "_" + df["image_id"].astype(str)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y, grupos))
    crudo = pd.read_csv(DATA / "dataset_frente2_hito1_final.csv")
    res: dict = {"filas_originales": int(len(crudo)),
                 "filas_etiquetadas_hito1": int(crudo["Nivel_Riesgo"].notna().sum()),
                 "filas_etiquetadas_hito3": int(len(df)),
                 "distribucion": {NIVELES[k]: int(v) for k, v in y.value_counts().sort_index().items()},
                 "clases": df["clase"].value_counts().to_dict()}
    comp, oof_mejor = {}, None
    for nombre, modelo in modelos_frente2().items():
        oof = np.zeros(len(y), dtype=int)
        proba = np.zeros((len(y), 4))
        f1s = []
        for tr, te in splits:
            pipe = Pipeline([("pre", preprocesador_f2()), ("m", modelo)]).fit(X.iloc[tr], y.iloc[tr])
            oof[te] = pipe.predict(X.iloc[te])
            proba[te] = pipe.predict_proba(X.iloc[te])
            f1s.append(f1_score(y.iloc[te], oof[te], average="macro"))
        comp[nombre] = {**metricas_clasificacion(y, oof, proba), "F1_macro_std": float(np.std(f1s))}
        if nombre == "Random Forest":
            oof_mejor = oof
        log(f"   {nombre}: F1 macro {comp[nombre]['F1_macro']:.4f}")
    res["comparativa"] = comp

    log("FRENTE 2 · GridSearch Random Forest")
    grid = GridSearchCV(
        Pipeline([("pre", preprocesador_f2()), ("m", RandomForestClassifier(class_weight="balanced", n_jobs=2,
                                                                            random_state=SEED))]),
        {"m__n_estimators": [150, 300], "m__max_depth": [None, 12], "m__min_samples_leaf": [1, 5]},
        scoring="f1_macro", cv=list(cv.split(X, y, grupos)), n_jobs=1,
    ).fit(X, y, groups=grupos)
    res["gridsearch"] = {"mejores_parametros": grid.best_params_, "F1_macro_cv": float(grid.best_score_),
                         "resultados": pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "std_test_score"]]
                         .assign(params=lambda t: t["params"].astype(str)).round(5).to_dict("records")}
    final = grid.best_estimator_

    # Prueba de estrés: modelo entrenado limpio, evaluado con sensores degradados
    log("FRENTE 2 · prueba de estrés")
    rng = np.random.default_rng(SEED)
    estres = {}
    tr, te = splits[0]
    pipe = Pipeline([("pre", preprocesador_f2()), ("m", RandomForestClassifier(**{k.replace("m__", ""): v for k, v in
                     grid.best_params_.items()}, class_weight="balanced", n_jobs=2, random_state=SEED))]).fit(X.iloc[tr], y.iloc[tr])
    for tasa in [0.0, 0.05, 0.10, 0.20]:
        Xs = estresar(X.iloc[te], tasa, rng) if tasa > 0 else X.iloc[te]
        estres[f"{int(tasa * 100)}%"] = metricas_clasificacion(y.iloc[te], pipe.predict(Xs))
    res["prueba_estres"] = estres

    # Importancia
    imp = permutation_importance(pipe, X.iloc[te], y.iloc[te], scoring="f1_macro", n_repeats=5, random_state=SEED,
                                 n_jobs=1)
    res["importancia_permutacion"] = (pd.DataFrame({"variable": X.columns, "importancia": imp.importances_mean})
                                      .sort_values("importancia", ascending=False).round(5).to_dict("records"))

    # Gráficos
    cm = confusion_matrix(y, oof_mejor, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(4), NIVELES); ax.set_yticks(range(4), NIVELES)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title("Frente 2 – Matriz de confusión (validación cruzada agrupada)")
    plt.tight_layout(); plt.savefig(REP / "f2_matriz_confusion.png", dpi=170); plt.close()

    t = pd.DataFrame(estres).T
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, c in [("F1_macro", COLORES["azul"]), ("Recall_Critico", COLORES["crit"]), ("Kappa_cuadratico", COLORES["bajo"])]:
        ax.plot(t.index, t[col], "o-", label=col, color=c, lw=2)
    ax.set_ylim(0.5, 1.02); ax.set_xlabel("Tasa de confusión del detector + GPS ±300 m")
    ax.set_title("Prueba de estrés del clasificador de riesgo"); ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(REP / "f2_prueba_estres.png", dpi=170); plt.close()

    ti = pd.DataFrame(res["importancia_permutacion"]).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(ti["variable"], ti["importancia"], color=COLORES["alto"])
    ax.set_title("Frente 2 – importancia por permutación (caída de F1 macro)"); ax.grid(axis="x", alpha=0.25)
    plt.tight_layout(); plt.savefig(REP / "f2_importancia.png", dpi=170); plt.close()

    # Predicciones del modelo final sobre todas las detecciones (insumo del dashboard)
    proba = final.predict_proba(X)
    out = df[["image_id", "fuente", "clase", "confidence_score", "latitud", "longitud", "en_zona_intangible",
              "en_zona_amortiguamiento", "distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m",
              "Nivel_Riesgo_v2"]].copy()
    out["nivel_predicho"] = proba.argmax(1)
    for k, n in enumerate(NIVELES):
        out[f"p_{n.lower()}"] = proba[:, k]
    out.to_csv(DATA / "predicciones_frente2.csv", index=False)
    joblib.dump({"modelo": final, "columnas": list(X.columns)}, MODELS / "frente2_random_forest.joblib")
    log("FRENTE 2 · listo")
    return res, out


def main() -> None:
    t0 = time.time()
    r1 = frente1()
    r2, _ = frente2()
    salida = {"generado": time.strftime("%Y-%m-%d %H:%M"), "frente_1": r1, "frente_2": r2,
              "segundos": round(time.time() - t0, 1)}
    with (REP / "metricas_hito3.json").open("w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False, default=str)
    log(f"Métricas guardadas en {REP / 'metricas_hito3.json'}")
    for k in ("perdida_7d", "perdida_30d"):
        h = r1[k]["comparativa"]["Hurdle LightGBM"]; d = r1[k]["deteccion"]
        print(f"{k}: AUC {d['ROC_AUC']:.3f} | PR-AUC {d['PR_AUC']:.3f} | MAE {h['MAE']:.3f} | "
              f"captura@20% {h['Captura_top20']:.1%} | mejora MAE vs persistencia {r1[k]['mejora_MAE_vs_persistencia']:.1%}")
    print("Frente 2:", {k: round(v, 4) for k, v in r2["comparativa"]["Random Forest"].items()})


if __name__ == "__main__":
    main()
