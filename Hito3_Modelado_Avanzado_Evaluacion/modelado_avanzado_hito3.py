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
Frente 2 (clasificación de riesgo, 5 277 eventos de una red de 58 estaciones):
    * StratifiedGroupKFold por estación (una estación completa queda en un solo pliegue)
      y validación temporal (entrena hasta 2025, prueba con 2026).
    * Comparativa: Regresión logística, Árbol, Random Forest, XGBoost, LightGBM;
      GridSearch del modelo elegido; prueba de estrés con fallas de sensor.

Salidas: models/*.txt + *_meta.json (formato portable de LightGBM), reports/hito3/*.png, reports/hito3/metricas_hito3.json,
data/predicciones_frente1.csv, data/predicciones_frente2.csv.

Ejecución: python Hito3_Modelado_Avanzado_Evaluacion/modelado_avanzado_hito3.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

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
    CATEGORICAS, CO2_POR_C, F2_VARIABLES, NIVELES, T_C_POR_HA, cargar_frente1,
    cargar_frente2, construir_features_frente1, construir_features_frente2, matriz_frente2,
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


SEMANAS_CAL = 16  # ventana reciente reservada para calibrar (no se usa para ajustar ese modelo auxiliar)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def aplicar_calibracion(p: np.ndarray, cal: dict) -> np.ndarray:
    return 1 / (1 + np.exp(-(cal["a"] * _logit(p) + cal["b"])))


def calibrar_hurdle(tr: pd.DataFrame, F: list[str], tgt: str, h: int, params: dict) -> dict:
    """Calibración temporal interna (sin tocar el bloque de prueba).

    Se entrena un Hurdle auxiliar con todo menos las últimas 16 semanas etiquetadas y se compara con lo
    que realmente ocurrió en ellas:
      * a, b: escalamiento de Platt de la probabilidad (monótono: no cambia el orden ni el ROC-AUC).
      * k: razón hectáreas reales / estimadas en los cuadrantes con pérdida. Con ella se construye el
        «escenario alto», cuyo total no queda sesgado a la baja como la estimación típica.
    """
    wk = np.sort(tr["semana_inicio"].unique())
    a_ = tr[tr["semana_inicio"] <= wk[-SEMANAS_CAL - h - 1]]
    b_ = tr[tr["semana_inicio"] > wk[-SEMANAS_CAL - 1]]
    aux = HurdleRegressor(params, params).fit(a_[F], a_[tgt])
    pb = aux.predict_proba(b_[F])
    yb = (b_[tgt] > 0).astype(int)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(pb).reshape(-1, 1), yb)
    pos = (b_[tgt] > 0).to_numpy()
    k = float(np.clip(b_[tgt][pos].sum() / max(aux.predict_magnitud(b_[F])[pos].sum(), 1e-6), 1.0, 3.0))
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "k": k}


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
        cal = calibrar_hurdle(tr, F, tgt, h, params)
        te["prob_cruda"] = hur.predict_proba(te[F])
        te["prob_evento"] = aplicar_calibracion(te["prob_cruda"].to_numpy(), cal)   # probabilidad calibrada
        te["Hurdle LightGBM"] = hur.predict(te[F])                                  # estimación típica
        te["Hurdle escenario alto"] = te["prob_evento"] * hur.predict_magnitud(te[F]) * cal["k"]
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
        calib = {"Brier_sin_calibrar": float(np.mean((R["prob_cruda"] - R[ev]) ** 2)),
                 "Brier_calibrada": float(np.mean((R["prob_evento"] - R[ev]) ** 2)),
                 "total_estimado_sobre_real_tipico": float(R["Hurdle LightGBM"].sum() / R[tgt].sum()),
                 "total_estimado_sobre_real_alto": float(R["Hurdle escenario alto"].sum() / R[tgt].sum()),
                 "MAE_escenario_alto": float(mean_absolute_error(R[tgt], R["Hurdle escenario alto"]))}
        res[tgt] = {"busqueda_hiperparametros": busqueda, "mejores_parametros": pc, "comparativa": comp,
                    "deteccion": det, "calibracion": calib, "mejora_MAE_vs_persistencia": float(mejora),
                    "filas_evaluadas": int(len(R)), "folds": int(R["fold"].nunique())}
        R.to_csv(DATA / f"walkforward_{tgt}.csv", index=False,
                 columns=["cuadrante_id", "semana_inicio", tgt, ev, "prob_cruda", "prob_evento", "Hurdle LightGBM",
                          "Hurdle escenario alto", "Persistencia"])
        # Modelo final con todo el histórico etiquetado
        D = df.dropna(subset=[tgt])
        finales[tgt] = HurdleRegressor(pc, pr).fit(D[F], D[tgt])
        finales[tgt].umbral = u
        finales[tgt].cal = calibrar_hurdle(D, F, tgt, h, pc)
        res[tgt]["calibracion"]["parametros_finales"] = finales[tgt].cal
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
        pred[f"prob_evento_{sufijo}"] = aplicar_calibracion(m.predict_proba(ultima[F]), m.cal)
        pred[f"ha_esperadas_{sufijo}"] = m.predict(ultima[F])
        pred[f"ha_alto_{sufijo}"] = pred[f"prob_evento_{sufijo}"] * m.predict_magnitud(ultima[F]) * m.cal["k"]
        pred[f"alerta_{sufijo}"] = (pred[f"prob_evento_{sufijo}"] >= m.umbral).astype(int)
    pred["co2_en_riesgo_30d_t"] = pred["ha_esperadas_30d"] * T_C_POR_HA * CO2_POR_C
    pred.to_csv(DATA / "predicciones_frente1.csv", index=False)
    guardar_frente1_portable(finales, F, df)
    log("FRENTE 1 · listo")
    return res


# =========================================================================== #
# FRENTE 2
# =========================================================================== #
LGB_F2 = dict(n_estimators=200, learning_rate=0.08, num_leaves=15, min_child_samples=5, class_weight="balanced",
              random_state=SEED, verbose=-1, n_jobs=2)


def modelos_frente2() -> dict:
    esc = lambda m: Pipeline([("esc", StandardScaler()), ("m", m)])  # noqa: E731
    return {
        "Regresión logística": lambda: esc(LogisticRegression(max_iter=3000, class_weight="balanced")),
        "Árbol baseline (Hito 2)": lambda: DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=2, random_state=SEED),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, tree_method="hist",
                                             n_jobs=2, random_state=SEED),
        "LightGBM": lambda: lgb.LGBMClassifier(**LGB_F2),
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


def estresar(df: pd.DataFrame, tasa: float, rng: np.random.Generator) -> pd.DataFrame:
    """Simula fallas del módulo IA y del sensor sobre las variables crudas:
    error de GPS (±300 m), reloj desfasado (±1 h) y confusión de clase dentro del mismo grupo
    (fauna↔fauna, amenaza↔amenaza), más probable cuanto menor es la confianza del detector."""
    X = df.copy()
    fuera = X["en_zona_intangible"] == 0
    borde = fuera & (X["distancia_limite_m"] < 300) & (rng.random(len(X)) < 0.5)
    for c in ["distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m"]:
        X[c] = np.clip(X[c] + rng.normal(0, 300, len(X)), 0, None)
    X.loc[~fuera, "distancia_limite_m"] = 0.0
    X.loc[borde, ["en_zona_intangible", "distancia_limite_m"]] = [1, 0.0]
    desfase = rng.random(len(X)) < tasa
    X.loc[desfase, "hora"] = (X.loc[desfase, "hora"] + rng.choice([-1, 1], int(desfase.sum()))) % 24
    X["es_noche"] = ((X["hora"] >= 19) | (X["hora"] < 5)).astype(int)
    cambiar = rng.random(len(X)) < tasa * (1.6 - X["confidence_score"].to_numpy())
    for grupo in [X["es_amenaza"] == 1, X["es_amenaza"] == 0]:
        idx = X.index[cambiar & grupo.to_numpy()]
        X.loc[idx, "clase"] = rng.choice(X.loc[grupo, "clase"].unique(), len(idx))
    return construir_features_frente2(X.drop(columns=["hora_sin", "hora_cos"]))


def frente2() -> tuple[dict, pd.DataFrame]:
    log("FRENTE 2 · cargando telemetría realista")
    df = cargar_frente2()
    y = df["Nivel_Riesgo_v2"]
    X = matriz_frente2(df)
    grupos = df["estacion_id"].astype(str)  # una estación completa queda en un solo pliegue
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, grupos))
    crudo = pd.read_csv(DATA / "dataset_frente2_hito1_final.csv")
    semanas_f2 = int(df["semana"].nunique())
    res: dict = {"filas_originales": int(len(crudo)),
                 "filas_etiquetadas_hito1": int(crudo["Nivel_Riesgo"].notna().sum()),
                 "eventos": int(len(df)), "estaciones": int(df["estacion_id"].nunique()), "semanas": semanas_f2,
                 "eventos_por_semana": float(len(df) / semanas_f2),
                 "porcentaje_amenazas": float(df["es_amenaza"].mean()),
                 "criticos_por_semana": float((y == 3).sum() / semanas_f2),
                 "distribucion": {NIVELES[k]: int(v) for k, v in y.value_counts().sort_index().items()},
                 "clases": df["clase"].value_counts().to_dict()}

    comp, oof_final = {}, None
    for nombre, fabrica in modelos_frente2().items():
        oof, proba, f1s = np.zeros(len(y), dtype=int), np.zeros((len(y), 4)), []
        for tr, te in splits:
            m = fabrica().fit(X.iloc[tr], y.iloc[tr])
            proba[te] = m.predict_proba(X.iloc[te]); oof[te] = proba[te].argmax(1)
            f1s.append(f1_score(y.iloc[te], oof[te], average="macro"))
        comp[nombre] = {**metricas_clasificacion(y, oof, proba), "F1_macro_std": float(np.std(f1s))}
        if nombre == "LightGBM":
            oof_final = oof
        log(f"   {nombre}: F1 macro {comp[nombre]['F1_macro']:.4f}")
    res["comparativa"] = comp

    log("FRENTE 2 · GridSearch LightGBM")
    grid = GridSearchCV(lgb.LGBMClassifier(**LGB_F2),
                        {"num_leaves": [7, 15, 31], "n_estimators": [100, 200, 400], "min_child_samples": [5, 20]},
                        scoring="f1_macro", cv=splits, n_jobs=1).fit(X, y)
    res["gridsearch"] = {"mejores_parametros": grid.best_params_, "F1_macro_cv": float(grid.best_score_),
                         "resultados": pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "std_test_score"]]
                         .assign(params=lambda t: t["params"].astype(str)).round(5).to_dict("records")}
    params = {**LGB_F2, **grid.best_params_}

    # Validación temporal: se entrena con lo ocurrido hasta 2025 y se prueba con 2026
    antes = (df["fecha_hora"] < "2026-01-01").to_numpy()
    pred_t = lgb.LGBMClassifier(**params).fit(X[antes], y[antes]).predict(X[~antes])
    res["validacion_temporal_2026"] = {**metricas_clasificacion(y[~antes], pred_t), "eventos_prueba": int((~antes).sum()),
                                       "matriz": confusion_matrix(y[~antes], pred_t, labels=[0, 1, 2, 3]).tolist()}

    # Prueba de estrés e importancia: modelo entrenado limpio, evaluado en cada pliegue con datos degradados
    log("FRENTE 2 · prueba de estrés e importancia")
    rng = np.random.default_rng(SEED)
    tasas = [0.0, 0.05, 0.10, 0.20]
    pred_estres = {t: np.zeros(len(y), dtype=int) for t in tasas}
    caidas = {v: [] for v in F2_VARIABLES}
    for tr, te in splits:
        m = lgb.LGBMClassifier(**params).fit(X.iloc[tr], y.iloc[tr])
        base_f1 = f1_score(y.iloc[te], m.predict(X.iloc[te]), average="macro")
        for t in tasas:
            d = df.iloc[te] if t == 0 else estresar(df.iloc[te], t, rng)
            pred_estres[t][te] = m.predict(matriz_frente2(d))
        for v in F2_VARIABLES:  # permutación sobre la variable cruda
            d = df.iloc[te].copy()
            d[v] = rng.permutation(d[v].to_numpy())
            caidas[v].append(base_f1 - f1_score(y.iloc[te], m.predict(matriz_frente2(d)), average="macro"))
    res["prueba_estres"] = {f"{int(t * 100)}%": metricas_clasificacion(y, pred_estres[t]) for t in tasas}
    res["importancia_permutacion"] = (pd.DataFrame({"variable": list(caidas), "importancia": [np.mean(c) for c in caidas.values()]})
                                      .sort_values("importancia", ascending=False).round(5).to_dict("records"))

    # Gráficos
    cm = confusion_matrix(y, oof_final, labels=[0, 1, 2, 3])
    res["matriz_confusion"] = cm.tolist()
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.imshow(np.log1p(cm), cmap="Greens")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > 60 else "black", fontweight="bold" if i == j else None)
    ax.set_xticks(range(4), NIVELES); ax.set_yticks(range(4), NIVELES)
    ax.set_xlabel("Nivel que asignó el sistema"); ax.set_ylabel("Nivel real del evento")
    ax.set_title("Frente 2 – aciertos (diagonal) y confusiones del clasificador")
    plt.tight_layout(); plt.savefig(REP / "f2_matriz_confusion.png", dpi=170); plt.close()

    t = pd.DataFrame(res["prueba_estres"]).T
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, lab, c in [("F1_macro", "F1 macro", COLORES["azul"]), ("Recall_Critico", "Críticos detectados", COLORES["crit"]),
                        ("Recall_Alto", "Altos detectados", COLORES["alto"])]:
        ax.plot(t.index, t[col], "o-", label=lab, color=c, lw=2)
    ax.set_ylim(0.5, 1.02); ax.set_xlabel("Porcentaje de fallas simuladas del sensor (clase, reloj) + GPS ±300 m")
    ax.set_title("Prueba de estrés del clasificador de riesgo"); ax.legend(); ax.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(REP / "f2_prueba_estres.png", dpi=170); plt.close()

    ti = pd.DataFrame(res["importancia_permutacion"]).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(ti["variable"], ti["importancia"], color=COLORES["alto"])
    ax.set_title("Frente 2 – qué variables pesan más (caída de F1 macro al desordenarlas)"); ax.grid(axis="x", alpha=0.25)
    plt.tight_layout(); plt.savefig(REP / "f2_importancia.png", dpi=170); plt.close()

    # Modelo final con todos los eventos, guardado en formato portable de LightGBM
    final = lgb.LGBMClassifier(**params).fit(X, y)
    final.booster_.save_model(str(MODELS / "frente2_clasificador.txt"))
    (MODELS / "frente2_meta.json").write_text(json.dumps(
        {"columnas": list(X.columns), "niveles": NIVELES, "parametros": {k: v for k, v in params.items()}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    proba = final.predict_proba(X)
    out = df[["evento_id", "fecha_hora", "semana", "estacion_id", "tipo_sensor", "cuadrante_id", "clase",
              "confidence_score", "latitud", "longitud", "en_zona_intangible", "en_zona_amortiguamiento",
              "distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m", "hora", "es_noche",
              "amenazas_7d_estacion", "Nivel_Riesgo_v2"]].copy()
    out["nivel_predicho"] = proba.argmax(1)
    for k, n in enumerate(NIVELES):
        out[f"p_{n.lower()}"] = proba[:, k]
    out.to_csv(DATA / "predicciones_frente2.csv", index=False)
    log("FRENTE 2 · listo")
    return res, out


def guardar_frente1_portable(finales: dict, F: list[str], df: pd.DataFrame) -> None:
    """Guarda los modelos del Frente 1 en el formato de texto nativo de LightGBM.
    A diferencia de un pickle, este formato no depende de la versión de scikit-learn ni de Python,
    por lo que cualquier integrante puede cargarlo sin reentrenar."""
    meta = {"features": F, "categorias": {c: list(df[c].cat.categories) for c in CATEGORICAS}, "horizontes": {}}
    for tgt, suf in [("perdida_7d", "7d"), ("perdida_30d", "30d")]:
        m = finales[tgt]
        m.clf.booster_.save_model(str(MODELS / f"frente1_{suf}_clasificador.txt"))
        m.reg.booster_.save_model(str(MODELS / f"frente1_{suf}_magnitud.txt"))
        meta["horizontes"][suf] = {"smear": m.smear, "umbral": m.umbral, "cal": m.cal}
    (MODELS / "frente1_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Entrena y evalúa los modelos del Hito 3")
    ap.add_argument("--frente", choices=["1", "2", "todo"], default="todo")
    cual = ap.parse_args().frente
    ruta = REP / "metricas_hito3.json"
    salida = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    t0 = time.time()
    if cual in ("1", "todo"):
        salida["frente_1"] = frente1()
    if cual in ("2", "todo"):
        salida["frente_2"], _ = frente2()
    salida["generado"] = time.strftime("%Y-%m-%d %H:%M")
    salida["segundos"] = round(time.time() - t0, 1)
    ruta.write_text(json.dumps(salida, indent=2, ensure_ascii=False, default=str).replace("NaN", "null"), encoding="utf-8")
    log(f"Métricas guardadas en {ruta}")
    r1, r2 = salida.get("frente_1"), salida.get("frente_2")
    if r1:
        for k in ("perdida_7d", "perdida_30d"):
            h = r1[k]["comparativa"]["Hurdle LightGBM"]; d = r1[k]["deteccion"]
            print(f"{k}: AUC {d['ROC_AUC']:.3f} | PR-AUC {d['PR_AUC']:.3f} | MAE {h['MAE']:.3f} | "
                  f"captura@20% {h['Captura_top20']:.1%} | mejora MAE vs persistencia {r1[k]['mejora_MAE_vs_persistencia']:.1%}")
    if r2:
        print("Frente 2:", {k: round(v, 4) for k, v in r2["comparativa"]["LightGBM"].items()})


if __name__ == "__main__":
    main()
