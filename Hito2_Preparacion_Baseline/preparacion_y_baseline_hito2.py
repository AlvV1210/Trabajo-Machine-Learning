"""Hito 2 (versión corregida) - Preparación de datos y baseline.

Corrección principal respecto a la versión entregada:
    * El Frente 1 ahora usa ``data/dataset_frente1_hito1_final.csv`` (273 cuadrantes,
      22 columnas: clima + alertas + MapBiomas, PNCB, ZEE, INEI) en lugar de
      ``dataset_frente1_hito1.csv`` (167 cuadrantes, solo clima y alertas).
    * La variable objetivo se desplaza al futuro (7 y 30 días) y el split es
      temporal (80 % semanas iniciales / 20 % finales): ya no se mezclan semanas
      futuras en el entrenamiento como ocurría con ``train_test_split`` aleatorio.
    * El Frente 2 normaliza la taxonomía (motosierra, arma blanca, vehículo ligero,
      persona, cazador), descarta el texto de exportación de Roboflow y deja de usar
      ``fuente`` como predictor (es metadato de procedencia, no una señal de campo).

Salidas: data/dataset_frente{1,2}_hito2_preparado.csv, reports/hito2_*.png,
reports/hito2_resumen.json, reports/hito2_dashboard.html.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Hito3_Modelado_Avanzado_Evaluacion"))
from features import (F2_BOOL, F2_NUM, NIVELES, cargar_frente1, cargar_frente2,  # noqa: E402
                      construir_features_frente1)

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FR1_IN = DATA_DIR / "dataset_frente1_hito1_final.csv"   # <- dataset correcto
FR1_OUT = DATA_DIR / "dataset_frente1_hito2_preparado.csv"
FR2_OUT = DATA_DIR / "dataset_frente2_hito2_preparado.csv"
SEED = 42
PESO_EVENTO = 30  # penalización de muestras con pérdida > 0 (inflación de ceros)


def preparar_frente1() -> tuple[pd.DataFrame, list[str], dict]:
    base = cargar_frente1(FR1_IN)
    nulos_antes = base[["temperatura_media", "humedad_media", "precipitacion_suma"]].isna().sum().to_dict()
    df, F = construir_features_frente1(base)
    df = pd.get_dummies(df, columns=["vocacion_zee", "causa_principal"], drop_first=True, dtype=float)
    F = [c for c in df.columns if c in F or c.startswith(("vocacion_zee_", "causa_principal_"))]
    df[["cuadrante_id", "semana", "semana_inicio", *F, "perdida_ha", "perdida_7d", "perdida_30d",
        "evento_7d", "evento_30d"]].to_csv(FR1_OUT, index=False)
    resumen = {
        "dataset": FR1_IN.name, "filas": int(len(df)), "cuadrantes": int(df["cuadrante_id"].nunique()),
        "semanas": int(df["semana_inicio"].nunique()), "variables_modelo": len(F),
        "nulos_clima_antes_de_imputar": {k: int(v) for k, v in nulos_antes.items()},
        "porcentaje_semanas_sin_perdida": float((df["perdida_ha"] == 0).mean()),
    }
    return df, F, resumen


def baseline_frente1(df: pd.DataFrame, F: list[str]) -> dict:
    semanas = np.sort(df["semana_inicio"].unique())
    corte = semanas[int(len(semanas) * 0.8)]
    out = {}
    for tgt, ev, h in [("perdida_7d", "evento_7d", 1), ("perdida_30d", "evento_30d", 4)]:
        D = df.dropna(subset=[tgt])
        tr = D[D["semana_inicio"] < semanas[int(len(semanas) * 0.8) - h]]  # sin solapamiento con test
        te = D[D["semana_inicio"] >= corte]
        sc = StandardScaler().fit(tr[F])  # se ajusta SOLO con entrenamiento
        Xtr, Xte = sc.transform(tr[F]), sc.transform(te[F])
        peso = np.where(tr[tgt] > 0, PESO_EVENTO, 1)
        modelos = {
            "Árbol de decisión": DecisionTreeRegressor(max_depth=6, random_state=SEED),
            "Random Forest ponderado": RandomForestRegressor(n_estimators=150, min_samples_leaf=10, max_features=0.5,
                                                             n_jobs=2, random_state=SEED),
        }
        res = {}
        naive = te["perdida_lag0"] if h == 1 else te["perdida_suma_4s"]
        res["Persistencia"] = {"MAE": float(mean_absolute_error(te[tgt], naive)),
                               "RMSE": float(np.sqrt(mean_squared_error(te[tgt], naive))),
                               "R2": float(r2_score(te[tgt], naive))}
        for nombre, m in modelos.items():
            m.fit(Xtr, tr[tgt], sample_weight=peso if "ponderado" in nombre else None)
            p = np.clip(m.predict(Xte), 0, None)
            clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED).fit(Xtr, tr[ev])
            res[nombre] = {"MAE": float(mean_absolute_error(te[tgt], p)),
                           "RMSE": float(np.sqrt(mean_squared_error(te[tgt], p))),
                           "R2": float(r2_score(te[tgt], p)),
                           "ROC_AUC_evento": float(roc_auc_score(te[ev], clf.predict_proba(Xte)[:, 1]))}
            if nombre == "Árbol de decisión":
                ultimo = te.assign(pred=p)
        out[tgt] = res
        serie = ultimo.groupby("semana_inicio")[[tgt, "pred"]].sum()
        plt.figure(figsize=(11, 4))
        plt.plot(serie.index, serie[tgt], color="black", lw=2, label="Real")
        plt.plot(serie.index, serie["pred"], ls="--", color="#1f4e79", lw=2, label="Árbol baseline")
        plt.title(f"Hito 2 – baseline {tgt} (suma de 273 cuadrantes, prueba temporal)")
        plt.ylabel("ha"); plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"hito2_regresion_{tgt}.png", dpi=170); plt.close()
    return out


def preparar_y_baseline_frente2() -> tuple[dict, dict]:
    df = cargar_frente2()
    df.to_csv(FR2_OUT, index=False)
    X = pd.get_dummies(df[F2_NUM + F2_BOOL + ["clase"]], columns=["clase"], drop_first=True, dtype=float)
    y = df["Nivel_Riesgo_v2"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_b, ytr_b = SMOTE(random_state=SEED).fit_resample(sc.transform(Xtr), ytr)
    res = {}
    for nombre, m in {"Árbol de decisión": DecisionTreeClassifier(max_depth=6, random_state=SEED),
                      "Random Forest + SMOTE": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                                      n_jobs=2, random_state=SEED)}.items():
        m.fit(Xtr_b, ytr_b)
        p = m.predict(sc.transform(Xte))
        res[nombre] = {"Accuracy": float(accuracy_score(yte, p)), "F1_macro": float(f1_score(yte, p, average="macro")),
                       "Recall_Critico": float(recall_score(yte, p, labels=[3], average="macro"))}
        cm = confusion_matrix(yte, p, labels=[0, 1, 2, 3])
    plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks(range(4), NIVELES); plt.yticks(range(4), NIVELES)
    plt.xlabel("Predicción"); plt.ylabel("Real"); plt.title("Hito 2 – Frente 2 (RF + SMOTE)")
    plt.tight_layout(); plt.savefig(REPORTS_DIR / "hito2_clasificacion.png", dpi=170); plt.close()
    resumen = {"filas": int(len(df)), "clases": {NIVELES[k]: int(v) for k, v in y.value_counts().sort_index().items()},
               "balance_tras_SMOTE": {NIVELES[k]: int(v) for k, v in pd.Series(ytr_b).value_counts().sort_index().items()}}
    return resumen, res


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    df1, F, r1 = preparar_frente1()
    m1 = baseline_frente1(df1, F)
    r2, m2 = preparar_y_baseline_frente2()
    salida = {"frente_1": {"summary": r1, "metrics": m1}, "frente_2": {"summary": r2, "metrics": m2}}
    (REPORTS_DIR / "hito2_resumen.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
