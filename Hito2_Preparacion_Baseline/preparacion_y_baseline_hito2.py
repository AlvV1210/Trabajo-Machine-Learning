from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
HITO2_DIR = Path(__file__).resolve().parent

FR1_IN = DATA_DIR / "dataset_frente1_hito1.csv"
FR2_IN = DATA_DIR / "dataset_frente2_hito1_final.csv"
FR1_OUT = DATA_DIR / "dataset_frente1_hito2_preparado.csv"
FR2_OUT = DATA_DIR / "dataset_frente2_hito2_preparado.csv"


def crear_directorios() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    HITO2_DIR.mkdir(parents=True, exist_ok=True)


def preparar_frente1() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(FR1_IN)
    print(f"Frente 1: {df.shape[0]} filas iniciales")

    df["semana_inicio"] = df["semana"].astype(str).str.split("/").str[0]
    df["semana_inicio"] = pd.to_datetime(df["semana_inicio"], errors="coerce")
    df["anio"] = df["semana_inicio"].dt.year
    df["mes"] = df["semana_inicio"].dt.month
    df["semana_anio"] = df["semana_inicio"].dt.isocalendar().week.astype(float)

    df["n_alertas"] = pd.to_numeric(df["n_alertas"], errors="coerce").fillna(0)
    df["alerta_observada"] = df["alerta_observada"].astype(bool).astype(int)
    df["perdida_ha"] = pd.to_numeric(df["perdida_ha"], errors="coerce").fillna(0)
    df["temperatura_media"] = pd.to_numeric(df["temperatura_media"], errors="coerce")
    df["humedad_media"] = pd.to_numeric(df["humedad_media"], errors="coerce")
    df["precipitacion_suma"] = pd.to_numeric(df["precipitacion_suma"], errors="coerce")

    medians = {
        "temperatura_media": df["temperatura_media"].median(),
        "humedad_media": df["humedad_media"].median(),
        "precipitacion_suma": df["precipitacion_suma"].median(),
    }
    df["temperatura_media"] = df["temperatura_media"].fillna(medians["temperatura_media"])
    df["humedad_media"] = df["humedad_media"].fillna(medians["humedad_media"])
    df["precipitacion_suma"] = df["precipitacion_suma"].fillna(medians["precipitacion_suma"])

    df["n_alertas_log"] = np.log1p(df["n_alertas"])
    df["perdida_ha_log"] = np.log1p(df["perdida_ha"])
    df["event_flag"] = (df["n_alertas"] > 0).astype(int)
    df["lluvia_flag"] = (df["precipitacion_suma"] > 0).astype(int)
    df["mes_sin_ciclo"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos_ciclo"] = np.cos(2 * np.pi * df["mes"] / 12)

    feature_cols = [
        "temperatura_media",
        "humedad_media",
        "precipitacion_suma",
        "n_alertas",
        "alerta_observada",
        "n_alertas_log",
        "event_flag",
        "lluvia_flag",
        "mes",
        "anio",
        "semana_anio",
        "mes_sin_ciclo",
        "mes_cos_ciclo",
    ]
    prepared = df[feature_cols + ["perdida_ha"]].copy()
    prepared.to_csv(FR1_OUT, index=False)

    target_summary = {
        "filas": int(len(df)),
        "nulos": int(df.isna().sum().sum()),
        "perdida_media": float(df["perdida_ha"].mean()),
        "perdida_maxima": float(df["perdida_ha"].max()),
        "filas_con_perdida": int((df["perdida_ha"] > 0).sum()),
    }
    return prepared, target_summary


def preparar_frente2() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(FR2_IN)
    print(f"Frente 2: {df.shape[0]} filas iniciales")

    df["clase_detectada"] = df["clase_detectada"].fillna("desconocida")
    df["fuente"] = df["fuente"].fillna("desconocida")
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df["distancia_limite_m"] = pd.to_numeric(df["distancia_limite_m"], errors="coerce")
    df["distancia_a_carretera_m"] = pd.to_numeric(df["distancia_a_carretera_m"], errors="coerce")
    df["distancia_a_rio_m"] = pd.to_numeric(df["distancia_a_rio_m"], errors="coerce")

    median_conf = df["confidence_score"].median()
    median_limite = df["distancia_limite_m"].median()
    median_carretera = df["distancia_a_carretera_m"].median()
    median_rio = df["distancia_a_rio_m"].median()

    df["confidence_score"] = df["confidence_score"].fillna(median_conf)
    df["distancia_limite_m"] = df["distancia_limite_m"].fillna(median_limite)
    df["distancia_a_carretera_m"] = df["distancia_a_carretera_m"].fillna(median_carretera)
    df["distancia_a_rio_m"] = df["distancia_a_rio_m"].fillna(median_rio)

    df["en_zona_intangible"] = df["en_zona_intangible"].astype(bool).astype(int)
    df["en_zona_amortiguamiento"] = df["en_zona_amortiguamiento"].astype(bool).astype(int)
    df["cerca_limite"] = (df["distancia_limite_m"] <= 1000).astype(int)
    df["cerca_carretera"] = (df["distancia_a_carretera_m"] <= 5000).astype(int)
    df["cerca_rio"] = (df["distancia_a_rio_m"] <= 5000).astype(int)
    df["distancia_total"] = df["distancia_limite_m"] + df["distancia_a_carretera_m"] + df["distancia_a_rio_m"]

    df = df.dropna(subset=["Nivel_Riesgo"]).copy()
    df["Nivel_Riesgo"] = df["Nivel_Riesgo"].astype(int)

    feature_cols = [
        "confidence_score",
        "en_zona_intangible",
        "en_zona_amortiguamiento",
        "cerca_limite",
        "cerca_carretera",
        "cerca_rio",
        "distancia_limite_m",
        "distancia_a_carretera_m",
        "distancia_a_rio_m",
        "distancia_total",
        "clase_detectada",
        "fuente",
    ]
    prepared = df[feature_cols + ["Nivel_Riesgo"]].copy()
    prepared.to_csv(FR2_OUT, index=False)

    summary = {
        "filas": int(len(df)),
        "clases": dict(df["Nivel_Riesgo"].value_counts().sort_index().astype(int).to_dict()),
        "clase_mayoritaria": int(df["Nivel_Riesgo"].value_counts().idxmax()),
    }
    return prepared, summary


def evaluar_regresion(df: pd.DataFrame) -> tuple[dict, Pipeline]:
    X = df.drop(columns=["perdida_ha"])
    y = df["perdida_ha"]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = [
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        )
    ]
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(max_depth=6, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "mape": float(np.mean(np.abs((y_test - pred) / np.maximum(y_test, 1e-6)))) * 100.0,
    }
    return metrics, pipeline


def evaluar_clasificacion(df: pd.DataFrame) -> tuple[dict, Pipeline]:
    X = df.drop(columns=["Nivel_Riesgo"])
    y = df["Nivel_Riesgo"].astype(int)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = [
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        )
    ]
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=6, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "classification_report": report,
    }
    return metrics, pipeline


def plot_regresion(y_true: pd.Series, y_pred: np.ndarray, metrics: dict) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, s=25, color="#2f6fed")
    min_y = min(y_true.min(), y_pred.min())
    max_y = max(y_true.max(), y_pred.max())
    plt.plot([min_y, max_y], [min_y, max_y], linestyle="--", color="darkred", linewidth=1.5, label="y = x")
    plt.title("Frente 1 - Predicción de perdida_ha")
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "hito2_regresion.png", dpi=200)
    plt.close()


def plot_clasificacion(y_true: pd.Series, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Frente 2 - Matriz de confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    tick_labels = ["Bajo", "Moderado", "Alto", "Crítico"]
    plt.xticks(range(4), tick_labels)
    plt.yticks(range(4), tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "hito2_clasificacion.png", dpi=200)
    plt.close()


def guardar_dashboard(reg_metrics: dict, cls_metrics: dict) -> None:
    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Hito 2 - Dashboard preliminar</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                color: #1f2937;
                margin: 0;
                padding: 24px;
            }}
            .container {{ max-width: 1100px; margin: auto; }}
            h1, h2 {{ color: #0f172a; }}
            .cards {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
                margin: 24px 0;
            }}
            .card {{
                background: white;
                border-radius: 10px;
                padding: 18px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            }}
            .metric {{ font-size: 1.8rem; font-weight: bold; color: #1d4ed8; margin-top: 8px; }}
            .panel {{ background: white; border-radius: 10px; padding: 18px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
            img {{ width: 100%; border-radius: 8px; border: 1px solid #dfe7f5; }}
            ul {{ line-height: 1.8; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Hito 2 - Preparación y Baseline</h1>
            <p>Informe inicial de la fase 3 y 4 de CRISP-DM para BIODIVERSITY-GUARD Predict.</p>

            <div class="cards">
                <div class="card">
                    <h3>Frente 1: Regresión</h3>
                    <div class="metric">MAE {reg_metrics['mae']:.2f}</div>
                    <div>RMSE: {reg_metrics['rmse']:.2f}</div>
                    <div>R²: {reg_metrics['r2']:.3f}</div>
                </div>
                <div class="card">
                    <h3>Frente 2: Clasificación</h3>
                    <div class="metric">Accuracy {cls_metrics['accuracy']:.3f}</div>
                    <div>Macro F1: {cls_metrics['macro_f1']:.3f}</div>
                    <div>Weighted F1: {cls_metrics['weighted_f1']:.3f}</div>
                </div>
            </div>

            <div class="panel">
                <h2>Data Preparation</h2>
                <ul>
                    <li>Se aplicó limpieza de valores nulos usando mediana para variables numéricas.</li>
                    <li>Se transformaron columnas de fecha y se generaron variables temporales (mes, año, semana del año).</li>
                    <li>Se agregaron indicadores de riesgo espacial y de cobertura para mejorar la señal predictiva.</li>
                    <li>Se codificaron variables categóricas para que el modelo pueda operar sobre texto y tipos de cobertura.</li>
                </ul>
            </div>

            <div class="panel">
                <h2>Baseline</h2>
                <ul>
                    <li>Frente 1: regresor de árbol de decisión con profundidad 6.</li>
                    <li>Frente 2: clasificador de árbol de decisión con profundidad 6.</li>
                    <li>Se dividió en train/test con semilla 42 para asegurar reproducibilidad.</li>
                </ul>
            </div>

            <div class="panel">
                <h2>Visualizaciones predictivas</h2>
                <img src="hito2_regresion.png" alt="Predicción Frente 1" />
                <img src="hito2_clasificacion.png" alt="Matriz de confusión Frente 2" style="margin-top: 18px;" />
            </div>
        </div>
    </body>
    </html>
    """
    with (REPORTS_DIR / "hito2_dashboard.html").open("w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    crear_directorios()

    fr1_df, fr1_summary = preparar_frente1()
    fr2_df, fr2_summary = preparar_frente2()

    reg_metrics, reg_model = evaluar_regresion(fr1_df)
    cls_metrics, cls_model = evaluar_clasificacion(fr2_df)

    X_reg = fr1_df.drop(columns=["perdida_ha"])
    y_reg = fr1_df["perdida_ha"]
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_model.fit(X_reg_train, y_reg_train)
    y_pred_reg = reg_model.predict(X_reg_test)
    plot_regresion(y_reg_test, y_pred_reg, reg_metrics)

    X_cls = fr2_df.drop(columns=["Nivel_Riesgo"])
    y_cls = fr2_df["Nivel_Riesgo"]
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    cls_model.fit(X_cls_train, y_cls_train)
    y_pred_cls = cls_model.predict(X_cls_test)
    plot_clasificacion(y_cls_test, y_pred_cls)

    guardar_dashboard(reg_metrics, cls_metrics)

    salida = {
        "frente_1": {
            "summary": fr1_summary,
            "metrics": reg_metrics,
            "dataset_path": str(FR1_OUT),
        },
        "frente_2": {
            "summary": fr2_summary,
            "metrics": cls_metrics,
            "dataset_path": str(FR2_OUT),
        },
    }
    with (REPORTS_DIR / "hito2_resumen.json").open("w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print("=== HITO 2 - PREPARACION Y BASELINE ===")
    print(f"Dataset Frente 1 preparado: {FR1_OUT}")
    print(f"Dataset Frente 2 preparado: {FR2_OUT}")
    print("Métricas Frente 1:")
    print(json.dumps(reg_metrics, indent=2, ensure_ascii=False))
    print("Métricas Frente 2:")
    print(json.dumps({k: v for k, v in cls_metrics.items() if k in {"accuracy", "weighted_f1", "macro_f1"}}, indent=2, ensure_ascii=False))
    print(f"Dashboard creado: {REPORTS_DIR / 'hito2_dashboard.html'}")


if __name__ == "__main__":
    main()
