from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

FR1_PATH = DATA_DIR / "dataset_frente1_hito2_preparado.csv"
FR2_PATH = DATA_DIR / "dataset_frente2_hito2_preparado.csv"
SUMMARY_PATH = REPORTS_DIR / "hito2_resumen.json"


@st.cache_data
def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fr1 = pd.read_csv(FR1_PATH)
    fr2 = pd.read_csv(FR2_PATH)

    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("r", encoding="utf-8") as archivo:
            resumen = json.load(archivo)
    else:
        resumen = {
            "frente_1": {"metrics": {"mae": 0.0, "rmse": 0.0, "r2": 0.0}},
            "frente_2": {"metrics": {"accuracy": 0.0, "weighted_f1": 0.0, "macro_f1": 0.0}},
        }
    return fr1, fr2, resumen


st.set_page_config(page_title="Hito 2 - Dashboard", layout="wide")
st.title("Hito 2: Preparación y Baseline")
st.caption("BIODIVERSITY-GUARD Predict - Fase 3 y 4 inicial de CRISP-DM")

fr1, fr2, resumen = cargar_datos()

reg = resumen["frente_1"]["metrics"]
cls = resumen["frente_2"]["metrics"]

st.subheader("Resumen ejecutivo")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{reg['mae']:.4f}")
col2.metric("RMSE", f"{reg['rmse']:.4f}")
col3.metric("R²", f"{reg['r2']:.4f}")
col4.metric("Macro F1", f"{cls['macro_f1']:.4f}")

st.write("""Se validó la preparación de datos con limpieza de nulos, imputación por mediana, generación de features temporales y de riesgo, y se entrenó un baseline funcional para ambos frentes analíticos.""")

st.subheader("1. Data Preparation")

st.markdown(
    """
    - Limpieza de valores nulos en variables numéricas.
    - Conversión de fechas para extraer mes, año y semana del año.
    - Generación de indicadores como `alerta_observada`, `event_flag`, `lluvia_flag` y señales temporales.
    - Selección de variables relevantes para regresión y clasificación.
    """
)

with st.expander("Vista previa del dataset del Frente 1"):
    st.dataframe(fr1.head(10), width="stretch")

with st.expander("Vista previa del dataset del Frente 2"):
    st.dataframe(fr2.head(10), width="stretch")

st.subheader("2. Baseline")
col_a, col_b = st.columns(2)
col_a.metric("Accuracy", f"{cls['accuracy']:.4f}")
col_b.metric("Weighted F1", f"{cls['weighted_f1']:.4f}")

st.markdown("""
- Frente 1: modelo base de regresión con árbol de decisión.
- Frente 2: modelo base de clasificación con árbol de decisión.
- Split train/test estratificado con semilla 42 para garantizar reproducibilidad.
""")

st.subheader("3. Demostración")

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.scatter(fr1["perdida_ha"], fr1["n_alertas"], alpha=0.6, color="#2f6fed")
ax1.set_title("Relación entre pérdida y número de alertas")
ax1.set_xlabel("perdida_ha")
ax1.set_ylabel("n_alertas")
ax1.grid(alpha=0.2)
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(fr1["perdida_ha"], bins=30, color="#0f766e", edgecolor="white")
ax2.set_title("Distribución de perdida_ha")
ax2.set_xlabel("perdida_ha")
ax2.set_ylabel("Frecuencia")
ax2.grid(axis="y", alpha=0.2)
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(8, 5))
conteo = fr2["Nivel_Riesgo"].value_counts().sort_index()
ax3.bar(["Bajo", "Moderado", "Alto", "Crítico"], [conteo.get(0, 0), conteo.get(1, 0), conteo.get(2, 0), conteo.get(3, 0)], color=["#60a5fa", "#fbbf24", "#f97316", "#ef4444"])
ax3.set_title("Distribución del nivel de riesgo")
ax3.set_ylabel("Cantidad")
ax3.grid(axis="y", alpha=0.25)
st.pyplot(fig3)

st.subheader("4. Indicadores clave")

fr1_summary = fr1[["temperatura_media", "humedad_media", "precipitacion_suma", "n_alertas", "perdida_ha"]].describe().T
st.dataframe(fr1_summary, width="stretch")

st.caption("Dashboard generado con Streamlit para presentación del Hito 2.")
