"""Dashboard preliminar del Hito 2 (versión corregida con el dataset final).

El dashboard completo del Hito 3 (mapa, plan de patrullaje, alertas y KPIs) está en
Hito3_Modelado_Avanzado_Evaluacion/dashboard_hito3_streamlit.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
resumen = json.loads((REPORTS / "hito2_resumen.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="Hito 2 - Baseline", layout="wide")
st.title("Hito 2 · Preparación de datos y baseline (corregido)")
s1 = resumen["frente_1"]["summary"]
st.caption(f"Frente 1: {s1['dataset']} · {s1['filas']:,} filas · {s1['cuadrantes']} cuadrantes · {s1['semanas']} semanas")

for tgt, titulo in [("perdida_7d", "Frente 1 · horizonte 7 días"), ("perdida_30d", "Frente 1 · horizonte 30 días")]:
    st.subheader(titulo)
    st.dataframe(pd.DataFrame(resumen["frente_1"]["metrics"][tgt]).T.round(4), width="stretch")
    img = REPORTS / f"hito2_regresion_{tgt}.png"
    if img.exists():
        st.image(str(img))

st.subheader("Frente 2 · clasificación de riesgo")
st.dataframe(pd.DataFrame(resumen["frente_2"]["metrics"]).T.round(4), width="stretch")
st.image(str(REPORTS / "hito2_clasificacion.png"), width=480)
