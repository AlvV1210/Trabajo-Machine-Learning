"""Dashboard preliminar del Hito 2 (Streamlit).

Requiere haber ejecutado antes:
    python Hito2_Preparacion_Baseline/preparacion_y_baseline_hito2.py
"""

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
RANKING_PATH = REPORTS_DIR / "hito2_ranking_cuadrantes.csv"
ETIQUETAS_RIESGO = ["Bajo", "Moderado", "Alto", "Crítico"]


@st.cache_data
def cargar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    faltan = [p.name for p in (FR1_PATH, FR2_PATH, SUMMARY_PATH, RANKING_PATH) if not p.exists()]
    if faltan:
        st.error(
            "Faltan artefactos: " + ", ".join(faltan)
            + ". Ejecute primero `python Hito2_Preparacion_Baseline/preparacion_y_baseline_hito2.py`."
        )
        st.stop()
    fr1 = pd.read_csv(FR1_PATH, parse_dates=["fecha"])
    fr2 = pd.read_csv(FR2_PATH)
    ranking = pd.read_csv(RANKING_PATH)
    resumen = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    return fr1, fr2, ranking, resumen


st.set_page_config(page_title="Hito 2 - BIODIVERSITY-GUARD Predict", layout="wide")
st.title("Hito 2: Preparación y Baseline")
st.caption("BIODIVERSITY-GUARD Predict · CRISP-DM fase 3 y 4 (inicial) · Reserva Nacional Tambopata")

fr1, fr2, ranking, resumen = cargar()
m1 = resumen["frente_1"]["metrics"]
m2 = resumen["frente_2"]["metrics"]
s1 = resumen["frente_1"]["summary"]

# ---------------------------------------------------------------------------
st.subheader("Resumen ejecutivo")
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC riesgo 30 días", f"{m1['30d']['riesgo']['roc_auc']:.3f}")
c2.metric("Recall riesgo 30 días", f"{m1['30d']['riesgo']['recall']:.3f}")
c3.metric(
    "Hectáreas captadas (top 20 % cuadrantes)",
    f"{m1['30d']['riesgo']['captura_ha_top20_pct']:.0f} %",
)
c4.metric("F2 · F1 macro / Recall Crítico", f"{m2['f1_macro']:.3f} / {m2['recall_por_clase']['Crítico']:.3f}")
st.write(
    f"Validación temporal: el modelo se entrena con semanas anteriores y se evalúa desde "
    f"**{m1['30d']['periodo_test_desde']}** (datos nunca vistos). Se usa el dataset "
    f"`{s1['archivo_entrada']}` ({s1['cuadrantes']} cuadrantes, {s1['semanas']} semanas)."
)

tab_alertas, tab_f1, tab_f2, tab_datos = st.tabs(
    ["Alertas por cuadrante", "Frente 1 · Deforestación", "Frente 2 · Riesgo de incursión", "Preparación de datos"]
)

# ---------------------------------------------------------------------------
with tab_alertas:
    st.markdown(f"**Predicción desde la semana {ranking['semana'].iloc[0]}** (última semana con alertas satelitales).")
    k = st.slider("Cuadrantes a priorizar", 5, 80, 20, step=5)
    mostrar = ranking.head(k)[
        ["cuadrante_id", "prob_7d", "prob_30d", "ha_riesgo_30d", "biomasa_riesgo_30d_t", "co2_riesgo_30d_t", "nivel_alerta"]
    ].rename(
        columns={
            "cuadrante_id": "Cuadrante",
            "prob_7d": "P(deforestación 7 d)",
            "prob_30d": "P(deforestación 30 d)",
            "ha_riesgo_30d": "ha estimadas 30 d",
            "biomasa_riesgo_30d_t": "Biomasa en riesgo (t)",
            "co2_riesgo_30d_t": "CO₂ en riesgo (t)",
            "nivel_alerta": "Nivel de alerta",
        }
    )
    st.dataframe(mostrar.style.format(precision=2), width="stretch", hide_index=True)
    st.caption(
        f"Cuadrantes con alerta activa a 30 días: {int(ranking['alerta_30d'].sum())} de {len(ranking)} "
        f"(umbral {m1['30d']['riesgo']['umbral']:.2f}, elegido en validación para Recall ≥ 0.80)."
    )

# ---------------------------------------------------------------------------
with tab_f1:
    filas = []
    for h in ("7d", "30d"):
        r, g = m1[h]["riesgo"], m1[h]["magnitud"]
        filas.append(
            {
                "Horizonte": h,
                "ROC-AUC": r["roc_auc"],
                "PR-AUC": r["pr_auc"],
                "Recall": r["recall"],
                "Precisión": r["precision"],
                "F1": r["f1"],
                "Captura ha top 20 %": r["captura_ha_top20_pct"],
                "AUC persistencia": r["persistencia_roc_auc"],
                "MAE (ha)": g["mae_ha"],
                "MAE ingenuo (ha)": g["naive_mae_ha"],
                "RMSE (ha)": g["rmse_ha"],
                "Spearman espacial": g["spearman_ranking_espacial"],
            }
        )
    st.dataframe(pd.DataFrame(filas).style.format(precision=3), width="stretch", hide_index=True)
    col_a, col_b = st.columns(2)
    col_a.image(str(REPORTS_DIR / "hito2_f1_roc.png"))
    col_b.image(str(REPORTS_DIR / "hito2_f1_captura.png"))
    st.image(str(REPORTS_DIR / "hito2_regresion.png"))
    st.image(str(REPORTS_DIR / "hito2_f1_importancia.png"), width=650)

    st.markdown("**Explorar un cuadrante**")
    cuadrante = st.selectbox("Cuadrante", sorted(fr1["cuadrante_id"].unique()), index=0)
    serie = fr1[fr1["cuadrante_id"] == cuadrante].set_index("fecha")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(serie.index, serie["perdida_ha"], width=6, color="#c2551b")
    ax.set(ylabel="ha perdidas por semana", title=f"Historial de pérdida - {cuadrante}")
    ax.grid(alpha=0.2)
    st.pyplot(fig)

# ---------------------------------------------------------------------------
with tab_f2:
    filas = [
        {"Clase": c, "Recall": m2["recall_por_clase"][c], "Precisión": m2["precision_por_clase"][c]}
        for c in ETIQUETAS_RIESGO
    ]
    col_a, col_b = st.columns([1, 1])
    col_a.dataframe(pd.DataFrame(filas).style.format(precision=3), width="stretch", hide_index=True)
    col_a.markdown(
        f"Accuracy {m2['accuracy']:.3f} · F1 macro {m2['f1_macro']:.3f} · ROC-AUC {m2['roc_auc_ovr_macro']:.3f}  \n"
        f"Validación: {m2['validacion']}."
    )
    col_a.info(
        "Las etiquetas de Nivel_Riesgo provienen de reglas heurísticas (clase detectada + zona). "
        "Un desempeño perfecto confirma que el modelo reproduce esas reglas; su validación real se hará "
        "con los datos del módulo de IA en el Hito 3."
    )
    col_b.image(str(REPORTS_DIR / "hito2_clasificacion.png"))
    conteo = fr2["Nivel_Riesgo"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(ETIQUETAS_RIESGO, [conteo.get(i, 0) for i in range(4)], color=["#60a5fa", "#fbbf24", "#f97316", "#ef4444"])
    ax.set(title="Distribución del nivel de riesgo (antes de SMOTE)", ylabel="Anotaciones")
    st.pyplot(fig)

# ---------------------------------------------------------------------------
with tab_datos:
    st.markdown(
        f"""
- Dataset Frente 1: `{s1['archivo_entrada']}` → {s1['filas_preparadas']:,} filas preparadas,
  {s1['n_features']} variables. Se excluyeron {s1['semanas_excluidas_sin_cobertura_satelital']} semanas
  posteriores al 31/08/2026 (sin cobertura del shapefile de alertas PNCB 2026).
- Clima: interpolación lineal temporal por cuadrante (+ relleno adelante/atrás).
- `n_alertas` se excluye de las entradas: `perdida_ha = n_alertas × 0.09` (sería fuga del target).
- Rezagos, sumas móviles (4–52 semanas), semanas desde la última alerta, clima acumulado y estacionalidad.
- Frente 2: imputación por mediana, índice de estrés hídrico, densidad de telemetría anómala,
  one-hot de clase y fuente, SMOTE dentro de cada fold y `class_weight='balanced'`.
"""
    )
    with st.expander("Vista previa Frente 1"):
        st.dataframe(fr1.head(20), width="stretch")
    with st.expander("Vista previa Frente 2"):
        st.dataframe(fr2.head(20), width="stretch")

st.caption("Dashboard generado con Streamlit para la presentación del Hito 2.")
