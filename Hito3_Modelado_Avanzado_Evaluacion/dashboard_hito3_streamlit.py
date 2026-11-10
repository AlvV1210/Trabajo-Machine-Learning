"""BIODIVERSITY-GUARD Predict · Centro de Mando (Dashboard Hito 3).

Ejecutar desde la raíz del repositorio:
    python Hito3_Modelado_Avanzado_Evaluacion/modelado_avanzado_hito3.py   # entrena y genera predicciones
    python Hito3_Modelado_Avanzado_Evaluacion/base_datos_comun.py init       # crea la BD común
    streamlit run Hito3_Modelado_Avanzado_Evaluacion/dashboard_hito3_streamlit.py

Funcionalidades para el guardaparque / jefe de ANP:
    1. Mapa de cuadrantes 5x5 km con la capa de riesgo elegida y las incursiones detectadas.
    2. Plan de patrullaje: ranking por Índice de Prioridad de Patrullaje (IPP) con la estrategia sugerida
       y la capacidad real de la jefatura (N cuadrantes por semana).
    3. Feed de alertas de la BD común (módulo IA -> modelo -> alerta), con simulación en vivo.
    4. KPIs de seguimiento y control (operativos, ambientales y de desempeño del modelo).
    5. Ficha del cuadrante: historia, pronóstico y perfil de presiones.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
sys.path.insert(0, str(AQUI))
DATA = ROOT / "data"
REP = ROOT / "reports" / "hito3"
DB = DATA / "biodiversity_guard.db"

T_C_POR_HA, CO2_POR_C = 138.9, 44 / 12
NIVELES = ["Bajo", "Moderado", "Alto", "Crítico"]
COL_NIVEL = {"Bajo": "#2e7d32", "Moderado": "#f9a825", "Alto": "#ef6c00", "Crítico": "#c62828"}
ESTRATEGIA = {
    "mineria": "Interdicción con FEMA/PNP + control fluvial de dragas",
    "agropecuario": "Patrullaje terrestre + verificación de quemas y cambio de uso",
    "infraestructura": "Control de trochas y puestos de vigilancia vial",
    "ninguna": "Patrullaje preventivo y mantenimiento de cámaras trampa",
}

st.set_page_config(page_title="BIODIVERSITY-GUARD Predict", page_icon="🌳", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem;}
div[data-testid="stMetric"] {background: rgba(31,78,121,.06); border: 1px solid rgba(31,78,121,.18);
  border-radius: 10px; padding: 10px 14px;}
div[data-testid="stMetricLabel"] p {font-size: .82rem;}
.kpi-note {font-size:.78rem; opacity:.75; margin-top:-6px}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cargar_estatico():
    from grilla_cuadrantes import obtener_grilla

    grilla = pd.read_csv(DATA / "cuadrantes_grid.csv") if (DATA / "cuadrantes_grid.csv").exists() else obtener_grilla()
    f1 = pd.read_csv(DATA / "predicciones_frente1.csv")
    hist = pd.read_csv(DATA / "dataset_frente1_hito1_final.csv",
                       usecols=["semana", "cuadrante_id", "perdida_ha", "precipitacion_suma", "temperatura_media"])
    hist["semana_inicio"] = pd.to_datetime(hist["semana"].str[:10])
    f2 = pd.read_csv(DATA / "predicciones_frente2.csv")
    metr = json.loads((REP / "metricas_hito3.json").read_text(encoding="utf-8"))
    wf = pd.read_csv(DATA / "walkforward_perdida_30d.csv", parse_dates=["semana_inicio"])
    return grilla, f1, hist, f2, metr, wf


def asignar_cuadrante(lat: pd.Series, lon: pd.Series, grilla: pd.DataFrame) -> pd.Series:
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, 32719, always_xy=True).transform(lon.to_numpy(), lat.to_numpy())
    llave = {(int(a // 5000), int(b // 5000)): c for c, a, b in grilla[["cuadrante_id", "x_utm", "y_utm"]].itertuples(index=False)}
    return pd.Series([llave.get((int(a // 5000), int(b // 5000)), None) for a, b in zip(x, y)], index=lat.index)


def leer_bd() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DB.exists():
        return pd.DataFrame(), pd.DataFrame()
    with sqlite3.connect(DB) as con:
        alertas = pd.read_sql("SELECT * FROM alertas ORDER BY creada_utc DESC", con)
        vivo = pd.read_sql("""SELECT d.*, p.nivel_predicho, p.nivel_nombre, p.p_critico, p.cuadrante_id
                              FROM detecciones d LEFT JOIN predicciones_riesgo p USING(evento_id)""", con)
    return alertas, vivo


def geojson(grilla: pd.DataFrame) -> dict:
    feats = []
    for r in grilla.itertuples():
        ring = [[r.lon_sw, r.lat_sw], [r.lon_se, r.lat_se], [r.lon_ne, r.lat_ne], [r.lon_nw, r.lat_nw], [r.lon_sw, r.lat_sw]]
        feats.append({"type": "Feature", "id": r.cuadrante_id, "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": feats}


grilla, f1, hist, f2, metr, wf = cargar_estatico()
alertas_bd, vivo = leer_bd()

# Riesgo de incursión por sector (Frente 2 histórico + detecciones vivas de la BD)
f2 = f2.assign(cuadrante_id=asignar_cuadrante(f2["latitud"], f2["longitud"], grilla),
               nivel=f2["nivel_predicho"].map(dict(enumerate(NIVELES))))
eventos = f2[["cuadrante_id", "clase", "confidence_score", "latitud", "longitud", "nivel", "p_alto", "p_crítico"]].rename(
    columns={"p_crítico": "p_critico"})
if not vivo.empty and "nivel_nombre" in vivo:
    v = vivo.dropna(subset=["nivel_nombre"])
    eventos = pd.concat([eventos, pd.DataFrame({
        "cuadrante_id": v["cuadrante_id"], "clase": v["clase_detectada"], "confidence_score": v["confidence_score"],
        "latitud": v["latitud"], "longitud": v["longitud"], "nivel": v["nivel_nombre"],
        "p_alto": np.where(v["nivel_nombre"] == "Alto", 1.0, 0.0), "p_critico": v["p_critico"]})], ignore_index=True)
iri = (eventos.assign(w=(0.5 * eventos["p_alto"] + eventos["p_critico"]) * eventos["confidence_score"])
       .groupby("cuadrante_id").agg(iri=("w", "sum"), incursiones_criticas=("nivel", lambda s: (s == "Crítico").sum()),
                                    incursiones_altas=("nivel", lambda s: (s == "Alto").sum()), detecciones=("nivel", "size")))

panel = grilla.merge(f1, on="cuadrante_id", how="left").merge(iri, on="cuadrante_id", how="left")
for c in ["iri", "incursiones_criticas", "incursiones_altas", "detecciones"]:
    panel[c] = panel[c].fillna(0)

# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🌳 BIODIVERSITY-GUARD")
    st.caption("Predict · Reserva Nacional Tambopata")
    horizonte = st.radio("Horizonte de pronóstico", ["30 días", "7 días"], horizontal=True)
    suf = "30d" if horizonte == "30 días" else "7d"
    zonas = st.multiselect("Zona", sorted(panel["zona"].unique()), default=sorted(panel["zona"].unique()))
    capacidad = st.slider("Capacidad de patrullaje (cuadrantes/semana)", 5, 80, 25, 5)
    peso_def = st.slider("Peso deforestación vs. incursión en el IPP", 0.0, 1.0, 0.6, 0.05)
    efectividad = st.slider("Efectividad esperada de la intervención", 0.1, 1.0, 0.5, 0.05,
                            help="Fracción de la pérdida esperada que se evita cuando se patrulla el cuadrante.")
    st.divider()
    st.subheader("Base de datos común")
    if st.button("📡 Recibir detecciones del módulo IA", width="stretch"):
        from base_datos_comun import procesar_nuevas, simular_modulo_ia

        simular_modulo_ia(20)
        procesar_nuevas()
        st.cache_data.clear()
        st.rerun()
    st.caption(f"BD: {'conectada' if DB.exists() else 'no inicializada'} · "
               f"{len(vivo)} detecciones en vivo · {len(alertas_bd)} alertas")

# Índice de Prioridad de Patrullaje
P = panel[panel["zona"].isin(zonas)].copy()
norm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)  # noqa: E731
P["IPP"] = 100 * (peso_def * norm(P[f"prob_evento_{suf}"] * np.log1p(P[f"ha_esperadas_{suf}"]) + P[f"prob_evento_{suf}"] * 0.2)
                  + (1 - peso_def) * norm(np.log1p(P["iri"])))
P = P.sort_values("IPP", ascending=False)
P["rank"] = np.arange(1, len(P) + 1)
P["priorizado"] = P["rank"] <= capacidad
P["estrategia"] = np.where(P["incursiones_criticas"] > 0, "Respuesta inmediata a incursión + ", "") + \
    P["causa_principal"].map(ESTRATEGIA).fillna(ESTRATEGIA["ninguna"])

# --------------------------------------------------------------------------- #
# Encabezado y KPIs
# --------------------------------------------------------------------------- #
semana = str(f1["semana"].iloc[0])
st.title("Centro de Mando · Alerta temprana de deforestación e incursiones")
st.caption(f"Pronóstico emitido al cierre de la semana {semana} · {len(P)} cuadrantes de 5×5 km · "
           f"geometría: {grilla['origen'].iloc[0]}")

ha_tot = P[f"ha_esperadas_{suf}"].sum()
ha_cubiertas = P.loc[P["priorizado"], f"ha_esperadas_{suf}"].sum()
co2_riesgo = ha_tot * T_C_POR_HA * CO2_POR_C
co2_evitable = ha_cubiertas * efectividad * T_C_POR_HA * CO2_POR_C
en_alerta = int(P[f"alerta_{suf}"].sum())
criticas_abiertas = int(((alertas_bd["nivel"] == "Crítico") & (alertas_bd["estado"] == "abierta")).sum()) if not alertas_bd.empty else 0
m30 = metr["frente_1"]["perdida_30d"]
k = st.columns(6)
k[0].metric(f"Hectáreas en riesgo ({horizonte})", f"{ha_tot:,.1f} ha")
k[1].metric("CO₂ en riesgo", f"{co2_riesgo/1000:,.1f} kt")
k[2].metric("Cuadrantes en alerta", f"{en_alerta}", f"{en_alerta/len(P):.0%} del área")
k[3].metric("Cobertura del plan", f"{ha_cubiertas/ha_tot:.0%}" if ha_tot else "–", f"con {capacidad} cuadrantes", delta_color="off")
k[4].metric("CO₂ evitable (C_evitado)", f"{co2_evitable/1000:,.1f} kt", f"efectividad {efectividad:.0%}", delta_color="off")
k[5].metric("Alertas críticas abiertas", f"{criticas_abiertas}")

tabs = st.tabs(["🗺️ Mapa de riesgo", "🚩 Plan de patrullaje", "🔔 Alertas en vivo", "📊 KPIs y desempeño", "🔎 Ficha de cuadrante"])

# --------------------------------------------------------------------------- #
# 1. Mapa
# --------------------------------------------------------------------------- #
with tabs[0]:
    c1, c2 = st.columns([3, 1])
    with c2:
        capa = st.radio("Capa", ["Índice de prioridad (IPP)", "Probabilidad de pérdida", "Hectáreas esperadas",
                                 "Riesgo de incursión (IRI)", "Presión minera (MapBiomas)"])
        mostrar = st.multiselect("Detecciones sobre el mapa", NIVELES, default=["Crítico"])
        solo_plan = st.checkbox("Resaltar cuadrantes del plan", value=True)
        fondo = st.radio("Fondo", ["Mapa base (OpenStreetMap)", "Cuadrícula UTM sin conexión"],
                         help="Use la cuadrícula UTM en campo, sin internet.")
        st.caption("Celdas con borde grueso = priorizadas en el plan de patrullaje de esta semana.")
    col = {"Índice de prioridad (IPP)": "IPP", "Probabilidad de pérdida": f"prob_evento_{suf}",
           "Hectáreas esperadas": f"ha_esperadas_{suf}", "Riesgo de incursión (IRI)": "iri",
           "Presión minera (MapBiomas)": "mb_mineria"}[capa]
    escala = "YlOrRd" if col != "mb_mineria" else "Oranges"
    ev = eventos[eventos["nivel"].isin(mostrar)]
    if len(ev) > 1500:
        ev = ev.sample(1500, random_state=1)
    hover = {"zona": True, f"prob_evento_{suf}": ":.0%", f"ha_esperadas_{suf}": ":.2f", "IPP": ":.0f",
             "incursiones_criticas": True, "causa_principal": True, "cuadrante_id": False}
    if fondo == "Mapa base (OpenStreetMap)":
        fig = px.choropleth_map(P, geojson=geojson(P), locations="cuadrante_id", color=col, color_continuous_scale=escala,
                                map_style="open-street-map", opacity=0.62, zoom=8.3,
                                center={"lat": P["lat_c"].mean(), "lon": P["lon_c"].mean()},
                                hover_name="cuadrante_id", hover_data=hover)
        Linea, Punto = go.Scattermap, go.Scattermap
    else:  # vectorial sin conexión: cuadrícula UTM 19S dibujada localmente (sin teselas, sin WebGL)
        from pyproj import Transformer

        utm = Transformer.from_crs(4326, 32719, always_xy=True)
        cx, cy = (P["x_utm"] // 5000).astype(int), (P["y_utm"] // 5000).astype(int)
        xs, ys = np.arange(cx.min(), cx.max() + 1), np.arange(cy.min(), cy.max() + 1)
        Z = np.full((len(ys), len(xs)), np.nan); T = np.full(Z.shape, "", dtype=object)
        for (a, b_), v, q in zip(zip(cx, cy), P[col], P["cuadrante_id"]):
            Z[b_ - ys[0], a - xs[0]] = v
            T[b_ - ys[0], a - xs[0]] = q
        fig = go.Figure(go.Heatmap(x=(xs * 5 + 2.5), y=(ys * 5 + 2.5), z=Z, text=T, colorscale=escala, xgap=1, ygap=1,
                                   colorbar=dict(title=capa.split(" (")[0], len=0.6),
                                   hovertemplate="%{text}<br>valor %{z:.2f}<extra></extra>"))
        if solo_plan:
            xl, yl = [], []
            for r in P[P["priorizado"]].itertuples():
                x0, y0 = r.x_utm / 1000, r.y_utm / 1000
                xl += [x0, x0 + 5, x0 + 5, x0, x0, None]; yl += [y0, y0, y0 + 5, y0 + 5, y0, None]
            fig.add_trace(go.Scatter(x=xl, y=yl, mode="lines", line=dict(color="#1f4e79", width=2.5),
                                     name="Plan de patrullaje", hoverinfo="skip"))
        for nivel in mostrar:
            e = ev[ev["nivel"] == nivel]
            ex, ey = utm.transform(e["longitud"].to_numpy(), e["latitud"].to_numpy())
            fig.add_trace(go.Scatter(x=np.asarray(ex) / 1000, y=np.asarray(ey) / 1000, mode="markers",
                                     name=f"Detección {nivel}", marker=dict(size=4, color=COL_NIVEL[nivel], opacity=0.6),
                                     text=e["clase"], hovertemplate="%{text}<extra>" + nivel + "</extra>"))
        fig.update_yaxes(scaleanchor="x", scaleratio=1, title="Norte UTM 19S (km)")
        fig.update_xaxes(title="Este UTM 19S (km)")
        fig.update_layout(plot_bgcolor="#eef2f5")
    if fondo == "Mapa base (OpenStreetMap)":
        if solo_plan:
            plan = P[P["priorizado"]]
            lon, lat = [], []
            for r in plan.itertuples():
                lon += [r.lon_sw, r.lon_se, r.lon_ne, r.lon_nw, r.lon_sw, None]
                lat += [r.lat_sw, r.lat_se, r.lat_ne, r.lat_nw, r.lat_sw, None]
            fig.add_trace(Linea(lon=lon, lat=lat, mode="lines", line=dict(width=2.5, color="#1f4e79"),
                                name="Plan de patrullaje", hoverinfo="skip"))
        for nivel in mostrar:
            e = ev[ev["nivel"] == nivel]
            fig.add_trace(Punto(lon=e["longitud"], lat=e["latitud"], mode="markers", name=f"Detección {nivel}",
                                marker=dict(size=5, color=COL_NIVEL[nivel], opacity=0.7),
                                text=e["clase"], hovertemplate="%{text}<extra>" + nivel + "</extra>"))
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(y=0.02, x=0.01, bgcolor="rgba(255,255,255,.8)"),
                      coloraxis_colorbar=dict(title=capa.split(" (")[0], len=0.6))
    if fondo != "Mapa base (OpenStreetMap)":
        fig.update_layout(margin=dict(l=60, r=10, t=30, b=50), legend=dict(orientation="h", x=0, y=1.06))
    with c1:
        st.plotly_chart(fig, width="stretch")
    if (grilla["geometria"] != "validada").any():
        st.info("Ubicación de cuadrantes: 89 celdas validadas contra la telemetría (error medio de cobertura 0.02); "
                "el resto se extrapola por el orden fila-columna de la cuadrícula. Coloque los shapefiles SERNANP en "
                "`dataset diverso foco2/` y ejecute `grilla_cuadrantes.py` para la geometría exacta.", icon="ℹ️")

# --------------------------------------------------------------------------- #
# 2. Plan de patrullaje
# --------------------------------------------------------------------------- #
with tabs[1]:
    st.subheader(f"Plan priorizado · {capacidad} cuadrantes ({capacidad * 25:,} km²)")
    a, b, c = st.columns(3)
    a.metric("Pérdida esperada cubierta", f"{ha_cubiertas:,.1f} ha", f"{ha_cubiertas/ha_tot:.0%} del total" if ha_tot else None)
    b.metric("Incursiones críticas en el plan", int(P.loc[P['priorizado'], 'incursiones_criticas'].sum()),
             f"de {int(P['incursiones_criticas'].sum())}", delta_color="off")
    cap_hist = m30["comparativa"]["Hurdle LightGBM"]["Captura_top20"]
    c.metric("Captura histórica con top-20 %", f"{cap_hist:.0%}", "vs 20 % patrullaje aleatorio")
    tabla = P[P["priorizado"]][["rank", "cuadrante_id", "zona", "IPP", f"prob_evento_{suf}", f"ha_esperadas_{suf}",
                                "incursiones_criticas", "causa_principal", "estrategia"]]
    st.dataframe(tabla, hide_index=True, width="stretch", column_config={
        "rank": st.column_config.NumberColumn("#", width="small"),
        "IPP": st.column_config.ProgressColumn("IPP", min_value=0, max_value=100, format="%.0f"),
        f"prob_evento_{suf}": st.column_config.ProgressColumn("P(pérdida)", min_value=0, max_value=1, format="percent"),
        f"ha_esperadas_{suf}": st.column_config.NumberColumn("ha esperadas", format="%.2f"),
        "incursiones_criticas": st.column_config.NumberColumn("Incursiones críticas"),
        "estrategia": st.column_config.TextColumn("Estrategia sugerida", width="large")})
    st.download_button("⬇️ Descargar hoja de ruta (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                       f"plan_patrullaje_{semana[:10]}.csv", "text/csv")
    resumen = P[P["priorizado"]].groupby("causa_principal")[f"ha_esperadas_{suf}"].sum().reset_index()
    st.plotly_chart(px.bar(resumen, x="causa_principal", y=f"ha_esperadas_{suf}", color="causa_principal",
                           title="Pérdida esperada del plan por causa principal (define el tipo de brigada)",
                           labels={f"ha_esperadas_{suf}": "ha", "causa_principal": ""}).update_layout(showlegend=False, height=330),
                    width="stretch")

# --------------------------------------------------------------------------- #
# 3. Alertas
# --------------------------------------------------------------------------- #
with tabs[2]:
    if alertas_bd.empty:
        st.warning("La BD común no está inicializada. Ejecute `python Hito3_Modelado_Avanzado_Evaluacion/base_datos_comun.py init`.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Alertas abiertas", int((alertas_bd["estado"] == "abierta").sum()))
        b.metric("Desde Frente 1 (deforestación)", int((alertas_bd["origen"] == "Frente 1").sum()))
        c.metric("Desde Frente 2 (incursión)", int((alertas_bd["origen"] == "Frente 2").sum()))
        d.metric("Detecciones procesadas", int((vivo["estado"] == "procesada").sum()) if not vivo.empty else 0)
        filtro = st.multiselect("Nivel", ["Crítico", "Alto"], default=["Crítico", "Alto"])
        vista = alertas_bd[alertas_bd["nivel"].isin(filtro)]
        st.dataframe(vista[["creada_utc", "nivel", "origen", "cuadrante_id", "detalle", "estado"]], hide_index=True,
                     width="stretch", height=360)
        if not vivo.empty and vivo["nivel_nombre"].notna().any():
            st.markdown("**Últimas detecciones del módulo IA clasificadas por el modelo**")
            st.dataframe(vivo.sort_values("timestamp_utc", ascending=False)[
                ["timestamp_utc", "sensor_id", "tipo_sensor", "clase_detectada", "confidence_score", "nivel_nombre",
                 "p_critico", "cuadrante_id"]].head(50), hide_index=True, width="stretch")
    st.caption("Flujo: módulo IA → tabla `detecciones` → modelo Frente 2 → `predicciones_riesgo` → `alertas`. "
               "Use el botón de la barra lateral para simular la llegada de nuevas detecciones.")

# --------------------------------------------------------------------------- #
# 4. KPIs
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Indicadores de seguimiento y control")
    d7, d30 = metr["frente_1"]["perdida_7d"]["deteccion"], m30["deteccion"]
    f2m = metr["frente_2"]["comparativa"]["Random Forest"]
    kp = pd.DataFrame([
        ["Operativo", "Cobertura del plan de patrullaje", f"{ha_cubiertas/ha_tot:.0%}" if ha_tot else "–",
         "% de la pérdida esperada dentro de los cuadrantes patrullados", "≥ 70 %"],
        ["Operativo", "Densidad de alertas", f"{en_alerta/len(P):.0%}", "% de cuadrantes en alerta (carga de trabajo)", "≤ 30 %"],
        ["Operativo", "Alertas críticas abiertas", f"{criticas_abiertas}", "Alertas críticas sin atender en la BD", "0 en < 48 h"],
        ["Ambiental", "Hectáreas en riesgo (30 d)", f"{P['ha_esperadas_30d'].sum():,.1f} ha", "Suma del pronóstico Hurdle", "tendencia ↓"],
        ["Ambiental", "C_evitado potencial", f"{co2_evitable/1000:,.1f} kt CO₂", "ha cubiertas × efectividad × 138.9 tC/ha × 44/12", "↑"],
        ["Ambiental", "Presión minera media en el plan", f"{P.loc[P['priorizado'],'mb_mineria'].mean():.1%}",
         "Fracción minera MapBiomas de los cuadrantes priorizados", "seguimiento"],
        ["Modelo", "ROC-AUC detección 7 d / 30 d", f"{d7['ROC_AUC']:.3f} / {d30['ROC_AUC']:.3f}", "Walk-forward fuera de muestra", "≥ 0.80 / 0.85"],
        ["Modelo", "Recall de pérdida 30 d", f"{d30['Recall']:.0%}", f"Umbral operativo {d30['umbral']:.2f}", "≥ 80 %"],
        ["Modelo", "Captura top-20 % (30 d)", f"{m30['comparativa']['Hurdle LightGBM']['Captura_top20']:.0%}",
         "Pérdida real cubierta patrullando 1 de cada 5 cuadrantes", "≥ 75 %"],
        ["Modelo", "F1 macro / Recall Crítico (incursión)", f"{f2m['F1_macro']:.3f} / {f2m['Recall_Critico']:.3f}",
         "Validación cruzada agrupada por imagen", "≥ 0.90 / 0.88"],
    ], columns=["Tipo", "KPI", "Valor actual", "Definición", "Meta"])
    st.dataframe(kp, hide_index=True, width="stretch")

    g1, g2 = st.columns(2)
    serie = wf.groupby("semana_inicio")[["perdida_30d", "Hurdle LightGBM", "Persistencia"]].sum().reset_index()
    fs = go.Figure()
    fs.add_trace(go.Scatter(x=serie["semana_inicio"], y=serie["perdida_30d"], name="Real", line=dict(color="black", width=2)))
    fs.add_trace(go.Scatter(x=serie["semana_inicio"], y=serie["Hurdle LightGBM"], name="Pronóstico", line=dict(color="#1f4e79", dash="dash", width=2)))
    fs.update_layout(title="Pérdida regional a 30 días: real vs pronóstico (fuera de muestra)", height=360,
                     yaxis_title="ha", margin=dict(t=50, b=10))
    g1.plotly_chart(fs, width="stretch")
    fr = np.linspace(0.02, 1, 40)
    sem = [g for _, g in wf.groupby("semana_inicio") if g["perdida_30d"].sum() > 0]
    cap = [np.mean([g.nlargest(max(1, int(len(g) * f)), "Hurdle LightGBM")["perdida_30d"].sum() / g["perdida_30d"].sum() for g in sem]) for f in fr]
    fc = go.Figure([go.Scatter(x=fr * 100, y=np.array(cap) * 100, name="Modelo", line=dict(color="#1f4e79", width=3)),
                    go.Scatter(x=[0, 100], y=[0, 100], name="Aleatorio", line=dict(color="grey", dash="dot"))])
    fc.add_vline(x=capacidad / len(panel) * 100, line_color="#c62828", annotation_text="capacidad actual")
    fc.update_layout(title="Curva de captura: % de pérdida cubierta vs % del área patrullada", height=360,
                     xaxis_title="% cuadrantes patrullados", yaxis_title="% pérdida cubierta", margin=dict(t=50, b=10))
    g2.plotly_chart(fc, width="stretch")
    comp = pd.DataFrame(m30["comparativa"]).T.reset_index().rename(columns={"index": "Modelo"})
    st.markdown("**Comparativa de modelos (Frente 1, 30 días, walk-forward)**")
    st.dataframe(comp[["Modelo", "MAE", "RMSE", "Spearman_espacial", "Captura_top20"]].round(3), hide_index=True,
                 width="stretch")

# --------------------------------------------------------------------------- #
# 5. Ficha de cuadrante
# --------------------------------------------------------------------------- #
with tabs[4]:
    sel = st.selectbox("Cuadrante", P["cuadrante_id"].tolist(), index=0)
    fila = P[P["cuadrante_id"] == sel].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Ranking IPP", f"#{int(fila['rank'])}", f"IPP {fila['IPP']:.0f}/100", delta_color="off")
    b.metric("P(pérdida 7 d / 30 d)", f"{fila['prob_evento_7d']:.0%} / {fila['prob_evento_30d']:.0%}")
    c.metric("ha esperadas 30 d", f"{fila['ha_esperadas_30d']:.2f}", f"{fila['co2_en_riesgo_30d_t']:,.0f} t CO₂", delta_color="off")
    d.metric("Detecciones (críticas)", f"{int(fila['detecciones'])} ({int(fila['incursiones_criticas'])})")
    h = hist[hist["cuadrante_id"] == sel]
    fh = px.bar(h, x="semana_inicio", y="perdida_ha", title=f"Historial semanal de pérdida en {sel}",
                labels={"perdida_ha": "ha", "semana_inicio": ""}, color_discrete_sequence=["#ef6c00"])
    fh.update_layout(height=320, margin=dict(t=50, b=10))
    st.plotly_chart(fh, width="stretch")
    perfil = pd.DataFrame({"Presión": ["Minería", "Agropecuario", "Tasa histórica de deforestación (ha/año)"],
                           "Valor": [fila["mb_mineria"], fila["mb_agropecuario"], fila["tasa_deforestacion_ha_anio"]]})
    st.markdown(f"**Zona:** {fila['zona']} · **Vocación ZEE:** {fila['vocacion_zee']} · **Causa principal:** {fila['causa_principal']} · "
                f"**Estrategia sugerida:** {fila['estrategia']}")
    st.dataframe(perfil, hide_index=True)
