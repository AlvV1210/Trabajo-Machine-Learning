"""BIODIVERSITY-GUARD Predict · Centro de Monitoreo (Hito 3).

Pensado para guardaparques y jefatura: lenguaje simple, niveles de atención por colores
y un menú por tareas. No necesita reentrenar modelos: usa los guardados en `models/`.

Cómo abrirlo (desde la raíz del proyecto):   python iniciar.py
o directamente:   streamlit run Hito3_Modelado_Avanzado_Evaluacion/dashboard_hito3_streamlit.py
"""

from __future__ import annotations

import json
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
import base_datos_comun as bd  # noqa: E402
import inferencia as inf  # noqa: E402
from features import CO2_POR_C, FAUNA, T_C_POR_HA  # noqa: E402

DATA, REP = ROOT / "data", ROOT / "reports" / "hito3"
NIVELES = ["Bajo", "Moderado", "Alto", "Crítico"]
COLOR = {"Bajo": "#7fbf7b", "Moderado": "#f2c94c", "Alto": "#f2994a", "Crítico": "#d64545"}
ICONO = {"Bajo": "🟢", "Moderado": "🟡", "Alto": "🟠", "Crítico": "🔴"}
VERDE = "#1b5e3b"
ESTRATEGIA = {
    "mineria": "Interdicción con Fiscalía ambiental y PNP; control del río",
    "agropecuario": "Patrullaje terrestre; verificar quemas y nuevas chacras",
    "infraestructura": "Control de trochas y accesos",
    "ninguna": "Patrullaje preventivo y revisión de cámaras",
}
CAUSA = {"mineria": "minería", "agropecuario": "agricultura y ganadería", "infraestructura": "caminos"}
PUERTO_MALDONADO = (-12.5933, -69.1891)
FONDOS = ["Mapa con calles y ríos (necesita internet)", "Solo cuadrícula (funciona sin internet)"]

st.set_page_config(page_title="BIODIVERSITY-GUARD · Tambopata", page_icon="🌳", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 2.2rem; max-width: 1400px;}}
h1, h2, h3 {{color: {VERDE};}}
section[data-testid="stSidebar"] {{background: #f1f6f1;}}
.tarjeta {{background:#fff; border:1px solid #dfe8df; border-radius:14px; padding:16px 18px; height:100%;
          box-shadow:0 1px 3px rgba(27,94,59,.08);}}
.tarjeta .num {{font-size:2.0rem; font-weight:700; line-height:1.15; color:#1f2d24;}}
.tarjeta .tit {{font-size:.82rem; color:#4d5f53; font-weight:600; text-transform:uppercase; letter-spacing:.03em;}}
.tarjeta .sub {{font-size:.86rem; color:#5f6f64; margin-top:4px;}}
.banda {{border-radius:14px; padding:16px 20px; margin-bottom:14px; color:#fff; font-size:1.15rem; font-weight:600;}}
.zona {{border-left:8px solid #999; background:#fff; border-radius:10px; padding:10px 14px; margin-bottom:10px;
       border-top:1px solid #e6ece6; border-right:1px solid #e6ece6; border-bottom:1px solid #e6ece6; color:#1f2d24;}}
.chip {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem; font-weight:700; color:#1f2d24;}}
.ayuda {{background:#eef5ee; border-radius:10px; padding:10px 14px; font-size:.92rem; color:#30423a; margin:6px 0 10px 0;}}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def datos_fijos():
    grilla = inf.grilla()
    f1 = pd.read_csv(DATA / "dataset_frente1_hito1_final.csv")
    fijos = f1.groupby("cuadrante_id")[["cobertura_anp", "causa_principal", "vocacion_zee", "mb_mineria", "mb_agropecuario",
                                        "tasa_deforestacion_ha_anio"]].first().reset_index()
    metr = json.loads((REP / "metricas_hito3.json").read_text(encoding="utf-8"))
    wf = pd.read_csv(DATA / "walkforward_perdida_30d.csv", parse_dates=["semana_inicio"])
    est = pd.read_csv(DATA / "estaciones_sensores.csv")
    return grilla, fijos, metr, wf, est


@st.cache_data(show_spinner=False)
def historia(_version: int) -> pd.DataFrame:
    return bd.panel_frente1()[["semana_inicio", "cuadrante_id", "perdida_ha"]]


def version_historia() -> int:
    return int(bd.leer("SELECT COUNT(*) n FROM semanas_extra")["n"].iloc[0])


def tarjeta(titulo: str, numero: str, sub: str = "", color: str | None = None) -> str:
    borde = f"border-top:5px solid {color};" if color else ""
    return (f'<div class="tarjeta" style="{borde}"><div class="tit">{titulo}</div>'
            f'<div class="num">{numero}</div><div class="sub">{sub}</div></div>')


def chip(nivel: str) -> str:
    return f'<span class="chip" style="background:{COLOR[nivel]}">{nivel}</span>'


def estado(peso_def: float = 0.6) -> dict:
    """Lee la BD y arma la foto actual: una fila por cuadrante + eventos + alertas."""
    grilla, fijos, _, _, _ = datos_fijos()
    pron = bd.leer("SELECT * FROM predicciones_deforestacion")
    ev = bd.leer("""SELECT d.evento_id, d.fecha_hora, d.origen, d.sensor_id, d.tipo_sensor, d.confidence_score, d.latitud,
                           d.longitud, p.cuadrante_id, p.clase, p.nivel, p.nivel_nombre, p.p_alto, p.p_critico,
                           p.explicacion, p.accion FROM detecciones d JOIN predicciones_riesgo p USING(evento_id)""")
    ev["fecha_hora"] = pd.to_datetime(ev["fecha_hora"])
    alertas = bd.leer("SELECT * FROM alertas ORDER BY fecha DESC")
    alertas["fecha"] = pd.to_datetime(alertas["fecha"], format="mixed")
    corte = bd.fecha_corte()
    rec = ev[ev["fecha_hora"] >= corte - pd.Timedelta(days=30)]
    graves = rec[rec["nivel"] >= 2]
    por_q = graves.assign(w=(0.5 * graves["p_alto"] + graves["p_critico"]) * graves["confidence_score"]).groupby("cuadrante_id").agg(
        iri=("w", "sum"), incursiones_30d=("nivel", "size"), criticas_30d=("nivel", lambda s: int((s == 3).sum())),
        nivel_incursion=("nivel", "max"))
    P = grilla.merge(fijos, on="cuadrante_id", how="left").merge(pron, on="cuadrante_id", how="left").merge(
        por_q, on="cuadrante_id", how="left")
    for c in ["iri", "incursiones_30d", "criticas_30d", "nivel_incursion"]:
        P[c] = P[c].fillna(0)
    P["nivel"] = np.maximum(P["nivel_deforestacion"].fillna(0), P["nivel_incursion"]).astype(int)
    P["nivel_nombre"] = P["nivel"].map(dict(enumerate(NIVELES)))
    norm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)  # noqa: E731
    P["prioridad"] = 100 * (peso_def * norm(P["prob_evento_30d"] * (0.2 + np.log1p(P["ha_esperadas_30d"])))
                            + (1 - peso_def) * norm(np.log1p(P["iri"])))
    P = P.sort_values("prioridad", ascending=False).reset_index(drop=True)
    P["puesto"] = np.arange(1, len(P) + 1)
    return {"P": P, "eventos": ev, "recientes": rec, "alertas": alertas, "corte": corte, "semana": str(pron["semana"].iloc[0])}


def frase_zona(r) -> str:
    """Explica en una frase por qué un cuadrante requiere atención."""
    partes = []
    if r.nivel_deforestacion >= 1:
        cuanta = "Muy probable" if r.prob_evento_30d >= 0.8 else "Probable" if r.prob_evento_30d >= 0.5 else "Posible"
        partes.append(f"{cuanta} pérdida de bosque en el próximo mes ({r.prob_evento_30d:.0%}), unas {r.ha_esperadas_30d:.1f} ha")
    if r.incursiones_30d > 0:
        partes.append(f"{int(r.incursiones_30d)} incursión(es) grave(s) en los últimos 30 días"
                      + (f", {int(r.criticas_30d)} crítica(s)" if r.criticas_30d else ""))
    if r.causa_principal in CAUSA and partes:
        partes.append(f"Presión habitual: {CAUSA[r.causa_principal]}")
    return (". ".join(partes) + ".") if partes else "Sin señales de riesgo."


# --------------------------------------------------------------------------- #
# Mapa
# --------------------------------------------------------------------------- #
CAPAS = {
    "Nivel de atención (recomendado)": ("nivel", "Combina el pronóstico de deforestación y las incursiones recientes."),
    "Probabilidad de perder bosque (próximo mes)": ("prob_evento_30d", "0 % = muy improbable, 100 % = casi seguro."),
    "Hectáreas de bosque en riesgo (próximo mes)": ("ha_esperadas_30d", "Cantidad de bosque que el sistema espera que se pierda."),
    "Incursiones graves (últimos 30 días)": ("incursiones_30d", "Detecciones de nivel Alto o Crítico de la red de sensores."),
    "Presencia de minería": ("mb_mineria", "Parte del cuadrante ocupada por minería según MapBiomas."),
    "Presencia de agricultura y ganadería": ("mb_agropecuario", "Parte del cuadrante con uso agropecuario según MapBiomas."),
}
ESCALA_NIVEL = [[0, COLOR["Bajo"]], [0.25, COLOR["Bajo"]], [0.25, COLOR["Moderado"]], [0.5, COLOR["Moderado"]],
                [0.5, COLOR["Alto"]], [0.75, COLOR["Alto"]], [0.75, COLOR["Crítico"]], [1, COLOR["Crítico"]]]


def _geojson(P: pd.DataFrame) -> dict:
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": r.cuadrante_id, "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
            [r.lon_sw, r.lat_sw], [r.lon_se, r.lat_se], [r.lon_ne, r.lat_ne], [r.lon_nw, r.lat_nw], [r.lon_sw, r.lat_sw]]]}}
        for r in P.itertuples()]}


def _borde_reserva(P: pd.DataFrame) -> tuple[list, list]:
    """Contorno del conjunto de cuadrantes de la reserva (sirve para ubicarse)."""
    R = P[P["zona"] == "Reserva Nacional Tambopata"]
    dentro = {(int(r.x_utm // 5000), int(r.y_utm // 5000)): r for r in R.itertuples()}
    lados = {(-1, 0): ("sw", "nw"), (1, 0): ("se", "ne"), (0, -1): ("sw", "se"), (0, 1): ("nw", "ne")}
    lon, lat = [], []
    for (cx, cy), r in dentro.items():
        for (dx, dy), (a, b) in lados.items():
            if (cx + dx, cy + dy) not in dentro:
                lon += [getattr(r, f"lon_{a}"), getattr(r, f"lon_{b}"), None]
                lat += [getattr(r, f"lat_{a}"), getattr(r, f"lat_{b}"), None]
    return lon, lat


def _cuadro(filas) -> tuple[list, list]:
    lon, lat = [], []
    for r in filas:
        lon += [r.lon_sw, r.lon_se, r.lon_ne, r.lon_nw, r.lon_sw, None]
        lat += [r.lat_sw, r.lat_se, r.lat_ne, r.lat_nw, r.lat_sw, None]
    return lon, lat


def mapa(P: pd.DataFrame, capa: str, eventos: pd.DataFrame | None = None, estaciones: pd.DataFrame | None = None,
         plan: pd.DataFrame | None = None, con_fondo: bool = True, alto: int = 620, resaltar: str | None = None) -> go.Figure:
    if not con_fondo:
        return mapa_sin_internet(P, capa, eventos, estaciones, plan, alto, resaltar)
    col = CAPAS[capa][0]
    texto = ("<b>" + P["cuadrante_id"] + "</b> · " + P["zona"] + "<br>Nivel de atención: <b>" + P["nivel_nombre"] + "</b><br>"
             + "Probabilidad de perder bosque: " + (P["prob_evento_30d"] * 100).round(0).astype(int).astype(str) + " %<br>"
             + "Bosque en riesgo: " + P["ha_esperadas_30d"].round(1).astype(str) + " ha<br>"
             + "Incursiones graves (30 días): " + P["incursiones_30d"].astype(int).astype(str))
    es_nivel = col == "nivel"
    barra = (dict(title="Nivel", tickvals=[0.375, 1.125, 1.875, 2.625], ticktext=NIVELES, len=0.5, thickness=14) if es_nivel
             else dict(title="", len=0.5, thickness=14, tickformat=".0%" if col in ("prob_evento_30d", "mb_mineria", "mb_agropecuario") else None))
    comun = dict(z=P[col], zmin=0, zmax=3 if es_nivel else float(max(P[col].max(), 1e-6)),
                 colorscale=ESCALA_NIVEL if es_nivel else "YlOrRd", colorbar=barra, text=texto, hovertemplate="%{text}<extra></extra>")
    if con_fondo:
        S = go.Scattermap
        fig = go.Figure(go.Choroplethmap(geojson=_geojson(P), locations=P["cuadrante_id"], marker_opacity=0.68,
                                         marker_line_width=0.4, marker_line_color="white", **comun))
        fig.update_layout(map=dict(style="open-street-map", zoom=8.2,
                                   center=dict(lat=float(P["lat_c"].mean()), lon=float(P["lon_c"].mean()))))
    else:
        S = go.Scattergeo
        fig = go.Figure(go.Choropleth(geojson=_geojson(P), locations=P["cuadrante_id"], marker_line_width=0.5,
                                      marker_line_color="white", **comun))
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="#f4f7f4")
    lon, lat = _borde_reserva(P)
    fig.add_trace(S(lon=lon, lat=lat, mode="lines", line=dict(width=2.5, color=VERDE), name="Límite de la reserva (aprox.)", hoverinfo="skip"))
    if plan is not None and len(plan):
        lo, la = _cuadro(plan.itertuples())
        fig.add_trace(S(lon=lo, lat=la, mode="lines", line=dict(width=3, color="#1f4e79"), name="Plan de patrullaje", hoverinfo="skip"))
    if resaltar is not None and resaltar in set(P["cuadrante_id"]):
        lo, la = _cuadro(P[P["cuadrante_id"] == resaltar].itertuples())
        fig.add_trace(S(lon=lo, lat=la, mode="lines", line=dict(width=4, color="#111"), name=f"Cuadrante {resaltar}", hoverinfo="skip"))
    if estaciones is not None:
        fig.add_trace(S(lon=estaciones["longitud"], lat=estaciones["latitud"], mode="markers", name="Cámaras y sensores",
                        marker=dict(size=7, color="#37474f"), text=estaciones["estacion_id"] + " · " + estaciones["tipo_sensor"],
                        hovertemplate="%{text}<extra></extra>"))
    if eventos is not None and len(eventos):
        for nv in ["Alto", "Crítico"]:
            e = eventos[eventos["nivel_nombre"] == nv]
            if len(e):
                marca = dict(size=13 if nv == "Crítico" else 10, color=COLOR[nv])
                if not con_fondo:
                    marca["line"] = dict(width=1.2, color="#222")
                fig.add_trace(S(lon=e["longitud"], lat=e["latitud"], mode="markers", name=f"Incursión de nivel {nv}", marker=marca,
                                text=e["fecha_hora"].dt.strftime("%d/%m %H:%M") + " · " + e["explicacion"],
                                hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(S(lon=[PUERTO_MALDONADO[1]], lat=[PUERTO_MALDONADO[0]], mode="markers+text", text=["Puerto Maldonado"],
                    textposition="top right", marker=dict(size=9, color="#111"), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=alto, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(orientation="h", y=-0.02, x=0, bgcolor="rgba(255,255,255,.85)"))
    return fig


def mapa_sin_internet(P: pd.DataFrame, capa: str, eventos, estaciones, plan, alto: int, resaltar: str | None) -> go.Figure:
    """Misma información que `mapa`, dibujada como cuadrícula en kilómetros UTM. No descarga nada de internet."""
    col = CAPAS[capa][0]
    es_nivel = col == "nivel"
    cx, cy = (P["x_utm"] // 5000).astype(int).to_numpy(), (P["y_utm"] // 5000).astype(int).to_numpy()
    xs, ys = np.arange(cx.min(), cx.max() + 1), np.arange(cy.min(), cy.max() + 1)
    Z = np.full((len(ys), len(xs)), np.nan); T = np.full(Z.shape, "", dtype=object)
    texto = ("<b>" + P["cuadrante_id"] + "</b> · " + P["zona"] + "<br>Nivel de atención: <b>" + P["nivel_nombre"] + "</b><br>"
             + "Probabilidad de perder bosque: " + (P["prob_evento_30d"] * 100).round(0).astype(int).astype(str) + " %<br>"
             + "Bosque en riesgo: " + P["ha_esperadas_30d"].round(1).astype(str) + " ha<br>"
             + "Incursiones graves (30 días): " + P["incursiones_30d"].astype(int).astype(str)).to_numpy()
    Z[cy - ys[0], cx - xs[0]] = P[col].to_numpy(dtype=float); T[cy - ys[0], cx - xs[0]] = texto
    barra = (dict(title="Nivel", tickvals=[0.375, 1.125, 1.875, 2.625], ticktext=NIVELES, len=0.5, thickness=14) if es_nivel
             else dict(title="", len=0.5, thickness=14, tickformat=".0%" if col in ("prob_evento_30d", "mb_mineria", "mb_agropecuario") else None))
    fig = go.Figure(go.Heatmap(x=xs * 5 + 2.5, y=ys * 5 + 2.5, z=Z, text=T, zmin=0, zmax=3 if es_nivel else float(max(P[col].max(), 1e-6)),
                               colorscale=ESCALA_NIVEL if es_nivel else "YlOrRd", colorbar=barra, xgap=1, ygap=1,
                               hovertemplate="%{text}<extra></extra>", hoverongaps=False))
    km = lambda lon, lat: tuple(np.asarray(v) / 1000 for v in inf.A_UTM.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)))  # noqa: E731

    def cuadros(filas):
        x, y = [], []
        for r in filas:
            a, b = r.x_utm / 1000, r.y_utm / 1000
            x += [a, a + 5, a + 5, a, a, None]; y += [b, b, b + 5, b + 5, b, None]
        return x, y
    dentro = {(int(r.x_utm // 5000), int(r.y_utm // 5000)) for r in P[P["zona"] == "Reserva Nacional Tambopata"].itertuples()}
    bx, by = [], []
    for a, b in dentro:  # contorno de la reserva
        for (dx, dy), (p0, p1) in {(-1, 0): ((0, 0), (0, 1)), (1, 0): ((1, 0), (1, 1)), (0, -1): ((0, 0), (1, 0)), (0, 1): ((0, 1), (1, 1))}.items():
            if (a + dx, b + dy) not in dentro:
                bx += [(a + p0[0]) * 5, (a + p1[0]) * 5, None]; by += [(b + p0[1]) * 5, (b + p1[1]) * 5, None]
    fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(width=2.5, color=VERDE), name="Límite de la reserva (aprox.)", hoverinfo="skip"))
    if plan is not None and len(plan):
        x, y = cuadros(plan.itertuples())
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=3, color="#1f4e79"), name="Plan de patrullaje", hoverinfo="skip"))
    if resaltar is not None and resaltar in set(P["cuadrante_id"]):
        x, y = cuadros(P[P["cuadrante_id"] == resaltar].itertuples())
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=4, color="#111"), name=f"Cuadrante {resaltar}", hoverinfo="skip"))
    if estaciones is not None:
        x, y = km(estaciones["longitud"], estaciones["latitud"])
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Cámaras y sensores", marker=dict(size=7, color="#37474f", symbol="diamond"),
                                 text=estaciones["estacion_id"] + " · " + estaciones["tipo_sensor"], hovertemplate="%{text}<extra></extra>"))
    if eventos is not None and len(eventos):
        for nv in ["Alto", "Crítico"]:
            e = eventos[eventos["nivel_nombre"] == nv]
            if len(e):
                x, y = km(e["longitud"], e["latitud"])
                fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=f"Incursión de nivel {nv}",
                                         marker=dict(size=13 if nv == "Crítico" else 10, color=COLOR[nv], line=dict(width=1.2, color="#222")),
                                         text=e["fecha_hora"].dt.strftime("%d/%m %H:%M") + " · " + e["explicacion"],
                                         hovertemplate="%{text}<extra></extra>"))
    px_, py_ = km([PUERTO_MALDONADO[1]], [PUERTO_MALDONADO[0]])
    if xs[0] * 5 - 10 <= px_[0] <= xs[-1] * 5 + 15 and ys[0] * 5 - 10 <= py_[0] <= ys[-1] * 5 + 15:
        fig.add_trace(go.Scatter(x=px_, y=py_, mode="markers+text", text=["Puerto Maldonado"], textposition="top center",
                                 marker=dict(size=9, color="#111"), hoverinfo="skip", showlegend=False))
    fig.add_annotation(x=0.995, y=0.99, xref="paper", yref="paper", text="▲ Norte", showarrow=False, font=dict(size=12, color="#4d5f53"))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=alto, margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="#eef3ee", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.02, x=0, bgcolor="rgba(255,255,255,.85)"))
    return fig


def con_fondo_mapa() -> bool:
    return st.session_state.get("fondo", FONDOS[0]) == FONDOS[0]


# --------------------------------------------------------------------------- #
# Páginas
# --------------------------------------------------------------------------- #
def pagina_inicio():
    E = estado(); P = E["P"]; al = E["alertas"]
    st.title("Situación de la Reserva Nacional Tambopata")
    st.caption(f"Información actualizada al {E['corte']:%d/%m/%Y} · pronóstico calculado con la semana del {E['semana'].replace('/', ' al ')}")
    n_crit, n_alto = int((P["nivel"] == 3).sum()), int((P["nivel"] == 2).sum())
    pend = al[al["estado"] == "abierta"]
    inc7 = E["eventos"][(E["eventos"]["fecha_hora"] >= E["corte"] - pd.Timedelta(days=7)) & (E["eventos"]["nivel"] >= 2)]
    if n_crit:
        color, msg = COLOR["Crítico"], f"🔴 {n_crit} zona(s) necesitan atención inmediata y {n_alto} deben visitarse esta semana."
    elif n_alto:
        color, msg = COLOR["Alto"], f"🟠 {n_alto} zona(s) deben visitarse esta semana. No hay zonas en nivel crítico."
    else:
        color, msg = "#3f9b63", "🟢 Sin zonas en nivel alto o crítico. Mantener el patrullaje de rutina."
    st.markdown(f'<div class="banda" style="background:{color}">{msg}</div>', unsafe_allow_html=True)

    ha = P["ha_esperadas_30d"].sum()
    c = st.columns(4)
    c[0].markdown(tarjeta("Zonas que requieren atención", f"{n_crit + n_alto}", f"{n_crit} en nivel Crítico · {n_alto} en nivel Alto<br>de {len(P)} cuadrantes vigilados", COLOR["Crítico"]), unsafe_allow_html=True)
    c[1].markdown(tarjeta("Bosque en riesgo el próximo mes", f"{ha:,.0f} ha", f"unas {ha * 1.4:,.0f} canchas de fútbol<br>≈ {ha * T_C_POR_HA * CO2_POR_C / 1000:,.0f} mil toneladas de CO₂", COLOR["Alto"]), unsafe_allow_html=True)
    c[2].markdown(tarjeta("Incursiones graves · últimos 7 días", f"{len(inc7)}", f"{int((inc7['nivel'] == 3).sum())} críticas (armas, motosierras, ingresos nocturnos)<br>detectadas por cámaras y sensores", COLOR["Moderado"]), unsafe_allow_html=True)
    c[3].markdown(tarjeta("Alertas pendientes de atender", f"{len(pend)}", f"{int((pend['nivel'] == 'Crítico').sum())} críticas · {int((pend['origen'] == 'Incursión').sum())} por incursión<br>vea el apartado «Alertas»", VERDE), unsafe_allow_html=True)

    st.write("")
    izq, der = st.columns([1.05, 1])
    with izq:
        st.subheader("Las zonas más urgentes ahora")
        for r in P[P["nivel"] >= 2].head(5).itertuples():
            st.markdown(f'<div class="zona" style="border-left-color:{COLOR[r.nivel_nombre]}"><b>Cuadrante {r.cuadrante_id}</b> · '
                        f'{r.zona} &nbsp;{chip(r.nivel_nombre)}<br>{frase_zona(r)}<br>'
                        f'<span style="color:#4d5f53">Qué hacer: {inf.ACCION[int(r.nivel)]}</span></div>', unsafe_allow_html=True)
        if not (P["nivel"] >= 2).any():
            st.info("No hay zonas en nivel Alto o Crítico.")
        st.markdown('<div class="ayuda">👉 En <b>Plan de patrullaje</b> está la lista completa, ordenada y lista para descargar. '
                    'En <b>Mapa</b> puede revisar cada zona con detalle.</div>', unsafe_allow_html=True)
    with der:
        st.subheader("Mapa de niveles de atención")
        st.plotly_chart(mapa(P, "Nivel de atención (recomendado)", E["recientes"][E["recientes"]["nivel"] >= 2],
                             con_fondo=con_fondo_mapa(), alto=470), width="stretch", key="mapa_inicio")
    with st.expander("¿Cómo leer esta pantalla?"):
        st.markdown("""
- **Cuadrante**: la reserva y su zona de amortiguamiento están divididas en 273 cuadros de 5 × 5 km. Cada uno tiene un código (por ejemplo Q199).
- **Nivel de atención**: 🟢 Bajo (sin señales) · 🟡 Moderado (vigilar) · 🟠 Alto (visitar esta semana) · 🔴 Crítico (actuar de inmediato).
- **Bosque en riesgo**: hectáreas que el sistema calcula que podrían perderse en los próximos 30 días si no se interviene.
- **Incursión**: detección de personas, vehículos, trampas, motosierras o armas por las cámaras y los sensores de sonido.""")


def pagina_mapa():
    E = estado(); P = E["P"]; est = datos_fijos()[4]
    st.title("Mapa de la reserva")
    a, b, c = st.columns([1.5, 1, 1.3])
    capa = a.selectbox("¿Qué desea ver en el mapa?", list(CAPAS))
    dias = b.selectbox("Incursiones a mostrar", ["Últimos 7 días", "Últimos 30 días", "No mostrar"], index=1)
    zonas = c.multiselect("Zona", sorted(P["zona"].unique()), default=sorted(P["zona"].unique()))
    d1, d2, d3 = st.columns([1, 1, 2.2])
    ver_est = d1.checkbox("Mostrar cámaras y sensores")
    ver_plan = d2.checkbox("Marcar plan de patrullaje")
    # la elección se guarda aparte del control para que se conserve al cambiar de página
    st.session_state["fondo"] = d3.radio("Fondo del mapa", FONDOS, index=FONDOS.index(st.session_state.get("fondo", FONDOS[0])), horizontal=True)
    Pz = P[P["zona"].isin(zonas)] if zonas else P
    ev = None
    if dias != "No mostrar":
        n = 7 if "7" in dias else 30
        ev = E["eventos"][(E["eventos"]["fecha_hora"] >= E["corte"] - pd.Timedelta(days=n)) & (E["eventos"]["nivel"] >= 2)]
    st.plotly_chart(mapa(Pz, capa, ev, est if ver_est else None, P.head(st.session_state.get("capacidad", 20)) if ver_plan else None,
                         con_fondo=con_fondo_mapa(), resaltar=st.session_state.get("cuadrante_sel")), width="stretch", key="mapa_principal")
    cuenta = Pz["nivel_nombre"].value_counts().reindex(NIVELES).fillna(0).astype(int)
    st.markdown(f'<div class="ayuda"><b>{capa}.</b> {CAPAS[capa][1]} Pase el cursor sobre un cuadro para ver su detalle. '
                f'Los puntos son incursiones de nivel Alto 🟠 o Crítico 🔴; la línea verde es el límite aproximado de la reserva.<br>'
                + " &nbsp;·&nbsp; ".join(f"{ICONO[n]} <b>{cuenta[n]}</b> cuadrantes en nivel {n}" for n in NIVELES) + "</div>", unsafe_allow_html=True)

    st.subheader("Detalle de un cuadrante")
    idx = P.set_index("cuadrante_id")
    sel = st.selectbox("Elija un cuadrante (ordenados del más al menos urgente)", Pz["cuadrante_id"].tolist(), key="cuadrante_sel",
                       format_func=lambda q: f"{q} · {idx.at[q, 'nivel_nombre']} · {idx.at[q, 'zona']}")
    r = P[P["cuadrante_id"] == sel].iloc[0]
    k = st.columns(4)
    k[0].markdown(tarjeta("Nivel de atención", f"{ICONO[r.nivel_nombre]} {r.nivel_nombre}", f"Puesto {int(r.puesto)} de {len(P)} en prioridad", COLOR[r.nivel_nombre]), unsafe_allow_html=True)
    k[1].markdown(tarjeta("Probabilidad de perder bosque", f"{r.prob_evento_30d:.0%}", f"en el próximo mes · {r.prob_evento_7d:.0%} en la próxima semana"), unsafe_allow_html=True)
    k[2].markdown(tarjeta("Bosque en riesgo", f"{r.ha_esperadas_30d:.1f} ha", f"hasta {max(r.ha_alto_30d, r.ha_esperadas_30d):.1f} ha en un escenario desfavorable<br>perdió {r.perdida_suma_4s:.1f} ha el último mes"), unsafe_allow_html=True)
    k[3].markdown(tarjeta("Incursiones graves", f"{int(r.incursiones_30d)}", f"en los últimos 30 días · {int(r.criticas_30d)} críticas"), unsafe_allow_html=True)
    st.write("")
    st.markdown(f"**En palabras simples:** {frase_zona(r)}  \n**Qué hacer:** {inf.ACCION[int(r.nivel)]}  \n"
                f"**Estrategia sugerida:** {ESTRATEGIA.get(r.causa_principal, ESTRATEGIA['ninguna'])}.  \n"
                f"**Uso del suelo:** {r.mb_mineria:.0%} minería, {r.mb_agropecuario:.0%} agropecuario · **Zonificación:** {r.vocacion_zee}.")
    h = historia(version_historia()); h = h[h["cuadrante_id"] == sel]
    fig = go.Figure(go.Bar(x=h["semana_inicio"], y=h["perdida_ha"], marker_color=COLOR["Alto"]))
    fig.update_layout(title=f"Bosque perdido cada semana en {sel} (alertas satelitales)", height=300, margin=dict(t=45, b=10, l=10, r=10),
                      yaxis_title="hectáreas", plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")
    evq = E["recientes"][(E["recientes"]["cuadrante_id"] == sel) & (E["recientes"]["nivel"] >= 1)]
    if len(evq):
        st.markdown("**Actividad humana registrada aquí en los últimos 30 días**")
        st.dataframe(evq.sort_values("fecha_hora", ascending=False)[["fecha_hora", "nivel_nombre", "explicacion", "sensor_id"]].rename(
            columns={"fecha_hora": "Fecha y hora", "nivel_nombre": "Nivel", "explicacion": "Qué se detectó", "sensor_id": "Sensor"}),
            hide_index=True, width="stretch")


def pagina_plan():
    st.title("Plan de patrullaje de la semana")
    st.markdown('<div class="ayuda">El sistema ordena los cuadrantes según dónde es más útil enviar a los guardaparques. '
                'Indique cuántos cuadrantes puede visitar su equipo y descargue la hoja de ruta.</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    capacidad = a.slider("¿Cuántos cuadrantes puede patrullar el equipo esta semana?", 5, 60, st.session_state.get("capacidad", 20), 5)
    st.session_state["capacidad"] = capacidad
    focos = {"Solo incursiones": 0.0, "Más incursiones": 0.3, "Equilibrado": 0.6, "Más deforestación": 0.8, "Solo deforestación": 1.0}
    foco = b.select_slider("¿Qué desea priorizar?", list(focos), value="Equilibrado")
    E = estado(focos[foco]); P = E["P"]; metr = datos_fijos()[2]
    plan = P.head(capacidad)
    ha_tot, ha_plan = P["ha_esperadas_30d"].sum(), plan["ha_esperadas_30d"].sum()
    inc_tot, inc_plan = int(P["incursiones_30d"].sum()), int(plan["incursiones_30d"].sum())
    k = st.columns(4)
    k[0].markdown(tarjeta("Área a recorrer", f"{capacidad * 25:,} km²", f"{capacidad} cuadrantes de 5 × 5 km"), unsafe_allow_html=True)
    k[1].markdown(tarjeta("Bosque en riesgo cubierto", f"{ha_plan / ha_tot:.0%}" if ha_tot else "–", f"{ha_plan:,.0f} de {ha_tot:,.0f} ha en riesgo", VERDE), unsafe_allow_html=True)
    k[2].markdown(tarjeta("Incursiones graves cubiertas", f"{inc_plan} de {inc_tot}", "de los últimos 30 días", VERDE), unsafe_allow_html=True)
    k[3].markdown(tarjeta("CO₂ que se puede evitar", f"{ha_plan * 0.5 * T_C_POR_HA * CO2_POR_C / 1000:,.1f} mil t", "si se detiene a tiempo la mitad de la pérdida"), unsafe_allow_html=True)
    st.write("")
    t = plan.assign(Prioridad=plan["prioridad"].round(0), Motivo=[frase_zona(r) for r in plan.itertuples()],
                    Estrategia=plan["causa_principal"].map(ESTRATEGIA).fillna(ESTRATEGIA["ninguna"]),
                    Nivel=plan["nivel_nombre"].map(lambda n: f"{ICONO[n]} {n}"))[
        ["puesto", "cuadrante_id", "zona", "Nivel", "Prioridad", "Motivo", "Estrategia", "lat_c", "lon_c"]].rename(
        columns={"puesto": "N.º", "cuadrante_id": "Cuadrante", "zona": "Zona", "lat_c": "Latitud", "lon_c": "Longitud"})
    st.subheader("Hoja de ruta")
    st.dataframe(t.drop(columns=["Latitud", "Longitud"]), hide_index=True, width="stretch", height=38 + 35 * min(len(t), 12), column_config={
        "N.º": st.column_config.NumberColumn("N.º", width="small"), "Cuadrante": st.column_config.TextColumn("Cuadrante", width="small"),
        "Prioridad": st.column_config.ProgressColumn("Prioridad", min_value=0, max_value=100, format="%.0f", width="small"),
        "Motivo": st.column_config.TextColumn("¿Por qué?", width="large"),
        "Estrategia": st.column_config.TextColumn("Qué hacer / con quién", width="large")})
    st.download_button("⬇️ Descargar hoja de ruta (CSV con coordenadas)", t.to_csv(index=False).encode("utf-8-sig"),
                       f"plan_patrullaje_{E['corte']:%Y-%m-%d}.csv", "text/csv")
    st.subheader("Dónde quedan los cuadrantes del plan")
    st.caption("Los cuadrantes del plan aparecen con borde azul.")
    st.plotly_chart(mapa(P, "Nivel de atención (recomendado)", None, plan=plan, con_fondo=con_fondo_mapa(), alto=520),
                    width="stretch", key="mapa_plan")
    cap = metr["frente_1"]["perdida_30d"]["comparativa"]["Hurdle LightGBM"]["Captura_top20"]
    st.markdown(f'<div class="ayuda">📌 <b>¿Sirve seguir este orden?</b> En la prueba con datos de 2025 y 2026 que el sistema no conocía, '
                f'visitar solo 1 de cada 5 cuadrantes (los que el sistema puso primero) habría cubierto el <b>{cap:.0%}</b> del bosque que realmente se perdió. '
                f'Eligiendo al azar se habría cubierto el 20 %.</div>', unsafe_allow_html=True)


def pagina_alertas():
    E = estado(); al = E["alertas"]
    st.title("Alertas")
    pend = al[al["estado"] == "abierta"]
    k = st.columns(4)
    k[0].markdown(tarjeta("Pendientes", f"{len(pend)}", "sin atender", COLOR["Crítico"]), unsafe_allow_html=True)
    k[1].markdown(tarjeta("Por incursión", f"{int((pend['origen'] == 'Incursión').sum())}", "cámaras, sensores y reportes"), unsafe_allow_html=True)
    k[2].markdown(tarjeta("Por deforestación", f"{int((pend['origen'] == 'Deforestación').sum())}", "pronóstico del próximo mes"), unsafe_allow_html=True)
    k[3].markdown(tarjeta("Por imagen satelital", f"{int((pend['origen'] == 'Imagen satelital').sum())}", "reportes cargados"), unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns(3)
    est_f = a.radio("Mostrar", ["Pendientes", "Atendidas", "Todas"], horizontal=True)
    tipos = ["Incursión", "Deforestación", "Imagen satelital"]
    tipo = b.multiselect("Tipo", tipos, default=tipos)
    niv = c.multiselect("Nivel", ["Crítico", "Alto"], default=["Crítico", "Alto"])
    v = al[al["origen"].isin(tipo) & al["nivel"].isin(niv)]
    if est_f != "Todas":
        v = v[v["estado"] == ("abierta" if est_f == "Pendientes" else "atendida")]
    v = v.assign(orden=v["nivel"].map({"Crítico": 0, "Alto": 1}), tipo_o=v["origen"].map({"Incursión": 0, "Imagen satelital": 1, "Deforestación": 2}))
    v = v.sort_values(["orden", "tipo_o", "fecha"], ascending=[True, True, False])
    st.caption(f"{len(v)} alerta(s). Primero las críticas; dentro de cada nivel, las incursiones más recientes.")
    for r in v.head(10).itertuples():
        x, y = st.columns([6, 1])
        x.markdown(f'<div class="zona" style="border-left-color:{COLOR[r.nivel]}">{chip(r.nivel)} &nbsp;<b>{r.origen}</b> · cuadrante '
                   f'<b>{r.cuadrante_id or "fuera de la cuadrícula"}</b> · {r.fecha:%d/%m/%Y %H:%M}<br>{r.detalle}<br>'
                   f'<span style="color:#4d5f53">Qué hacer: {r.accion}</span></div>', unsafe_allow_html=True)
        if r.estado == "abierta":
            if y.button("✔ Atendida", key=f"at_{r.alerta_id}", help="Marcar esta alerta como atendida"):
                bd.marcar_alerta(r.alerta_id, "atendida"); st.rerun()
        else:
            y.markdown("✅ Atendida")
    if len(v) > 10:
        with st.expander(f"Ver las {len(v)} alertas en una tabla"):
            st.dataframe(v[["fecha", "nivel", "origen", "cuadrante_id", "detalle", "estado"]].rename(columns={
                "fecha": "Fecha", "nivel": "Nivel", "origen": "Tipo", "cuadrante_id": "Cuadrante", "detalle": "Detalle", "estado": "Estado"}),
                hide_index=True, width="stretch")
    ev = E["eventos"]; ev = ev[(ev["nivel"] >= 2) & (ev["fecha_hora"] >= E["corte"] - pd.Timedelta(weeks=26))]
    s = ev.groupby([pd.Grouper(key="fecha_hora", freq="W-MON"), "nivel_nombre"]).size().reset_index(name="n")
    fig = px.bar(s, x="fecha_hora", y="n", color="nivel_nombre", color_discrete_map=COLOR, category_orders={"nivel_nombre": ["Alto", "Crítico"]},
                 labels={"fecha_hora": "", "n": "incursiones", "nivel_nombre": "Nivel"}, title="Incursiones graves por semana (últimos 6 meses)")
    fig.update_layout(height=320, plot_bgcolor="white", margin=dict(t=45, b=10))
    st.plotly_chart(fig, width="stretch")


# ---- Ingreso de datos ------------------------------------------------------
ALIAS = {"lat": "latitud", "latitude": "latitud", "lon": "longitud", "lng": "longitud", "long": "longitud", "longitude": "longitud",
         "clase": "clase_detectada", "class": "clase_detectada", "label": "clase_detectada", "etiqueta": "clase_detectada",
         "especie": "clase_detectada", "objeto": "clase_detectada", "deteccion": "clase_detectada", "detección": "clase_detectada",
         "fecha": "fecha_hora", "datetime": "fecha_hora", "timestamp": "fecha_hora", "timestamp_utc": "fecha_hora",
         "confianza": "confidence_score", "score": "confidence_score", "confidence": "confidence_score",
         "sensor": "sensor_id", "camara": "sensor_id", "cámara": "sensor_id", "estacion": "sensor_id", "estacion_id": "sensor_id"}
OPCIONES_CLASE = ["persona", "cazador", "vehiculo_ligero", "vehiculo_maderero", "trampa_caza", "arma_blanca", "motosierra",
                  "arma_fuego"] + sorted(FAUNA)


def leer_tabla(arch) -> pd.DataFrame | None:
    try:
        return pd.read_excel(arch) if arch.name.lower().endswith("xlsx") else pd.read_csv(arch, sep=None, engine="python")
    except Exception as e:  # archivo dañado o formato inesperado
        st.error(f"No se pudo leer el archivo: {e}")
        return None


def mostrar_resultado(res: pd.DataFrame, clave: str):
    if res is None or res.empty:
        st.warning("No se pudo registrar ninguna detección válida."); return
    cuenta = res["nivel_nombre"].value_counts().reindex(NIVELES).fillna(0).astype(int)
    st.success(f"Se registraron {len(res)} detección(es): " + " · ".join(f"{ICONO[n]} {cuenta[n]} {n}" for n in NIVELES if cuenta[n]))
    for r in res.sort_values("nivel_predicho", ascending=False).head(4).itertuples():
        st.markdown(f'<div class="zona" style="border-left-color:{COLOR[r.nivel_nombre]}">{chip(r.nivel_nombre)} &nbsp;{r.explicacion} '
                    f'Cuadrante <b>{r.cuadrante_id or "fuera de la cuadrícula"}</b>.<br>'
                    f'<span style="color:#4d5f53">Qué hacer: {r.accion}</span></div>', unsafe_allow_html=True)
    if len(res) > 4:
        t = res[["fecha_hora", "clase", "nivel_nombre", "cuadrante_id", "explicacion", "accion"]].rename(columns={
            "fecha_hora": "Fecha y hora", "clase": "Detección", "nivel_nombre": "Nivel", "cuadrante_id": "Cuadrante",
            "explicacion": "Por qué", "accion": "Qué hacer"})
        st.dataframe(t, hide_index=True, width="stretch", key=f"tabla_{clave}")
        st.download_button("⬇️ Descargar resultados", t.to_csv(index=False).encode("utf-8-sig"), "detecciones_clasificadas.csv", "text/csv", key=f"desc_{clave}")
    st.caption("Ya están guardadas: aparecen en el Mapa y, si son de nivel Alto o Crítico, en Alertas.")


def selector_lugar(clave: str, P: pd.DataFrame, est: pd.DataFrame, exif: dict | None = None) -> tuple[float, float, str]:
    opciones = ["Cerca de una cámara o sensor", "En un cuadrante", "Con coordenadas GPS"]
    modo = st.radio("¿Dónde ocurrió?", opciones, horizontal=True, key=f"modo_{clave}", index=2 if exif and "latitud" in exif else 0)
    if modo == opciones[0]:
        e_idx = est.set_index("estacion_id")
        e = st.selectbox("Cámara o sensor", est["estacion_id"], key=f"est_{clave}",
                         format_func=lambda s: f"{s} · {e_idx.at[s, 'tipo_sensor']} · {e_idx.at[s, 'zona']}")
        return float(e_idx.at[e, "latitud"]), float(e_idx.at[e, "longitud"]), e
    if modo == opciones[1]:
        q = st.selectbox("Cuadrante", sorted(P["cuadrante_id"]), key=f"q_{clave}")
        f = P[P["cuadrante_id"] == q].iloc[0]
        return float(f.lat_c), float(f.lon_c), "sin sensor"
    a, b = st.columns(2)
    lat0 = float(np.clip((exif or {}).get("latitud", -12.95), -13.6, -12.2))
    lon0 = float(np.clip((exif or {}).get("longitud", -69.30), -70.2, -68.5))
    lat = a.number_input("Latitud", -13.60, -12.20, lat0, format="%.5f", key=f"lat_{clave}")
    lon = b.number_input("Longitud", -70.20, -68.50, lon0, format="%.5f", key=f"lon_{clave}")
    return lat, lon, "sin sensor"


def pagina_ingreso():
    E = estado(); P = E["P"]; est = datos_fijos()[4]
    st.title("Ingresar datos")
    st.markdown('<div class="ayuda">Elija la forma que le resulte más cómoda. Todo lo que ingrese se analiza al instante con los modelos ya '
                'entrenados y aparece en el mapa y en las alertas.</div>', unsafe_allow_html=True)
    t = st.tabs(["📄 Subir un archivo", "✍️ Registrar un avistamiento", "📷 Subir una foto", "🛰️ Imagen satelital o de dron",
                 "🌳 Alertas de deforestación de la semana", "📡 Simular sensores"])

    with t[0]:
        st.subheader("Subir un archivo con muchas detecciones")
        st.write("Acepta CSV o Excel. Columnas obligatorias: **clase_detectada**, **latitud**, **longitud**. "
                 "Opcionales: fecha_hora, confidence_score, sensor_id, tipo_sensor.")
        ej = DATA / "ejemplos" / "detecciones_ejemplo.csv"
        if ej.exists():
            st.download_button("⬇️ Descargar archivo de ejemplo (sirve de plantilla)", ej.read_bytes(), ej.name, "text/csv")
        arch = st.file_uploader("Archivo de detecciones", type=["csv", "xlsx"], key="arch_det")
        df = leer_tabla(arch) if arch is not None else None
        if df is not None:
            df.columns = [ALIAS.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
            faltan = [c for c in ["clase_detectada", "latitud", "longitud"] if c not in df.columns]
            if faltan:
                st.error(f"Faltan columnas obligatorias: {', '.join(faltan)}. Columnas encontradas: {', '.join(map(str, df.columns))}.")
            else:
                df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce"); df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
                clase_ok = inf.normalizar_clase(df["clase_detectada"]).notna()
                lugar_ok = df["latitud"].between(-13.6, -12.2) & df["longitud"].between(-70.2, -68.5)
                ok = clase_ok & lugar_ok
                a, b, c = st.columns(3)
                a.metric("Filas leídas", len(df)); b.metric("Filas válidas", int(ok.sum())); c.metric("Filas con problemas", int((~ok).sum()))
                if (~clase_ok).any():
                    st.warning(f"Detección no reconocida en {int((~clase_ok).sum())} fila(s): "
                               + ", ".join(df.loc[~clase_ok, "clase_detectada"].astype(str).unique()[:6])
                               + ". Valores aceptados: " + ", ".join(OPCIONES_CLASE) + ".")
                if (~lugar_ok).any():
                    st.warning(f"{int((~lugar_ok).sum())} fila(s) tienen coordenadas vacías o fuera del ámbito de Tambopata.")
                st.dataframe(df[ok].head(8), hide_index=True, width="stretch")
                if ok.any() and st.button(f"Analizar y guardar {int(ok.sum())} detecciones", type="primary"):
                    st.session_state["res_archivo"] = bd.registrar_detecciones(df[ok], "Archivo")
        if "res_archivo" in st.session_state:
            mostrar_resultado(st.session_state["res_archivo"], "archivo")

    with t[1]:
        st.subheader("Registrar un avistamiento o un reporte de patrulla")
        a, b = st.columns(2)
        clase = a.selectbox("¿Qué se observó?", OPCIONES_CLASE, format_func=lambda c: inf.NOMBRE_CLASE[c])
        certeza = b.radio("¿Qué tan seguro está?", ["Seguro", "Probable", "Dudoso"], horizontal=True)
        c, d, e = st.columns(3)
        fecha = c.date_input("Fecha", E["corte"].date()); hora = d.time_input("Hora", E["corte"].time().replace(second=0, microsecond=0))
        fuente = e.selectbox("¿Cómo se detectó?", ["patrulla", "camara", "acustico", "dron"],
                             format_func={"patrulla": "Guardaparque en patrulla", "camara": "Cámara trampa", "acustico": "Sensor de sonido", "dron": "Dron"}.get)
        lat, lon, sensor = selector_lugar("manual", P, est)
        if st.button("Analizar y guardar", type="primary", key="btn_manual"):
            st.session_state["res_manual"] = bd.registrar_detecciones(pd.DataFrame([{
                "clase_detectada": clase, "latitud": lat, "longitud": lon, "fecha_hora": pd.Timestamp.combine(fecha, hora),
                "confidence_score": {"Seguro": 0.95, "Probable": 0.8, "Dudoso": 0.6}[certeza], "sensor_id": sensor,
                "tipo_sensor": fuente}]), "Registro manual")
        if "res_manual" in st.session_state:
            mostrar_resultado(st.session_state["res_manual"], "manual")

    with t[2]:
        st.subheader("Subir una foto de cámara trampa o de celular")
        st.caption("El reconocimiento automático de imágenes corresponde al módulo de IA del proyecto hermano, que aquí es referencial. "
                   "Por eso se pide confirmar qué muestra la foto; el sistema lee la fecha y el GPS de la foto si los trae.")
        foto = st.file_uploader("Foto", type=["jpg", "jpeg", "png"], key="foto")
        if foto is not None:
            exif = inf.leer_exif(foto); foto.seek(0)
            a, b = st.columns([1, 1.3])
            a.image(foto, width="stretch")
            with b:
                if exif:
                    st.info("Datos leídos de la foto: " + ", ".join(f"{k} = {v}" for k, v in exif.items()))
                else:
                    st.caption("La foto no trae fecha ni GPS; complételos aquí.")
                clase = st.selectbox("¿Qué muestra la foto?", OPCIONES_CLASE, format_func=lambda c: inf.NOMBRE_CLASE[c], key="clase_foto")
                f0 = exif.get("fecha_hora"); f0 = f0 if isinstance(f0, pd.Timestamp) and pd.notna(f0) else E["corte"]
                c, d = st.columns(2)
                fecha = c.date_input("Fecha", f0.date(), key="f_foto"); hora = d.time_input("Hora", f0.time().replace(second=0, microsecond=0), key="h_foto")
                lat, lon, sensor = selector_lugar("foto", P, est, exif)
                if st.button("Analizar y guardar", type="primary", key="btn_foto"):
                    res = bd.registrar_detecciones(pd.DataFrame([{
                        "clase_detectada": clase, "latitud": lat, "longitud": lon, "fecha_hora": pd.Timestamp.combine(fecha, hora),
                        "confidence_score": 0.9, "sensor_id": sensor, "tipo_sensor": "camara"}]), "Foto")
                    if not res.empty:
                        (DATA / "evidencias").mkdir(exist_ok=True)
                        (DATA / "evidencias" / f"{res['evento_id'].iloc[0]}{Path(foto.name).suffix.lower()}").write_bytes(foto.getvalue())
                    st.session_state["res_foto"] = res
        if "res_foto" in st.session_state:
            mostrar_resultado(st.session_state["res_foto"], "foto")

    with t[3]:
        st.subheader("Analizar una imagen satelital o de dron")
        st.caption("El sistema pinta de rojo las áreas sin vegetación (suelo descubierto, claros, playas de minería) y estima cuántas hectáreas son. "
                   "Es una ayuda para decidir si conviene verificar en campo; no reemplaza a las alertas oficiales de Geobosques.")
        ej = DATA / "ejemplos" / "imagen_satelital_ejemplo.png"
        if ej.exists():
            st.download_button("⬇️ Descargar imagen de ejemplo (simulada)", ej.read_bytes(), ej.name, "image/png")
        img = st.file_uploader("Imagen a color", type=["jpg", "jpeg", "png", "tif", "tiff"], key="sat")
        if img is not None:
            r = inf.analizar_imagen_satelital(img)
            a, b = st.columns(2)
            a.image(r["imagen"], caption="Imagen original", width="stretch")
            b.image(r["resaltado"], caption="En rojo: sin cobertura vegetal", width="stretch")
            c, d, e = st.columns(3)
            c.markdown(tarjeta("Con vegetación", f"{r['fraccion_vegetacion']:.0%}", "bosque o cultivos verdes", "#3f9b63"), unsafe_allow_html=True)
            d.markdown(tarjeta("Sin cobertura", f"{r['fraccion_sin_cobertura']:.0%}", "suelo descubierto o claros", COLOR["Crítico"]), unsafe_allow_html=True)
            e.markdown(tarjeta("Agua o sombra", f"{r['fraccion_agua_sombra']:.0%}", "ríos, cochas o sombras"), unsafe_allow_html=True)
            st.write("")
            f, g = st.columns(2)
            q = f.selectbox("¿A qué cuadrante corresponde?", sorted(P["cuadrante_id"]), key="q_sat")
            area = g.number_input("¿Cuántas hectáreas cubre la imagen?", 1.0, 2500.0, 100.0, 10.0,
                                  help="Un cuadrante completo de 5 × 5 km son 2 500 ha. 1 km² = 100 ha.")
            st.markdown(f"**Estimación:** unas **{area * r['fraccion_sin_cobertura']:,.1f} ha** sin cobertura vegetal en la imagen.")
            if st.button("Guardar reporte", type="primary", key="btn_sat"):
                st.session_state["res_sat"] = bd.registrar_reporte_satelital(q, area, r["fraccion_sin_cobertura"], img.name)
        if "res_sat" in st.session_state:
            r = st.session_state["res_sat"]
            st.success(f"Reporte guardado: {r['ha']:.1f} ha sin cobertura · nivel {r['nivel']}. "
                       + ("Se abrió una alerta para verificación en campo." if r["nivel"] in ("Alto", "Crítico") else "No supera el mínimo para abrir una alerta."))

    with t[4]:
        st.subheader("Cargar las alertas de deforestación de una semana nueva")
        st.write("Cada semana Geobosques publica alertas satelitales. Suba aquí el conteo por cuadrante y el sistema "
                 "**actualizará el pronóstico de los próximos 7 y 30 días** con el modelo ya entrenado.")
        ini = pd.Timestamp(E["semana"][:10]) + pd.Timedelta(days=7)
        st.info(f"La próxima semana por cargar es la del {ini:%d/%m/%Y} al {ini + pd.Timedelta(days=6):%d/%m/%Y}.")
        plantilla = pd.DataFrame({"cuadrante_id": sorted(P["cuadrante_id"]), "n_alertas": 0, "temperatura_media": "",
                                  "humedad_media": "", "precipitacion_suma": ""})
        a, b = st.columns(2)
        a.download_button("⬇️ Descargar plantilla (273 cuadrantes)", plantilla.to_csv(index=False).encode("utf-8-sig"), "plantilla_semana.csv", "text/csv")
        ej = DATA / "ejemplos" / "semana_ejemplo.csv"
        if ej.exists():
            b.download_button("⬇️ Descargar semana de ejemplo", ej.read_bytes(), ej.name, "text/csv")
        st.caption("n_alertas = número de píxeles con alerta (cada uno equivale a 0,09 ha). El clima es opcional; si se deja vacío se repite el de la última semana.")
        arch = st.file_uploader("Archivo de la semana", type=["csv", "xlsx"], key="arch_sem")
        df = leer_tabla(arch) if arch is not None else None
        if df is not None:
            df.columns = [str(c).strip().lower() for c in df.columns]
            if not {"cuadrante_id", "n_alertas"} <= set(df.columns):
                st.error("El archivo necesita las columnas cuadrante_id y n_alertas.")
            else:
                df["n_alertas"] = pd.to_numeric(df["n_alertas"], errors="coerce").fillna(0)
                validos = df["cuadrante_id"].isin(P["cuadrante_id"])
                c, d, e = st.columns(3)
                c.metric("Cuadrantes reconocidos", int(validos.sum())); d.metric("Con alertas", int((df.loc[validos, "n_alertas"] > 0).sum()))
                e.metric("Bosque perdido reportado", f"{df.loc[validos, 'n_alertas'].sum() * 0.09:,.1f} ha")
                if validos.any() and st.button("Guardar y actualizar el pronóstico", type="primary", key="btn_sem"):
                    antes = P["ha_esperadas_30d"].sum()
                    with st.spinner("Recalculando el pronóstico…"):
                        r = bd.registrar_semana_frente1(df[validos])
                    r["antes"], r["despues"] = antes, estado()["P"]["ha_esperadas_30d"].sum()
                    st.session_state["res_sem"] = r
        if "res_sem" in st.session_state:
            r = st.session_state["res_sem"]
            st.success(f"Semana del {r['semana'].replace('/', ' al ')} guardada ({r['ha_reportadas']:.1f} ha reportadas). "
                       f"El bosque en riesgo para el próximo mes pasó de {r['antes']:,.0f} a {r['despues']:,.0f} ha; "
                       f"{r['alertas']} cuadrantes quedan en nivel Alto o Crítico por deforestación.")

    with t[5]:
        st.subheader("Simular la llegada de datos de cámaras y sensores")
        st.write("Sirve para mostrar el flujo completo cuando no hay sensores reales conectados: se generan detecciones "
                 "(en su mayoría fauna y algunas amenazas) como las enviaría el módulo de IA.")
        n = st.slider("Número de detecciones a simular", 5, 50, 15, 5)
        if st.button("Simular", type="primary", key="btn_sim"):
            st.session_state["res_sim"] = bd.simular_modulo_ia(n)
        if "res_sim" in st.session_state:
            mostrar_resultado(st.session_state["res_sim"], "sim")


def semaforo(valor: float, meta: float, mayor_es_mejor: bool = True, tolerancia: float = 0.15) -> str:
    if pd.isna(valor):
        return "—"
    bien = valor >= meta if mayor_es_mejor else valor <= meta
    casi = valor >= meta * (1 - tolerancia) if mayor_es_mejor else valor <= meta * (1 + tolerancia)
    return "🟢 Cumple" if bien else "🟡 Cerca" if casi else "🔴 No cumple"


def pagina_indicadores():
    E = estado(); P = E["P"]; al = E["alertas"]; ev = E["eventos"]; corte = E["corte"]
    _, _, metr, wf, est = datos_fijos()
    st.title("Indicadores de seguimiento")
    st.markdown('<div class="ayuda">Con estos indicadores la jefatura controla, semana a semana, si la vigilancia está funcionando. '
                'El semáforo compara cada valor con su meta.</div>', unsafe_allow_html=True)
    capacidad = st.session_state.get("capacidad", 20)
    inc = al[al["origen"] == "Incursión"]
    aten = inc[inc["estado"] == "atendida"].copy()
    aten["horas"] = (pd.to_datetime(aten["atendida_fecha"], format="mixed") - aten["fecha"]).dt.total_seconds() / 3600
    ult = aten[aten["fecha"] >= corte - pd.Timedelta(days=90)]
    t_medio = float(ult["horas"].clip(lower=0).median()) if len(ult) else float("nan")
    pct48 = float((ult["horas"] <= 48).mean()) if len(ult) else float("nan")
    pend_crit = int(((al["estado"] == "abierta") & (al["nivel"] == "Crítico") & (al["origen"] == "Incursión")).sum())
    g = ev[ev["nivel"] >= 2]
    inc12 = len(g[g["fecha_hora"] >= corte - pd.Timedelta(weeks=12)]) / 12
    inc12p = len(g[(g["fecha_hora"] < corte - pd.Timedelta(weeks=12)) & (g["fecha_hora"] >= corte - pd.Timedelta(weeks=24))]) / 12
    h = historia(version_historia()).groupby("semana_inicio")["perdida_ha"].sum()
    ha4, ha4p = float(h.iloc[-4:].sum()), float(h.iloc[-8:-4].sum())
    ha_riesgo, ha_plan = P["ha_esperadas_30d"].sum(), P.head(capacidad)["ha_esperadas_30d"].sum()
    activas = ev[(ev["fecha_hora"] >= corte - pd.Timedelta(days=30)) & ev["sensor_id"].isin(est["estacion_id"])]["sensor_id"].nunique()
    m30 = metr["frente_1"]["perdida_30d"]; f2 = metr["frente_2"]; rec_crit = f2["comparativa"]["LightGBM"]["Recall_Critico"]
    filas = [
        ["Respuesta", "Tiempo de atención de una incursión grave", f"{t_medio:.0f} horas", "48 horas o menos", semaforo(t_medio, 48, False), "Mediana de los últimos 90 días"],
        ["Respuesta", "Incursiones graves atendidas en 48 horas", f"{pct48:.0%}", "80 % o más", semaforo(pct48, 0.8), "Últimos 90 días"],
        ["Respuesta", "Incursiones críticas sin atender", f"{pend_crit}", "0", "🟢 Cumple" if pend_crit == 0 else "🔴 No cumple", "En este momento"],
        ["Presión", "Incursiones graves por semana", f"{inc12:.1f}", "que baje", "🟢 Bajó" if inc12 <= inc12p else "🔴 Subió", f"Últimas 12 semanas (las 12 anteriores: {inc12p:.1f})"],
        ["Presión", "Bosque perdido en el último mes", f"{ha4:,.0f} ha", "que baje", "🟢 Bajó" if ha4 <= ha4p else "🔴 Subió", f"Mes anterior: {ha4p:,.0f} ha"],
        ["Presión", "Bosque en riesgo el próximo mes", f"{ha_riesgo:,.0f} ha", "que baje", "—", f"≈ {ha_riesgo * T_C_POR_HA * CO2_POR_C / 1000:,.0f} mil t de CO₂"],
        ["Planificación", "Bosque en riesgo cubierto por el plan", f"{ha_plan / ha_riesgo:.0%}", "60 % o más", semaforo(ha_plan / ha_riesgo, 0.6), f"Con {capacidad} cuadrantes por semana"],
        ["Planificación", "CO₂ que se puede evitar con el plan", f"{ha_plan * 0.5 * T_C_POR_HA * CO2_POR_C / 1000:,.1f} mil t", "que suba", "—", "Si se detiene la mitad de la pérdida"],
        ["Monitoreo", "Cámaras y sensores que reportaron este mes", f"{activas} de {len(est)}", "90 % o más", semaforo(activas / len(est), 0.9), "Estaciones con algún registro en 30 días"],
        ["Sistema", "Pérdidas de bosque que el sistema anticipó", f"{m30['deteccion']['Recall']:.0%}", "80 % o más", semaforo(m30["deteccion"]["Recall"], 0.8), "Prueba con datos 2025–2026"],
        ["Sistema", "Incursiones críticas reconocidas", f"{rec_crit:.0%}", "88 % o más", semaforo(rec_crit, 0.88), "Prueba en estaciones que el sistema no conocía"],
    ]
    st.dataframe(pd.DataFrame(filas, columns=["Tema", "Indicador", "Valor actual", "Meta", "Estado", "Cómo se mide"]),
                 hide_index=True, width="stretch", height=425)

    a, b = st.columns(2)
    s = h.iloc[-52:].reset_index()
    f1 = go.Figure(go.Bar(x=s["semana_inicio"], y=s["perdida_ha"], marker_color=COLOR["Alto"]))
    f1.update_layout(title="Bosque perdido por semana (último año)", height=300, yaxis_title="hectáreas", plot_bgcolor="white", margin=dict(t=45, b=10))
    a.plotly_chart(f1, width="stretch")
    w = g[g["fecha_hora"] >= corte - pd.Timedelta(weeks=52)].groupby(pd.Grouper(key="fecha_hora", freq="W-MON")).size().reset_index(name="n")
    f2g = go.Figure(go.Scatter(x=w["fecha_hora"], y=w["n"], mode="lines+markers", line=dict(color=COLOR["Crítico"], width=2)))
    f2g.update_layout(title="Incursiones graves por semana (último año)", height=300, yaxis_title="incursiones", plot_bgcolor="white", margin=dict(t=45, b=10))
    b.plotly_chart(f2g, width="stretch")

    st.subheader("¿Qué tan confiable es el sistema?")
    st.write("El sistema se puso a prueba con datos de junio de 2025 a septiembre de 2026 que **no había visto** al aprender. Estos fueron los resultados:")
    d = m30["deteccion"]; cap = m30["comparativa"]["Hurdle LightGBM"]["Captura_top20"]
    k = st.columns(4)
    k[0].markdown(tarjeta("Avisó a tiempo", f"{round(d['Recall'] * 10)} de cada 10", "veces que un cuadrante perdió bosque, el sistema lo había anticipado un mes antes", VERDE), unsafe_allow_html=True)
    k[1].markdown(tarjeta("Avisos acertados", f"{round(d['Precision'] * 10)} de cada 10", "avisos de pérdida de bosque se confirmaron después", VERDE), unsafe_allow_html=True)
    k[2].markdown(tarjeta("Dónde buscar", f"{cap:.0%}", "del bosque perdido estaba en la quinta parte de cuadrantes que el sistema puso primero", VERDE), unsafe_allow_html=True)
    k[3].markdown(tarjeta("Incursiones críticas", f"{round(rec_crit * 100)} de cada 100", "fueron reconocidas como críticas por el sistema", VERDE), unsafe_allow_html=True)
    st.write("")
    nv = inf.nivel_deforestacion(wf["prob_evento"], wf["Hurdle LightGBM"], d["umbral"]).to_numpy()
    tn = wf.assign(nivel=nv).groupby("nivel")["evento_30d"].mean().reindex(range(4)).fillna(0)
    fig = go.Figure(go.Bar(x=NIVELES, y=tn.to_numpy() * 100, marker_color=[COLOR[n] for n in NIVELES],
                           text=[f"{v:.0%}" for v in tn], textposition="outside"))
    fig.update_layout(title="Cuando el sistema asignó cada nivel, ¿en qué porcentaje de los casos se perdió bosque el mes siguiente?",
                      height=340, yaxis=dict(title="% de casos con pérdida", range=[0, 112]), plot_bgcolor="white", margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.markdown('<div class="ayuda"><b>Lo que el sistema no puede hacer:</b> no adivina la semana exacta ni la cantidad exacta de hectáreas; '
                'indica <i>dónde</i> es más probable la pérdida y aproximadamente <i>cuánta</i>. En los picos más grandes las hectáreas reales suelen superar a las estimadas.</div>', unsafe_allow_html=True)
    with st.expander("Detalle técnico (para el equipo de ciencia de datos)"):
        comp = pd.DataFrame(m30["comparativa"]).T.reset_index().rename(columns={"index": "Modelo"})
        st.markdown("**Frente 1 · 30 días · validación walk-forward**")
        st.dataframe(comp[["Modelo", "MAE", "RMSE", "R2", "Spearman_espacial", "Captura_top20"]].round(3), hide_index=True, width="stretch")
        st.markdown(f"Detector: ROC-AUC {d['ROC_AUC']:.3f} · PR-AUC {d['PR_AUC']:.3f} · recall {d['Recall']:.2f} · precisión {d['Precision']:.2f} · "
                    f"umbral {d['umbral']:.2f}. A 7 días: ROC-AUC {metr['frente_1']['perdida_7d']['deteccion']['ROC_AUC']:.3f}.")
        st.markdown("**Frente 2 · validación cruzada agrupada por estación**")
        st.dataframe(pd.DataFrame(f2["comparativa"]).T.round(4).reset_index().rename(columns={"index": "Modelo"}), hide_index=True, width="stretch")
        for img in ["f1_roc_calibracion.png", "f1_curva_captura.png", "f2_matriz_confusion.png", "f2_prueba_estres.png"]:
            if (REP / img).exists():
                st.image(str(REP / img))


def pagina_ayuda():
    grilla, _, _, _, est = datos_fijos()
    st.title("Ayuda")
    st.subheader("¿Qué hace este sistema?")
    st.markdown("""
BIODIVERSITY-GUARD Predict ayuda a decidir **dónde enviar a los guardaparques**. Hace dos cosas:

1. **Anticipa la pérdida de bosque.** Con las alertas satelitales de los últimos meses, el clima y el uso del suelo, calcula para cada cuadrante
   la probabilidad de que pierda bosque en los próximos 7 y 30 días y cuántas hectáreas están en riesgo.
2. **Clasifica las incursiones.** Cada detección de las cámaras y sensores (o cada reporte que usted ingrese) recibe un nivel de riesgo según
   qué se detectó, dónde, a qué hora y si se repite en el mismo lugar.""")
    st.subheader("Niveles de atención")
    st.dataframe(pd.DataFrame([
        ["🟢 Bajo", "Sin señales de riesgo o solo fauna.", "Nada especial; patrullaje de rutina."],
        ["🟡 Moderado", "Alguna señal: pérdida posible o presencia humana en zonas permitidas.", "Mantener en observación."],
        ["🟠 Alto", "Pérdida de bosque probable o incursión en zonas protegidas.", "Visitar el sector en 48 horas."],
        ["🔴 Crítico", "Pérdida de bosque casi segura y de varias hectáreas, o armas, motosierras o ingresos nocturnos reiterados.", "Actuar de inmediato con apoyo."],
    ], columns=["Nivel", "Qué significa", "Qué hacer"]), hide_index=True, width="stretch")
    st.subheader("Palabras que aparecen en las pantallas")
    st.markdown("""
- **Cuadrante**: cuadro de 5 × 5 km (2 500 ha). Hay 273: 167 tocan la reserva y 106 son de la zona de amortiguamiento.
- **Hectárea (ha)**: 100 × 100 metros; algo más que una cancha de fútbol.
- **Probabilidad**: qué tan seguro está el sistema, de 0 % a 100 %.
- **Incursión**: ingreso o actividad humana detectada (personas, vehículos, trampas, motosierras, armas).
- **Reincidencia**: dos o más amenazas en el mismo lugar en una semana; sube el nivel de riesgo.
- **CO₂ evitado**: carbono que no llega a la atmósfera si se impide la tala (138,9 toneladas de carbono por hectárea).""")
    st.subheader("De dónde vienen los datos")
    st.markdown(f"""
- Alertas de deforestación: Geobosques (MINAM), semanales, de agosto de 2023 a septiembre de 2026.
- Clima: estación Malinowski del SENAMHI. Uso del suelo: MapBiomas Perú. Límites: SERNANP.
- Cámaras y sensores: red simulada de {len(est)} estaciones. Las detecciones provienen de anotaciones reales de fotos (Roboflow), pero su ubicación y su fecha son simuladas porque aún no hay sensores instalados.
- Ubicación de los cuadrantes: {grilla['origen'].iloc[0]}. {int((grilla['geometria'] == 'validada').sum())} cuadrantes están verificados; el resto, sobre todo en la zona de amortiguamiento, tiene posición aproximada.""")
    st.subheader("Datos de demostración")
    st.write("Si cargó datos de prueba y quiere volver al estado inicial, use este botón. No borra los modelos ni los archivos del proyecto.")
    if st.button("Restablecer datos de demostración"):
        bd.init(); st.cache_data.clear()
        for k in [k for k in st.session_state if k.startswith("res_")]:
            del st.session_state[k]
        st.success("Datos restablecidos.")


# --------------------------------------------------------------------------- #
# Arranque
# --------------------------------------------------------------------------- #
with st.spinner("Preparando los datos (solo la primera vez)…"):
    bd.asegurar()

pagina = st.navigation({
    "🌳 BIODIVERSITY-GUARD": [
        st.Page(pagina_inicio, title="Inicio", icon="🏠", url_path="inicio", default=True),
        st.Page(pagina_mapa, title="Mapa", icon="🗺️", url_path="mapa"),
        st.Page(pagina_plan, title="Plan de patrullaje", icon="🚩", url_path="plan"),
        st.Page(pagina_alertas, title="Alertas", icon="🔔", url_path="alertas")],
    "Datos": [st.Page(pagina_ingreso, title="Ingresar datos", icon="📥", url_path="ingresar")],
    "Seguimiento": [
        st.Page(pagina_indicadores, title="Indicadores", icon="📊", url_path="indicadores"),
        st.Page(pagina_ayuda, title="Ayuda", icon="❓", url_path="ayuda")],
})
with st.sidebar:
    st.divider()
    st.caption(f"Datos al {bd.fecha_corte():%d/%m/%Y}")
    st.caption("🟢 Bajo · 🟡 Moderado · 🟠 Alto · 🔴 Crítico")
pagina.run()
