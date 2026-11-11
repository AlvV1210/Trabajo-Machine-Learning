"""Figuras explicativas del Hito 3 para el informe (no entrena nada: usa los resultados guardados).

Uso: python Hito3_Modelado_Avanzado_Evaluacion/figuras_informe.py
Salidas en reports/hito3/: informe_f2_antes_despues.png, informe_f1_niveles.png, informe_f1_resumen.png
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inferencia as inf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, REP = ROOT / "data", ROOT / "reports" / "hito3"
COLOR = {"Bajo": "#7fbf7b", "Moderado": "#f2c94c", "Alto": "#f2994a", "Crítico": "#d64545"}
NIVELES = list(COLOR)


def frente2_antes_despues() -> None:
    """Mismo mes de datos: telemetría del Hito 1 (puntos al azar) frente a la red de sensores del Hito 3."""
    h1 = pd.read_csv(DATA / "dataset_frente2_hito1_final.csv")
    h3 = pd.read_csv(DATA / "dataset_frente2_hito3_realista.csv", parse_dates=["fecha_hora"])
    est = pd.read_csv(DATA / "estaciones_sensores.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    g = h1[h1["Nivel_Riesgo"] >= 2]
    ax[0].scatter(h1["longitud"], h1["latitud"], s=2, color="#c9d3c9")
    ax[0].scatter(g["longitud"], g["latitud"], s=5, color=COLOR["Crítico"], alpha=0.6)
    ax[0].set_title(f"Hito 1: {len(h1):,} puntos repartidos al azar, sin fecha\n{len(g):,} amenazas altas o críticas a la vez", fontsize=11)
    ult = h3[h3["fecha_hora"] >= h3["fecha_hora"].max() - pd.Timedelta(days=30)]
    gu = ult[ult["Nivel_Riesgo"] >= 2]
    ax[1].scatter(est["longitud"], est["latitud"], s=38, marker="D", color="#37474f", label=f"{len(est)} cámaras y sensores")
    ax[1].scatter(gu["longitud"], gu["latitud"], s=110, color=COLOR["Crítico"], edgecolor="black", label=f"{len(gu)} incursiones graves en 30 días")
    ax[1].set_title(f"Hito 3: red de {len(est)} estaciones con fecha y hora\n{len(ult)} eventos en el último mes, la mayoría fauna", fontsize=11)
    ax[1].legend(loc="lower left", fontsize=9)
    for a in ax:
        a.set_xlabel("longitud"); a.grid(alpha=0.2)
    ax[0].set_ylabel("latitud")
    plt.tight_layout(); plt.savefig(REP / "informe_f2_antes_despues.png", dpi=170); plt.close()


def frente1_niveles() -> None:
    """¿Se cumplió lo que anunciaba cada nivel de atención? (prueba 2025-2026)."""
    m = json.loads((REP / "metricas_hito3.json").read_text(encoding="utf-8"))["frente_1"]["perdida_30d"]
    wf = pd.read_csv(DATA / "walkforward_perdida_30d.csv")
    nv = inf.nivel_deforestacion(wf["prob_evento"], wf["Hurdle LightGBM"], m["deteccion"]["umbral"]).to_numpy()
    t = wf.assign(n=nv).groupby("n").agg(pct=("evento_30d", "mean"), ha=("perdida_30d", "mean"), casos=("n", "size")).reindex(range(4))
    sem = wf["semana_inicio"].nunique()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    b = ax.bar(NIVELES, t["pct"] * 100, color=list(COLOR.values()))
    ax.bar_label(b, [f"{v:.0%}" for v in t["pct"]], padding=3, fontsize=12)
    ax.set_ylim(0, 115); ax.set_ylabel("% de casos en que sí se perdió bosque")
    ax.set_title("De cada 100 cuadrantes puestos en un nivel,\n¿cuántos perdieron bosque el mes siguiente?", fontsize=11)
    ax.set_xticks(range(4), [f"{n}\n({c / sem:.0f} por semana)" if c / sem >= 3 else f"{n}\n({c / sem:.1f} por semana)" for n, c in zip(NIVELES, t["casos"])])
    ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(REP / "informe_f1_niveles.png", dpi=170); plt.close()
    (REP / "niveles_atencion.json").write_text(json.dumps({
        NIVELES[k]: {"acierto": float(t.at[k, "pct"]), "ha_promedio": float(t.at[k, "ha"]), "cuadrantes_por_semana": float(t.at[k, "casos"] / sem)}
        for k in range(4)}, indent=2, ensure_ascii=False), encoding="utf-8")


def frente1_resumen() -> None:
    """Comparación sencilla: error en hectáreas y bosque cubierto patrullando 1 de cada 5 cuadrantes."""
    m = json.loads((REP / "metricas_hito3.json").read_text(encoding="utf-8"))["frente_1"]["perdida_30d"]["comparativa"]
    nombres = {"Persistencia": "Repetir el mes\nanterior", "Árbol baseline (Hito 2)": "Árbol\n(Hito 2)", "Random Forest": "Random\nForest",
               "XGBoost": "XGBoost", "LightGBM": "LightGBM", "Hurdle LightGBM": "Modelo elegido\n(dos etapas)"}
    col = ["#1b5e3b" if k == "Hurdle LightGBM" else "#b9c7bd" for k in nombres]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
    b = ax[0].bar(list(nombres.values()), [m[k]["MAE"] for k in nombres], color=col)
    ax[0].bar_label(b, [f"{m[k]['MAE']:.2f}" for k in nombres], padding=3)
    ax[0].set_title("Error promedio al estimar hectáreas (menos es mejor)", fontsize=11); ax[0].set_ylabel("hectáreas por cuadrante")
    b = ax[1].bar(list(nombres.values()), [m[k]["Captura_top20"] * 100 for k in nombres], color=col)
    ax[1].bar_label(b, [f"{m[k]['Captura_top20']:.0%}" for k in nombres], padding=3)
    ax[1].axhline(20, color="#888", ls="--", lw=1); ax[1].text(-0.4, 22, "eligiendo al azar: 20 %", ha="left", fontsize=9, color="#555")
    ax[1].set_title("Bosque perdido que se habría cubierto patrullando\n1 de cada 5 cuadrantes (más es mejor)", fontsize=11); ax[1].set_ylabel("% de la pérdida real")
    ax[1].set_ylim(0, 100)
    for a in ax:
        a.grid(axis="y", alpha=0.25); a.spines[["top", "right"]].set_visible(False); a.tick_params(axis="x", labelsize=8.5)
    plt.tight_layout(); plt.savefig(REP / "informe_f1_resumen.png", dpi=170); plt.close()


if __name__ == "__main__":
    frente2_antes_despues(); frente1_niveles(); frente1_resumen()
    print("Figuras guardadas en", REP)
