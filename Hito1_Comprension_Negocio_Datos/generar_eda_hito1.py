"""EDA del Hito 1 regenerado sobre los datasets FINALES (corrección Hito 3).

Frente 1: data/dataset_frente1_hito1_final.csv (273 cuadrantes x 164 semanas)
Frente 2: data/dataset_frente2_hito1_final.csv

Genera reports/eda/*.png y reports/eda/resumen_eda.json con las cifras citadas
en el informe (nulos, outliers IQR, normalidad, correlaciones).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SALIDA = ROOT / "reports" / "eda"
SALIDA.mkdir(parents=True, exist_ok=True)
AZUL, NARANJA, ROJO, VERDE = "#1f4e79", "#ef6c00", "#c62828", "#2e7d32"


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(SALIDA / nombre, dpi=170, bbox_inches="tight")
    plt.close(fig)


def iqr_outliers(s: pd.Series) -> int:
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())


def main() -> None:
    f1 = pd.read_csv(DATA / "dataset_frente1_hito1_final.csv")
    f2 = pd.read_csv(DATA / "dataset_frente2_hito1_final.csv")
    f1["semana_inicio"] = pd.to_datetime(f1["semana"].str[:10])
    clima = ["temperatura_media", "humedad_media", "precipitacion_suma"]
    num = clima + ["perdida_ha", "tasa_deforestacion_ha_anio", "mb_mineria", "mb_agropecuario",
                   "mb_bosque", "cobertura_anp", "densidad_poblacional"]
    r = {"frente1": {"filas": len(f1), "semanas": int(f1["semana"].nunique()),
                     "cuadrantes": int(f1["cuadrante_id"].nunique()),
                     "nulos_pct": (f1.isna().mean() * 100).round(2)[lambda s: s > 0].to_dict(),
                     "descriptivos": f1[clima + ["perdida_ha", "n_alertas"]].describe().round(3).to_dict(),
                     "outliers_iqr": {c: iqr_outliers(f1[c]) for c in clima + ["n_alertas", "perdida_ha"]},
                     "pct_semanas_sin_perdida": float((f1["perdida_ha"] == 0).mean() * 100),
                     "pearson_con_perdida": f1[num].corr()["perdida_ha"].round(3).to_dict(),
                     "spearman_con_perdida": f1[num].corr("spearman")["perdida_ha"].round(3).to_dict(),
                     "normalidad": {}}}
    for c in clima + ["perdida_ha"]:
        s = f1[c].dropna()
        k2, p = stats.normaltest(s)
        r["frente1"]["normalidad"][c] = {"K2": float(k2), "p": float(p), "asimetria": float(stats.skew(s)),
                                         "curtosis": float(stats.kurtosis(s))}
    # Pérdida estructural por zona
    cel = f1.groupby("cuadrante_id").agg(perdida=("perdida_ha", "sum"), anp=("cobertura_anp", "first"),
                                         tasa=("tasa_deforestacion_ha_anio", "first"), causa=("causa_principal", "first"))
    r["frente1"]["perdida_por_causa"] = cel.groupby("causa")["perdida"].sum().round(1).to_dict()
    r["frente1"]["corr_tasa_historica_vs_perdida_2023_2026"] = float(cel[["tasa", "perdida"]].corr("spearman").iloc[0, 1])

    # ---------------- Figuras Frente 1
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(f1["perdida_ha"], bins=60, color=AZUL); ax[0].set_yscale("log")
    ax[0].set_title("Distribución de perdida_ha (escala log)"); ax[0].set_xlabel("ha por cuadrante-semana")
    pos = f1.loc[f1["perdida_ha"] > 0, "perdida_ha"]
    ax[1].hist(np.log1p(pos), bins=40, color=NARANJA); ax[1].set_title("log1p(perdida_ha) solo semanas con pérdida")
    guardar(fig, "figura_1_histograma_perdida_ha.png")

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, c in zip(ax, clima + ["perdida_ha"]):
        a.boxplot(f1[c].dropna(), patch_artist=True, boxprops=dict(facecolor="#d6e4f0"))
        a.set_title(c, fontsize=10)
    guardar(fig, "figura_2_boxplots_frente1.png")

    cm = f1[num].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num)), num, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(len(num)), num, fontsize=8)
    for i in range(len(num)):
        for j in range(len(num)):
            ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    plt.colorbar(im, shrink=0.8); ax.set_title("Matriz de correlación de Pearson – Frente 1")
    guardar(fig, "figura_3_correlacion_pearson_frente1.png")

    serie = f1.groupby("semana_inicio")["perdida_ha"].sum()
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.fill_between(serie.index, serie.values, color=NARANJA, alpha=0.5); ax.plot(serie.index, serie.values, color=NARANJA)
    ax.plot(serie.rolling(8).mean(), color=AZUL, lw=2, label="media móvil 8 semanas")
    ax.set_title("Pérdida semanal total en los 273 cuadrantes"); ax.set_ylabel("ha"); ax.legend(); ax.grid(alpha=0.25)
    guardar(fig, "figura_4_evolucion_temporal_frente1.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(cel["anp"], cel["tasa"], c=np.log1p(cel["perdida"]), cmap="YlOrRd", s=30, edgecolor="grey", lw=0.3)
    ax.set_xlabel("cobertura_anp (fracción en la reserva)"); ax.set_ylabel("tasa_deforestacion_ha_anio (2001–2022)")
    ax.set_title("Protección vs. deforestación histórica (color = pérdida 2023–2026)"); ax.grid(alpha=0.25)
    guardar(fig, "figura_8_anp_vs_tasa.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    cel.groupby("causa")["perdida"].sum().sort_values().plot.barh(ax=ax, color=[VERDE, AZUL, NARANJA, ROJO])
    ax.set_title("Pérdida acumulada 2023–2026 por causa principal del cuadrante"); ax.set_xlabel("ha")
    guardar(fig, "figura_9_perdida_por_causa.png")

    # ---------------- Frente 2
    vc = f2["Nivel_Riesgo"].map({0: "Bajo", 1: "Moderado", 2: "Alto", 3: "Crítico"}).fillna("Sin regla").value_counts()
    r["frente2"] = {"filas": len(f2), "clases_detectadas": int(f2["clase_detectada"].nunique()),
                    "nivel_riesgo": vc.to_dict(), "image_id_unicos": int(f2["image_id"].nunique())}
    fig, ax = plt.subplots(figsize=(7, 4))
    orden = ["Bajo", "Moderado", "Alto", "Crítico", "Sin regla"]
    ax.bar(orden, [vc.get(k, 0) for k in orden], color=[VERDE, "#f9a825", NARANJA, ROJO, "#9e9e9e"])
    ax.set_title("Distribución de Nivel_Riesgo (Frente 2, Hito 1)")
    guardar(fig, "figura_5_distribucion_nivel_riesgo.png")
    fig, ax = plt.subplots(figsize=(9, 6))
    f2["clase_detectada"].str.slice(0, 40).value_counts().head(25).iloc[::-1].plot.barh(ax=ax, color=AZUL)
    ax.set_title("Clases detectadas (incluye etiquetas residuales de exportación)")
    guardar(fig, "figura_6_clases_frente2.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    f2["fuente"].value_counts().iloc[::-1].plot.barh(ax=ax, color=NARANJA); ax.set_title("Registros por fuente")
    guardar(fig, "figura_7_fuentes_frente2.png")

    (SALIDA / "resumen_eda.json").write_text(json.dumps(r, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(r, indent=1, ensure_ascii=False, default=float)[:3000])


if __name__ == "__main__":
    main()
