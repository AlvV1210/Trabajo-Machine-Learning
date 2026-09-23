"""Genera las figuras reproducibles del EDA del Hito 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATA = RAIZ / "data"
SALIDA = RAIZ / "reports" / "eda"


def guardar(figura: plt.Figure, nombre: str, ajustar: bool = True) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    if ajustar:
        figura.tight_layout()
    figura.savefig(SALIDA / nombre, dpi=180, bbox_inches="tight")
    plt.close(figura)


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    frente1 = pd.read_csv(DATA / "dataset_frente1_hito1.csv")
    frente2 = pd.read_csv(DATA / "dataset_frente2_hito1_final.csv")
    return frente1, frente2


def figura_1(frente1: pd.DataFrame) -> None:
    figura, eje = plt.subplots(figsize=(8, 5))
    eje.hist(frente1["perdida_ha"], bins=40, color="#2f6f61", edgecolor="white")
    eje.set(title="Distribucion de perdida estimada", xlabel="perdida_ha (ha)", ylabel="Frecuencia")
    guardar(figura, "figura_1_histograma_perdida_ha.png")


def figura_2(frente1: pd.DataFrame) -> None:
    columnas = ["temperatura_media", "humedad_media", "precipitacion_suma", "perdida_ha"]
    figura, ejes = plt.subplots(1, len(columnas), figsize=(12, 4))
    for eje, columna in zip(ejes, columnas):
        eje.boxplot(frente1[columna].dropna(), vert=True)
        eje.set_title(columna.replace("_", " "))
        eje.grid(axis="y", alpha=0.25)
    guardar(figura, "figura_2_boxplots_frente1.png")


def figura_3(frente1: pd.DataFrame) -> None:
    columnas = ["temperatura_media", "humedad_media", "precipitacion_suma", "perdida_ha"]
    figura, ejes = plt.subplots(1, 2, figsize=(13, 6))
    imagen = None
    for eje, metodo in zip(ejes, ("pearson", "spearman")):
        correlacion = frente1[columnas].corr(method=metodo)
        imagen = eje.imshow(correlacion, vmin=-1, vmax=1, cmap="RdYlBu_r")
        eje.set_xticks(range(len(columnas)), columnas, rotation=45, ha="right")
        eje.set_yticks(range(len(columnas)), columnas)
        for fila in range(len(columnas)):
            for columna in range(len(columnas)):
                eje.text(columna, fila, f"{correlacion.iloc[fila, columna]:.2f}", ha="center", va="center")
        eje.set_title(metodo.capitalize())
    figura.subplots_adjust(left=0.08, right=0.82, bottom=0.22, top=0.86, wspace=0.48)
    eje_barra = figura.add_axes([0.86, 0.22, 0.025, 0.64])
    figura.colorbar(imagen, cax=eje_barra, label="Correlacion")
    figura.suptitle("Correlaciones del Frente 1")
    guardar(figura, "figura_3_correlacion_pearson_frente1.png", ajustar=False)


def figura_4(frente1: pd.DataFrame) -> None:
    tabla = frente1.assign(
        fecha_inicio=pd.to_datetime(frente1["semana"].str.split("/").str[0])
    )
    semanal = tabla.groupby("fecha_inicio", as_index=False).agg(
        perdida_ha=("perdida_ha", "sum"),
        cuadrantes_con_alerta=("alerta_observada", "sum"),
    )
    figura, eje = plt.subplots(figsize=(11, 5))
    eje.plot(semanal["fecha_inicio"], semanal["perdida_ha"], color="#b45f06", label="perdida_ha total")
    eje.set(xlabel="Semana", ylabel="Hectareas estimadas", title="Evolucion temporal de la perdida estimada")
    eje.grid(alpha=0.25)
    segundo_eje = eje.twinx()
    segundo_eje.plot(
        semanal["fecha_inicio"],
        semanal["cuadrantes_con_alerta"],
        color="#3c78a8",
        label="cuadrantes con alerta",
    )
    segundo_eje.set_ylabel("Cuadrantes con alerta")
    lineas, etiquetas = eje.get_legend_handles_labels()
    lineas_2, etiquetas_2 = segundo_eje.get_legend_handles_labels()
    eje.legend(lineas + lineas_2, etiquetas + etiquetas_2, loc="upper left")
    guardar(figura, "figura_4_evolucion_temporal_frente1.png")


def figura_5(frente2: pd.DataFrame) -> None:
    etiquetas = frente2["Nivel_Riesgo"].map({0: "Bajo", 1: "Moderado", 2: "Alto", 3: "Critico"}).fillna("Nulo")
    conteo = etiquetas.value_counts().reindex(["Bajo", "Moderado", "Alto", "Critico", "Nulo"], fill_value=0)
    figura, eje = plt.subplots(figsize=(8, 5))
    eje.bar(conteo.index, conteo.values, color=["#70ad47", "#ffc000", "#ed7d31", "#c00000", "#7f7f7f"])
    eje.set(title="Distribucion de Nivel_Riesgo", ylabel="Anotaciones")
    eje.tick_params(axis="x", rotation=15)
    guardar(figura, "figura_5_distribucion_nivel_riesgo.png")


def figura_6(frente2: pd.DataFrame) -> None:
    principales = frente2["clase_detectada"].value_counts().head(12).sort_values()
    figura, eje = plt.subplots(figsize=(9, 6))
    eje.barh(principales.index, principales.values, color="#3c78a8")
    eje.set(title="Principales clases detectadas", xlabel="Anotaciones")
    guardar(figura, "figura_6_clases_frente2.png")


def figura_7(frente2: pd.DataFrame) -> None:
    fuentes = frente2["fuente"].value_counts().sort_values()
    figura, eje = plt.subplots(figsize=(9, 6))
    eje.barh(fuentes.index, fuentes.values, color="#8064a2")
    eje.set(title="Anotaciones por fuente", xlabel="Anotaciones")
    guardar(figura, "figura_7_fuentes_frente2.png")


def main() -> None:
    frente1, frente2 = cargar_datos()
    figura_1(frente1)
    figura_2(frente1)
    figura_3(frente1)
    figura_4(frente1)
    figura_5(frente2)
    figura_6(frente2)
    figura_7(frente2)
    print(f"Figuras generadas en: {SALIDA}")


if __name__ == "__main__":
    main()