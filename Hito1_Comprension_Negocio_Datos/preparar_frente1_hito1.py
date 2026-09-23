"""Define la variable objetivo del Frente 1 para el Hito 1.

Parte del panel semanal por cuadrante. perdida_ha es la pérdida observada en
esa misma semana y cuadrante: cada alerta ATD es un píxel de la malla medida
(30 m), es decir 0.09 ha. No arma horizontes a 7 ni a 30 días.

alerta_observada es una columna de auditoría: no debe utilizarse como predictor
contemporáneo porque se deriva del mismo conteo que origina perdida_ha.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "data" / "dataset_frente1_crudo.csv"
SALIDA = RAIZ / "data" / "dataset_frente1_hito1.csv"
# Separación entre puntos vecinos del shapefile ATD: 30 m. 30 x 30 m = 0.09 ha.
HECTAREAS_POR_ALERTA = 0.09


def cargar_panel(ruta: Path) -> pd.DataFrame:
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el panel del Frente 1: {ruta}")
    try:
        panel = pd.read_csv(ruta)
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Fallo al leer {ruta}") from exc

    requeridas = {"semana", "cuadrante_id", "n_alertas"}
    faltan = requeridas.difference(panel.columns)
    if faltan:
        raise ValueError(f"Al panel le faltan columnas: {sorted(faltan)}")
    return panel


def definir_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Hectáreas de pérdida observadas en la misma semana y cuadrante.

    La semana sin alerta ATD queda en cero: ese producto no registró pérdida.
    alerta_observada conserva si existía un conteo original antes del relleno.
    """
    tabla = panel.sort_values(["cuadrante_id", "semana"]).reset_index(drop=True)
    tabla["alerta_observada"] = tabla["n_alertas"].notna()
    if tabla["alerta_observada"].isna().any():
        raise ValueError("alerta_observada no puede contener valores nulos.")
    tabla["perdida_ha"] = tabla["n_alertas"].fillna(0) * HECTAREAS_POR_ALERTA
    return tabla


def exportar(tabla: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)


def main() -> None:
    panel = cargar_panel(ENTRADA)
    resultado = definir_target(panel)
    exportar(resultado, SALIDA)

    print(f"Filas de entrada: {len(panel)}")
    print(f"Filas exportadas: {len(resultado)}")
    print(f"Cuadrantes: {resultado['cuadrante_id'].nunique()}")
    print(f"Semanas distintas: {resultado['semana'].nunique()}")
    print(f"perdida_ha nula: {int(resultado['perdida_ha'].isna().sum())}")
    print(f"Filas con pérdida: {int((resultado['perdida_ha'] > 0).sum())}")
    print(f"Dataset escrito en {SALIDA}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
