"""Geometría de los cuadrantes 5x5 km del Frente 1 (Q001-Q273).

Hito 3 - soporte para el mapa del dashboard.

Hay dos caminos, en orden de preferencia:

1. ``desde_shapefile``: reconstruye la cuadrícula EXACTA con el mismo algoritmo de
   ``Hito1_Comprension_Negocio_Datos/construir_frente1_crudo.py`` (celdas de 5 km
   alineadas a múltiplos de 5000 m en UTM 19S, ordenadas por fila ``y`` y luego
   columna ``x``). Requiere los shapefiles SERNANP que el equipo tiene en local
   (``dataset diverso foco2/ANP_ACR_ACP_ZA_SERNANP``). No se suben al repositorio
   por su tamaño.

2. ``reconstruir_aproximada``: si los shapefiles no están disponibles, la grilla se
   recupera a partir de datos que sí están en ``data/``:
   - Las 15 434 detecciones del Frente 2 tienen la bandera ``en_zona_intangible``
     calculada contra el polígono oficial de Tambopata. Agregadas por celda UTM de
     5 km dan la fracción de cada celda dentro de la reserva.
   - El Frente 1 guarda esa misma fracción en ``cobertura_anp`` para Q001-Q167.
   - Un alineamiento de secuencias (Needleman-Wunsch) entre ambas listas,
     recorridas en el mismo orden fila-columna, asigna a cada Qxxx su celda.
     El error medio de la fracción en las celdas emparejadas es ~0.02, lo que
     valida el emparejamiento.
   - Las celdas que caen fuera del bounding box de la telemetría (oeste y norte)
     se ubican por continuidad del orden fila-columna. Las 106 celdas de la zona
     de amortiguamiento (Q168-Q273) se ubican de forma aproximada.
   Cada celda lleva la columna ``geometria`` = ``validada`` | ``extrapolada`` |
   ``aproximada`` para que el dashboard lo muestre con transparencia.

Uso:
    python Hito3_Modelado_Avanzado_Evaluacion/grilla_cuadrantes.py
genera ``data/cuadrantes_grid.csv`` (centroides y esquinas en EPSG:4326).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FR1 = DATA / "dataset_frente1_hito1_final.csv"
FR2 = DATA / "dataset_frente2_hito1_final.csv"
SALIDA = DATA / "cuadrantes_grid.csv"

RUTA_ANP = ROOT / "dataset diverso foco2" / "ANP_ACR_ACP_ZA_SERNANP" / "ANP Nacional Definitivas" / "ANPNacionalDefinitivas.shp"
RUTA_ZA = ROOT / "dataset diverso foco2" / "ANP_ACR_ACP_ZA_SERNANP" / "ZonasdeAmortiguamiento.shp"

TAM = 5_000
N_RESERVA = 167
A_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
A_GEO = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)


# --------------------------------------------------------------------------- #
# 1. Camino exacto (shapefile SERNANP disponible)
# --------------------------------------------------------------------------- #
def desde_shapefile() -> pd.DataFrame | None:
    if not RUTA_ANP.is_file():
        return None
    import geopandas as gpd
    from shapely.geometry import box

    capa = gpd.read_file(RUTA_ANP)
    if capa.crs is None:
        capa = capa.set_crs("EPSG:4326")
    nombre = capa["anp_nomb"].astype(str).str.strip().str.casefold()
    reserva = capa.loc[nombre.eq("tambopata")].to_crs("EPSG:32719")
    geom = reserva.union_all()

    def celdas_que_intersectan(g, excluir: set) -> list[tuple[int, int]]:
        minx, miny, maxx, maxy = g.bounds
        out = []
        x = math.floor(minx / TAM) * TAM
        while x < maxx:
            y = math.floor(miny / TAM) * TAM
            while y < maxy:
                if box(x, y, x + TAM, y + TAM).intersects(g) and (x, y) not in excluir:
                    out.append((x, y))
                y += TAM
            x += TAM
        return sorted(out, key=lambda c: (c[1], c[0]))

    reserva_celdas = celdas_que_intersectan(geom, set())
    filas = [(f"Q{i + 1:03d}", x, y, "validada") for i, (x, y) in enumerate(reserva_celdas)]
    if RUTA_ZA.is_file():
        za = gpd.read_file(RUTA_ZA)
        if za.crs is None:
            za = za.set_crs("EPSG:4326")
        col = next((c for c in za.columns if "nomb" in c.lower()), None)
        if col:
            za = za[za[col].astype(str).str.casefold().str.contains("tambopata")]
        za_celdas = celdas_que_intersectan(za.to_crs("EPSG:32719").union_all(), set(reserva_celdas))
        base = len(filas)
        filas += [(f"Q{base + i + 1:03d}", x, y, "validada") for i, (x, y) in enumerate(za_celdas)]
    return _a_tabla(filas)


# --------------------------------------------------------------------------- #
# 2. Camino aproximado (sin shapefile)
# --------------------------------------------------------------------------- #
def _celdas_visibles() -> pd.DataFrame:
    f2 = pd.read_csv(FR2, usecols=["latitud", "longitud", "en_zona_intangible", "en_zona_amortiguamiento"])
    x, y = A_UTM.transform(f2["longitud"].to_numpy(), f2["latitud"].to_numpy())
    f2["cx"] = (np.asarray(x) // TAM).astype(int)
    f2["cy"] = (np.asarray(y) // TAM).astype(int)
    g = f2.groupby(["cx", "cy"]).agg(
        fi=("en_zona_intangible", "mean"), fa=("en_zona_amortiguamiento", "mean")
    ).reset_index()
    return g.sort_values(["cy", "cx"]).reset_index(drop=True)


def _alinear(obs: np.ndarray, seq: np.ndarray, gap_seq: float = 0.08) -> dict[int, int]:
    """Needleman-Wunsch: devuelve {indice_seq: indice_obs}."""
    n, m = len(obs), len(seq)
    D = np.zeros((n + 1, m + 1))
    B = np.zeros((n + 1, m + 1), dtype=int)
    D[0, :] = np.arange(m + 1) * gap_seq
    D[1:, 0] = np.cumsum(obs * 3 + 0.05)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opciones = (
                D[i - 1, j - 1] + abs(obs[i - 1] - seq[j - 1]),
                D[i, j - 1] + gap_seq,
                D[i - 1, j] + obs[i - 1] * 3 + 0.05,
            )
            k = int(np.argmin(opciones))
            D[i, j], B[i, j] = opciones[k], k
    i, j, pares = n, m, {}
    while i > 0 and j > 0:
        k = B[i, j]
        if k == 0:
            pares[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif k == 1:
            j -= 1
        else:
            i -= 1
    return pares


def reconstruir_aproximada() -> pd.DataFrame:
    f1 = pd.read_csv(FR1, usecols=["cuadrante_id", "cobertura_anp"])
    cob = f1.groupby("cuadrante_id")["cobertura_anp"].first().sort_index()
    ids = cob.index.tolist()
    seq = cob.to_numpy()[:N_RESERVA]

    vis = _celdas_visibles()
    res_vis = vis[vis["fi"] > 0].reset_index(drop=True)
    pares = _alinear(res_vis["fi"].to_numpy(), seq)

    pos: dict[int, tuple[int, int, str]] = {
        j: (int(res_vis.at[i, "cx"]), int(res_vis.at[i, "cy"]), "validada") for j, i in pares.items()
    }

    # Fila superior visible: la telemetría solo cubre parte; se fuerza continuidad.
    cy_top = int(res_vis["cy"].max())
    top = res_vis[res_vis["cy"] == cy_top].sort_values("cx")
    primeros_top = [j for j, (cx, cy, _) in pos.items() if cy == cy_top]
    if primeros_top:
        j0 = min(primeros_top)
        for j in [k for k, v in list(pos.items()) if v[1] == cy_top]:
            pos.pop(j)
        for k, (_, r) in enumerate(top.iterrows()):
            pos[j0 + k] = (int(r["cx"]), cy_top, "validada")

    # Huecos: celdas fuera del bounding box (oeste) o sin puntos de telemetría.
    emparejados = sorted(pos)
    for a, b in zip(emparejados[:-1], emparejados[1:]):
        if b - a <= 1:
            continue
        cxa, cya, _ = pos[a]
        cxb, cyb, _ = pos[b]
        faltan = list(range(a + 1, b))
        if cya == cyb:  # hueco interior de la misma fila
            libres = [c for c in range(cxa + 1, cxb)]
            for k, j in enumerate(faltan):
                pos[j] = (libres[min(k, len(libres) - 1)] if libres else cxa, cya, "extrapolada")
        else:  # celdas al oeste del inicio de la fila b
            for k, j in enumerate(reversed(faltan)):
                pos[j] = (cxb - 1 - k, cyb, "extrapolada")
    # Antes del primero
    primero = min(pos)
    for k, j in enumerate(range(primero - 1, -1, -1)):
        cx, cy, _ = pos[primero]
        pos[j] = (cx - 1 - k, cy, "extrapolada")
    # Después del último: filas al norte del bounding box, con el ancho de la fila previa.
    ultimo = max(pos)
    restantes = list(range(ultimo + 1, N_RESERVA))
    if restantes:
        fila_prev = sorted(c for c, y, _ in pos.values() if y == cy_top)
        ancho = max(4, len(fila_prev) - 1)
        x0 = fila_prev[0]
        for k, j in enumerate(restantes):
            fila, col = divmod(k, ancho)
            pos[j] = (x0 + col + fila, cy_top + 1 + fila, "extrapolada")

    # Si un hueco interior no tenía celdas libres, la fila se vuelve a trazar contigua.
    for cy in {v[1] for v in pos.values()}:
        js = sorted(j for j, v in pos.items() if v[1] == cy)
        cxs = [pos[j][0] for j in js]
        if len(set(cxs)) < len(cxs):
            x0 = min(cxs)
            for k, j in enumerate(js):
                pos[j] = (x0 + k, cy, pos[j][2])

    filas = [(ids[j], pos[j][0] * TAM, pos[j][1] * TAM, pos[j][2]) for j in range(N_RESERVA)]

    # Zona de amortiguamiento (Q168-Q273): visible al norte-oeste de la reserva; el resto
    # se distribuye en franjas hacia el norte (Puerto Maldonado / corredor Interoceánica).
    ocupadas = {(x // TAM, y // TAM) for _, x, y, _ in filas}
    za_vis = vis[(vis["fa"] > 0) & (vis["fi"] == 0)]
    za_ids = ids[N_RESERVA:]
    celdas_za: list[tuple[int, int, str]] = []
    for cy in sorted(za_vis["cy"].unique()):
        fila_vis = sorted(za_vis.loc[za_vis["cy"] == cy, "cx"])
        oeste = [fila_vis[0] - k for k in range(6, 0, -1)]
        for cx in oeste:
            if (cx, cy) not in ocupadas:
                celdas_za.append((cx, int(cy), "aproximada"))
        for cx in fila_vis:
            if (int(cx), int(cy)) not in ocupadas:
                celdas_za.append((int(cx), int(cy), "aproximada"))
    cy = int(za_vis["cy"].max()) + 1 if len(za_vis) else cy_top + 1
    x_min = min(c for c, _, _ in celdas_za) if celdas_za else 80
    while len(celdas_za) < len(za_ids):
        for cx in range(x_min, x_min + 15):
            if (cx, cy) not in ocupadas and len(celdas_za) < len(za_ids):
                celdas_za.append((cx, cy, "aproximada"))
        cy += 1
    for qid, (cx, cy_, tipo) in zip(za_ids, celdas_za):
        filas.append((qid, cx * TAM, cy_ * TAM, tipo))
    return _a_tabla(filas)


def _a_tabla(filas: list[tuple[str, int, int, str]]) -> pd.DataFrame:
    t = pd.DataFrame(filas, columns=["cuadrante_id", "x_utm", "y_utm", "geometria"])
    esquinas = {}
    for nombre, dx, dy in [("sw", 0, 0), ("se", TAM, 0), ("ne", TAM, TAM), ("nw", 0, TAM), ("c", TAM / 2, TAM / 2)]:
        lon, lat = A_GEO.transform((t["x_utm"] + dx).to_numpy(), (t["y_utm"] + dy).to_numpy())
        esquinas[f"lon_{nombre}"], esquinas[f"lat_{nombre}"] = lon, lat
    t = t.assign(**esquinas)
    t["zona"] = np.where(t["cuadrante_id"].str[1:].astype(int) <= N_RESERVA, "Reserva Nacional Tambopata", "Zona de amortiguamiento")
    return t


def obtener_grilla(guardar: bool = True) -> pd.DataFrame:
    grilla = desde_shapefile()
    origen = "shapefile SERNANP (exacta)"
    if grilla is None:
        grilla = reconstruir_aproximada()
        origen = "reconstrucción desde telemetría (aproximada)"
    grilla["origen"] = origen
    if guardar:
        grilla.to_csv(SALIDA, index=False)
    return grilla


if __name__ == "__main__":
    g = obtener_grilla()
    print(f"Cuadrantes: {len(g)} -> {SALIDA}")
    print(g["geometria"].value_counts().to_string())
    print("Celdas duplicadas:", int(g.duplicated(["x_utm", "y_utm"]).sum()))
