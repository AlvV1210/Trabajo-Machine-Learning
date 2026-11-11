"""Frente 2 · telemetría realista (Hito 3).

Problema del dataset del Hito 1: las 15 434 anotaciones se repartieron con
coordenadas uniformes en todo el rectángulo de Tambopata, sin fecha, y más de la
mitad eran amenazas (3 177 «críticas»). En el mapa eso se veía como miles de
incursiones simultáneas, algo que no ocurre en una reserva real.

Este script reconstruye la telemetría como la produciría una red de monitoreo:

1. RED DE SENSORES. 60 estaciones fijas (cámaras trampa y sensores acústicos)
   elegidas entre los puntos del Hito 1 que están cerca de ríos, trochas o del
   límite de la reserva (donde realmente se instalan), separadas al menos 4 km.
   Cada estación conserva sus distancias reales a río, carretera y límite
   (calculadas en el Hito 1 con SERNANP + OpenStreetMap).
2. FECHAS. Cada evento recibe fecha y hora dentro del mismo periodo del Frente 1
   (agosto 2023 – septiembre 2026).
3. PROPORCIÓN REALISTA. Se conservan las 4 647 anotaciones de fauna y se
   submuestrean las amenazas hasta ~12 % de los eventos. Las incursiones graves
   quedan en el orden de 1 a 2 por semana, coherente con el balance oficial:
   256 interdicciones apoyadas por SERNANP entre 2024 y 2026, más de 70
   guardaparques y 10 puestos de control (Caretas, 19-05-2026).
4. DÓNDE Y CUÁNDO. Las amenazas se concentran en estaciones con mayor presión
   (minería, agro, tasa histórica del cuadrante del Frente 1; cercanía a vías) y
   en las semanas con mayor deforestación regional. Llegan en «incidentes» de
   1 a 3 detecciones en pocas horas (p. ej. persona + motosierra).
5. REGLA DE RIESGO. La del Hito 1 (clase x zona) más dos agravantes operativos:
   actividad humana de noche y reincidencia en la misma estación (>= 2 amenazas
   en los 7 días previos). Ver features.asignar_riesgo.

No se inventan anotaciones: cada fila sigue siendo una anotación real de los
datasets de Roboflow (image_id y fuente se conservan). Lo simulado es la
ubicación, la fecha y el sensor, igual que en el Hito 1.

Uso: python Hito3_Modelado_Avanzado_Evaluacion/construir_frente2_realista.py
Salidas: data/estaciones_sensores.csv, data/dataset_frente2_hito3_realista.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import AMENAZAS, FAUNA, asignar_riesgo, normalizar_frente2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEMILLA = 42
N_ESTACIONES = 60
SEPARACION_M = 4_000
A_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
A_GEO = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)

# Amenazas a conservar (de las 9 346 disponibles) y horas típicas
CUPO_AMENAZAS = {"persona": 190, "vehiculo_ligero": 130, "vehiculo_maderero": 85, "trampa_caza": 75,
                 "arma_blanca": 60, "motosierra": 45, "arma_fuego": 35, "cazador": 10}
NOCTURNAS = {"rana", "snake", "jaguar", "capibara"}


def elegir_estaciones(pts: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Muestreo por punto más lejano entre candidatos accesibles (río, trocha o límite)."""
    cand = pts[(pts["distancia_a_rio_m"] < 800) | (pts["distancia_a_carretera_m"] < 1500)
               | ((pts["distancia_limite_m"] < 1000) & (pts["distancia_limite_m"] > 0))].copy()
    cand["zona"] = np.where(cand["en_zona_intangible"] == 1, "Reserva",
                            np.where(cand["en_zona_amortiguamiento"] == 1, "Amortiguamiento", "Exterior"))
    cupos = {"Reserva": 34, "Amortiguamiento": 16, "Exterior": 10}
    elegidas = []
    for zona, n in cupos.items():
        c = cand[cand["zona"] == zona].sample(frac=1, random_state=int(rng.integers(1e9)))
        xy = c[["x", "y"]].to_numpy()
        idx = [0]
        while len(idx) < n:
            d = np.min(np.linalg.norm(xy[:, None, :] - xy[idx][None, :, :], axis=2), axis=1)
            if elegidas:
                prev = pd.concat(elegidas)[["x", "y"]].to_numpy()
                d = np.minimum(d, np.min(np.linalg.norm(xy[:, None, :] - prev[None, :, :], axis=2), axis=1))
            j = int(np.argmax(d))
            if d[j] < SEPARACION_M:
                break
            idx.append(j)
        elegidas.append(c.iloc[idx])
    est = pd.concat(elegidas).reset_index(drop=True)
    est["estacion_id"] = [f"EST-{i + 1:02d}" for i in range(len(est))]
    est["tipo_sensor"] = rng.choice(["camara", "acustico"], len(est), p=[0.7, 0.3])
    return est


def presion_por_estacion(est: pd.DataFrame) -> pd.DataFrame:
    """Presión del cuadrante del Frente 1 donde cae la estación (0-1)."""
    from grilla_cuadrantes import obtener_grilla

    grilla = pd.read_csv(DATA / "cuadrantes_grid.csv") if (DATA / "cuadrantes_grid.csv").exists() else obtener_grilla()
    llave = {(int(a // 5000), int(b // 5000)): c for c, a, b in grilla[["cuadrante_id", "x_utm", "y_utm"]].itertuples(index=False)}
    est["cuadrante_id"] = [llave.get((int(a // 5000), int(b // 5000))) for a, b in zip(est["x"], est["y"])]
    f1 = pd.read_csv(DATA / "dataset_frente1_hito1_final.csv",
                     usecols=["cuadrante_id", "tasa_deforestacion_ha_anio", "mb_mineria", "mb_agropecuario"]).groupby("cuadrante_id").first()
    p = (f1["tasa_deforestacion_ha_anio"] / f1["tasa_deforestacion_ha_anio"].max() + 2 * f1["mb_mineria"] + f1["mb_agropecuario"])
    p = (p / p.quantile(0.98)).clip(0, 1)
    est["presion"] = est["cuadrante_id"].map(p).fillna(p.median())
    # fuera de la grilla o sin presión estructural: la cercanía a vías define el acceso
    acceso = np.exp(-est["distancia_a_carretera_m"] / 4000)
    est["presion"] = (0.75 * est["presion"] + 0.25 * acceso).clip(0.02, 1)
    return est


def construir() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEMILLA)
    base = normalizar_frente2(pd.read_csv(DATA / "dataset_frente2_hito1_final.csv"))
    base["x"], base["y"] = A_UTM.transform(base["longitud"].to_numpy(), base["latitud"].to_numpy())
    est = presion_por_estacion(elegir_estaciones(base, rng))

    # Semanas y actividad regional (del Frente 1)
    f1 = pd.read_csv(DATA / "dataset_frente1_hito1_final.csv", usecols=["semana", "perdida_ha"])
    f1["ini"] = pd.to_datetime(f1["semana"].str[:10])
    reg = f1.groupby("ini")["perdida_ha"].sum().rolling(4, min_periods=1).mean()
    semanas = reg.index.to_numpy()
    w_semana = 0.5 + reg.to_numpy() / reg.max()
    w_semana /= w_semana.sum()
    meses_secos = pd.Series(semanas).dt.month.isin([5, 6, 7, 8, 9, 10]).to_numpy()

    # ---- Fauna: todas las anotaciones, más frecuentes donde hay menos presión
    fauna = base[base["clase"].isin(FAUNA)].copy()
    w_est_fauna = (1.15 - est["presion"]) * np.where(est["zona"] == "Reserva", 1.5, np.where(est["zona"] == "Amortiguamiento", 1.0, 0.5))
    fauna["k_est"] = rng.choice(len(est), len(fauna), p=(w_est_fauna / w_est_fauna.sum()).to_numpy())
    w_sem_fauna = np.where(meses_secos, 1.15, 0.9); w_sem_fauna = w_sem_fauna / w_sem_fauna.sum()
    fauna["k_sem"] = rng.choice(len(semanas), len(fauna), p=w_sem_fauna)
    noct = fauna["clase"].isin(NOCTURNAS).to_numpy()
    fauna["hora"] = np.where(noct, rng.choice([19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6], len(fauna)),
                             rng.choice(np.arange(6, 18), len(fauna)))
    fauna["dia"] = rng.integers(0, 7, len(fauna))

    # ---- Amenazas: submuestra por clase, agrupadas en incidentes
    partes = []
    for clase, n in CUPO_AMENAZAS.items():
        pool = base[base["clase"] == clase]
        partes.append(pool.sample(min(n, len(pool)), random_state=int(rng.integers(1e9))))
    amen = pd.concat(partes).sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
    tam = rng.choice([1, 2, 3], len(amen), p=[0.55, 0.3, 0.15])
    incidente = np.repeat(np.arange(len(tam)), tam)[: len(amen)]
    n_inc = incidente.max() + 1
    w_est_am = est["presion"].to_numpy() ** 1.5
    inc_est = rng.choice(len(est), n_inc, p=w_est_am / w_est_am.sum())
    inc_sem = rng.choice(len(semanas), n_inc, p=w_semana)
    inc_dia = rng.integers(0, 7, n_inc)
    inc_hora = np.where(rng.random(n_inc) < 0.45, rng.choice([19, 20, 21, 22, 23, 0, 1, 2, 3, 4], n_inc),
                        rng.choice(np.arange(5, 19), n_inc))
    amen["k_est"], amen["k_sem"], amen["dia"] = inc_est[incidente], inc_sem[incidente], inc_dia[incidente]
    amen["hora"] = (inc_hora[incidente] + rng.integers(0, 3, len(amen))) % 24

    ev = pd.concat([fauna, amen], ignore_index=True)
    e = est.iloc[ev["k_est"].to_numpy()].reset_index(drop=True)
    ev["fecha_hora"] = (pd.to_datetime(semanas[ev["k_sem"].to_numpy()]) + pd.to_timedelta(ev["dia"], unit="D")
                        + pd.to_timedelta(ev["hora"], unit="h") + pd.to_timedelta(rng.integers(0, 60, len(ev)), unit="m"))
    # posición: la estación más un desplazamiento menor a 150 m (alcance del sensor)
    ang, r = rng.uniform(0, 2 * np.pi, len(ev)), rng.uniform(0, 150, len(ev))
    lon, lat = A_GEO.transform(e["x"].to_numpy() + r * np.cos(ang), e["y"].to_numpy() + r * np.sin(ang))
    ev["latitud"], ev["longitud"] = lat, lon
    for c in ["estacion_id", "tipo_sensor", "cuadrante_id", "en_zona_intangible", "en_zona_amortiguamiento",
              "distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m"]:
        ev[c] = e[c].to_numpy()
    ev = ev.sort_values("fecha_hora").reset_index(drop=True)
    ev["evento_id"] = [f"EV-{i + 1:05d}" for i in range(len(ev))]
    ev["semana"] = ev["fecha_hora"].dt.to_period("W-MON").dt.start_time.dt.date.astype(str)
    ev["es_noche"] = ((ev["hora"] >= 19) | (ev["hora"] < 5)).astype(int)
    ev["amenazas_7d_estacion"] = reincidencia(ev)
    ev["Nivel_Riesgo"] = asignar_riesgo(ev)

    cols = ["evento_id", "fecha_hora", "semana", "estacion_id", "tipo_sensor", "image_id", "fuente", "clase_detectada",
            "clase", "confidence_score", "latitud", "longitud", "cuadrante_id", "en_zona_intangible",
            "en_zona_amortiguamiento", "distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m", "hora",
            "es_noche", "amenazas_7d_estacion", "Nivel_Riesgo"]
    lon_e, lat_e = A_GEO.transform(est["x"].to_numpy(), est["y"].to_numpy())
    est_out = est.assign(latitud=lat_e, longitud=lon_e)[
        ["estacion_id", "tipo_sensor", "zona", "cuadrante_id", "latitud", "longitud", "en_zona_intangible",
         "en_zona_amortiguamiento", "distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m", "presion"]]
    return ev[cols], est_out


def reincidencia(ev: pd.DataFrame) -> np.ndarray:
    """Número de amenazas en la misma estación durante los 7 días previos (sin contar el evento)."""
    out = np.zeros(len(ev), dtype=int)
    es_am = ev["clase"].isin(AMENAZAS).to_numpy()
    t = ev["fecha_hora"].to_numpy()
    for _, idx in ev.groupby("estacion_id").indices.items():
        ti, ai = t[idx], es_am[idx]
        orden = np.argsort(ti, kind="stable")
        ti, ai, idx = ti[orden], ai[orden], np.asarray(idx)[orden]
        tam = ti[ai]
        izq = np.searchsorted(tam, ti - np.timedelta64(7, "D"), side="left")
        der = np.searchsorted(tam, ti, side="left")
        out[idx] = der - izq
    return out


if __name__ == "__main__":
    ev, est = construir()
    est.to_csv(DATA / "estaciones_sensores.csv", index=False)
    ev.to_csv(DATA / "dataset_frente2_hito3_realista.csv", index=False)
    nivel = ev["Nivel_Riesgo"].map(dict(enumerate(["Bajo", "Moderado", "Alto", "Crítico"]))).value_counts()
    semanas = ev["semana"].nunique()
    print(f"Estaciones: {len(est)} ({est['zona'].value_counts().to_dict()})")
    print(f"Eventos: {len(ev)} en {semanas} semanas ({len(ev) / semanas:.1f} por semana)")
    print(f"Amenazas: {ev['clase'].isin(AMENAZAS).mean():.1%} de los eventos")
    print("Niveles:", nivel.to_dict())
    print(f"Críticos por semana: {nivel.get('Crítico', 0) / semanas:.2f} · Altos por semana: {nivel.get('Alto', 0) / semanas:.2f}")
