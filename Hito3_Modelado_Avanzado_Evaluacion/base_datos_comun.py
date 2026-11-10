"""Integración con la Base de Datos Común de BIODIVERSITY-GUARD (Hito 3, semana 12).

Arquitectura:

    Módulo IA (curso 1ASI0404: visión/audio)            BIODIVERSITY-GUARD Predict (1ACC0057)
    ───────────────────────────────────────            ─────────────────────────────────────
    cámara/sensor -> detección JSON  ──INSERT──►  detecciones (estado='nueva')
                                                      │
                                                      ▼  procesar_nuevas()
                                     enriquecimiento espacial + modelo Frente 2
                                                      │
                                                      ▼
                                   predicciones_riesgo ─► alertas ◄─ predicciones_deforestacion
                                                      │
                                                      ▼
                                               Dashboard Streamlit

El contrato de interfaz (``CONTRATO``) define los campos que el módulo IA debe
escribir. Se usa SQLite para la demo local; la misma DDL corre en PostgreSQL
(cambiando ``conectar``) cuando el equipo de IA exponga su servidor.

Uso:
    python base_datos_comun.py init           # crea la BD y carga cuadrantes + pronóstico F1
    python base_datos_comun.py simular 25     # el "módulo IA" inserta 25 detecciones nuevas
    python base_datos_comun.py procesar       # el motor ML clasifica lo nuevo y genera alertas
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import AMENAZAS, FAUNA, NIVELES, asignar_riesgo, construir_features_frente2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "biodiversity_guard.db"
MODELO_F2 = ROOT / "models" / "frente2_random_forest.joblib"

CONTRATO = {
    "evento_id": "TEXT  · UUID generado por el módulo IA",
    "timestamp_utc": "TEXT  · ISO-8601 de la captura",
    "sensor_id": "TEXT  · cámara trampa / sensor acústico / dron",
    "tipo_sensor": "TEXT  · camara | acustico | dron",
    "clase_detectada": "TEXT  · etiqueta normalizada (ver features.FAUNA y features.AMENAZAS)",
    "confidence_score": "REAL  · 0.0 a 1.0",
    "latitud": "REAL  · EPSG:4326",
    "longitud": "REAL  · EPSG:4326",
}

DDL = """
CREATE TABLE IF NOT EXISTS cuadrantes (
    cuadrante_id TEXT PRIMARY KEY, zona TEXT, x_utm REAL, y_utm REAL,
    lat_c REAL, lon_c REAL, geometria TEXT
);
CREATE TABLE IF NOT EXISTS detecciones (
    evento_id TEXT PRIMARY KEY, timestamp_utc TEXT, sensor_id TEXT, tipo_sensor TEXT,
    clase_detectada TEXT, confidence_score REAL, latitud REAL, longitud REAL,
    estado TEXT DEFAULT 'nueva', recibido_utc TEXT
);
CREATE TABLE IF NOT EXISTS predicciones_riesgo (
    evento_id TEXT PRIMARY KEY REFERENCES detecciones(evento_id), cuadrante_id TEXT,
    nivel_predicho INTEGER, nivel_nombre TEXT, p_critico REAL, en_zona_intangible INTEGER,
    en_zona_amortiguamiento INTEGER, distancia_limite_m REAL, modelo TEXT, procesado_utc TEXT
);
CREATE TABLE IF NOT EXISTS predicciones_deforestacion (
    cuadrante_id TEXT, semana TEXT, prob_evento_7d REAL, ha_esperadas_7d REAL,
    prob_evento_30d REAL, ha_esperadas_30d REAL, co2_en_riesgo_30d_t REAL, modelo TEXT,
    PRIMARY KEY (cuadrante_id, semana)
);
CREATE TABLE IF NOT EXISTS alertas (
    alerta_id TEXT PRIMARY KEY, creada_utc TEXT, cuadrante_id TEXT, origen TEXT,
    nivel TEXT, detalle TEXT, estado TEXT DEFAULT 'abierta'
);
"""

A_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)


def conectar() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(DDL)
    return con


def _referencia_espacial() -> pd.DataFrame:
    """Detecciones históricas con zona y distancias ya calculadas (Hito 1, OSM + SERNANP).
    Permiten enriquecer una detección nueva por su vecino más cercano cuando el
    shapefile no está en la máquina de inferencia."""
    ref = pd.read_csv(ROOT / "data" / "dataset_frente2_hito1_final.csv",
                      usecols=["latitud", "longitud", "en_zona_intangible", "en_zona_amortiguamiento",
                               "distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m"])
    x, y = A_UTM.transform(ref["longitud"].to_numpy(), ref["latitud"].to_numpy())
    return ref.assign(x=x, y=y)


def init() -> None:
    con = conectar()
    from grilla_cuadrantes import obtener_grilla

    g = obtener_grilla()
    g[["cuadrante_id", "zona", "x_utm", "y_utm", "lat_c", "lon_c", "geometria"]].to_sql(
        "cuadrantes", con, if_exists="replace", index=False)
    pred = pd.read_csv(ROOT / "data" / "predicciones_frente1.csv")
    pred.assign(modelo="Hurdle LightGBM v3")[
        ["cuadrante_id", "semana", "prob_evento_7d", "ha_esperadas_7d", "prob_evento_30d", "ha_esperadas_30d",
         "co2_en_riesgo_30d_t", "modelo"]].to_sql("predicciones_deforestacion", con, if_exists="replace", index=False)
    ahora = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    alertas = pred[pred["alerta_30d"] == 1]
    con.execute("DELETE FROM alertas WHERE origen='Frente 1'")
    con.executemany(
        "INSERT OR REPLACE INTO alertas VALUES (?,?,?,?,?,?,?)",
        [(f"F1-{r.cuadrante_id}-{r.semana[:10]}", ahora, r.cuadrante_id, "Frente 1",
          "Crítico" if r.prob_evento_30d >= 0.8 else "Alto",
          f"P(pérdida 30d)={r.prob_evento_30d:.0%}; {r.ha_esperadas_30d:.2f} ha esperadas", "abierta")
         for r in alertas.itertuples()])
    con.commit()
    print(f"BD inicializada en {DB_PATH}: {len(g)} cuadrantes, {len(pred)} pronósticos, {len(alertas)} alertas F1")
    print("Contrato de interfaz del módulo IA:\n" + json.dumps(CONTRATO, indent=2, ensure_ascii=False))


def simular_modulo_ia(n: int = 25, semilla: int | None = None) -> pd.DataFrame:
    """Imita al módulo de visión del curso hermano escribiendo detecciones con el contrato."""
    rng = np.random.default_rng(semilla)
    ref = pd.read_csv(ROOT / "data" / "dataset_frente2_hito1_final.csv", usecols=["latitud", "longitud"])
    clases = sorted(FAUNA) + sorted(AMENAZAS)
    pesos = np.array([3.0] * len(FAUNA) + [1.0] * len(AMENAZAS)); pesos /= pesos.sum()
    base = ref.sample(n, random_state=int(rng.integers(1e9)))
    ahora = datetime.now(timezone.utc).replace(tzinfo=None)
    filas = pd.DataFrame({
        "evento_id": [str(uuid.uuid4()) for _ in range(n)],
        "timestamp_utc": [(ahora - timedelta(minutes=int(m))).isoformat(timespec="seconds")
                          for m in rng.integers(0, 720, n)],
        "sensor_id": [f"CAM-{k:03d}" for k in rng.integers(1, 60, n)],
        "tipo_sensor": rng.choice(["camara", "acustico", "dron"], n, p=[0.6, 0.3, 0.1]),
        "clase_detectada": rng.choice(clases, n, p=pesos),
        "confidence_score": rng.uniform(0.6, 0.99, n).round(3),
        "latitud": base["latitud"].to_numpy() + rng.normal(0, 0.002, n),
        "longitud": base["longitud"].to_numpy() + rng.normal(0, 0.002, n),
        "estado": "nueva", "recibido_utc": ahora.isoformat(timespec="seconds"),
    })
    con = conectar()
    filas.to_sql("detecciones", con, if_exists="append", index=False)
    con.close()
    return filas


def procesar_nuevas() -> pd.DataFrame:
    """Lee detecciones 'nuevas', las enriquece, las clasifica con el modelo del Frente 2,
    escribe predicciones y genera alertas Alto/Crítico."""
    con = conectar()
    nuevas = pd.read_sql("SELECT * FROM detecciones WHERE estado='nueva'", con)
    if nuevas.empty:
        con.close()
        return nuevas
    ref = _referencia_espacial()
    x, y = A_UTM.transform(nuevas["longitud"].to_numpy(), nuevas["latitud"].to_numpy())
    from scipy.spatial import cKDTree

    _, idx = cKDTree(ref[["x", "y"]].to_numpy()).query(np.c_[x, y])
    vecino = ref.iloc[idx].reset_index(drop=True)
    df = nuevas.reset_index(drop=True).assign(
        clase=nuevas["clase_detectada"].str.lower(),
        en_zona_intangible=vecino["en_zona_intangible"].astype(int),
        en_zona_amortiguamiento=vecino["en_zona_amortiguamiento"].astype(int),
        distancia_limite_m=vecino["distancia_limite_m"], distancia_a_carretera_m=vecino["distancia_a_carretera_m"],
        distancia_a_rio_m=vecino["distancia_a_rio_m"])
    df = construir_features_frente2(df)
    paquete = joblib.load(MODELO_F2)
    proba = paquete["modelo"].predict_proba(df[paquete["columnas"]])
    df["nivel_predicho"] = proba.argmax(1)
    df["p_critico"] = proba[:, 3]
    df["regla"] = asignar_riesgo(df)

    grilla = pd.read_sql("SELECT cuadrante_id, x_utm, y_utm FROM cuadrantes", con)
    llave = {(int(a // 5000), int(b // 5000)): c for c, a, b in grilla.itertuples(index=False)}
    df["cuadrante_id"] = [llave.get((int(a // 5000), int(b // 5000)), "fuera_grilla") for a, b in zip(x, y)]
    ahora = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    con.executemany("INSERT OR REPLACE INTO predicciones_riesgo VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (r.evento_id, r.cuadrante_id, int(r.nivel_predicho), NIVELES[int(r.nivel_predicho)], float(r.p_critico),
         int(r.en_zona_intangible), int(r.en_zona_amortiguamiento), float(r.distancia_limite_m),
         "Random Forest v3", ahora) for r in df.itertuples()])
    graves = df[df["nivel_predicho"] >= 2]
    con.executemany("INSERT OR REPLACE INTO alertas VALUES (?,?,?,?,?,?,?)", [
        (f"F2-{r.evento_id[:8]}", ahora, r.cuadrante_id, "Frente 2", NIVELES[int(r.nivel_predicho)],
         f"{r.clase} ({r.confidence_score:.0%}) {r.tipo_sensor} {r.sensor_id}", "abierta") for r in graves.itertuples()])
    con.executemany("UPDATE detecciones SET estado='procesada' WHERE evento_id=?", [(e,) for e in df["evento_id"]])
    con.commit(); con.close()
    print(f"Procesadas {len(df)} detecciones · {len(graves)} alertas Alto/Crítico · "
          f"coincidencia con regla de negocio {np.mean(df['regla'] == df['nivel_predicho']):.1%}")
    return df


if __name__ == "__main__":
    accion = sys.argv[1] if len(sys.argv) > 1 else "init"
    if accion == "init":
        init()
    elif accion == "simular":
        print(simular_modulo_ia(int(sys.argv[2]) if len(sys.argv) > 2 else 25)[
                  ["evento_id", "clase_detectada", "confidence_score", "latitud", "longitud"]].head())
    elif accion == "procesar":
        procesar_nuevas()
