"""Base de datos común de BIODIVERSITY-GUARD (Hito 3) e ingesta de datos.

El módulo de IA del curso hermano (visión / audio / OCR) es referencial: aquí se simula.
Lo que importa es el contrato: cualquier fuente que entregue una detección con

    clase_detectada, latitud, longitud  [+ fecha_hora, confidence_score, sensor_id, tipo_sensor]

entra a la tabla `detecciones`; el motor ML la enriquece (zona, cuadrante, reincidencia),
la clasifica con el modelo del Frente 2 y, si el nivel es Alto o Crítico, abre una alerta.

Vías de ingesta (todas terminan en `registrar_detecciones` o `registrar_semana_frente1`):
    1. Archivo CSV/Excel con muchas detecciones.
    2. Registro manual de un avistamiento.
    3. Foto de cámara trampa / celular (la etiqueta la pone el módulo IA simulado o el usuario).
    4. Imagen satelital o de dron (estimación de suelo sin cobertura -> reporte satelital).
    5. Archivo semanal de alertas de deforestación (actualiza el pronóstico del Frente 1).
    6. Simulador del módulo IA.

SQLite para la demo; el DDL es estándar y migra a PostgreSQL cambiando `conectar`.

Línea de comandos:
    python base_datos_comun.py init      # (re)crea la BD con el histórico y el pronóstico
    python base_datos_comun.py simular 10
"""

from __future__ import annotations

import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inferencia as inf  # noqa: E402
from features import FAUNA, NIVELES, cargar_frente1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB_PATH = DATA / "biodiversity_guard.db"
VENTANA_DIAS = 30  # las alertas de incursión más antiguas se consideran ya gestionadas

CONTRATO = {
    "clase_detectada": "obligatorio · qué se detectó (jaguar, persona, motosierra, arma_fuego, ...)",
    "latitud": "obligatorio · grados decimales (EPSG:4326)",
    "longitud": "obligatorio · grados decimales (EPSG:4326)",
    "fecha_hora": "opcional · AAAA-MM-DD HH:MM (si falta se usa el momento del registro)",
    "confidence_score": "opcional · certeza del detector de 0 a 1 (por defecto 0.8)",
    "sensor_id": "opcional · identificador de la cámara, sensor acústico o dron",
    "tipo_sensor": "opcional · camara | acustico | dron | patrulla",
}

DDL = """
CREATE TABLE IF NOT EXISTS cuadrantes (
    cuadrante_id TEXT PRIMARY KEY, zona TEXT, x_utm REAL, y_utm REAL, lat_c REAL, lon_c REAL, geometria TEXT);
CREATE TABLE IF NOT EXISTS detecciones (
    evento_id TEXT PRIMARY KEY, fecha_hora TEXT, origen TEXT, sensor_id TEXT, tipo_sensor TEXT,
    clase_detectada TEXT, confidence_score REAL, latitud REAL, longitud REAL, estado TEXT DEFAULT 'nueva',
    registrado_utc TEXT);
CREATE TABLE IF NOT EXISTS predicciones_riesgo (
    evento_id TEXT PRIMARY KEY REFERENCES detecciones(evento_id), cuadrante_id TEXT, clase TEXT,
    nivel INTEGER, nivel_nombre TEXT, p_alto REAL, p_critico REAL, en_zona_intangible INTEGER,
    en_zona_amortiguamiento INTEGER, es_noche INTEGER, reincidencia INTEGER, explicacion TEXT, accion TEXT,
    modelo TEXT);
CREATE TABLE IF NOT EXISTS predicciones_deforestacion (
    cuadrante_id TEXT PRIMARY KEY, semana TEXT, prob_evento_7d REAL, ha_esperadas_7d REAL, alerta_7d INTEGER,
    prob_evento_30d REAL, ha_esperadas_30d REAL, ha_alto_30d REAL, alerta_30d INTEGER, co2_en_riesgo_30d_t REAL,
    nivel_deforestacion INTEGER, perdida_suma_4s REAL, perdida_suma_52s REAL);
CREATE TABLE IF NOT EXISTS semanas_extra (
    semana TEXT, cuadrante_id TEXT, n_alertas REAL, temperatura_media REAL, humedad_media REAL,
    precipitacion_suma REAL, cargado_utc TEXT, PRIMARY KEY (semana, cuadrante_id));
CREATE TABLE IF NOT EXISTS reportes_satelitales (
    reporte_id TEXT PRIMARY KEY, fecha TEXT, cuadrante_id TEXT, area_imagen_ha REAL, fraccion_sin_cobertura REAL,
    ha_sin_cobertura REAL, nota TEXT);
CREATE TABLE IF NOT EXISTS alertas (
    alerta_id TEXT PRIMARY KEY, fecha TEXT, cuadrante_id TEXT, origen TEXT, nivel TEXT, detalle TEXT,
    accion TEXT, estado TEXT DEFAULT 'abierta', atendida_fecha TEXT);
"""


def ahora() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def conectar() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(DDL)
    return con


def leer(sql: str, params: tuple = ()) -> pd.DataFrame:
    with conectar() as con:
        return pd.read_sql(sql, con, params=params)


def asegurar() -> None:
    """Crea la BD la primera vez (unos segundos). No reentrena modelos."""
    if not DB_PATH.exists() or leer("SELECT COUNT(*) n FROM cuadrantes")["n"].iloc[0] == 0:
        init()


# --------------------------------------------------------------------------- #
# Frente 1
# --------------------------------------------------------------------------- #
def panel_frente1() -> pd.DataFrame:
    """Histórico del proyecto más las semanas cargadas desde el dashboard."""
    panel = cargar_frente1()
    extra = leer("SELECT * FROM semanas_extra ORDER BY semana")
    for _, bloque in extra.groupby("semana", sort=True):
        panel = inf.agregar_semana(panel, bloque)
    return panel


def _guardar_pronostico(con: sqlite3.Connection, pred: pd.DataFrame) -> int:
    cols = ["cuadrante_id", "semana", "prob_evento_7d", "ha_esperadas_7d", "alerta_7d", "prob_evento_30d",
            "ha_esperadas_30d", "ha_alto_30d", "alerta_30d", "co2_en_riesgo_30d_t", "nivel_deforestacion", "perdida_suma_4s",
            "perdida_suma_52s"]
    con.execute("DELETE FROM predicciones_deforestacion")
    pred[cols].to_sql("predicciones_deforestacion", con, if_exists="append", index=False)
    con.execute("DELETE FROM alertas WHERE origen='Deforestación' AND estado='abierta'")
    graves = pred[pred["nivel_deforestacion"] >= 2]
    fin = str(pred["semana"].iloc[0])[-10:]
    con.executemany("INSERT OR REPLACE INTO alertas VALUES (?,?,?,?,?,?,?,?,?)", [
        (f"DEF-{r.cuadrante_id}-{fin}", f"{fin} 00:00:00", r.cuadrante_id, "Deforestación", NIVELES[int(r.nivel_deforestacion)],
         f"Probabilidad de perder bosque en 30 días: {r.prob_evento_30d:.0%} · unas {r.ha_esperadas_30d:.1f} ha",
         inf.ACCION[int(r.nivel_deforestacion)], "abierta", None) for r in graves.itertuples()])
    return len(graves)


def registrar_semana_frente1(nueva: pd.DataFrame) -> dict:
    """Guarda una semana nueva de alertas satelitales y recalcula el pronóstico (sin reentrenar)."""
    panel = panel_frente1()
    ini = panel["semana_inicio"].max() + pd.Timedelta(days=7)
    semana = f"{ini.date()}/{(ini + pd.Timedelta(days=6)).date()}"
    validos = set(panel["cuadrante_id"].unique())
    nueva = nueva[nueva["cuadrante_id"].isin(validos)].drop_duplicates("cuadrante_id").copy()
    for c in ["temperatura_media", "humedad_media", "precipitacion_suma"]:
        if c not in nueva:
            nueva[c] = np.nan
    nueva["n_alertas"] = pd.to_numeric(nueva["n_alertas"], errors="coerce").fillna(0).clip(lower=0)
    with conectar() as con:
        con.executemany("INSERT OR REPLACE INTO semanas_extra VALUES (?,?,?,?,?,?,?)", [
            (semana, r.cuadrante_id, float(r.n_alertas), _f(r.temperatura_media), _f(r.humedad_media),
             _f(r.precipitacion_suma), ahora()) for r in nueva.itertuples()])
    pred = inf.pronosticar_frente1(panel_frente1())
    with conectar() as con:
        n = _guardar_pronostico(con, pred)
    return {"semana": semana, "cuadrantes": len(nueva), "ha_reportadas": float(nueva["n_alertas"].sum() * 0.09),
            "alertas": n}


def _f(v):
    return None if pd.isna(v) else float(v)


# --------------------------------------------------------------------------- #
# Frente 2
# --------------------------------------------------------------------------- #
def historial_amenazas(dias: int = 10) -> pd.DataFrame:
    return leer(f"""SELECT d.fecha_hora, d.latitud, d.longitud, p.clase FROM detecciones d
                    JOIN predicciones_riesgo p USING(evento_id)
                    WHERE p.nivel >= 1 AND d.fecha_hora >= (SELECT datetime(MAX(fecha_hora), '-{dias} days') FROM detecciones)""")


def fecha_corte() -> pd.Timestamp:
    """«Hoy» del sistema: el dato más reciente (última detección o fin de la última semana pronosticada)."""
    d = leer("SELECT MAX(fecha_hora) f FROM detecciones")["f"].iloc[0]
    s = leer("SELECT MAX(semana) s FROM predicciones_deforestacion")["s"].iloc[0]
    fechas = [pd.Timestamp(x) for x in [d, s[-10:] + " 23:59" if s else None] if x]
    return max(fechas) if fechas else pd.Timestamp.now()


def registrar_detecciones(tabla: pd.DataFrame, origen: str) -> pd.DataFrame:
    """Punto único de entrada de detecciones. Devuelve la tabla clasificada."""
    df = tabla.copy()
    if "clase_detectada" not in df and "clase" in df:
        df["clase_detectada"] = df["clase"]
    if "fecha_hora" not in df:
        df["fecha_hora"] = pd.NaT
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce").fillna(fecha_corte())
    res = inf.clasificar_detecciones(df.assign(clase=df["clase_detectada"]), historial_amenazas())
    if res.empty:
        return res
    res["evento_id"] = [f"{origen[:3].upper()}-{uuid.uuid4().hex[:10]}" for _ in range(len(res))]
    for c, v in [("sensor_id", "sin sensor"), ("tipo_sensor", "patrulla")]:
        if c not in res:
            res[c] = v
        res[c] = res[c].fillna(v)
    reg = ahora()
    with conectar() as con:
        con.executemany("INSERT INTO detecciones VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
            (r.evento_id, str(r.fecha_hora)[:19], origen, str(r.sensor_id), str(r.tipo_sensor), r.clase,
             float(r.confidence_score), float(r.latitud), float(r.longitud), "procesada", reg) for r in res.itertuples()])
        con.executemany("INSERT INTO predicciones_riesgo VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            (r.evento_id, r.cuadrante_id, r.clase, int(r.nivel_predicho), r.nivel_nombre, float(r.p_alto),
             float(r.p_critico), int(r.en_zona_intangible), int(r.en_zona_amortiguamiento), int(r.es_noche),
             int(r.amenazas_7d_estacion), r.explicacion, r.accion, "LightGBM v3") for r in res.itertuples()])
        graves = res[res["nivel_predicho"] >= 2]
        con.executemany("INSERT INTO alertas VALUES (?,?,?,?,?,?,?,?,?)", [
            (f"INC-{r.evento_id}", str(r.fecha_hora)[:19], r.cuadrante_id, "Incursión", r.nivel_nombre, r.explicacion,
             r.accion, "abierta", None) for r in graves.itertuples()])
    return res


def simular_modulo_ia(n: int = 10, semilla: int | None = None) -> pd.DataFrame:
    """Imita al módulo de IA: detecciones nuevas en estaciones de la red, en su mayoría fauna."""
    rng = np.random.default_rng(semilla)
    est = pd.read_csv(DATA / "estaciones_sensores.csv")
    es_amenaza = rng.random(n) < 0.12
    p_am = est["presion"].to_numpy() ** 1.5
    k = np.where(es_amenaza, rng.choice(len(est), n, p=p_am / p_am.sum()), rng.integers(0, len(est), n))
    e = est.iloc[k].reset_index(drop=True)
    am = {"persona": 30, "vehiculo_ligero": 21, "vehiculo_maderero": 14, "trampa_caza": 12, "arma_blanca": 9,
          "motosierra": 7, "arma_fuego": 5, "cazador": 2}  # misma mezcla que la telemetría histórica
    clases = np.where(es_amenaza, rng.choice(list(am), n, p=np.array(list(am.values())) / sum(am.values())),
                      rng.choice(sorted(FAUNA), n))
    base = fecha_corte()
    return registrar_detecciones(pd.DataFrame({
        "clase_detectada": clases, "confidence_score": rng.uniform(0.6, 0.99, n).round(3),
        "latitud": e["latitud"] + rng.normal(0, 0.0008, n), "longitud": e["longitud"] + rng.normal(0, 0.0008, n),
        "fecha_hora": [base + pd.Timedelta(minutes=int(m)) for m in np.sort(rng.integers(10, 2880, n))],
        "sensor_id": e["estacion_id"], "tipo_sensor": e["tipo_sensor"]}), "Módulo IA (simulado)")


def registrar_reporte_satelital(cuadrante_id: str, area_ha: float, fraccion: float, nota: str = "") -> dict:
    ha = float(area_ha * fraccion)
    fecha = str(fecha_corte())[:19]
    rid = f"SAT-{uuid.uuid4().hex[:8]}"
    nivel = 3 if ha >= 10 else 2 if ha >= 2 else 1
    with conectar() as con:
        con.execute("INSERT INTO reportes_satelitales VALUES (?,?,?,?,?,?,?)", (rid, fecha, cuadrante_id, area_ha, fraccion, ha, nota))
        if nivel >= 2:
            con.execute("INSERT INTO alertas VALUES (?,?,?,?,?,?,?,?,?)", (
                rid, fecha, cuadrante_id, "Imagen satelital", NIVELES[nivel],
                f"La imagen muestra unas {ha:.1f} ha sin cobertura vegetal ({fraccion:.0%} del área analizada). Requiere verificación.",
                inf.ACCION[nivel], "abierta", None))
    return {"reporte_id": rid, "ha": ha, "nivel": NIVELES[nivel]}


def marcar_alerta(alerta_id: str, estado: str) -> None:
    with conectar() as con:
        # la hora de atención se registra en el calendario de los datos (fecha de corte), no en el reloj del equipo
        cuando = str(fecha_corte())[:19] if estado == "atendida" else None
        con.execute("UPDATE alertas SET estado=?, atendida_fecha=? WHERE alerta_id=?", (estado, cuando, alerta_id))


# --------------------------------------------------------------------------- #
# Inicialización
# --------------------------------------------------------------------------- #
def init() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = conectar()
    g = inf.grilla()
    g[["cuadrante_id", "zona", "x_utm", "y_utm", "lat_c", "lon_c", "geometria"]].to_sql("cuadrantes", con, if_exists="append", index=False)

    # Pronóstico del Frente 1 con el modelo guardado
    n_def = _guardar_pronostico(con, inf.pronosticar_frente1())

    # Histórico de la red de sensores (ya clasificado al entrenar)
    h = pd.read_csv(DATA / "predicciones_frente2.csv", parse_dates=["fecha_hora"])
    h["nivel_nombre"] = h["nivel_predicho"].map(dict(enumerate(NIVELES)))
    h["explicacion"] = h.apply(inf._explicar, axis=1)
    reg = ahora()
    con.executemany("INSERT INTO detecciones VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
        (r.evento_id, str(r.fecha_hora)[:19], "Red de sensores", r.estacion_id, r.tipo_sensor, r.clase,
         float(r.confidence_score), float(r.latitud), float(r.longitud), "procesada", reg) for r in h.itertuples()])
    con.executemany("INSERT INTO predicciones_riesgo VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (r.evento_id, r.cuadrante_id if isinstance(r.cuadrante_id, str) else None, r.clase, int(r.nivel_predicho),
         r.nivel_nombre, float(r.p_alto), float(getattr(r, "p_crítico")), int(r.en_zona_intangible),
         int(r.en_zona_amortiguamiento), int(r.es_noche), int(r.amenazas_7d_estacion), r.explicacion,
         inf.ACCION[int(r.nivel_predicho)], "LightGBM v3") for r in h.itertuples()])
    corte = h["fecha_hora"].max()
    graves = h[h["nivel_predicho"] >= 2]
    rng = np.random.default_rng(7)
    filas = []
    for r in graves.itertuples():
        antigua = (corte - r.fecha_hora).days > VENTANA_DIAS
        # las alertas antiguas figuran como atendidas, con un tiempo de respuesta de 6 a 72 horas
        atendida = str(r.fecha_hora + pd.Timedelta(hours=float(rng.uniform(6, 72))))[:19] if antigua else None
        filas.append((f"INC-{r.evento_id}", str(r.fecha_hora)[:19], r.cuadrante_id if isinstance(r.cuadrante_id, str) else None,
                      "Incursión", r.nivel_nombre, r.explicacion, inf.ACCION[int(r.nivel_predicho)],
                      "atendida" if antigua else "abierta", atendida))
    con.executemany("INSERT INTO alertas VALUES (?,?,?,?,?,?,?,?,?)", filas)
    con.commit(); con.close()
    abiertas = sum(1 for f in filas if f[7] == "abierta")
    print(f"BD creada en {DB_PATH.name}: {len(g)} cuadrantes · {len(h)} detecciones históricas · "
          f"{n_def} alertas de deforestación · {abiertas} alertas de incursión abiertas (últimos {VENTANA_DIAS} días)")


if __name__ == "__main__":
    accion = sys.argv[1] if len(sys.argv) > 1 else "init"
    if accion == "init":
        init()
    elif accion == "simular":
        asegurar()
        r = simular_modulo_ia(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
        print(r[["fecha_hora", "sensor_id", "clase", "nivel_nombre", "explicacion"]].to_string(index=False))
