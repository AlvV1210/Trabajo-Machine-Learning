"""Inferencia con los modelos ya entrenados (no reentrena nada).

Los modelos se guardan en `models/` en el formato de texto de LightGBM, que no depende
de la versión de Python ni de scikit-learn. Este módulo los carga y ofrece:

    pronosticar_frente1(panel)      -> pronóstico por cuadrante para la última semana del panel
    clasificar_detecciones(tabla)   -> nivel de riesgo de cada detección, con explicación en palabras
    analizar_imagen_satelital(img)  -> fracción de suelo sin vegetación en una imagen RGB

Lo usan la base de datos común (base_datos_comun.py) y el dashboard.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (  # noqa: E402
    AMENAZAS, CATEGORICAS, CO2_POR_C, FAUNA, MAPEO_CLASES, MOVILES, NIVELES, REGLA, T_C_POR_HA, cargar_frente1,
    construir_features_frente1, construir_features_frente2, matriz_frente2,
)

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "data", ROOT / "models"
A_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)

NOMBRE_CLASE = {
    "mono": "Mono", "capibara": "Capibara (ronsoco)", "snake": "Serpiente", "rana": "Rana", "guacamayo": "Guacamayo",
    "lobo_de_rio": "Lobo de río", "jaguar": "Jaguar", "persona": "Persona no identificada", "cazador": "Cazador",
    "vehiculo_ligero": "Moto, auto o bote", "vehiculo_maderero": "Camión o vehículo maderero",
    "trampa_caza": "Trampa de caza", "arma_blanca": "Machete o arma blanca", "motosierra": "Motosierra",
    "arma_fuego": "Arma de fuego / disparo",
}
SINONIMOS = {
    **MAPEO_CLASES, "serpiente": "snake", "culebra": "snake", "ronsoco": "capibara", "nutria": "lobo_de_rio",
    "lobo de rio": "lobo_de_rio", "lobo de río": "lobo_de_rio", "moto": "vehiculo_ligero", "bote": "vehiculo_ligero",
    "auto": "vehiculo_ligero", "camion": "vehiculo_maderero", "camión": "vehiculo_maderero",
    "camion maderero": "vehiculo_maderero", "trampa": "trampa_caza", "machete": "arma_blanca", "cuchillo": "arma_blanca",
    "disparo": "arma_fuego", "escopeta": "arma_fuego", "arma": "arma_fuego", "arma de fuego": "arma_fuego",
    "motosierra": "motosierra", "tala": "motosierra", "humano": "persona", "gente": "persona",
    **{v.lower(): k for k, v in NOMBRE_CLASE.items()},
}
ACCION = {
    0: "Sin acción. Queda registrado para el monitoreo de fauna.",
    1: "Mantener en observación y revisar en el próximo patrullaje de rutina.",
    2: "Programar un patrullaje al sector dentro de las próximas 48 horas.",
    3: "Avisar de inmediato a la jefatura y coordinar la salida con apoyo (PNP / Fiscalía ambiental).",
}


# --------------------------------------------------------------------------- #
# Frente 1
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def modelo_frente1() -> dict:
    meta = json.loads((MODELS / "frente1_meta.json").read_text(encoding="utf-8"))
    for suf in meta["horizontes"]:
        meta["horizontes"][suf]["clf"] = lgb.Booster(model_file=str(MODELS / f"frente1_{suf}_clasificador.txt"))
        meta["horizontes"][suf]["reg"] = lgb.Booster(model_file=str(MODELS / f"frente1_{suf}_magnitud.txt"))
    return meta


def nivel_deforestacion(prob: pd.Series, ha: pd.Series, umbral: float) -> pd.Series:
    """Traduce el pronóstico a 30 días en un nivel de atención fácil de leer.

    Los cortes se eligieron para que la lista sea atendible por la jefatura: en la prueba 2025-2026
    dan en promedio 1 a 2 cuadrantes críticos y unos 7 altos por semana, y se confirmaron con pérdida
    real el 100 % de los críticos y el 93 % de los altos."""
    n = np.zeros(len(prob), dtype=int)
    n[(prob >= umbral).to_numpy()] = 1
    n[((prob >= 0.80) & (ha >= 3.0)).to_numpy()] = 2   # pérdida muy probable y de al menos 3 ha
    n[((prob >= 0.95) & (ha >= 6.0)).to_numpy()] = 3   # pérdida casi segura y de al menos 6 ha
    return pd.Series(n, index=prob.index)


def pronosticar_frente1(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pronóstico a 7 y 30 días para la última semana del panel (por defecto, el histórico del proyecto)."""
    meta = modelo_frente1()
    base = cargar_frente1() if panel is None else panel
    df, F = construir_features_frente1(base)
    for c in CATEGORICAS:
        df[c] = pd.Categorical(df[c].astype(str), categories=meta["categorias"][c])
    ult = df[df["semana_inicio"] == df["semana_inicio"].max()]
    X = ult[meta["features"]]
    out = ult[["cuadrante_id", "semana", "semana_inicio", "cobertura_anp", "causa_principal", "vocacion_zee", "mb_mineria",
               "mb_agropecuario", "tasa_deforestacion_ha_anio", "perdida_suma_4s", "perdida_suma_52s",
               "densidad_poblacional"]].copy()
    for suf, h in meta["horizontes"].items():
        cruda = np.clip(h["clf"].predict(X), 1e-4, 1 - 1e-4)
        p = 1 / (1 + np.exp(-(h["cal"]["a"] * np.log(cruda / (1 - cruda)) + h["cal"]["b"])))   # probabilidad calibrada
        mag = np.clip(np.exp(h["reg"].predict(X)) * h["smear"] - 1, 0, None)
        out[f"prob_evento_{suf}"] = p
        out[f"ha_esperadas_{suf}"] = cruda * mag             # estimación típica
        out[f"ha_alto_{suf}"] = p * mag * h["cal"]["k"]      # escenario alto (total sin sesgo a la baja)
        out[f"alerta_{suf}"] = (p >= h["umbral"]).astype(int)
    out["co2_en_riesgo_30d_t"] = out["ha_esperadas_30d"] * T_C_POR_HA * CO2_POR_C
    out["nivel_deforestacion"] = nivel_deforestacion(out["prob_evento_30d"], out["ha_esperadas_30d"],
                                                     meta["horizontes"]["30d"]["umbral"]).to_numpy()
    return out.reset_index(drop=True)


def agregar_semana(panel: pd.DataFrame, nueva: pd.DataFrame) -> pd.DataFrame:
    """Añade al panel una semana nueva de alertas satelitales.

    `nueva` necesita `cuadrante_id` y `n_alertas`; el clima es opcional (si falta se repite el de la
    última semana). Las variables fijas del cuadrante (MapBiomas, ZEE, etc.) se copian del histórico."""
    ult = panel[panel["semana_inicio"] == panel["semana_inicio"].max()].copy()
    ini = ult["semana_inicio"].iloc[0] + pd.Timedelta(days=7)
    fila = ult.drop(columns=["n_alertas", "perdida_ha"]).merge(
        nueva[["cuadrante_id", "n_alertas"] + [c for c in ["temperatura_media", "humedad_media", "precipitacion_suma"] if c in nueva]],
        on="cuadrante_id", how="left", suffixes=("", "_nuevo"))
    for c in ["temperatura_media", "humedad_media", "precipitacion_suma"]:
        if f"{c}_nuevo" in fila:
            fila[c] = fila[f"{c}_nuevo"].fillna(fila[c])
            fila = fila.drop(columns=f"{c}_nuevo")
    fila["n_alertas"] = pd.to_numeric(fila["n_alertas"], errors="coerce").fillna(0).clip(lower=0)
    fila["perdida_ha"] = fila["n_alertas"] * 0.09
    fila["alerta_observada"] = (fila["n_alertas"] > 0).astype(int)
    fila["semana_inicio"] = ini
    fila["semana"] = f"{ini.date()}/{(ini + pd.Timedelta(days=6)).date()}"
    return pd.concat([panel, fila[panel.columns]], ignore_index=True)


# --------------------------------------------------------------------------- #
# Frente 2
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def modelo_frente2() -> dict:
    meta = json.loads((MODELS / "frente2_meta.json").read_text(encoding="utf-8"))
    meta["booster"] = lgb.Booster(model_file=str(MODELS / "frente2_clasificador.txt"))
    return meta


@lru_cache(maxsize=1)
def _referencia_espacial():
    """Puntos con zona y distancias ya calculadas en el Hito 1 (SERNANP + OpenStreetMap)."""
    from scipy.spatial import cKDTree

    ref = pd.read_csv(DATA / "dataset_frente2_hito1_final.csv",
                      usecols=["latitud", "longitud", "en_zona_intangible", "en_zona_amortiguamiento",
                               "distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m"])
    x, y = A_UTM.transform(ref["longitud"].to_numpy(), ref["latitud"].to_numpy())
    for b in ["en_zona_intangible", "en_zona_amortiguamiento"]:
        ref[b] = ref[b].astype(str).str.lower().isin(["true", "1"]).astype(int)
    return ref, cKDTree(np.c_[x, y])


@lru_cache(maxsize=1)
def grilla() -> pd.DataFrame:
    ruta = DATA / "cuadrantes_grid.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    from grilla_cuadrantes import obtener_grilla

    return obtener_grilla()


def cuadrante_de(lat, lon) -> list:
    g = grilla()
    llave = {(int(a // 5000), int(b // 5000)): c for c, a, b in g[["cuadrante_id", "x_utm", "y_utm"]].itertuples(index=False)}
    x, y = A_UTM.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return [llave.get((int(a // 5000), int(b // 5000))) for a, b in zip(np.atleast_1d(x), np.atleast_1d(y))]


def _plano(texto: str) -> str:
    """minúsculas, sin tildes y con espacios en lugar de guiones bajos."""
    import unicodedata

    t = unicodedata.normalize("NFKD", str(texto).strip().lower().replace("_", " "))
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


_SIN = {_plano(k): v for k, v in SINONIMOS.items()}
_SIN.update({_plano(c): c for c in FAUNA | AMENAZAS})
_SIN.update({"camion maderero": "vehiculo_maderero", "vehiculo": "vehiculo_ligero", "camioneta": "vehiculo_ligero",
             "tractor": "vehiculo_maderero", "draga": "vehiculo_maderero", "rifle": "arma_fuego", "tiro": "arma_fuego"})


def normalizar_clase(valores: pd.Series) -> pd.Series:
    """Lleva cualquier etiqueta conocida (español/inglés, con o sin tildes) a la taxonomía del modelo."""
    return valores.map(lambda v: _SIN.get(_plano(v)))


def _explicar(f: pd.Series) -> str:
    if f["clase"] in FAUNA:
        return f"{NOMBRE_CLASE[f['clase']]}: es fauna silvestre, no representa una amenaza."
    zona = ("dentro de la reserva" if f["en_zona_intangible"] == 1 else
            "en la zona de amortiguamiento" if f["en_zona_amortiguamiento"] == 1 else "fuera de las zonas protegidas")
    partes = [f"{NOMBRE_CLASE[f['clase']]} {zona}"]
    base = REGLA[f["clase"]][2 if f["en_zona_intangible"] == 1 else 1 if f["en_zona_amortiguamiento"] == 1 else 0]
    if base >= 1 and f["es_noche"] == 1 and f["clase"] in MOVILES:
        partes.append("detectado de noche")
    if base >= 1 and f["amenazas_7d_estacion"] >= 2:
        partes.append(f"con {int(f['amenazas_7d_estacion'])} amenazas en el mismo lugar en los últimos 7 días")
    return ", ".join(partes) + "."


def clasificar_detecciones(tabla: pd.DataFrame, historial: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clasifica detecciones nuevas.

    `tabla`: columnas `clase` (o `clase_detectada`), `latitud`, `longitud`; opcionales `fecha_hora`,
    `confidence_score`. `historial`: eventos previos (`fecha_hora`, `latitud`, `longitud`, `clase`) para
    calcular la reincidencia en un radio de 1 km.
    Devuelve la tabla enriquecida con zona, cuadrante, nivel, probabilidades, explicación y acción sugerida.
    """
    df = tabla.copy().reset_index(drop=True)
    df["clase"] = normalizar_clase(df["clase"] if "clase" in df else df["clase_detectada"])
    df = df.dropna(subset=["clase", "latitud", "longitud"]).reset_index(drop=True)
    if df.empty:
        return df
    if "confidence_score" not in df:
        df["confidence_score"] = 0.8
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce").fillna(0.8).clip(0, 1)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce") if "fecha_hora" in df else pd.NaT
    df["hora"] = df["fecha_hora"].dt.hour.fillna(12).astype(int)
    df["es_noche"] = ((df["hora"] >= 19) | (df["hora"] < 5)).astype(int)

    # Zona y distancias: vecino más cercano de la capa de referencia
    ref, arbol = _referencia_espacial()
    x, y = A_UTM.transform(df["longitud"].to_numpy(dtype=float), df["latitud"].to_numpy(dtype=float))
    dist, idx = arbol.query(np.c_[x, y])
    vec = ref.iloc[idx].reset_index(drop=True)
    for c in ["en_zona_intangible", "en_zona_amortiguamiento", "distancia_limite_m", "distancia_a_carretera_m", "distancia_a_rio_m"]:
        df[c] = vec[c].to_numpy()
    df["cuadrante_id"] = cuadrante_de(df["latitud"], df["longitud"])
    lejos = dist > 3000  # fuera del área con referencia: la zona se toma del cuadrante
    if lejos.any():
        g = grilla().set_index("cuadrante_id")["zona"]
        zona = df.loc[lejos, "cuadrante_id"].map(g)
        df.loc[lejos, "en_zona_intangible"] = (zona == "Reserva Nacional Tambopata").astype(int).to_numpy()
        df.loc[lejos, "en_zona_amortiguamiento"] = (zona == "Zona de amortiguamiento").astype(int).to_numpy()
    df["ubicacion_aproximada"] = lejos.astype(int)

    # Reincidencia: amenazas a menos de 1 km en los 7 días previos (historial + el propio lote)
    previos = df[["fecha_hora", "clase"]].assign(x=x, y=y)
    if historial is not None and len(historial):
        hx, hy = A_UTM.transform(historial["longitud"].to_numpy(dtype=float), historial["latitud"].to_numpy(dtype=float))
        previos = pd.concat([previos, pd.DataFrame({"fecha_hora": pd.to_datetime(historial["fecha_hora"], errors="coerce"),
                                                    "clase": historial["clase"].to_numpy(), "x": hx, "y": hy})])
    previos = previos[previos["clase"].isin(AMENAZAS) & previos["fecha_hora"].notna()]
    rein = np.zeros(len(df), dtype=int)
    if len(previos):
        pt, px_, py_ = previos["fecha_hora"].to_numpy(), previos["x"].to_numpy(), previos["y"].to_numpy()
        for i, t in enumerate(df["fecha_hora"].to_numpy()):
            if pd.isna(t):
                continue
            m = (pt < t) & (pt >= t - np.timedelta64(7, "D")) & (np.hypot(px_ - x[i], py_ - y[i]) <= 1000)
            rein[i] = int(m.sum())
    df["amenazas_7d_estacion"] = rein

    df = construir_features_frente2(df)
    meta = modelo_frente2()
    proba = np.asarray(meta["booster"].predict(matriz_frente2(df)[meta["columnas"]]))
    df["nivel_predicho"] = proba.argmax(1)
    df["nivel_nombre"] = [NIVELES[k] for k in df["nivel_predicho"]]
    df["p_alto"], df["p_critico"] = proba[:, 2], proba[:, 3]
    df["certeza_modelo"] = proba.max(1)
    df["explicacion"] = df.apply(_explicar, axis=1)
    df["accion"] = df["nivel_predicho"].map(ACCION)
    return df


# --------------------------------------------------------------------------- #
# Imagen satelital / dron
# --------------------------------------------------------------------------- #
def analizar_imagen_satelital(imagen) -> dict:
    """Estimación simple de suelo sin vegetación en una imagen RGB (satélite o dron).

    Usa el índice VARI = (G - R) / (G + R - B), que resalta vegetación verde con solo las bandas
    visibles. Píxeles con VARI bajo y no oscuros (agua/sombra) se marcan como «sin cobertura».
    Es una ayuda visual para priorizar una verificación, no reemplaza a las alertas oficiales de Geobosques.
    """
    from PIL import Image

    im = Image.open(imagen).convert("RGB") if not hasattr(imagen, "convert") else imagen.convert("RGB")
    im.thumbnail((900, 900))
    a = np.asarray(im).astype(float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    vari = (g - r) / (g + r - b + 1e-6)
    brillo = a.mean(axis=2)
    agua_sombra = (brillo < 45) | ((b > r) & (b > g))
    sin_cobertura = (vari < 0.02) & ~agua_sombra & (brillo > 60)
    vegetacion = (vari >= 0.02) & ~agua_sombra
    mascara = a.copy()
    mascara[sin_cobertura] = 0.45 * mascara[sin_cobertura] + 0.55 * np.array([230, 60, 30])
    return {"fraccion_sin_cobertura": float(sin_cobertura.mean()), "fraccion_vegetacion": float(vegetacion.mean()),
            "fraccion_agua_sombra": float(agua_sombra.mean()), "imagen": im,
            "resaltado": Image.fromarray(mascara.astype("uint8"))}


def leer_exif(imagen) -> dict:
    """Fecha y coordenadas GPS guardadas por la cámara o el celular (si existen)."""
    from PIL import ExifTags, Image

    out: dict = {}
    try:
        exif = Image.open(imagen).getexif()
        fecha = exif.get(306) or exif.get_ifd(ExifTags.IFD.Exif).get(36867)
        if fecha:
            out["fecha_hora"] = pd.to_datetime(str(fecha), format="%Y:%m:%d %H:%M:%S", errors="coerce")
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        if gps and 2 in gps and 4 in gps:
            gms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600  # noqa: E731
            out["latitud"] = gms(gps[2]) * (-1 if gps.get(1) == "S" else 1)
            out["longitud"] = gms(gps[4]) * (-1 if gps.get(3) == "W" else 1)
    except Exception:  # imagen sin metadatos o formato no soportado
        pass
    return out
