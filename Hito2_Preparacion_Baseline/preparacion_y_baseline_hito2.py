import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

ruta_raiz = Path(__file__).resolve().parents[1]
ruta_datos = ruta_raiz / "data"
ruta_reportes = ruta_raiz / "reports"
ruta_hito2 = Path(__file__).resolve().parent

archivo_fr1_in = ruta_datos / "dataset_frente1_hito1.csv"
archivo_fr2_in = ruta_datos / "dataset_frente2_hito1_final.csv"

# Función para crear las carpetas necesarias si no existen
def crear_directorios():
    ruta_datos.mkdir(parents=True, exist_ok=True)
    ruta_reportes.mkdir(parents=True, exist_ok=True)
    ruta_hito2.mkdir(parents=True, exist_ok=True)

# Limpieza e ingeniería de características para series de tiempo
def preparar_frente_uno():
    datos = pd.read_csv(archivo_fr1_in)
    datos = datos.sort_values(by=['cuadrante_id', 'semana'])
    
    caracteristicas_clima = ['temperatura_media', 'humedad_media', 'precipitacion_suma', 'n_alertas']
    for col in caracteristicas_clima:
        datos[col] = datos.groupby('cuadrante_id')[col].transform(
            lambda x: x.interpolate(method='linear').ffill().bfill()
        )
        
    for col in caracteristicas_clima:
        datos[f'{col}_media_3sem'] = datos.groupby('cuadrante_id')[col].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        datos[f'{col}_rezago_1sem'] = datos.groupby('cuadrante_id')[col].shift(1)
        
    datos['target_7d'] = datos.groupby('cuadrante_id')['perdida_ha'].shift(-1)
    datos['target_28d'] = datos.groupby('cuadrante_id')['perdida_ha'].shift(-4)
    
    datos_limpios = datos.dropna(subset=['target_7d', 'target_28d', 'temperatura_media_rezago_1sem']).copy()
    
    return datos_limpios, caracteristicas_clima

# Limpieza e ingeniería de características espaciales para clasificación
def preparar_frente_dos():
    datos = pd.read_csv(archivo_fr2_in)
    datos = datos.dropna(subset=['Nivel_Riesgo'])
    datos['Nivel_Riesgo'] = datos['Nivel_Riesgo'].astype(int)
    
    caracteristicas_num = ['confidence_score', 'distancia_limite_m', 'distancia_a_carretera_m', 'distancia_a_rio_m']
    caracteristicas_cat = ['clase_detectada', 'en_zona_intangible', 'en_zona_amortiguamiento']
    
    for col in caracteristicas_num:
        datos[col] = datos[col].fillna(datos[col].median())
        
    datos['en_zona_intangible'] = datos['en_zona_intangible'].astype(int)
    datos['en_zona_amortiguamiento'] = datos['en_zona_amortiguamiento'].astype(int)
    
    datos['densidad_telemetria_anomala'] = datos['confidence_score'] / (datos['distancia_limite_m'] + 1)
    datos['indice_estres_hidrico'] = datos['distancia_a_rio_m'] / 1000.0
    
    caracteristicas_num.extend(['densidad_telemetria_anomala', 'indice_estres_hidrico'])
    
    datos_modelo = pd.get_dummies(datos[caracteristicas_num + caracteristicas_cat], columns=['clase_detectada'], drop_first=True)
    objetivo = datos['Nivel_Riesgo']
    
    return datos_modelo, objetivo

# Entrenamiento del modelo base de regresión
def evaluar_regresion(datos, variables_clima):
    nuevas_variables = variables_clima + [f'{col}_media_3sem' for col in variables_clima] + [f'{col}_rezago_1sem' for col in variables_clima]
    X = datos[nuevas_variables]
    y_7d = datos['target_7d']
    y_28d = datos['target_28d']
    
    X_train, X_test, y_train_7d, y_test_7d = train_test_split(X, y_7d, test_size=0.3, shuffle=False)
    _, _, y_train_28d, y_test_28d = train_test_split(X, y_28d, test_size=0.3, shuffle=False)
    
    escalador = StandardScaler()
    X_train_escalado = escalador.fit_transform(X_train)
    X_test_escalado = escalador.transform(X_test)
    
    pesos_7d = np.where(y_train_7d > 0, 30, 1)
    pesos_28d = np.where(y_train_28d > 0, 30, 1)
    
    modelo_7d = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    modelo_7d.fit(X_train_escalado, y_train_7d, sample_weight=pesos_7d)
    
    modelo_28d = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    modelo_28d.fit(X_train_escalado, y_train_28d, sample_weight=pesos_28d)
    
    pred_7d = modelo_7d.predict(X_test_escalado)
    pred_28d = modelo_28d.predict(X_test_escalado)
    
    metricas = {
        "mae_7d": float(mean_absolute_error(y_test_7d, pred_7d)),
        "rmse_7d": float(np.sqrt(mean_squared_error(y_test_7d, pred_7d))),
        "mae_28d": float(mean_absolute_error(y_test_28d, pred_28d)),
        "rmse_28d": float(np.sqrt(mean_squared_error(y_test_28d, pred_28d)))
    }
    
    resultados_7d = pd.DataFrame({'Real': y_test_7d.values, 'Prediccion': pred_7d})
    resultados_28d = pd.DataFrame({'Real': y_test_28d.values, 'Prediccion': pred_28d})
    resultados_7d.to_csv(ruta_reportes / "resultados_f1_7d.csv", index=False)
    resultados_28d.to_csv(ruta_reportes / "resultados_f1_28d.csv", index=False)
    
    return metricas

# Manejo del desbalanceo de clases y evaluación del modelo base de clasificación
def evaluar_clasificacion(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    escalador = StandardScaler()
    X_train_escalado = escalador.fit_transform(X_train)
    X_test_escalado = escalador.transform(X_test)
    
    conteo_antes = y_train.value_counts().to_dict()
    
    smote = SMOTE(random_state=42)
    X_train_balanceado, y_train_balanceado = smote.fit_resample(X_train_escalado, y_train)
    
    conteo_despues = y_train_balanceado.value_counts().to_dict()
    
    modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    modelo.fit(X_train_balanceado, y_train_balanceado)
    
    prediccion = modelo.predict(X_test_escalado)
    
    metricas = {
        "accuracy": float(accuracy_score(y_test, prediccion)),
        "weighted_f1": float(f1_score(y_test, prediccion, average="weighted")),
        "macro_f1": float(f1_score(y_test, prediccion, average="macro")),
        "conteo_antes": conteo_antes,
        "conteo_despues": conteo_despues
    }
    
    resultados_clasificacion = pd.DataFrame({'Real': y_test.values, 'Prediccion': prediccion})
    resultados_clasificacion.to_csv(ruta_reportes / "resultados_f2.csv", index=False)
    
    return metricas

def principal():
    print("Iniciando la preparación de datos y modelado base.")
    crear_directorios()
    
    datos_fr1, variables_clima = preparar_frente_uno()
    X_fr2, y_fr2 = preparar_frente_dos()
    
    metricas_regresion = evaluar_regresion(datos_fr1, variables_clima)
    metricas_clasificacion = evaluar_clasificacion(X_fr2, y_fr2)
    
    resumen_final = {
        "frente_uno": metricas_regresion,
        "frente_dos": metricas_clasificacion
    }
    
    with open(ruta_reportes / "hito2_resumen.json", "w", encoding="utf-8") as archivo:
        json.dump(resumen_final, archivo, indent=4, ensure_ascii=False)
        
    print("Proceso finalizado y datos guardados.")

if __name__ == "__main__":
    principal()