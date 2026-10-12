import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ruta_raiz = Path(__file__).resolve().parents[1]
ruta_reportes = ruta_raiz / "reports"
archivo_resumen = ruta_reportes / "hito2_resumen.json"

st.set_page_config(page_title="Panel Interactivo Hito Dos", layout="wide")

def cargar_metricas():
    if archivo_resumen.exists():
        with open(archivo_resumen, "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    return None

def cargar_resultados(nombre_archivo):
    ruta = ruta_reportes / nombre_archivo
    if ruta.exists():
        return pd.read_csv(ruta)
    return None

st.title("Proyecto Biodiversity Guard Predict")
st.caption("Fase de modelado base y preparación de datos")

metricas = cargar_metricas()

pestaña_uno, pestaña_dos = st.tabs(["Frente 1: Predicción continua", "Frente 2: Clasificación de riesgo"])

with pestaña_uno:
    st.header("Análisis de pérdida de cobertura forestal")
    
    st.subheader("Preparación de datos y selección de características")
    st.write("1. Limpieza y preprocesamiento")
    st.write("Técnicas de imputación de valores nulos utilizando forward fill y backward fill para series de tiempo, lo que preserva la continuidad climática.")
    st.write("Técnicas de estandarización empleando StandardScaler para los datos climáticos continuos.")
    st.write("Codificación de variables categóricas mediante transformaciones directas y variables dummy.")
    
    st.write("2. Ingeniería de características")
    st.write("Se procedió con la creación de rezagos predictivos a siete y veintiocho días para proyectar los resultados al futuro. Además se incorporaron medias móviles sobre las variables climáticas para capturar el comportamiento temporal acumulado.")
    
    if metricas:
        st.subheader("Indicadores de éxito técnico")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Error absoluto medio a 7 días", f"{metricas['frente_uno']['mae_7d']:.4f}")
        col2.metric("Raíz error cuadrático a 7 días", f"{metricas['frente_uno']['rmse_7d']:.4f}")
        col3.metric("Error absoluto medio a 28 días", f"{metricas['frente_uno']['mae_28d']:.4f}")
        col4.metric("Raíz error cuadrático a 28 días", f"{metricas['frente_uno']['rmse_28d']:.4f}")

    st.subheader("Visualización del horizonte predictivo")
    opcion_horizonte = st.radio(
        "Seleccione el horizonte predictivo a evaluar",
        ["Proyección a siete días", "Proyección a veintiocho días"]
    )
    
    if opcion_horizonte == "Proyección a siete días":
        datos_grafico = cargar_resultados("resultados_f1_7d.csv")
        titulo_dinamico = "Comparativa de eventos críticos Proyección a 7 días"
        etiqueta_dinamica = "Predicción a 7 días"
        color_dinamico = "orange"
    else:
        datos_grafico = cargar_resultados("resultados_f1_28d.csv")
        titulo_dinamico = "Comparativa de eventos críticos Proyección a 28 días"
        etiqueta_dinamica = "Predicción a 28 días"
        color_dinamico = "salmon"
        
    if datos_grafico is not None:
        picos = datos_grafico[datos_grafico['Real'] > 0].head(40)
        figura_regresion, eje_regresion = plt.subplots(figsize=(10, 4))
        eje_regresion.plot(picos['Real'].values, label='Pérdida de cobertura real', color='teal', marker='o')
        eje_regresion.plot(picos['Prediccion'].values, label=etiqueta_dinamica, color=color_dinamico, linestyle='dashed', marker='x')
        eje_regresion.set_title(titulo_dinamico)
        eje_regresion.set_xlabel("Eventos registrados")
        eje_regresion.set_ylabel("Hectáreas perdidas")
        eje_regresion.legend()
        eje_regresion.grid(alpha=0.3)
        st.pyplot(figura_regresion)

with pestaña_dos:
    st.header("Análisis del nivel de riesgo de amenaza")
    
    st.subheader("Manejo del desbalanceo y selección de características")
    st.write("1. Manejo de desbalanceo de clases")
    st.write("Se explica la aplicación de técnicas de sobremuestreo sintético y el uso de pesos para la clasificación de riesgo. Esto garantiza que las alertas críticas no sean ignoradas.")
    
    if metricas:
        col_antes, col_despues = st.columns(2)
        with col_antes:
            st.write("Tabla de clases antes del balanceo")
            df_antes = pd.DataFrame(list(metricas['frente_dos']['conteo_antes'].items()), columns=["Nivel de riesgo", "Cantidad"])
            st.table(df_antes)
        with col_despues:
            st.write("Tabla de clases después del balanceo")
            df_despues = pd.DataFrame(list(metricas['frente_dos']['conteo_despues'].items()), columns=["Nivel de riesgo", "Cantidad"])
            st.table(df_despues)
            
    st.write("2. Ingeniería de características")
    st.write("Se crearon nuevas variables derivadas como el índice de estrés hídrico y la densidad de telemetría anómala. Estas métricas mejoran el poder predictivo ya que combinan la vulnerabilidad geográfica directa con la precisión analítica de los sensores.")

    if metricas:
        st.subheader("Indicadores de éxito técnico")
        col_acc, col_f1 = st.columns(2)
        col_acc.metric("Exactitud global del modelo", f"{metricas['frente_dos']['accuracy']:.4f}")
        col_f1.metric("Métrica F1 Macro priorizada", f"{metricas['frente_dos']['macro_f1']:.4f}")

    st.subheader("Matriz de confusión de riesgo")
    datos_clasificacion = cargar_resultados("resultados_f2.csv")
    
    if datos_clasificacion is not None:
        matriz = confusion_matrix(datos_clasificacion['Real'], datos_clasificacion['Prediccion'])
        visualizacion = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["Bajo", "Moderado", "Alto", "Crítico"])
        
        figura_clasificacion, eje_clasificacion = plt.subplots(figsize=(6, 4))
        visualizacion.plot(cmap=plt.cm.Blues, ax=eje_clasificacion)
        st.pyplot(figura_clasificacion)