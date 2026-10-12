# Hito 2 - Preparación y Baseline

Este entregable corresponde a la fase 3 y 4 inicial de la metodología CRISP-DM para el proyecto BIODIVERSITY-GUARD Predict.

## Objetivo
Demostrar que los datos están listos para el modelado y que existe un modelo base funcional para nuestros dos frentes analíticos:
* Frente 1: regresión de la pérdida de cobertura forestal (`perdida_ha`)
* Frente 2: clasificación del nivel de riesgo de incursión (`Nivel_Riesgo`)

## Contenido aplicado
* Preparación de datos con imputación por interpolación lineal para el clima y medianas para las distancias espaciales.
* Ingeniería de características avanzada creando rezagos predictivos a siete y veintiocho días, medias móviles climáticas y variables derivadas como el índice de estrés hídrico.
* Manejo riguroso del desbalanceo de clases aplicando la técnica de sobremuestreo sintético para la clasificación y penalización por pesos de muestra para la regresión.
* Modelado base con algoritmos de bosques aleatorios evaluados con métricas exactas y previniendo la fuga de datos temporales.
* Panel interactivo dinámico para la visualización en vivo de las proyecciones y matrices de confusión.

## Archivos principales
* `preparacion_y_baseline_hito2.py`: script central con todo el flujo de limpieza, ingeniería, balanceo y entrenamiento.
* `dashboard_hito2_streamlit.py` aplicación web dinámica desarrollada para consumir los resultados.
* `reports/` carpeta de destino donde el código autogenerará los reportes en formato json y los datos de proyección tras su primera ejecución.
* `data/` directorio que contiene los conjuntos de datos base provenientes de la etapa anterior.

## Cómo ejecutarlo

### Pipeline base y generación de artefactos
```bash
python Hito2_Preparacion_Baseline/preparacion_y_baseline_hito2.py
```

### Dashboard interactivo con Streamlit
```bash
streamlit run Hito2_Preparacion_Baseline/dashboard_hito2_streamlit.py
```

## Criterio de evaluación cubiertos
* Recolección de datos utilizando los conjuntos validados del hito previo.
* Limpieza de datos aplicando técnicas específicas para series de tiempo y datos espaciales sin mezclar el orden cronológico.
* Transformación de datos mediante estandarización y codificación de variables categóricas.
* Balanceo de clases justificado matemáticamente para evitar sesgos hacia las categorías mayoritarias asegurando la detección de amenazas críticas.
* Modelado base funcional con resultados cuantificables que apoyan directamente a los objetivos de desarrollo sostenible 13 y 15.
* Demostración visual mediante una interfaz web interactiva que grafica los horizontes predictivos y el rendimiento del clasificador en tiempo real.
