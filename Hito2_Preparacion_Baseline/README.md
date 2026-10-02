# Hito 2 - Preparación y Baseline

Este entregable corresponde a la fase 3 y 4 inicial de CRISP-DM para el proyecto BIODIVERSITY-GUARD Predict.

## Objetivo
Demostrar que los datos ya están preparados para modelado y que existe un modelo base funcional para ambos frentes analíticos:
- Frente 1: regresión de pérdida de cobertura forestal (`perdida_ha`)
- Frente 2: clasificación del nivel de riesgo (`Nivel_Riesgo`)

## Contenido
- Preparación de datos con limpieza, imputación y feature engineering
- Baseline inicial con métricas reales
- Dashboard preliminar con visualizaciones predictivas

## Archivos principales
- `preparacion_y_baseline_hito2.py`: pipeline completo de preparación y baseline
- `reports/hito2_dashboard.html`: dashboard preliminar PDF/HTML
- `reports/hito2_regresion.png`: visualización del desempeño del frente 1
- `reports/hito2_clasificacion.png`: visualización del desempeño del frente 2
- `data/dataset_frente1_hito2_preparado.csv`: dataset preparado del frente 1
- `data/dataset_frente2_hito2_preparado.csv`: dataset preparado del frente 2

## Cómo ejecutarlo

### Pipeline base y generación de artefactos
```bash
python Hito2_Preparacion_Baseline/preparacion_y_baseline_hito2.py
```

### Dashboard interactivo con Streamlit
```bash
streamlit run Hito2_Preparacion_Baseline/dashboard_hito2_streamlit.py
```

## Criterio de evaluación
Se cubren los elementos de la rúbrica:
- Data Collecting: uso del dataset validado del Hito 1
- Data Cleaning: manejo de nulos y limpieza de registros
- Data Transformation: feature engineering temporal y categórica
- Data Reduction: selección de variables relevantes para baseline
- Data Integration: preparación de datasets unificados y listos para modelado
- Baseline: modelo base funcional y métricas iniciales
- Demostración: dashboard con visualizaciones preliminares
