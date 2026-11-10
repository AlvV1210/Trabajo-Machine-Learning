# Hito 3 – Refinamiento, Modelado Avanzado e Integración

CRISP-DM fase 4 (avanzada) y fase 5 de BIODIVERSITY-GUARD Predict.

| Archivo | Qué hace |
|---|---|
| `features.py` | Ingeniería de características compartida (Frente 1 y 2). Objetivos `perdida_7d` / `perdida_30d` desplazados al futuro, sin fuga. |
| `modelado_avanzado_hito3.py` | Búsqueda de hiperparámetros, validación walk-forward, comparativa (persistencia, árbol, RF, XGBoost, LightGBM, Hurdle LightGBM), ROC/calibración, captura, análisis de errores, importancia por permutación, Frente 2 con StratifiedGroupKFold, GridSearch y prueba de estrés. |
| `base_datos_comun.py` | BD común (SQLite, DDL portable a PostgreSQL), contrato de interfaz del módulo IA, simulación de detecciones e inferencia que genera alertas. |
| `grilla_cuadrantes.py` | Geometría de los 273 cuadrantes (exacta con shapefiles SERNANP; aproximada desde la telemetría si no están). |
| `dashboard_hito3_streamlit.py` | Centro de Mando: mapa de riesgo, plan de patrullaje, alertas en vivo, KPIs y ficha de cuadrante. |

## Ejecución (desde la raíz del repositorio)

```bash
pip install -r requirements.txt
python Hito2_Preparacion_Baseline/preparacion_y_baseline_hito2.py      # baseline corregido
python Hito3_Modelado_Avanzado_Evaluacion/modelado_avanzado_hito3.py   # ~6 min, genera modelos y métricas
python Hito3_Modelado_Avanzado_Evaluacion/base_datos_comun.py init
streamlit run Hito3_Modelado_Avanzado_Evaluacion/dashboard_hito3_streamlit.py
```

Simular el flujo del módulo IA: `base_datos_comun.py simular 25` y luego `base_datos_comun.py procesar`
(o el botón «Recibir detecciones del módulo IA» del dashboard).

## Resultados (walk-forward fuera de muestra)

Ver `reports/hito3/metricas_hito3.json`. Frente 1 a 30 días: ROC-AUC 0.895, recall 80 %, captura top-20 % 81.5 %,
MAE −21 % vs persistencia. Frente 2: F1 macro 0.9997, recall Crítico 1.0, F1 0.97 con 10 % de errores de sensor.
