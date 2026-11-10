# BIODIVERSITY-GUARD Predict · Trabajo Machine Learning (1ACC0057)

Motor analítico de alerta temprana de deforestación (Frente 1) y riesgo de incursión ilegal (Frente 2)
para la Reserva Nacional Tambopata y su zona de amortiguamiento.

- `data/` datasets finales del Hito 1 (`dataset_frente1_hito1_final.csv`, `dataset_frente2_hito1_final.csv`) y salidas de los modelos
- `Hito1_Comprension_Negocio_Datos/` construcción de datasets y EDA
- `Hito2_Preparacion_Baseline/` preparación y baseline (corregido para usar el dataset final)
- `Hito3_Modelado_Avanzado_Evaluacion/` modelado avanzado, evaluación, BD común y dashboard (ver su README)
- `models/` modelos entrenados (`.joblib`)
- `reports/` figuras y métricas (`reports/eda`, `reports/hito3`)

```bash
pip install -r requirements.txt
python Hito3_Modelado_Avanzado_Evaluacion/modelado_avanzado_hito3.py
python Hito3_Modelado_Avanzado_Evaluacion/base_datos_comun.py init
streamlit run Hito3_Modelado_Avanzado_Evaluacion/dashboard_hito3_streamlit.py
```
