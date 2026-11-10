# Hito 2 - Preparación y Baseline (versión corregida)

Fases 3 (Data Preparation) y 4 inicial (Modeling) de CRISP-DM para BIODIVERSITY-GUARD Predict.

## Qué cambió respecto a la versión anterior

| Problema | Corrección |
|---|---|
| El Frente 1 leía `dataset_frente1_hito1.csv` (27 388 filas, 167 cuadrantes, sin variables MapBiomas/ZEE) | Ahora lee `dataset_frente1_hito1_final.csv` (44 772 filas, 273 cuadrantes, 22 columnas) |
| `n_alertas` era variable de entrada, pero `perdida_ha = n_alertas × 0.09` (fuga del target, R² ficticio de 0.98) | Se excluye `n_alertas`; el modelo solo usa información disponible **antes** del horizonte a predecir |
| Se predecía la pérdida de la misma semana con un split aleatorio | Se predice la pérdida **futura** a 7 días (1 semana) y 30 días (4 semanas) con validación **temporal** y purga |
| Las 3 semanas de septiembre de 2026 valían 0 por falta de cobertura satelital (shapefile PNCB 2026 = días 1–243) | Se excluyen las semanas posteriores al 31/08/2026 |
| Imputación por mediana global del clima | Interpolación lineal temporal por cuadrante (como describe el informe) |
| Árbol de decisión sin balanceo | Random Forest (`class_weight`) en el Frente 1; Random Forest + SMOTE (dentro de cada fold) en el Frente 2 |
| Frente 2 evaluado con un split aleatorio por anotación | Validación cruzada `StratifiedGroupKFold` (5 folds) agrupada por imagen |

## Frente 1: alerta temprana de deforestación

Para cada cuadrante y semana *t* el sistema entrega:

1. **Riesgo**: probabilidad de pérdida de cobertura en los próximos 7 y 30 días (clasificación).
2. **Magnitud**: hectáreas esperadas en ese horizonte, junto con la biomasa (× 295.5 t/ha) y el CO₂ (× 509.3 t/ha) en riesgo.

Resultados en el periodo de prueba (sep‑2025 a ago‑2026, semanas que el modelo nunca vio):

| Métrica | 7 días | 30 días |
|---|---|---|
| ROC‑AUC | **0.832** | **0.881** |
| Recall (umbral elegido en validación) | **0.915** | **0.862** |
| Precisión | 0.223 | 0.571 |
| F1 | 0.359 | 0.687 |
| % de hectáreas perdidas que cae en el 20 % de cuadrantes priorizados | **80.1 %** | **77.5 %** |
| % de hectáreas perdidas dentro de cuadrantes alertados | 92.4 % | 88.2 % |
| MAE hectáreas (persistencia) | 0.451 (0.532) | 1.307 (1.607) |
| RMSE hectáreas (persistencia) | 2.698 (2.734) | 5.786 (5.960) |
| Spearman del ranking espacial | 0.939 | 0.954 |

**Por qué no se usa R² ≥ 0.82 como criterio:** el 1 % de las filas cuadrante‑semana concentra el 70 % de la pérdida. Ni siquiera un “oráculo” que supiera de antemano el total exacto de cada semana y el peso histórico de cada cuadrante pasaría de R² ≈ 0.45 a nivel cuadrante‑semana. El R² del baseline es 0.04 (7 d) y 0.09 (30 d). Para un sistema de alerta temprana las métricas relevantes son Recall, AUC y el porcentaje de hectáreas captadas por los cuadrantes priorizados.

**Referencia de persistencia:** ordenar los cuadrantes por su pérdida de las últimas 12 semanas obtiene un AUC similar (0.833 / 0.889). La deforestación en Tambopata es muy persistente en el espacio. En este hito el RF aporta un umbral de alerta calibrado (Recall 0.86–0.92 frente a 0.63–0.68 de la regla “hubo alerta en las últimas 4 semanas”), MAE un 15–19 % menor y la estimación de hectáreas, biomasa y CO₂. Superar claramente la persistencia es el objetivo del Hito 3.

**Limitación conocida:** en el periodo de prueba hubo picos de deforestación sin precedentes (oct‑2025, mar–may‑2026). El regresor subestima el total: estima 6 876 ha contra 18 048 ha reales a 30 días. Se corregirá en el Hito 3 con reentrenamiento por ventanas móviles (walk‑forward) y con las detecciones del módulo de IA.

## Frente 2: nivel de riesgo de incursión

Random Forest + SMOTE + `class_weight='balanced'`, con validación cruzada agrupada por imagen: Accuracy 1.000, F1 macro 1.000, Recall Crítico 1.000. Estas métricas son perfectas **por construcción**: `Nivel_Riesgo` se generó con reglas heurísticas a partir de `clase_detectada` y de la zona (intangible o de amortiguamiento), y el modelo reproduce esas reglas. Su validación real vendrá con los datos del módulo de IA (Hito 3).

## Archivos

- `preparacion_y_baseline_hito2.py`: preparación, feature engineering, baseline y evaluación.
- `dashboard_hito2_streamlit.py`: dashboard interactivo (ranking de cuadrantes, métricas y gráficos).
- `data/dataset_frente1_hito2_preparado.csv`, `data/dataset_frente2_hito2_preparado.csv`: datasets preparados.
- `reports/hito2_resumen.json`: todas las métricas.
- `reports/hito2_ranking_cuadrantes.csv`: riesgo a 7 y 30 días por cuadrante desde la última semana.
- `reports/hito2_dashboard.html` y figuras `reports/hito2_*.png`.
- `models/hito2_frente1_rf.joblib`, `models/hito2_frente2_rf_smote.joblib`: modelos entrenados.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python Hito2_Preparacion_Baseline/preparacion_y_baseline_hito2.py   # ~2 min
streamlit run Hito2_Preparacion_Baseline/dashboard_hito2_streamlit.py
```
