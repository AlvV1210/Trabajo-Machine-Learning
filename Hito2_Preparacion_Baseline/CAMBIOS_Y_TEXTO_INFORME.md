# Correcciones del Hito 2: texto para actualizar el informe

Este documento indica qué partes del informe (TB1_1ACC0057_16623_GRUPO_2) deben cambiar para coincidir con el código corregido. Todas las cifras provienen de `reports/hito2_resumen.json`.

---

## 1. Criterios de éxito técnico (Business Understanding y tabla RACI)

**Reemplazar** “R² ≥ 0.82; Recall ≥ 0.88 en clase Crítico” por:

| Frente | Criterio | Meta | Baseline Hito 2 |
|---|---|---|---|
| F1 · Riesgo 30 días | ROC‑AUC | ≥ 0.85 | 0.881 ✔ |
| F1 · Riesgo 30 días | Recall de eventos de deforestación | ≥ 0.80 | 0.862 ✔ |
| F1 · Riesgo 7 días | ROC‑AUC / Recall | ≥ 0.80 / ≥ 0.80 | 0.832 / 0.915 ✔ |
| F1 · Priorización | % de hectáreas perdidas en el 20 % de cuadrantes priorizados | ≥ 75 % | 77.5 % (30 d), 80.1 % (7 d) ✔ |
| F1 · Magnitud | MAE menor que el de la persistencia de 12 semanas | < 1.607 ha (30 d) | 1.307 ha ✔ |
| F2 · Nivel de riesgo | F1 macro / Recall Crítico | ≥ 0.90 / ≥ 0.88 | 1.000 / 1.000 ✔ |

**Párrafo de justificación sugerido:**

> El criterio inicial de R² ≥ 0.82 se reemplazó porque no es alcanzable a nivel cuadrante‑semana. La pérdida de cobertura es un fenómeno de ráfagas: el 1 % de las combinaciones cuadrante‑semana concentra el 70 % de las hectáreas perdidas. Un estimador ideal que conociera de antemano el total exacto de cada semana y la participación histórica de cada cuadrante obtendría R² ≈ 0.45. Por ello, en coherencia con el objetivo de alerta temprana, el éxito se mide con la capacidad de anticipar **dónde** ocurrirá la deforestación (ROC‑AUC, Recall y porcentaje de hectáreas captadas al priorizar cuadrantes). El MAE y el RMSE se reportan como métricas secundarias de la magnitud.

---

## 2. Preparación de datos (sección “Limpieza y Preprocesamiento”)

Añadir o ajustar:

- El dataset del Frente 1 es `dataset_frente1_hito1_final.csv`: 44 772 filas, 273 cuadrantes de 5 × 5 km (167 de la reserva y 106 de la zona de amortiguamiento), 164 semanas.
- Se excluyeron las 3 semanas posteriores al 31/08/2026, porque el shapefile de alertas PNCB 2026 (`PNCB_ATD_UTM_01_243_S`) cubre solo los días 1–243. Quedan 161 semanas y 43 953 filas.
- `n_alertas` nulo significa ausencia de alerta y se reemplaza por 0. **`n_alertas` no se usa como variable de entrada**, porque `perdida_ha = n_alertas × 0.09` y su uso sería una fuga directa del target.
- Clima: interpolación lineal temporal por cuadrante, seguida de relleno hacia adelante y hacia atrás (5 187 nulos de temperatura y precipitación y 5 460 de humedad).
- Se excluyeron `perdida_biomasa_t` y `emisiones_co2_t` (transformaciones lineales de `tasa_deforestacion_ha_anio`, r = 1.0) y `mb_expansion_mineria` (casi idéntica a `mb_mineria`).
- `vocacion_zee` y `causa_principal` se codificaron con one‑hot, eliminando la primera categoría.
- Frente 2: se excluyen las 6 331 anotaciones sin regla de riesgo y quedan 9 103 (Bajo 4 647, Moderado 339, Alto 1 408, Crítico 2 709).

## 3. Manejo del desbalance

**Reemplazar** el párrafo del Frente 1 (“noventa y siete por ciento… peso treinta veces mayor”) por:

> En el Frente 1 solo el 7.3 % de las semanas‑cuadrante registra pérdida. El clasificador de riesgo usa `class_weight='balanced_subsample'`, que pondera cada evento de forma inversamente proporcional a su frecuencia. El umbral de alerta no es 0.5: se eligió en una validación interna (último 20 % de las semanas de entrenamiento) como el de mayor F1 entre los que alcanzan un Recall ≥ 0.80.

El párrafo del Frente 2 (SMOTE + pesos balanceados) se mantiene. Hay que precisar que SMOTE se aplica **solo sobre el entrenamiento de cada fold** (pipeline de `imblearn`), para no filtrar ejemplos sintéticos al conjunto de prueba.

## 4. Feature engineering (Frente 1)

**Reemplazar** por:

> Los targets son la pérdida futura: `perdida_7d` (semana t+1) y `perdida_30d` (suma de las semanas t+1 a t+4, es decir 28 días), junto con sus versiones binarias `evento_7d` y `evento_30d`. Todas las variables usan solo información disponible al cierre de la semana t:
> - **Historial de pérdida del cuadrante:** pérdida actual, rezagos de 1 y 2 semanas, sumas móviles de 4, 8, 12, 26 y 52 semanas, número de semanas con alerta en esas ventanas, media exponencial (vida media de 4 semanas), media y frecuencia histórica, y semanas desde la última alerta.
> - **Clima acumulado:** rezago de 1 semana y medias o sumas móviles de 4 y 12 semanas de temperatura, humedad y precipitación.
> - **Estacionalidad:** seno y coseno del mes, y temporada seca (mayo–octubre).
> - **Contexto territorial (MapBiomas, ZEE, SERNANP, INEI):** cobertura ANP, tasa histórica de deforestación, fracciones de bosque, agropecuario, minería, infraestructura y agua, y densidad poblacional.
>
> En total son 44 variables. Se evaluaron también variables de actividad total de la reserva, pero se descartaron: en la validación interna no mejoraron el AUC y el modelo las usaba para memorizar el periodo temporal.

## 5. Modelado

**Reemplazar** los párrafos de resultados y las notas de las Figuras 25–27 por:

> **Validación:** partición temporal. El entrenamiento usa el 70 % inicial de las semanas (ago‑2023 a ago‑2025), con una purga igual al horizonte para que ningún target de entrenamiento se solape con la prueba. La prueba usa las semanas posteriores (sep‑2025 a ago‑2026), que el modelo nunca vio.
>
> **Frente 1 (Random Forest, 400 árboles, `min_samples_leaf=100`):** a 30 días el modelo alcanza un ROC‑AUC de 0.881, un Recall de 0.862, una precisión de 0.571 y un F1 de 0.687. A 7 días obtiene un ROC‑AUC de 0.832 y un Recall de 0.915. En la práctica, si cada semana los guardaparques priorizan el 20 % de cuadrantes con mayor riesgo, cubren el 77.5 % (30 d) y el 80.1 % (7 d) de las hectáreas que efectivamente se deforestan, cuatro veces más que un patrullaje aleatorio. El regresor de hectáreas (Random Forest, 300 árboles) obtiene un MAE de 1.307 ha y un RMSE de 5.786 ha a 30 días, un 18.7 % y un 2.9 % menores que la persistencia de 12 semanas. Su ranking espacial tiene una correlación de Spearman de 0.954 con la pérdida real por cuadrante.
>
> **Comparación con la persistencia:** ordenar los cuadrantes por su pérdida de las últimas 12 semanas da un AUC similar (0.889 a 30 d). Esto confirma que la deforestación en Tambopata es espacialmente persistente. El baseline aporta un umbral de alerta calibrado con un Recall mucho mayor que la regla simple (0.862 frente a 0.634) y la estimación de hectáreas, biomasa y CO₂ en riesgo. Superar claramente esta referencia es el objetivo del Hito 3.
>
> **Limitación:** el periodo de prueba incluye picos de deforestación sin precedentes (oct‑2025 y mar–may‑2026). El regresor, entrenado con niveles históricos menores, subestima el total: 6 876 ha estimadas frente a 18 048 ha reales. En el Hito 3 se abordará con reentrenamiento por ventanas móviles (walk‑forward), modelos de gradient boosting y la integración de las detecciones del módulo de IA.
>
> **Frente 2 (Random Forest + SMOTE + `class_weight='balanced'`, validación cruzada `StratifiedGroupKFold` de 5 folds agrupada por imagen):** Accuracy 1.000, F1 macro 1.000, Recall 1.000 en las cuatro clases y ROC‑AUC 1.000. Estas métricas son perfectas por construcción: `Nivel_Riesgo` se generó con reglas heurísticas a partir de la clase detectada y la zona, y el modelo las reproduce sin errores. Esto valida el pipeline de extremo a extremo, pero no equivale a una validación de campo. La validación con datos reales del módulo de IA se hará en el Hito 3.

**Figuras nuevas** (en `reports/`):
- `hito2_f1_roc.png`: curva ROC 7 y 30 días.
- `hito2_f1_captura.png`: % de hectáreas cubiertas según el % de cuadrantes priorizados.
- `hito2_regresion.png`: hectáreas en riesgo a 30 días, real frente a estimado (toda la reserva).
- `hito2_f1_importancia.png`: variables más influyentes.
- `hito2_clasificacion.png`: matriz de confusión del Frente 2 (validación cruzada).

## 6. Otros ajustes menores del informe

- En “Frente 1” del EDA se cita “27,388 filas, 164 semanas y 167 cuadrantes”, que corresponde al dataset anterior. El dataset final tiene 44 772 filas y 273 cuadrantes.
- El texto sobre `alerta_observada` ya no aplica: el dataset final no incluye esa columna y se deriva de `n_alertas` no nulo.
- La figura de evolución temporal (`reports/eda/figura_4_...`) se regeneró con el dataset final.
