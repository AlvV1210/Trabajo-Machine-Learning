# Hito 3 – Refinamiento, modelado avanzado e integración

| Archivo | Qué hace |
|---|---|
| `construir_frente2_realista.py` | Convierte las anotaciones del Hito 1 en la telemetría de una red de 58 estaciones con fecha y hora (≈32 eventos por semana, 12 % amenazas). |
| `features.py` | Variables de ambos frentes; se usan igual al entrenar y al predecir. |
| `modelado_avanzado_hito3.py` | Entrena, ajusta hiperparámetros, valida (walk-forward y por estación) y guarda modelos y métricas. `--frente 1` o `--frente 2` para uno solo. |
| `inferencia.py` | Carga los modelos guardados y predice. No entrena. |
| `base_datos_comun.py` | Base de datos común (SQLite) y todas las vías de ingreso de datos. |
| `grilla_cuadrantes.py` | Ubicación de los 273 cuadrantes. |
| `figuras_informe.py` | Figuras explicativas del informe. |
| `dashboard_hito3_streamlit.py` | Centro de monitoreo. Se abre con `python iniciar.py` desde la raíz. |

## Dashboard

Inicio · Mapa · Plan de patrullaje · Alertas · Ingresar datos · Indicadores · Ayuda.

Formas de ingresar datos: archivo CSV/Excel de detecciones, registro manual de un avistamiento, foto (lee fecha y GPS),
imagen satelital o de dron (estima suelo sin cobertura), alertas de deforestación de una semana nueva (recalcula el
pronóstico) y simulador de sensores. En `data/ejemplos/` hay un archivo de cada tipo para probar.

## Resultados (datos que los modelos no vieron al entrenar)

Frente 1, 30 días: ROC-AUC 0.88, detecta 8 de cada 10 pérdidas, 82 % de la pérdida real cae en el 20 % de cuadrantes
priorizados, error en hectáreas 21 % menor que repetir el mes anterior.
Frente 2: F1 macro 0.99 y 99 % de críticos reconocidos en estaciones no vistas; 0.96 con 10 % de fallas de sensor.
Detalle en `reports/hito3/metricas_hito3.json`.
