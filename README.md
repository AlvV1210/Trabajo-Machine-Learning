# BIODIVERSITY-GUARD Predict · Machine Learning (1ACC0057)

Sistema de alerta temprana para la Reserva Nacional Tambopata y su zona de amortiguamiento:

- **Frente 1:** anticipa en qué cuadrantes de 5 × 5 km se perderá bosque en los próximos 7 y 30 días y cuántas hectáreas.
- **Frente 2:** clasifica el riesgo (Bajo, Moderado, Alto, Crítico) de cada detección de cámaras, sensores o patrullas.

## Abrir el dashboard (2 pasos, no hay que entrenar nada)

```bash
pip install -r requirements.txt
python iniciar.py
```

`iniciar.py` comprueba las librerías y los archivos, prepara la base de datos de demostración (unos segundos) y abre el
dashboard en el navegador. Los modelos ya entrenados están en `models/`.
Para comprobar la instalación sin abrir el navegador: `python iniciar.py --revisar`.

Requiere Python 3.10 o superior. El mapa con calles necesita internet; sin conexión, elija «Solo cuadrícula» en la página Mapa.

## Carpetas

| Carpeta | Contenido |
|---|---|
| `data/` | Datasets finales, red de sensores, pronósticos guardados y `ejemplos/` para probar la carga de datos |
| `models/` | Modelos entrenados en formato de texto de LightGBM (no dependen de la versión de Python) |
| `Hito1_Comprension_Negocio_Datos/` | Construcción de datasets y EDA |
| `Hito2_Preparacion_Baseline/` | Preparación de datos y modelo base |
| `Hito3_Modelado_Avanzado_Evaluacion/` | Modelado avanzado, evaluación, base de datos común y dashboard |
| `reports/` | Figuras y métricas (`reports/hito3/metricas_hito3.json`) |

## Volver a generar todo (opcional, unos 10 minutos)

```bash
python Hito1_Comprension_Negocio_Datos/generar_eda_hito1.py
python Hito3_Modelado_Avanzado_Evaluacion/construir_frente2_realista.py
python Hito2_Preparacion_Baseline/preparacion_y_baseline_hito2.py
python Hito3_Modelado_Avanzado_Evaluacion/modelado_avanzado_hito3.py
python Hito3_Modelado_Avanzado_Evaluacion/figuras_informe.py
python Hito3_Modelado_Avanzado_Evaluacion/base_datos_comun.py init
```
