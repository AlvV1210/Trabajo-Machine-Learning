"""Abre el Centro de Monitoreo de BIODIVERSITY-GUARD Predict.

    python iniciar.py            # comprueba el entorno, prepara la base de datos y abre el dashboard
    python iniciar.py --revisar  # solo comprueba que todo esté en orden (no abre nada)

No entrena modelos: usa los que ya vienen en la carpeta `models/`. La primera vez crea la base de
datos de demostración (unos segundos). Si falta alguna librería, indica el comando para instalarla.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
H3 = ROOT / "Hito3_Modelado_Avanzado_Evaluacion"
LIBRERIAS = {"pandas": "pandas", "numpy": "numpy", "scipy": "scipy", "lightgbm": "lightgbm", "plotly": "plotly",
             "streamlit": "streamlit", "pyproj": "pyproj", "PIL": "Pillow", "openpyxl": "openpyxl"}
ARCHIVOS = ["data/dataset_frente1_hito1_final.csv", "data/dataset_frente2_hito1_final.csv",
            "data/dataset_frente2_hito3_realista.csv", "data/estaciones_sensores.csv", "data/cuadrantes_grid.csv",
            "data/predicciones_frente2.csv", "data/walkforward_perdida_30d.csv", "reports/hito3/metricas_hito3.json",
            "models/frente1_meta.json", "models/frente1_7d_clasificador.txt", "models/frente1_7d_magnitud.txt",
            "models/frente1_30d_clasificador.txt", "models/frente1_30d_magnitud.txt", "models/frente2_meta.json",
            "models/frente2_clasificador.txt"]


def revisar() -> bool:
    ok = True
    if sys.version_info < (3, 10):
        print(f"✗ Se necesita Python 3.10 o superior (tiene {sys.version.split()[0]})."); ok = False
    faltan = []
    for modulo, paquete in LIBRERIAS.items():
        try:
            importlib.import_module(modulo)
        except ImportError:
            faltan.append(paquete)
    if faltan:
        print("✗ Faltan librerías: " + ", ".join(faltan) + "\n  Instálelas con:  pip install -r requirements.txt"); ok = False
    else:
        print("✓ Librerías instaladas")
    perdidos = [a for a in ARCHIVOS if not (ROOT / a).exists()]
    if perdidos:
        print("✗ Faltan archivos del proyecto:\n   - " + "\n   - ".join(perdidos)
              + "\n  Descargue de nuevo el repositorio completo o regenérelos con:\n"
                "  python Hito3_Modelado_Avanzado_Evaluacion/modelado_avanzado_hito3.py"); ok = False
    else:
        print("✓ Datos y modelos entrenados presentes (no hace falta entrenar)")
    if not ok:
        return False
    sys.path.insert(0, str(H3))
    import base_datos_comun as bd
    import inferencia as inf

    pron = inf.pronosticar_frente1()
    print(f"✓ Modelo de deforestación operativo: {len(pron)} cuadrantes pronosticados, "
          f"{pron['ha_esperadas_30d'].sum():.0f} ha en riesgo a 30 días")
    import pandas as pd
    prueba = inf.clasificar_detecciones(pd.DataFrame([{"clase": "motosierra", "latitud": -12.95, "longitud": -69.30,
                                                       "fecha_hora": "2026-09-21 10:00"}]))
    print(f"✓ Modelo de incursiones operativo: una motosierra en la reserva se clasifica como «{prueba['nivel_nombre'].iloc[0]}»")
    bd.asegurar()
    n = bd.leer("SELECT COUNT(*) n FROM detecciones")["n"].iloc[0]
    print(f"✓ Base de datos lista ({n} detecciones registradas)")
    return True


if __name__ == "__main__":
    if not revisar():
        sys.exit(1)
    if "--revisar" in sys.argv:
        print("\nTodo en orden. Para abrir el dashboard:  python iniciar.py")
        sys.exit(0)
    print("\nAbriendo el dashboard en el navegador (http://localhost:8501). Para cerrarlo: Ctrl + C\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(H3 / "dashboard_hito3_streamlit.py")], cwd=ROOT)
