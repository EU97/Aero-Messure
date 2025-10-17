# Aero-Messure
 Repository for aero messurments
 
 
 ## Practicas
 Consulta el resumen y metodologías en `Practicas/README.md`.
 
 ## Quickstart (Windows, PowerShell)
 
 1. Instalar Python 3.10+ desde Microsoft Store o python.org y asegúrate de que `python`/`pip` estén en PATH.
 2. Crear entorno virtual e instalar dependencias:
 
 ```powershell
 python -m venv .venv
 # Si PowerShell bloquea el script de activación:
 # Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
 .\.venv\Scripts\Activate.ps1
 pip install --upgrade pip
 pip install -r requirements.txt
 ```
 
 3. Abrir Jupyter y ejecutar los notebooks de las prácticas:
 
 ```powershell
 # Opción A: estándar
 jupyter notebook
 # Opción B (si A falla):
 python -m notebook
 ```
 
 Navega a `Practicas/Pxx_*/notebooks/` y abre el notebook correspondiente.
 
 ### Cómo ejecutar scripts de Python (alternativa a notebooks)
 Cada práctica incluye utilidades en `src/`. Puedes crear un pequeño script driver o usar Python interactivo.
 
 - Ejemplo (P02 Calibración):
   ```powershell
   # Crear un driver temporal en la consola interactiva de Python
   python
   >>> from pathlib import Path
   >>> import pandas as pd
   >>> from Practicas.P02_Calibracion_Tunel.src.calib_utils import velocity_from_dp
   >>> cal = pd.read_csv(r"Practicas\P02_Calibracion_Tunel\data\calibracion.csv")
   >>> V = velocity_from_dp(cal['dp'], rho=cal.get('rho', 1.225))
   >>> print('V mean =', V.mean())
   >>> exit()
   ```
 
 - Si prefieres un archivo, crea `run_calibracion.py` con ese contenido y ejecútalo con:
   ```powershell
   python Practicas\P02_Calibracion_Tunel\src\run_calibracion.py
   ```
 
 ### Notas
 - Los datos crudos van en `Practicas/Pxx_*/data/` y no se versionan por defecto. Añade tus archivos allí.
 - Si usas PIV, puede tardar en instalar `openpiv`; si fuese un problema, avísame y ajustamos la dependencia.
 
 ### Solución de problemas comunes
 - "jupyter: command not found" o falla con código 1:
   ```powershell
   pip install notebook jupyter
   python -m notebook
   ```
 - Activación de entorno bloqueada:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```
 - ImportError (cv2, skimage, openpiv):
   ```powershell
   pip install -r requirements.txt
   # Alternativa para entornos limitados
   pip install opencv-python-headless
   ```
