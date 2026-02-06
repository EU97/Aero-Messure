# Aero-Messure

> **Tool repository for the Measurement Techniques (*Técnicas de Medida*) class and laboratory** in Aerospace Engineering. This repo centralises raw experimental data, Jupyter notebooks, Python utilities, and documentation for all lab sessions, providing a reproducible environment to acquire, process, analyse, and visualise aerodynamic and thermal measurements.

---

## Overview

The repository is organised around **9 laboratory practices** covering the principal measurement techniques used in aerospace experimentation — from wind-tunnel calibration and hot-wire anemometry to PIV, thermography, and combustion-gas analysis. Each practice follows a consistent layout:

| Folder | Contents |
|---|---|
| `notebooks/` | Jupyter notebook with the full analysis pipeline |
| `src/` | Reusable Python modules and utility functions |
| `data/` | Raw / processed data files (`.txt`, `.csv`, images) |

### Practices

| # | Practice | Description |
|---|---|---|
| P01 | **Protección Láser** | Laser safety — risk assessment, OD selection, and regulatory compliance |
| P02 | **Calibración del Túnel de Viento** | Wind-tunnel calibration — Pitot-static pressure, velocity curves |
| P03 | **Hilo Caliente** | Hot-wire anemometry — King's law calibration, turbulence statistics |
| P04 | **LDA / Perfil de Velocidad** | Laser Doppler Anemometry — non-intrusive velocity profiles |
| P05 | **Análisis de Imagen** | Image analysis — filtering, segmentation, edge detection |
| P06 | **PIV** | Particle Image Velocimetry — velocity fields from image pairs |
| P07 | **Termografía** | Infrared thermography — surface temperature mapping |
| P08 | **Productos de Combustión** | Combustion products — gas analysis, AFR, emission characterisation |
| P09 | **Promedio Temporal** | Temporal averaging — statistical treatment of non-stationary signals |

> Full methodology summaries for every practice are available in [`Practicas/README.md`](Practicas/README.md).

### Raw Data (`files/`)

Legacy and supplementary experimental datasets used across practices:

| Folder | Content |
|---|---|
| `files/p2/` | Wind-tunnel acquisitions at 5–55 Hz (`.txt`) |
| `files/P3/AO/` | Hot-wire voltage samples at multiple conditions |
| `files/P4/` | LDA measurement files (multiple traverses) |
| `files/p5/` | Fatigue-crack images for image-analysis practice |
| `files/p6/` | TIFF image pairs (Rankine vortex) for PIV |
| `files/p7/` | Infrared / thermal images (PNG, JPEG) |

---

## Quickstart (Windows, PowerShell)

1. Install **Python 3.10+** from Microsoft Store or [python.org](https://python.org) and make sure `python` / `pip` are on your PATH.
2. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
# If PowerShell blocks the activation script:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

3. Launch Jupyter and open any practice notebook:

```powershell
# Option A — standard
jupyter notebook
# Option B (if A fails):
python -m notebook
```

Navigate to `Practicas/Pxx_*/notebooks/` and open the corresponding `.ipynb`.

### Running Python scripts (alternative to notebooks)

Each practice includes utilities in `src/`. You can write a small driver script or use the interactive interpreter.

- Example (P02 — Wind-Tunnel Calibration):
  ```powershell
  python
  >>> from Practicas.P02_Calibracion_Tunel.src.calib_utils import velocity_from_dp
  >>> import pandas as pd
  >>> cal = pd.read_csv(r"Practicas\P02_Calibracion_Tunel\data\calibracion.csv")
  >>> V = velocity_from_dp(cal['dp'], rho=cal.get('rho', 1.225))
  >>> print('V mean =', V.mean())
  >>> exit()
  ```

- Or create a file and run it directly:
  ```powershell
  python Practicas\P02_Calibracion_Tunel\src\run_calibracion.py
  ```

### Notes
- Raw data goes in `Practicas/Pxx_*/data/` and is not version-controlled by default. Place your files there.
- Installing `openpiv` can be slow on some machines; if that is an issue the dependency can be made optional.

### Troubleshooting

- **"jupyter: command not found"** or exit code 1:
  ```powershell
  pip install notebook jupyter
  python -m notebook
  ```
- **Environment activation blocked**:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```
- **ImportError** (`cv2`, `skimage`, `openpiv`):
  ```powershell
  pip install -r requirements.txt
  # Lighter alternative:
  pip install opencv-python-headless
  ```
