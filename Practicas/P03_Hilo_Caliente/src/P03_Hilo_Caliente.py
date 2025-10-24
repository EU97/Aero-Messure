#!/usr/bin/env python3
"""
Práctica 3: Análisis de Hilo Caliente (CTA/HWA) - Script de Calibración

Este script es el equivalente al notebook P03_Hilo_Caliente.ipynb, diseñado
para ser ejecutado desde la carpeta 'src' del repositorio AERO-MESSURE.

Realiza las siguientes operaciones:
- Descubre datos en 'files/p3'.
- Procesa series de voltaje E(t), descartando transientes.
- Agrega por condición de controlador (SD).
- Calcula velocidades de referencia U_ref(SD) del túnel.
- Realiza el ajuste lineal de la Ley de King: U^(1/n) = AE^2 + B.
- Genera gráficas de calibración y ajuste (PNG/SVG).
- Guarda tablas de resultados (CSV/Parquet) y un resumen (JSON).
- Actualiza automáticamente el archivo 'p3.md' en la raíz del repo.
"""

# --- 1. Importaciones y Configuración ---
import sys
import os
import re
import json
import math
import logging
import datetime as dt
from pathlib import Path
from typing import Iterable, Iterator, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# tqdm es opcional pero útil para la barra de progreso
try:
    from tqdm.auto import tqdm
except ImportError:
    logger.info("tqdm no encontrado. Para barras de progreso, instálalo (pip install tqdm)")
    tqdm = lambda x, **kwargs: x  # Fallback a un iterador simple

# --- 2. Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('P03_Script')

# --- 3. Constantes y Rutas ---

# Lógica de rutas adaptada para un script en 'src/'
try:
    # __file__ es la ubicación de este script (AERO-MESSURE/src/p03_analisis.py)
    SCRIPT_DIR = Path(__file__).parent.resolve()
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent.resolve()
except NameError:
    # Fallback si se ejecuta interactivamente (p.ej., en una consola)
    logger.warning("No se pudo usar __file__, asumiendo CWD es la raíz del repo.")
    REPO_ROOT = Path.cwd().resolve()

logger.info(f"Raíz del Repositorio detectada: {REPO_ROOT}")

# Rutas del proyecto (basadas en el notebook original)
PROJECT_ROOT = REPO_ROOT / 'Practicas' / 'P03_Hilo_Caliente'
IMAGES_DIR = PROJECT_ROOT / 'data' / 'images'
DATA_DIR = PROJECT_ROOT / 'data'

# Expresiones Regulares (de Celda 4)
RE_SD_DIR = re.compile(r'(?:AO[_-]0*0?-?)(\d+)$', re.IGNORECASE)  # p.ej., AO_00-15 -> 15
RE_SD_NUM = re.compile(r'(?:^|[^\d])(1[5-9]|2[0-9]|3[0-5])([^\d]|$)')  # 15..35 en texto

# Parámetros de Análisis (de Celda 4)
SD_RANGE = list(range(15, 36))  # 15..35
N_KING_DEFAULT = 3.0  # n=3 (se puede alternar a 2)
COEF_M_SISTEMA = 7.6243
COEF_B_SISTEMA = -1.8926
DISCARD_HEAD = 0.10
DISCARD_TAIL = 0.10


# --- 4. Definiciones de Funciones ---

def ensure_dir(p: Path):
    """Crea un directorio si no existe."""
    p.mkdir(parents=True, exist_ok=True)

def find_p3_data_root() -> Path:
    """Encuentra la carpeta de datos 'files/P3' o 'files/p3'."""
    data_root_p3_upper = REPO_ROOT / 'files' / 'P3'
    data_root_p3_lower = REPO_ROOT / 'files' / 'p3'
    
    if data_root_p3_upper.is_dir():
        return data_root_p3_upper
    if data_root_p3_lower.is_dir():
        return data_root_p3_lower
    
    raise FileNotFoundError(f"No se encontró 'files/p3' o 'files/P3' en {REPO_ROOT}")

def parse_sd_from_path(p: Path) -> Optional[int]:
    """Extrae el valor SD (15-35) del nombre/ruta del archivo."""
    m = RE_SD_DIR.search(p.parent.name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    for part in [p.name, p.stem, p.parent.name]:
        m2 = RE_SD_NUM.search(part)
        if m2:
            try:
                return int(m2.group(1))
            except Exception:
                continue
    return None

def iter_hwa_files(p3_root: Path) -> Iterator[Tuple[int, Path]]:
    """Itera sobre archivos .txt válidos y extrae su SD."""
    for f in p3_root.rglob('*.txt'):
        sd = parse_sd_from_path(f)
        if sd is not None:
            yield sd, f

def read_voltage_series(filepath: Path) -> np.ndarray:
    """Lee series de voltaje, tolerando encabezados no numéricos."""
    try:
        arr = np.loadtxt(str(filepath), dtype=float)
        if arr.ndim == 0:
            arr = np.array([float(arr)])
        return np.ravel(arr).astype(float)
    except Exception:
        lines = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    lines.append(float(line.strip()))
                except ValueError:
                    continue # Saltar encabezado
        arr = np.array(lines, dtype=float)
        if arr.ndim == 0:
            arr = np.array([float(arr)])
        return np.ravel(arr).astype(float)

def trim_series(arr: np.ndarray, discard_head: float, discard_tail: float) -> np.ndarray:
    """Descarta porcentajes del inicio y fin de una serie."""
    n = len(arr)
    if n == 0:
        return arr
    i0 = int(max(0, min(n, round(discard_head * n))))
    i1 = int(max(i0, min(n, round(n * (1.0 - discard_tail)))))
    sub = arr[i0:i1]
    if sub.size == 0 and n > 0: # Si el descarte fue excesivo, devolver todo
        return arr
    return sub

def tunnel_velocity_from_sd(sd_percent: float, m: float, b: float) -> float:
    """Calcula U_ref [mm/s] desde el SD [%] usando coefs. del túnel."""
    return max(0.0, m * float(sd_percent) + b)

def process_iteration_file(fpath: Path, discard_head=DISCARD_HEAD, discard_tail=DISCARD_TAIL) -> Tuple[float, float, int]:
    """Calcula E_mean y E_rms para la porción útil de un archivo."""
    raw = read_voltage_series(fpath)
    sub = trim_series(raw, discard_head, discard_tail)
    if sub.size == 0:
        return float('nan'), float('nan'), 0
    return float(np.mean(sub)), float(np.std(sub, ddof=1) if sub.size>1 else 0.0), int(sub.size)

def aggregate_by_sd(per_iter: pd.DataFrame) -> pd.DataFrame:
    """Agrega los resultados de iteraciones por cada valor SD."""
    rows = []
    for sd, grp in per_iter.groupby('SD'):
        valid = grp[grp['n_points']>0]
        if valid.empty:
            continue
        e_means = valid['E_mean_V'].to_numpy()
        e_rms_iter = valid['E_rms_V'].to_numpy()
        e_sd_mean = float(np.mean(e_means))
        e_sd_std_between = float(np.std(e_means, ddof=1)) if e_means.size>1 else 0.0
        n_files = int(valid.shape[0])
        n_total_pts = int(valid['n_points'].sum())
        U_ref = tunnel_velocity_from_sd(sd, COEF_M_SISTEMA, COEF_B_SISTEMA)
        rows.append({
            'SD': int(sd),
            'E_SD_V': e_sd_mean,
            'E_SD_std_between_V': e_sd_std_between,
            'E_iter_rms_mean_V': float(np.mean(e_rms_iter)) if e_rms_iter.size>0 else 0.0,
            'U_ref_mm_s': float(U_ref),
            'n_files': n_files,
            'n_total_points': n_total_pts
        })
    return pd.DataFrame(rows).sort_values('SD').reset_index(drop=True)

def build_king_table(per_sd: pd.DataFrame, n_exp: float = N_KING_DEFAULT) -> pd.DataFrame:
    """Prepara el DataFrame para la transformación lineal de King."""
    df = per_sd.copy()
    df['E_SD_sq_V2'] = df['E_SD_V']**2
    df['U_ref_pow_1_over_n'] = df['U_ref_mm_s'].clip(lower=1e-12)**(1.0/float(n_exp))
    return df

def fit_king_linear(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Realiza el ajuste lineal U^(1/n) = A*E^2 + B y calcula R^2."""
    X = df['E_SD_sq_V2'].to_numpy()
    Y = df['U_ref_pow_1_over_n'].to_numpy()
    if X.size < 2:
        return float('nan'), float('nan'), 0.0
    
    A, B = np.polyfit(X, Y, 1)
    
    Y_pred = A * X + B
    ss_res = np.sum((Y - Y_pred)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    
    if ss_tot < 1e-12:
        R2 = 1.0 if ss_res < 1e-12 else 0.0
    else:
        R2 = 1.0 - (ss_res / ss_tot)
        
    return float(A), float(B), float(R2)

def dU_dE(E: np.ndarray, A: float, B: float, n: float) -> np.ndarray:
    """Calcula la sensibilidad dU/dE."""
    base = np.maximum(A*(E**2) + B, 0.0)
    return n * (np.maximum(base, 1e-12)**(n-1.0)) * (2.0*A*E)

def propagate_sigma_U(per_sd: pd.DataFrame, A: float, B: float, n: float) -> pd.DataFrame:
    """Estima la incertidumbre en U (U_sigma) basada en sigma_E."""
    df = per_sd.copy()
    E = df['E_SD_V'].to_numpy()
    sigma_E = np.maximum(df['E_SD_std_between_V'].to_numpy(), 1e-12)
    sens = dU_dE(E, A, B, n)
    df['U_calib_mm_s'] = np.maximum((A*(E**2) + B)**n, 0.0)
    df['U_sigma_mm_s'] = np.abs(sens) * sigma_E
    return df

def create_and_save_plots(KING_DF: pd.DataFrame, A: float, B: float, R2: float):
    """Genera y guarda las gráficas de ajuste y calibración."""
    logger.info("Generando gráficas...")
    sns.set(context='notebook', style='whitegrid', palette='deep')
    plt.rcParams['figure.dpi'] = 110

    KING_DF_VALID = KING_DF[KING_DF['U_ref_mm_s'] > 1e-9].copy()

    # --- Figura 1: Ajuste Transformado (de Celda 9) ---
    fig1, ax1 = plt.subplots(figsize=(9,6))
    ax1.scatter(KING_DF_VALID['E_SD_sq_V2'], KING_DF_VALID['U_ref_pow_1_over_n'], c='C0', label='Datos transformados (U_ref > 0)')
    if KING_DF['U_ref_mm_s'].min() <= 1e-9:
        ax1.scatter(KING_DF[KING_DF['U_ref_mm_s'] <= 1e-9]['E_SD_sq_V2'], KING_DF[KING_DF['U_ref_mm_s'] <= 1e-9]['U_ref_pow_1_over_n'], 
                    c='gray', marker='s', label='Datos (U_ref = 0)', alpha=0.5)

    if not math.isnan(A):
        x_min_all = KING_DF['E_SD_sq_V2'].min()
        x_max_all = KING_DF['E_SD_sq_V2'].max()
        xline = np.linspace(x_min_all * 0.95, x_max_all * 1.05, 200)
        yline = A * xline + B
        ax1.plot(xline, yline, 'r-', label=f'Ajuste: Y={A:.3f} X + {B:.3f}\nR²={R2:.4f}')
        
    ax1.set_xlabel(r'$E_{SD}^2$ [V$^2$]')
    ax1.set_ylabel(r'$U_{ref}^{1/n}$ [(mm/s)$^{1/n}$]')
    ax1.set_title('Ajuste lineal (transformación de King)')
    ax1.grid(True, alpha=0.3); ax1.legend(); fig1.tight_layout()
    fig1_png = IMAGES_DIR / 'p3_king_transform_fit.png'
    fig1_svg = IMAGES_DIR / 'p3_king_transform_fit.svg'
    fig1.savefig(fig1_png); fig1.savefig(fig1_svg)
    logger.info(f'Guardado: {fig1_png}')

    # --- Figura 2: Curva de Calibración (de Celda 10) ---
    fig2, ax2 = plt.subplots(figsize=(9,6))
    ax2.scatter(KING_DF['E_SD_V'], KING_DF['U_ref_mm_s'], c='C2', marker='x', s=80, label='Puntos de calibración')
    if not math.isnan(A):
        egrid = np.linspace(max(1e-9, float(KING_DF['E_SD_V'].min())*0.95), float(KING_DF['E_SD_V'].max())*1.05, 200)
        ucurve = np.maximum((A*(egrid**2) + B), 0.0)**N_KING_DEFAULT # Asegurar base >= 0
        ax2.plot(egrid, ucurve, 'm-', label=fr'Modelo: $U = (MAX(0, {A:.3f} E^2 + {B:.3f}))^{int(N_KING_DEFAULT)}$')
        
    ax2.set_xlabel('E_SD (V)')
    ax2.set_ylabel('U (mm/s)')
    ax2.set_title('Curva de Calibración (HWA, Ley de King)')
    ax2.grid(True, alpha=0.3); ax2.legend(); ax2.set_ylim(bottom=-max(KING_DF['U_ref_mm_s'].max()*0.05, 1.0))
    fig2.tight_layout()
    fig2_png = IMAGES_DIR / 'p3_king_curve.png'
    fig2_svg = IMAGES_DIR / 'p3_king_curve.svg'
    fig2.savefig(fig2_png); fig2.savefig(fig2_svg)
    logger.info(f'Guardado: {fig2_png}')
    
    plt.close('all') # Cerrar figuras para que el script termine

def update_markdown_report(A: float, B: float, R2: float):
    """Actualiza el archivo p3.md con los resultados y gráficas."""
    logger.info(f"Actualizando {REPO_ROOT.name}/p3.md...")
    p3_md_path = REPO_ROOT / 'p3.md'
    backup_path = p3_md_path.with_suffix('.md.bak')

    try:
        if p3_md_path.exists():
            original_content = p3_md_path.read_text(encoding='utf-8')
            backup_path.write_text(original_content, encoding='utf-8')
            logger.info(f'Respaldo creado: {backup_path.name}')
        else:
            original_content = f"# Práctica 3: Hilo Caliente (P03)\n\n"
            logger.info(f'Creando nuevo archivo: {p3_md_path.name}')

        # Marcadores para reemplazar el contenido (de Celda 13)
        marker_start = ""
        marker_end = ""
        
        sec = []
        sec.append(marker_start + "\n")
        sec.append("## Resultados Automáticos (Generados por Script)\n")
        sec.append(f"*(Actualizado: {dt.datetime.now().isoformat()})*\n\n")
        sec.append(f"Ajuste Ley de King con $n={N_KING_DEFAULT:.0f}$:\n")
        sec.append(f"- **$A$**: `{A:.7g}`\n")
        sec.append(f"- **$B$**: `{B:.7g}`\n")
        sec.append(f"- **$R^2$**: `{R2:.7f}`\n\n")
        
        # Calcular ruta relativa desde p3.md (en REPO_ROOT) a IMAGES_DIR
        try:
            md_parent_dir = p3_md_path.parent
            img_rel_path = os.path.relpath(IMAGES_DIR.resolve(), md_parent_dir.resolve())
            img_rel_path = img_rel_path.replace(os.path.sep, '/')
        except Exception:
            img_rel_path = f"Practicas/P03_Hilo_Caliente/data/images"

        sec.append(f"### Curva de Calibración $U(E)$ \n")
        sec.append(f"![Curva de King]({img_rel_path}/p3_king_curve.png)\n\n")
        sec.append(f"### Ajuste Lineal $U^{{1/n}}(E^2)$ \n")
        sec.append(f"![Ajuste lineal]({img_rel_path}/p3_king_transform_fit.png)\n\n")
        sec.append(marker_end + "\n")
        
        new_results = "".join(sec)
        
        re_marker = re.compile(f"{re.escape(marker_start)}.*{re.escape(marker_end)}", re.DOTALL)
        
        if re_marker.search(original_content):
            final_content = re_marker.sub(new_results, original_content)
        else:
            final_content = original_content + "\n" + new_results

        with open(p3_md_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        logger.info(f'Resultados actualizados en: {p3_md_path.name}')

    except Exception as e_md:
        logger.error(f'No se pudo actualizar {p3_md_path.name}: {e_md}')
        if backup_path.exists():
            backup_path.replace(p3_md_path)
            logger.warning(f'Fallo al escribir MD, backup restaurado.')


# --- 5. Flujo Principal de Ejecución ---

def main() -> int:
    """Función principal del script."""
    logger.info("Iniciando análisis de P03 (Hilo Caliente)...")
    
    # --- Configurar Directorios ---
    try:
        ensure_dir(IMAGES_DIR)
        ensure_dir(DATA_DIR)
    except Exception as e:
        logger.error(f"Error creando directorios de salida: {e}")
        return 1

    # --- Celda 5: Descubrimiento de Datos ---
    try:
        p3_root = find_p3_data_root()
        logger.info(f'Raíz P3: {p3_root}')
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1  # Salir con error

    disc = []
    for sd, f in iter_hwa_files(p3_root):
        disc.append({'file': str(f), 'SD': int(sd)})
    
    if not disc:
        logger.error(f"No se encontraron archivos .txt válidos en {p3_root}")
        return 1

    DF_FILES = pd.DataFrame(disc).sort_values(['SD','file']).reset_index(drop=True)
    print("\n--- Archivos Detectados (Head) ---")
    print(DF_FILES.head())
    logger.info(f'Archivos detectados: {len(DF_FILES)}')
    missing = sorted(set(SD_RANGE) - set(DF_FILES['SD'].unique().tolist()))
    if missing:
        logger.warning(f'SD esperados sin archivos: {missing}')

    # --- Celda 6: Procesamiento por Iteración ---
    rows = []
    logger.info("Procesando archivos...")
    iterator = tqdm(DF_FILES.iterrows(), total=len(DF_FILES), desc="Procesando archivos")
    for _, r in iterator:
        f = Path(r['file'])
        try:
            e_mean, e_rms, n_pts = process_iteration_file(f, DISCARD_HEAD, DISCARD_TAIL)
            rows.append({'file': str(f), 'SD': int(r['SD']), 'E_mean_V': e_mean, 'E_rms_V': e_rms, 'n_points': n_pts})
        except Exception as ex:
            logger.warning(f'Fallo procesando {f.name}: {ex}')

    PER_ITER = pd.DataFrame(rows).sort_values(['SD','file']).reset_index(drop=True)
    print("\n--- Datos por Iteración (Head) ---")
    print(PER_ITER.head())
    logger.info(f"Iteraciones válidas (n_points > 0): {(PER_ITER['n_points'] > 0).sum()}")

    # --- Celda 7: Agregación por SD ---
    PER_SD = aggregate_by_sd(PER_ITER)
    print("\n--- Datos Agregados por SD ---")
    print(PER_SD.to_string()) # Imprimir tabla completa
    logger.info(f'Condiciones SD válidas: {len(PER_SD)}')

    # --- Celda 8: Transformación King y Guardado de Tabla ---
    KING_DF = build_king_table(PER_SD, n_exp=N_KING_DEFAULT)
    calib_csv = DATA_DIR / 'calibracion_p3.csv'
    calib_parquet = DATA_DIR / 'calibracion_p3.parquet'
    try:
        KING_DF.to_csv(calib_csv, index=False)
        KING_DF.to_parquet(calib_parquet, index=False)
        logger.info(f'Tablas de calibración guardadas en: {DATA_DIR}')
    except Exception as ex:
        logger.warning(f'No se pudo guardar Parquet (opcional): {ex}')
        if not calib_csv.exists():
             logger.error(f'Error fatal: No se pudo guardar CSV en {calib_csv}')
             return 1

    # --- Celda 9: Ajuste Lineal ---
    KING_DF_VALID = KING_DF[KING_DF['U_ref_mm_s'] > 1e-9].copy()
    if len(KING_DF_VALID) < 2:
        logger.error("No hay suficientes puntos válidos (U_ref > 0) para el ajuste.")
        A, B, R2 = np.nan, np.nan, 0.0
    else:
        A, B, R2 = fit_king_linear(KING_DF_VALID)

    print("\n--- Resultados del Ajuste ---")
    print(f'Ley de King (lin.): U^(1/n) = A*E^2 + B, n={N_KING_DEFAULT:.0f}')
    print(f'A = {A:.6f}, B = {B:.6f}, R^2 = {R2:.6f}')

    # --- Celdas 9 & 10: Generación de Gráficas ---
    if not math.isnan(A):
        create_and_save_plots(KING_DF, A, B, R2)
    else:
        logger.warning("Ajuste fallido (NaN), no se generarán gráficas.")

    # --- Celda 11: Propagación de Incertidumbre ---
    UNC = propagate_sigma_U(PER_SD, A, B, N_KING_DEFAULT) if not math.isnan(A) else PER_SD.copy()
    print("\n--- Datos con Incertidumbre (Head) ---")
    print(UNC.head())

    # --- Celda 13: Exportación de Resumen y Actualización de MD ---
    summary = {
        'timestamp': dt.datetime.now().isoformat(),
        'N_KING_default': N_KING_DEFAULT,
        'model_king_linear': {'A': A, 'B': B, 'R2': R2},
        'SDs': PER_SD['SD'].tolist(),
        'images': {
            'transform_fit': str((IMAGES_DIR/'p3_king_transform_fit.png').resolve()),
            'king_curve': str((IMAGES_DIR/'p3_king_curve.png').resolve()),
        },
        'tables': {
            'calibracion_csv': str(calib_csv.resolve()),
            'calibracion_parquet': str(calib_parquet.resolve()) if calib_parquet.exists() else None,
        },
        'libs': {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'matplotlib': plt.matplotlib.__version__,
            'seaborn': sns.__version__,
        },
    }
    summary_path = DATA_DIR / 'calibracion_p3_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f'Resumen guardado en: {summary_path}')
    
    # Actualizar p3.md
    if not math.isnan(A):
        update_markdown_report(A, B, R2)
    else:
        logger.warning("Ajuste fallido, no se actualizará p3.md.")

    # --- Celda 14: Comprobaciones Finales ---
    try:
        assert not DF_FILES.empty, 'No se encontraron archivos en files/p3'
        assert (PER_ITER['n_points']>0).any(), 'Ningún archivo produjo muestras válidas'
        assert PER_SD.shape[0] >= 2, 'Se requieren >=2 SD para regresión'
        assert 0.0 <= R2 <= 1.0, 'R^2 fuera de [0,1]'
        logger.info('Checks básicos: OK')
    except AssertionError as e:
        logger.error(f"Fallo la comprobación final: {e}")
        return 1

    logger.info('Análisis de P03 completado exitosamente.')
    return 0

# --- 6. Punto de Entrada del Script ---
if __name__ == "__main__":
    sys.exit(main())