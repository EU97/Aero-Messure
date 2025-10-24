# -*- coding: utf-8 -*-
# p02_calibracion_script.py
# Equivalente en script de Python del notebook P02_Calibracion_Tunel.ipynb
# CORREGIDO PARA INCLUIR FUNCIONES FALTANTES Y ACLARAR RUTAS

import sys
import os
import json
import math
import logging
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from importlib import reload

# --- 1) Configuración del entorno y dependencias ---

# Añadir el directorio 'src' al path
# ASUNCIÓN: Este script está en un directorio (p.ej., 'notebooks') y 'src' está un nivel arriba.
# Ajusta 'src_path' si tu estructura es diferente.
script_dir = Path(__file__).parent.resolve()
src_path = (script_dir / '../src').resolve() # Sube un nivel y busca 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))
    print(f"Añadido a sys.path: {src_path}")
else:
    print(f"Ruta ya en sys.path: {src_path}")

try:
    import calib_utils as cu
    reload(cu) # Recarga por si hay cambios
except ModuleNotFoundError:
    print(f"ERROR: No se pudo encontrar el módulo 'calib_utils'. Asegúrate de que '{src_path}' exista y sea accesible.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR al importar calib_utils: {e}")
    sys.exit(1)

# Configuración global de gráficos y logging
sns.set(context='notebook', style='whitegrid', palette='deep')
plt.rcParams['figure.dpi'] = 110
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("P02_Script")

# Directorios base (relativos al script)
# ASUNCIÓN: El script está un nivel por debajo del raíz del proyecto.
# Si el script está en la raíz, cambia 'script_dir.parent' a 'script_dir'.
project_root = script_dir.parent
data_dir = project_root / 'data'
images_dir = data_dir / 'images' # Ahora usa project_root definido aquí

# Asegurar que los directorios existan
cu.ensure_dir(images_dir)
cu.ensure_dir(data_dir) # Asegura data_dir también

# Lista de requerimientos (informativo)
REQS = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'ipywidgets', 'scipy', 'tqdm', 'pyarrow'] # pyarrow para parquet
print(f"Directorio raíz del proyecto asumido: {project_root}")
print(f"Directorio de datos: {data_dir}")
print(f"Directorio de imágenes: {images_dir}")
print(f"Asegúrate de tener instaladas las dependencias: {', '.join(REQS)}")
print("-" * 30)

# --- Funciones definidas en el notebook (y faltantes en el script original) ---

# 3) y 4) Carga y estadísticas por archivo (ya estaba)
def load_and_stats_per_file(df_files: pd.DataFrame, abs_values: bool = True) -> pd.DataFrame:
    rows = []
    logger.info(f"Procesando {len(df_files)} archivos descubiertos...")
    for _, r in df_files.iterrows():
        f = Path(r['file'])
        try:
            s = cu.read_lda_speeds(f, abs_values=abs_values)
            rows.append({
                'file': str(f),
                'frequency_hz': int(r['frequency_hz']),
                'mean_ms': float(s.mean()),
                'std_ms': float(s.std(ddof=1)),
                'n_samples': int(s.shape[0]),
            })
        except Exception as ex:
            logger.warning(f"No se pudo leer {f.name}: {ex}")
            continue
    logger.info(f"Se procesaron {len(rows)} archivos válidamente.")
    if not rows:
        raise ValueError("No se pudo procesar ningún archivo LDA.")
    return pd.DataFrame(rows).sort_values(['frequency_hz', 'file']).reset_index(drop=True)

# 5) Agregación por frecuencia (ya estaba)
def aggregate_by_frequency(per_file: pd.DataFrame, method: str = 'weighted') -> pd.DataFrame:
    rows = []
    logger.info(f"Agregando estadísticas por frecuencia usando el método: {method}")
    for freq, grp in per_file.groupby('frequency_hz'):
        means = grp['mean_ms'].to_numpy()
        ns = grp['n_samples'].to_numpy()
        if method == 'weighted' and ns.sum() > 0:
            mean_avg = float(np.average(means, weights=ns))
        else:
            mean_avg = float(np.mean(means))

        stds = grp['std_ms'].to_numpy()
        denom = np.sum(np.maximum(ns - 1, 1)) # Evita dividir por cero si n=1
        if denom > 0:
             # Varianza ponderada (pooled variance)
            pooled_var = float(np.sum((np.maximum(ns - 1, 1)) * (stds ** 2)) / denom)
            pooled_std = float(np.sqrt(pooled_var))
        else:
            # Si todos los n son 1 o 0, usa std de las medias (menos preciso) o 0
            pooled_std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0

        rows.append({
            'frequency_hz': int(freq),
            'U_mean_ms': mean_avg,
            'U_std_ms': pooled_std,
            'n_total': int(ns.sum()),
            'n_files': int(len(grp)),
            'aggregation': method,
        })
    return pd.DataFrame(rows).sort_values('frequency_hz').reset_index(drop=True)

# 9) Ajuste Polinomial Grado 2 (ya estaba)
def fit_poly2(x, y):
    x = np.asarray(x, dtype=float) # Asegurar que sean numpy arrays
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        logger.warning("Se requieren al menos 3 puntos para un ajuste polinomial de grado 2.")
        return float('nan'), float('nan'), float('nan'), float('nan')
    coeffs = np.polyfit(x, y, deg=2)  # a, b, c
    a, b2, c_poly = coeffs # Renombrar 'c' para evitar conflicto con la 'c' lineal
    yhat = a*x**2 + b2*x + c_poly
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - float(np.mean(y)))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 1e-9 else 1.0 # Evita división por cero si y es constante
    return a, b2, c_poly, r2

# ****** FUNCIONES FALTANTES AÑADIDAS AQUÍ ******
# Funciones del caso de estudio (Cell 11 del notebook)
def v_from_f_lineal(f, m, b):
    """Calcula V usando el modelo lineal V = m*f + b."""
    return m * float(f) + b

def f_from_v_lineal(V, m, b):
    """Calcula f usando la inversa del modelo lineal f = (V - b) / m."""
    return cu.invert_calibration(V, m, b) # Usa la función de calib_utils

def v_from_f_poly2(f, a, b2, c_poly):
    """Calcula V usando el modelo polinomial V = a*f^2 + b2*f + c_poly."""
    f = float(f)
    return a * f**2 + b2 * f + c_poly
# La inversa polinomial se calcula directamente en main()
# ************************************************

# --- Ejecución Principal ---

def main():
    # 2) Descubrimiento de datos
    logger.info("Buscando archivos de datos LDA...")
    try:
        # ASUNCIÓN: find_p2_data_root busca correctamente relativo a calib_utils
        # o encuentra la carpeta 'files/p2' subiendo desde el directorio actual.
        # Si falla, verifica la lógica en find_p2_data_root o la ubicación de 'files/p2'.
        p2_root = cu.find_p2_data_root()
        logger.info(f"Directorio de datos P2 encontrado en: {p2_root}")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Asegúrate de que la carpeta 'files/p2' exista en una ubicación detectable (p.ej., en {project_root} o un directorio padre).")
        return # Salir si no hay datos

    discovered = []
    for freq, fpath in cu.iter_lda_files(p2_root):
        discovered.append({'file': str(fpath), 'frequency_hz': int(freq)})

    if not discovered:
        logger.error(f"No se encontraron archivos .txt en '{p2_root}'. Verifica la ruta y los archivos.")
        return

    df_files = pd.DataFrame(discovered).sort_values(['frequency_hz', 'file']).reset_index(drop=True)
    print("\nPrimeros 5 archivos descubiertos:")
    print(df_files.head())

    expected_freqs = list(range(5, 60, 5))
    present = sorted(df_files['frequency_hz'].unique().tolist())
    missing = sorted(set(expected_freqs) - set(present))
    logger.info(f"Frecuencias detectadas: {present}")
    if missing:
        logger.warning(f"Frecuencias esperadas no encontradas: {missing}")
    print("-" * 30)

    # 3 & 4) Estadísticas por archivo
    try:
        per_file_stats = load_and_stats_per_file(df_files)
        print("\nEstadísticas por archivo (primeros 5):")
        print(per_file_stats.head())
        print("-" * 30)
    except ValueError as e:
        logger.error(e)
        return


    # 5) Agregación por frecuencia
    per_freq_weighted = aggregate_by_frequency(per_file_stats, method='weighted')
    per_freq_simple = aggregate_by_frequency(per_file_stats, method='simple') # Calculado pero no usado por defecto después
    print("\nEstadísticas agregadas por frecuencia (ponderado):")
    print(per_freq_weighted)
    print("-" * 30)

    # 6) Guardado de la tabla de calibración
    calib_csv = data_dir / 'calibracion.csv'
    calib_parquet = data_dir / 'calibracion.parquet'
    try:
        per_freq_weighted.to_csv(calib_csv, index=False)
        logger.info(f"Tabla de calibración guardada en: {calib_csv}")
    except Exception as e:
        logger.error(f"No se pudo guardar {calib_csv}: {e}")
    try:
        per_freq_weighted.to_parquet(calib_parquet, index=False)
        logger.info(f"Tabla de calibración guardada en: {calib_parquet}")
    except ImportError:
         logger.warning("No se guardó Parquet. Necesitas instalar 'pyarrow': pip install pyarrow")
    except Exception as ex:
        logger.warning(f"No se guardó Parquet (opcional): {ex}")
    print("-" * 30)

    # 7) Gráfica Scatter + Error Bars
    logger.info("Generando gráfica scatter con barras de error...")
    # Renombrar columnas para que coincidan con lo esperado por plot_calibration
    df_plot = per_freq_weighted.rename(columns={'U_mean_ms': 'mean_speed_ms', 'U_std_ms': 'std_speed_ms'})
    fig1, ax1 = cu.plot_calibration(df_plot,
                                    include_errorbars=True,
                                    title='Curva de Calibración (media ± std)')
    png1, svg1 = cu.save_figure(fig1, images_dir, 'calibracion_scatter')
    logger.info(f"Gráfica guardada en: {png1}, {svg1}")
    plt.close(fig1) # Cierra la figura para no mostrarla si no es necesario
    print("-" * 30)

    # 8) Regresión Lineal y Residuales
    logger.info("Realizando ajuste lineal y graficando...")
    x = per_freq_weighted['frequency_hz'].to_numpy()
    y = per_freq_weighted['U_mean_ms'].to_numpy()
    if len(x) < 2:
        logger.error("No hay suficientes puntos para realizar un ajuste lineal.")
        return

    m, b, r2 = cu.fit_linear(x, y)
    print(f"Ecuación (lineal): V = {m:.6f} f + {b:.6f}  |  R^2 = {r2:.6f}")

    # Gráfica de ajuste lineal (Re-creada aquí para claridad, aunque plot_calibration ya la hace)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.errorbar(x, y, yerr=per_freq_weighted['U_std_ms'].to_numpy(), fmt='o', capsize=4, label='Datos (media ± std)')
    xf_plot = np.linspace(x.min(), x.max(), 200) if len(x) > 1 else x # Evita error si solo hay 1 punto
    ax2.plot(xf_plot, m * xf_plot + b, 'r-', label=f'Ajuste lineal: V = {m:.3f} f + {b:.3f}\\nR² = {r2:.4f}')
    ax2.set_xlabel('Frecuencia del motor (Hz)')
    ax2.set_ylabel('Velocidad media (m/s)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    png2, svg2 = cu.save_figure(fig2, images_dir, 'calibracion_ajuste_lineal')
    logger.info(f"Gráfica de ajuste lineal guardada en: {png2}, {svg2}")
    plt.close(fig2)

    # Gráfica de residuales
    resid = y - (m * x + b)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.axhline(0, color='k', lw=1)
    ax3.plot(x, resid, 'o-')
    ax3.set_xlabel('Frecuencia del motor (Hz)')
    ax3.set_ylabel('Residual (m/s)')
    ax3.set_title('Residuales del ajuste lineal')
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    png3, svg3 = cu.save_figure(fig3, images_dir, 'calibracion_residuales')
    logger.info(f"Gráfica de residuales guardada en: {png3}, {svg3}")
    plt.close(fig3)
    print("-" * 30)

    # 9) Regresión Polinomial Grado 2 y Comparación
    logger.info("Realizando ajuste polinomial de grado 2 y graficando comparación...")
    a2, b2, c2, r2q = fit_poly2(x, y) # Usa la c renombrada c2 aquí internamente
    if not np.isnan(a2): # Verificar si el ajuste fue posible
        print(f"Polinomial grado 2: V = {a2:.6e} f^2 + {b2:.6f} f + {c2:.6f}  |  R^2 = {r2q:.6f}")

        # Gráfica de comparación
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.errorbar(x, y, yerr=per_freq_weighted['U_std_ms'].to_numpy(), fmt='o', capsize=4, label='Datos (media ± std)')
        ax4.plot(xf_plot, m * xf_plot + b, 'r-', label=f'Lineal R²={r2:.4f}')
        ax4.plot(xf_plot, a2 * xf_plot**2 + b2 * xf_plot + c2, 'g--', label=f'Grado 2 R²={r2q:.4f}')
        ax4.set_xlabel('Frecuencia del motor (Hz)')
        ax4.set_ylabel('Velocidad media (m/s)')
        ax4.set_title('Comparación de modelos')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        fig4.tight_layout()
        png4, svg4 = cu.save_figure(fig4, images_dir, 'calibracion_modelos_comparacion')
        logger.info(f"Gráfica de comparación guardada en: {png4}, {svg4}")
        plt.close(fig4)
    else:
        logger.warning("No se pudo realizar el ajuste polinomial de grado 2.")
    print("-" * 30)

    # 10) Widgets interactivos - Funcionalidad reemplazada por ejecución única.
    logger.info("La funcionalidad interactiva de widgets para reajuste no aplica en el script.")
    print("-" * 30)

    # 11) Caso de estudio - Cálculos con valores de ejemplo
    # ****** AHORA USA LAS FUNCIONES DEFINIDAS ARRIBA ******
    logger.info("Ejecutando caso de estudio con valores de ejemplo...")
    V_obj_ex = 12.0
    f_val_ex = 28.0
    L_ex = 0.1
    nu_ex = 1.5e-5

    # Lineal
    V_pred_lin = v_from_f_lineal(f_val_ex, m, b)
    f_req_lin = f_from_v_lineal(V_obj_ex, m, b)
    Re_lin = V_pred_lin * L_ex / max(nu_ex, 1e-12) if nu_ex > 0 else float('inf')
    print(f"[Lineal] Para f={f_val_ex:.2f} Hz, V ≈ {V_pred_lin:.3f} m/s")
    print(f"[Lineal] Para V={V_obj_ex:.3f} m/s -> f ≈ {f_req_lin:.3f} Hz")
    print(f"[Lineal] Re (V={V_pred_lin:.3f} m/s, L={L_ex}, nu={nu_ex:.1e}) ≈ {Re_lin:.3e}")

    # Polinomial Grado 2 (si aplica)
    if not np.isnan(a2):
        V_pred_poly = v_from_f_poly2(f_val_ex, a2, b2, c2)
        # Inversión (resolviendo ax^2 + bx + (c-V) = 0 para x=f)
        A_poly, B_poly, C_v = a2, b2, (c2 - V_obj_ex)
        disc = B_poly**2 - 4*A_poly*C_v
        f_req_poly = float('nan')
        if abs(A_poly) < 1e-9: # Caso lineal degenerado
             f_req_poly = (V_obj_ex - c2) / B_poly if abs(B_poly) > 1e-9 else float('nan')
        elif disc >= 0:
            # Calcular ambas raíces
            sqrt_disc = math.sqrt(disc)
            r1 = (-B_poly + sqrt_disc) / (2 * A_poly)
            r2 = (-B_poly - sqrt_disc) / (2 * A_poly)
            # Elegir la raíz positiva o la más cercana al rango de frecuencias original si ambas son positivas/negativas
            positive_roots = [r for r in [r1, r2] if r >= 0]
            if len(positive_roots) == 1:
                f_req_poly = positive_roots[0]
            elif len(positive_roots) == 2:
                # Si ambas son positivas, elige la que está dentro o más cerca del rango original [x.min(), x.max()]
                x_min, x_max = (x.min(), x.max()) if len(x)>0 else (0, float('inf'))
                in_range = [r for r in positive_roots if x_min <= r <= x_max]
                if in_range:
                    f_req_poly = in_range[0] # Si una está en rango, úsala (raro que ambas lo estén si la curva es monótona)
                else:
                     # Si ninguna está en rango, elige la más cercana a la media o un valor típico como f_val_ex
                     f_req_poly = min(positive_roots, key=lambda r: abs(r - np.mean(x) if len(x)>0 else f_val_ex))
            else: # Ambas raíces negativas o complejas (disc<0 ya manejado)
                 # Podrías devolver la menos negativa si tiene sentido físico, o NaN
                 f_req_poly = max(r1, r2) # La menos negativa (más cercana a 0)

        Re_poly = V_pred_poly * L_ex / max(nu_ex, 1e-12) if nu_ex > 0 else float('inf')
        print(f"[Grado 2] Para f={f_val_ex:.2f} Hz, V ≈ {V_pred_poly:.3f} m/s")
        print(f"[Grado 2] Para V={V_obj_ex:.3f} m/s -> f ≈ {f_req_poly:.3f} Hz")
        print(f"[Grado 2] Re (V={V_pred_poly:.3f} m/s, L={L_ex}, nu={nu_ex:.1e}) ≈ {Re_poly:.3e}")

    print("-" * 30)

    # 12) Exportación de artefactos JSON
    logger.info("Generando resumen JSON...")
    # Obtener versiones de librerías
    try: import matplotlib; matplotlib_version = matplotlib.__version__
    except: matplotlib_version = "No disponible"
    try: import seaborn; seaborn_version = seaborn.__version__
    except: seaborn_version = "No disponible"

    summary = {
        'timestamp': dt.datetime.now().isoformat(),
        'model_linear': {'m': m, 'b': b, 'R2': r2},
        'model_poly2': {'a': a2, 'b': b2, 'c': c2, 'R2': r2q} if not np.isnan(a2) else None,
        'frequencies_hz_detected': present,
        'frequencies_hz_processed': per_freq_weighted['frequency_hz'].tolist(),
        'data_files_count': int(df_files.shape[0]), # Total descubierto
        'valid_files_count': int(len(per_file_stats)), # Total procesado
        'calibration_table_csv': str(calib_csv.resolve()),
        'calibration_table_parquet': str(calib_parquet.resolve()) if calib_parquet.exists() else None,
        'images_dir': str(images_dir.resolve()),
        'image_files': {p.name: str(p.resolve()) for p in images_dir.glob('calibracion_*.png')},
         'libs': {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'matplotlib': matplotlib_version,
            'seaborn': seaborn_version,
         },
    }
    summary_path = data_dir / 'calibracion_summary.json'
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Resumen guardado en: {summary_path}")
    except Exception as e:
        logger.error(f"No se pudo guardar {summary_path}: {e}")
    print("-" * 30)

    # 13) Actualización de p2.md (Opcional)
    logger.info("Intentando actualizar p2.md...")
    # ASUNCIÓN: p2.md está en la raíz del proyecto. Ajusta si es diferente.
    p2_md_path = project_root / 'p2.md'
    backup_path = p2_md_path.with_suffix('.md.bak')
    try:
        original_text = ''
        if p2_md_path.exists():
            original_text = p2_md_path.read_text(encoding='utf-8', errors='ignore')
            # Crear backup
            backup_path.write_text(original_text, encoding='utf-8')
            logger.info(f'Respaldo de {p2_md_path.name} creado: {backup_path.name}')

        # Intentar eliminar sección antigua si existe
        base_text = original_text
        start_marker = '## Resultados Automáticos'
        end_marker = '\n## ' # Asume que hay otra sección después
        start_idx = base_text.find(start_marker)
        if start_idx != -1:
             end_idx = base_text.find(end_marker, start_idx + len(start_marker))
             if end_idx != -1:
                 # Conserva texto antes del marcador y después del siguiente marcador
                 base_text = base_text[:start_idx].strip() + '\n\n' + base_text[end_idx:].strip()
             else: # Si es la última sección
                 base_text = base_text[:start_idx].strip() # Elimina desde el marcador hasta el final
        else:
             # Si no hay marcador, usa el texto original como base
             base_text = original_text.strip()

        # Construir nueva sección
        section = []
        section.append('\n\n## Resultados Automáticos (generados por script)\n') # Asegura un salto de línea antes
        section.append(f"Fecha de ejecución: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        section.append(f"Ecuación (lineal): `V [m/s] = {m:.6f} f [Hz] + {b:.6f}` (R² = {r2:.6f})\n")
        if not np.isnan(a2):
            section.append(f"Ecuación (grado 2): `V = {a2:.4e} f² + {b2:.6f} f + {c2:.6f}` (R² = {r2q:.6f})\n")
        section.append('\nTabla de calibración (agregación ponderada):\n')
        # Usar floatfmt para controlar decimales en markdown
        section.append(per_freq_weighted.to_markdown(index=False, floatfmt=".4f"))
        section.append('\n\nFiguras generadas en `data/images`:\n')

        # Crear rutas relativas al archivo md para las imágenes
        try:
            # Calcula la ruta relativa desde el directorio del MD al directorio de imágenes
            img_rel_dir = Path(os.path.relpath(images_dir.resolve(), p2_md_path.parent.resolve()))
        except ValueError: # Si están en unidades diferentes en Windows
             # Fallback: usa solo el nombre del directorio de imágenes (asume que está al mismo nivel o usa ruta absoluta)
             # O mejor, asume una estructura común como 'data/images' relativo al proyecto
             img_rel_dir = Path('data/images') # Asume estructura común si relpath falla

        image_files_sorted = sorted(summary.get('image_files', {}).keys()) # Usa .get para evitar error si no hay imágenes
        for img_name in image_files_sorted:
             if img_name.endswith('.png'): # Solo incluir PNGs en markdown por compatibilidad
                 alt_text = img_name.replace('.png','').replace('calibracion_','').replace('_',' ').title()
                 # Construye la ruta relativa usando la carpeta relativa y el nombre del archivo
                 md_img_path = (img_rel_dir / img_name).as_posix() # Usa / como separador
                 section.append(f"- {alt_text}: `![]({md_img_path})`") # Sintaxis Markdown para imagen

        section_text = '\n'.join(section)
        # Añade la nueva sección al final del texto base (que ya tiene la sección vieja eliminada)
        new_text = base_text + '\n' + section_text + '\n' # Asegura saltos de línea
        p2_md_path.write_text(new_text, encoding='utf-8')
        logger.info(f'{p2_md_path.name} ({p2_md_path.resolve()}) actualizado con resultados automáticos.')

    except Exception as ex:
        logger.warning(f'No se pudo actualizar {p2_md_path} automáticamente: {ex}')
        # Opcional: restaurar desde backup si la actualización falló
        # if backup_path.exists():
        #     backup_path.replace(p2_md_path)
        #     logger.info(f'Restaurado {p2_md_path.name} desde backup debido a error.')
    print("-" * 30)

    # 14) Comprobaciones automáticas
    logger.info("Realizando checks básicos...")
    try:
        assert not df_files.empty, 'No se encontraron archivos en files/p2'
        assert not per_file_stats.empty, 'No se pudieron procesar estadísticas por archivo'
        assert per_file_stats['n_samples'].min() > 0, 'Algún archivo válido no tiene muestras'
        assert not per_freq_weighted.empty, 'No se pudieron agregar estadísticas por frecuencia'
        assert per_freq_weighted.shape[0] >= 2, 'Se requieren al menos 2 frecuencias para ajuste lineal'
        if not np.isnan(a2):
             assert per_freq_weighted.shape[0] >= 3, 'Se requieren al menos 3 frecuencias para ajuste grado 2'
        assert 0.0 <= r2 <= 1.0, f'R^2 lineal fuera de [0,1]: {r2}'
        if not np.isnan(r2q):
             assert 0.0 <= r2q <= 1.0, f'R^2 grado 2 fuera de [0,1]: {r2q}'
        logger.info('Checks básicos: OK')
    except AssertionError as e:
        logger.error(f"Check fallido: {e}")
    except Exception as e:
        logger.error(f"Error durante los checks: {e}")
    print("-" * 30)

    # 15) Detección de cabecera (ejemplo)
    if not df_files.empty:
        sample_file_path = Path(df_files.iloc[0]['file'])
        try:
            skip = cu._detect_header_rows(sample_file_path)
            logger.info(f"Diagnóstico: Fila de encabezado ('Row#') detectada en línea {skip+1} (skiprows={skip}) para archivo de ejemplo: {sample_file_path.name}")
        except Exception as ex:
            logger.warning(f'No se pudo detectar encabezado en {sample_file_path.name}: {ex}')
    else:
        logger.warning("No hay archivos descubiertos para el diagnóstico de cabecera.")
    print("-" * 30)

    # 16) Generación de histogramas (uno por frecuencia)
    logger.info("Generando histogramas de distribución (uno por frecuencia)...")
    hist_images_dir = images_dir / 'histograms' # Subdirectorio opcional para histogramas
    cu.ensure_dir(hist_images_dir)

    files_by_freq = per_file_stats.groupby('frequency_hz')['file'].apply(list).to_dict()
    processed_freqs = sorted(per_freq_weighted['frequency_hz'].unique().tolist()) # Frecuencias que realmente se usaron

    for freq_val in processed_freqs:
        files_for_freq = files_by_freq.get(freq_val, [])
        if files_for_freq:
            # Grafica solo el primer archivo válido encontrado para esa frecuencia
            file_to_plot = Path(files_for_freq[0])
            try:
                s_hist = cu.read_lda_speeds(file_to_plot, abs_values=True)
                if not s_hist.empty:
                    fig_hist, ax_hist = cu.plot_frequency_distribution(freq_val, s_hist, bins=50) # Bins fijos
                    # Guardar en el subdirectorio
                    png_hist, svg_hist = cu.save_figure(fig_hist, hist_images_dir, f"hist_{freq_val}Hz")
                    logger.info(f"Histograma para {freq_val}Hz guardado en: {png_hist.name}")
                    plt.close(fig_hist) # Cierra la figura
                else:
                     logger.warning(f"Archivo {file_to_plot.name} para {freq_val}Hz estaba vacío después de leer velocidades.")
            except Exception as e:
                logger.warning(f"No se pudo generar histograma para {freq_val}Hz (archivo: {file_to_plot.name}): {e}")
        else:
            # Esto no debería ocurrir si usamos processed_freqs, pero por si acaso
            logger.warning(f"No se encontraron archivos en per_file_stats para la frecuencia {freq_val}Hz procesada.")

    logger.info("Proceso completado.")

# --- Punto de entrada del script ---
if __name__ == "__main__":
    # Añadir manejo básico de argumentos si fuera necesario en el futuro
    # import argparse
    # parser = argparse.ArgumentParser(description="Script de calibración del túnel de viento.")
    # args = parser.parse_args()
    main()