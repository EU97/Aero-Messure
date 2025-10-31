# -*- coding: utf-8 -*-
"""
Script para análisis PIV (Velocimetría de Partículas por Imagen) usando Flujo Óptico.

Este script convierte la libreta P06_PIV.ipynb en un módulo ejecutable.
Realiza el cálculo de flujo óptico (Farnebäck), calcula magnitud y vorticidad,
y genera visualizaciones y reportes.

Nuevas características añadidas:
- Búsqueda automática y procesamiento en bucle de TODOS los pares de imágenes 
  (convención basename_0.tif y basename_1.tif).
- Guardado de datos vectoriales en formato CSV.
- Análisis de puntos extremos (máx/mín velocidad y vorticidad).
- Generación de histogramas de distribución de velocidad y vorticidad.

Estructura de directorios esperada (según corrección):
Aero-Messure/
├── Practicas/
│   └── P06_PIV/  (<- PROYECT_DIR)
│       ├── src/  (<- SCRIPT_DIR)
│       │   └── piv_analysis.py (<- SCRIPT_FILE)
│       ├── data/ (<- DATA_DIR)
│       │   └── (salida_vectores.csv)
│       │   └── images/ (<- OUTPUT_DIR)
│       │       └── (salida_figuras.png/svg)
├── files/
│   └── p6/ (<- P6_FILES_DIR)
│       ├── B005_0.tif
│       ├── B005_1.tif
│       ├── vortex_street_0.tif
│       ├── vortex_street_1.tif
│       └── ...

Ejecución desde la terminal (dentro del dir 'src'):
> python piv_analysis.py
(Opcional, para cambiar parámetros de Farnebäck o visualización)
> python piv_analysis.py --skip 15 --winsize 20
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import pandas as pd
from pathlib import Path
import glob
import argparse  # Reemplaza a ipywidgets

# --- 1. CONFIGURACIÓN DE RUTAS (AJUSTADO A LA ESTRUCTURA CORREGIDA) ---

# Ruta absoluta al script actual
SCRIPT_FILE = Path(__file__).resolve()
# Ruta al directorio 'src'
SCRIPT_DIR = SCRIPT_FILE.parent
# Ruta al directorio raíz del proyecto (P06_PIV)
PROYECT_DIR = SCRIPT_DIR.parent
# Ruta al directorio 'data' (salidas)
DATA_DIR = PROYECT_DIR / 'data'
# Ruta al directorio de salida de imágenes
OUTPUT_DIR = DATA_DIR / 'images'

# Navegamos "hacia arriba" para encontrar la raíz 'Aero-Messure'
AERO_MESSURE_DIR = PROYECT_DIR.parent.parent
# Ruta a los archivos de entrada (p6)
P6_FILES_DIR = AERO_MESSURE_DIR / 'files' / 'p6'

# --- 2. FUNCIONES ORIGINALES Y MODIFICADAS ---

def encontrar_pares_piv(p6_files_dir: Path):
    """
    Escanea el directorio de entrada y encuentra todos los pares PIV
    (basename_0, basename_1).
    Devuelve una lista de 'basenames' válidos (ej. ['B005', 'vortex_street']).
    """
    print(f"Buscando pares de imágenes en: {p6_files_dir}")
    pares_encontrados = []
    
    # Busca todos los archivos que terminen en _0.*
    archivos_0 = list(p6_files_dir.glob('*_0.*'))
    
    if not archivos_0:
        print(f"Advertencia: No se encontraron archivos '_0' (ej. B005_0.tif) en {p6_files_dir}.")
        return []

    for path_0 in archivos_0:
        # Extraer el basename: '.../B005_0.tif' -> 'B005_0' -> 'B005'
        basename = path_0.stem.rsplit('_0', 1)[0]
        
        # Verificar si existe el archivo _1 correspondiente
        # Usamos glob para ser flexibles con la extensión (tif, png, etc.)
        path_1_glob = list(p6_files_dir.glob(f'{basename}_1.*'))
        
        if path_1_glob:
            pares_encontrados.append(basename)
            print(f"  > Par encontrado: {basename} (archivos {path_0.name} y {path_1_glob[0].name})")
        else:
            print(f"  > Advertencia: Se encontró '{path_0.name}' pero falta su par '{basename}_1.*'.")
            
    return pares_encontrados

def cargar_par_imagenes(basename: str, p6_files_dir: Path):
    """
    Carga un par de imágenes (_0 y _1) desde el directorio de archivos P6.
    (MODIFICADA para buscar _0 y _1)
    """
    img_a_path = list(p6_files_dir.glob(f'{basename}_0.*'))
    img_b_path = list(p6_files_dir.glob(f'{basename}_1.*'))

    # Esta comprobación es redundante si 'encontrar_pares_piv' ya la hizo,
    # pero es una buena práctica de seguridad.
    if not img_a_path or not img_b_path:
        print(f"Error: No se encontraron imágenes para el basename '{basename}' en {p6_files_dir}")
        return None, None, None

    img1 = cv2.imread(str(img_a_path[0]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img_b_path[0]), cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"Error: No se pudieron leer las imágenes para {basename}.")
        return None, None, None

    print(f"Imágenes cargadas: {img_a_path[0].name} y {img_b_path[0].name}")
    return img1, img2, basename

def calcular_flujo_optico(img1, img2, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma):
    """Calcula el flujo óptico denso usando el algoritmo de Farnebäck."""
    print("Calculando flujo óptico de Farnebäck...")
    flow = cv2.calcOpticalFlowFarneback(
        img1, img2, None,
        pyr_scale, levels, winsize,
        iterations, poly_n, poly_sigma,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    return flow

def calcular_metricas_flujo(flow):
    """Calcula u, v, magnitud de velocidad y vorticidad."""
    print("Calculando métricas de flujo (u, v, magnitud, vorticidad)...")
    u = flow[..., 0]
    v = flow[..., 1]
    magnitude = np.sqrt(u**2 + v**2)

    grad_v = np.gradient(v) # (dv/dy, dv/dx)
    grad_u = np.gradient(u) # (du/dy, du/dx)
    dv_dx = grad_v[1]
    du_dy = grad_u[0]
    vorticity = dv_dx - du_dy
    
    return u, v, magnitude, vorticity

def preparar_visualizacion(shape, skip):
    """Prepara las mallas (grids) X, Y para visualización."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    y_q, x_q = np.mgrid[skip//2:h:skip, skip//2:w:skip]
    return x, y, (x_q, y_q)

def _guardar_figura(fig, base_name, tag, output_dir):
    """Función helper para guardar figuras en PNG y SVG."""
    filename_png = output_dir / f"{base_name}_{tag}.png"
    filename_svg = output_dir / f"{base_name}_{tag}.svg"
    fig.savefig(filename_png, dpi=150, bbox_inches='tight')
    fig.savefig(filename_svg, bbox_inches='tight')
    print(f"Figura guardada en: {filename_png}")

def visualizar_vectores(x_q, y_q, u, v, skip, img, base_name, output_dir):
    """Visualiza el campo de vectores (quiver plot)."""
    print("Generando visualización de vectores (quiver)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    u_q = u[skip//2:img.shape[0]:skip, skip//2:img.shape[1]:skip]
    v_q = v[skip//2:img.shape[0]:skip, skip//2:img.shape[1]:skip]

    ax.imshow(img, cmap='gray', alpha=0.6)
    ax.quiver(x_q, y_q, u_q, v_q, color='blue',
              scale=None, scale_units='xy', angles='xy', units='xy',
              width=0.1, headwidth=3, headlength=4, alpha=0.8)
    
    ax.set_title(f'Campo de Vectores (Quiver) - {base_name}')
    ax.set_xlabel('Posición X (píxeles)')
    ax.set_ylabel('Posición Y (píxeles)')
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    
    _guardar_figura(fig, base_name, "vectores", output_dir)
    plt.close(fig)

def visualizar_mapa_calor(data, x, y, title, cmap, tag, base_name, output_dir, points_to_mark=None):
    """Visualiza un mapa de calor (imshow) para una métrica (magnitud, vorticidad)."""
    print(f"Generando mapa de calor: {title}...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'vorticidad' in tag.lower():
        vmax = np.nanmax(np.abs(data))
        vmin = -vmax
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(data, cmap=cmap, extent=(x.min(), x.max(), y.min(), y.max()),
                   origin='lower', norm=norm, interpolation='bilinear')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(title)
    
    if points_to_mark:
        for (y_idx, x_idx), marker, label in points_to_mark:
            ax.plot(x_idx, y_idx, marker, markersize=10, label=label, markeredgecolor='black')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    ax.set_title(f'{title} - {base_name}')
    ax.set_xlabel('Posición X (píxeles)')
    ax.set_ylabel('Posición Y (píxeles)')
    ax.set_aspect('equal')
    
    _guardar_figura(fig, base_name, tag, output_dir)
    plt.close(fig)

def visualizar_streamlines(x, y, u, v, img, base_name, output_dir):
    """Visualiza líneas de corriente (streamplot)."""
    print("Generando visualización de líneas de corriente...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    magnitude = np.sqrt(u**2 + v**2)
    ax.imshow(img, cmap='gray', alpha=0.4, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower')
    strm = ax.streamplot(x, y, u, v, color=magnitude, cmap=cm.jet,
                         linewidth=1.5, density=1.5, arrowstyle='->')
    
    cbar = fig.colorbar(strm.lines, ax=ax, shrink=0.8)
    cbar.set_label('Magnitud de Velocidad')
    ax.set_title(f'Líneas de Corriente (Streamlines) - {base_name}')
    ax.set_xlabel('Posición X (píxeles)')
    ax.set_ylabel('Posición Y (píxeles)')
    ax.set_aspect('equal')
    
    _guardar_figura(fig, base_name, "streamlines", output_dir)
    plt.close(fig)

def generar_reporte_md(base_name, output_dir):
    """Genera un archivo Markdown simple enlazando las figuras generadas."""
    print("Generando reporte markdown...")
    tags = ['vectores', 'magnitud', 'vorticidad', 'streamlines', 'hist_magnitud', 'hist_vorticidad']
    lines = [f"# Reporte PIV: {base_name}\n"]
    
    for tag in tags:
        png_file = f"{base_name}_{tag}.png"
        svg_file = f"{base_name}_{tag}.svg"
        if (output_dir / png_file).exists():
            lines.append(f"## {tag.replace('_', ' ').capitalize()}")
            lines.append(f"![{tag}]({output_dir.name}/{png_file})")
            lines.append(f"[Descargar SVG]({output_dir.name}/{svg_file})")
            lines.append("\n")

    md_content = '\n'.join(lines)
    out_md = output_dir.parent / f'{base_name}_reporte.md'
    out_md.write_text(md_content, encoding='utf-8')
    print(f"Reporte generado en: {out_md}")

# --- 3. NUEVAS FUNCIONES DE ANÁLISIS ---

def guardar_datos_csv(u, v, magnitude, vorticity, base_name, data_dir):
    """Guarda los campos 2D (u, v, mag, vort) en un archivo CSV."""
    print(f"Guardando datos en CSV en {data_dir}...")
    
    h, w = u.shape
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    df = pd.DataFrame({
        'x': x_grid.flatten(),
        'y': y_grid.flatten(),
        'u': u.flatten(),
        'v': v.flatten(),
        'magnitud': magnitude.flatten(),
        'vorticidad': vorticity.flatten()
    })
    
    csv_path = data_dir / f"{base_name}_datos_piv.csv"
    df.to_csv(csv_path, index=False)
    print(f"Datos guardados en: {csv_path}")

def visualizar_histograma(metric_data, title, tag, base_name, output_dir):
    """Crea y guarda un histograma de la distribución de una métrica."""
    print(f"Generando histograma: {title}...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    data_flat = metric_data.flatten()
    data_flat = data_flat[np.isfinite(data_flat)]
    
    ax.hist(data_flat, bins=100, density=True, color='c', edgecolor='k', alpha=0.7)
    mean_val = np.mean(data_flat)
    ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
    
    ax.set_title(f'Distribución de {title} - {base_name}')
    ax.set_xlabel(title)
    ax.set_ylabel('Densidad de Probabilidad')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    _guardar_figura(fig, base_name, tag, output_dir)
    plt.close(fig)

def analizar_puntos_extremos(magnitude, vorticity):
    """Encuentra y reporta los puntos de máx/mín magnitud y vorticidad."""
    print("\n--- Análisis de Puntos Extremos ---")
    
    max_mag_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    min_mag_idx = np.unravel_index(np.argmin(magnitude), magnitude.shape)
    print(f"Velocidad Máxima: {magnitude[max_mag_idx]:.3f} en (y={max_mag_idx[0]}, x={max_mag_idx[1]})")

    max_vort_idx = np.unravel_index(np.argmax(vorticity), vorticity.shape)
    min_vort_idx = np.unravel_index(np.argmin(vorticity), vorticity.shape)
    print(f"Vorticidad Máxima (+): {vorticity[max_vort_idx]:.3f} en (y={max_vort_idx[0]}, x={max_vort_idx[1]}) (Giro anti-horario)")
    print(f"Vorticidad Mínima (-): {vorticity[min_vort_idx]:.3f} en (y={min_vort_idx[0]}, x={min_vort_idx[1]}) (Giro horario)")
    print("-----------------------------------\n")

    return {
        "max_mag": (max_mag_idx, 'w*', 'V máx'), 
        "min_mag": (min_mag_idx, 'wx', 'V mín'),
        "max_vort": (max_vort_idx, 'c+', 'Vort máx (+)'),
        "min_vort": (min_vort_idx, 'cx', 'Vort mín (-)')
    }


# --- 4. FUNCIÓN PRINCIPAL Y EJECUCIÓN ---

def main(args):
    """
    Flujo principal del script de análisis PIV.
    (MODIFICADO para buscar y procesar todos los pares en un bucle)
    """
    
    # Asegurar que los directorios de salida existan
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True) # Para el CSV

    print("--- Iniciando Análisis PIV ---")
    print(f"Directorio del Script: {SCRIPT_DIR}")
    print(f"Directorio del Proyecto: {PROYECT_DIR}")
    print(f"Directorio de Datos (Entrada): {P6_FILES_DIR}")
    print(f"Directorio de Datos (Salida CSV): {DATA_DIR}")
    print(f"Directorio de Imágenes (Salida): {OUTPUT_DIR}")
    print("----------------------------------\n")
    
    # --- ¡NUEVA LÓGICA DE BÚSQUEDA DE PARES! ---
    basenames_a_procesar = encontrar_pares_piv(P6_FILES_DIR)
    
    if not basenames_a_procesar:
        print("No se encontraron pares válidos (ej. basename_0.tif y basename_1.tif) para procesar.")
        print("Terminando script.")
        return
        
    print(f"\nSe procesarán {len(basenames_a_procesar)} pares de imágenes.")

    # Guardar los parámetros de Farnebäck (se usarán en todos los bucles)
    params_farneback = {
        "pyr_scale": args.pyr_scale,
        "levels": args.levels,
        "winsize": args.winsize,
        "iterations": args.iterations,
        "poly_n": args.poly_n,
        "poly_sigma": args.poly_sigma
    }
    
    # --- ¡NUEVO BUCLE DE PROCESAMIENTO! ---
    for i, basename in enumerate(basenames_a_procesar):
        print(f"\n---=================================================---")
        print(f"--- Procesando Par {i+1}/{len(basenames_a_procesar)}: {basename}")
        print(f"---=================================================---")

        # 1. Cargar Imágenes
        img1, img2, base_name = cargar_par_imagenes(basename, P6_FILES_DIR)
        if img1 is None:
            print(f"Error fatal al cargar {basename}, saltando al siguiente par.")
            continue # Salta al siguiente basename en el bucle

        # 2. Calcular Flujo Óptico
        flow = calcular_flujo_optico(img1, img2, **params_farneback)

        # 3. Calcular Métricas
        u, v, magnitude, vorticity = calcular_metricas_flujo(flow)
        
        # 4. Preparar Mallas de Visualización
        x, y, (x_q, y_q) = preparar_visualizacion(img1.shape, args.skip)

        # 5. Guardar datos en CSV
        guardar_datos_csv(u, v, magnitude, vorticity, base_name, DATA_DIR)

        # 6. Analizar puntos extremos
        extremos = analizar_puntos_extremos(magnitude, vorticity)
        mag_points = [extremos["max_mag"], extremos["min_mag"]]
        vort_points = [extremos["max_vort"], extremos["min_vort"]]

        # 7. Generar Visualizaciones
        visualizar_vectores(x_q, y_q, u, v, args.skip, img1, base_name, OUTPUT_DIR)
        
        visualizar_mapa_calor(magnitude, x, y, 'Magnitud de Velocidad |V|', cm.jet, 
                              'magnitud', base_name, OUTPUT_DIR,
                              points_to_mark=mag_points)
        
        visualizar_mapa_calor(vorticity, x, y, 'Vorticidad $\omega_z$', cm.seismic, 
                              'vorticidad', base_name, OUTPUT_DIR,
                              points_to_mark=vort_points)
        
        visualizar_streamlines(x, y, u, v, img1, base_name, OUTPUT_DIR)
        
        # 8. Generar Histogramas
        visualizar_histograma(magnitude, 'Magnitud de Velocidad', 'hist_magnitud', base_name, OUTPUT_DIR)
        visualizar_histograma(vorticity, 'Vorticidad', 'hist_vorticidad', base_name, OUTPUT_DIR)

        # 9. Generar Reporte MD
        generar_reporte_md(base_name, OUTPUT_DIR)
        
        print(f"--- Fin del procesamiento de {basename} ---")

    print("\n--- Análisis PIV Completado Exitosamente para todos los pares ---")


if __name__ == "__main__":
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Análisis PIV con Flujo Óptico (Farnebäck)")
    
    # --- Argumento 'basename' ELIMINADO ---
    
    # Argumentos de visualización
    parser.add_argument('--skip', type=int, default=10,
                        help='Factor de submuestreo (skip) para el gráfico de vectores (quiver).')

    # Argumentos de Farnebäck (se aplican a todos los pares procesados)
    parser.add_argument('--pyr_scale', type=float, default=0.5, 
                        help='Escala de la pirámide (< 1).')
    parser.add_argument('--levels', type=int, default=3, 
                        help='Número de niveles de la pirámide.')
    parser.add_argument('--winsize', type=int, default=15, 
                        help='Tamaño de la ventana de promedio.')
    parser.add_argument('--iterations', type=int, default=3, 
                        help='Iteraciones en cada nivel de la pirámide.')
    parser.add_argument('--poly_n', type=int, default=5, 
                        help='Tamaño del vecindario polinomial (5 o 7).')
    parser.add_argument('--poly_sigma', type=float, default=1.2, 
                        help='Sigma de la gaussiana para suavizado (1.1 o 1.5).')

    args = parser.parse_args()
    
    main(args)