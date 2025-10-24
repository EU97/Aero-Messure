#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de Python para el Análisis de Imagen (Fractografía SEM)
Convertido desde P05_Analisis_Imagen.ipynb.

Este script realiza el análisis de procesamiento de imágenes (Histograma,
CLAHE, Sobel, Canny, Otsu, Morfología) para todas las imágenes
encontradas en el directorio de entrada y guarda los resultados
en el directorio de salida.

LIMITACIÓN IMPORTANTE:
La funcionalidad de "Medición Interactiva" (la clase CalculadoraDistancia
y los widgets de ipywidgets) ha sido eliminada. Esta funcionalidad
depende intrínsecamente del entorno interactivo de Jupyter Notebook
y no se puede replicar en un script de Python simple sin construir
una aplicación de GUI completa (por ejemplo, con Tkinter o PyQt).

El script se enfoca en la parte automatizada del análisis.
"""

# --- Importación de Librerías ---
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import glob
import re
import sys  # Importado para sys.exit

# Importaciones de Scikit-image
from skimage import exposure, filters, measure, morphology
from skimage.io import imread
from skimage.util import img_as_float, img_as_ubyte
from skimage.feature import canny

# Importaciones de SciPy
from scipy import ndimage as ndi

## --- Configuración de Directorios ---

# NOTA: Rutas ajustadas para ejecutarse desde la carpeta raíz
# AERO-MESSURE/
# ├── analisis_imagen.py      <-- (Este script)
# ├── files/
# │   └── p5/                 <-- (Imágenes de entrada)
# └── Practicas/
#     └── P05_Analisis_Imagen/
#         └── data/
#             └── images/     <-- (Resultados)

# Directorio donde están tus imágenes de entrada
# (Desde la raíz, solo entramos a 'files/p5')
FILES_DIR = Path('files/p5') 

# Directorio donde se guardarán los resultados
# (Desde la raíz, entramos a 'Practicas/P05.../data/images')
RESULTS_DIR = Path('Practicas/P05_Analisis_Imagen/data/images')
# --- Funciones Auxiliares (Carga y Guardado) ---

def cargar_imagen_gris(path):
    """Carga una imagen, la convierte a gris y la normaliza a float [0, 1]."""
    try:
        # Usamos as_gray=True para que imread maneje la conversión
        img_gray = imread(path, as_gray=True)
        # 'imread' con 'as_gray=True' ya devuelve float [0, 1]
        img_norm = img_as_float(img_gray)
        return img_norm
    except Exception as e:
        print(f"Error cargando la imagen {path}: {e}")
        return None

def save_fig(path): 
    """Guarda la figura actual en el directorio de resultados."""
    try:
        full_path = RESULTS_DIR / path
        plt.savefig(full_path, bbox_inches='tight', dpi=150)
        print(f"  > Figura guardada en: {full_path.name}")
    except Exception as e:
        print(f"Error guardando figura: {e}")

# --- Funciones de Procesamiento de Imagen ---

def mostrar_histograma(ax, img_gray_norm, title='Histograma'):
    """Muestra el histograma de una imagen en un eje (ax) dado."""
    ax.hist(img_gray_norm.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    ax.set_title(title)
    ax.set_xlabel('Intensidad de Píxel (0-1)')
    ax.set_ylabel('Frecuencia')
    ax.set_xlim(0, 1)

def procesar_clahe(img_gray_norm, clip_limit=0.03):
    """Aplica CLAHE (Ecualización Adaptativa) a la imagen."""
    return exposure.equalize_adapthist(img_gray_norm, clip_limit=clip_limit)

def procesar_sobel(img_gray_norm):
    """Aplica el filtro Sobel para detección de bordes."""
    return filters.sobel(img_gray_norm)

def procesar_canny(img_gray_norm, sigma=1.0):
    """Aplica el detector de bordes Canny."""
    return canny(img_gray_norm, sigma=sigma)

def procesar_otsu(img_gray_norm):
    """Aplica umbralización global de Otsu."""
    thresh = filters.threshold_otsu(img_gray_norm)
    return img_gray_norm > thresh

def procesar_morfologia(img_bin, op_type='opening', disk_size=3):
    """Aplica operaciones morfológicas (opening, closing, erosion, dilation)."""
    selem = morphology.disk(disk_size)
    if op_type == 'opening':
        return morphology.binary_opening(img_bin, selem)
    elif op_type == 'closing':
        return morphology.binary_closing(img_bin, selem)
    elif op_type == 'erosion':
        return morphology.binary_erosion(img_bin, selem)
    elif op_type == 'dilation':
        return morphology.binary_dilation(img_bin, selem)
    return img_bin

def procesar_etiquetado(img_bin):
    """Etiqueta regiones conectadas e imprime sus propiedades."""
    label_image, num_features = ndi.label(img_bin)
    props = measure.regionprops_table(label_image, 
                                      properties=['label', 'area', 'perimeter', 'eccentricity', 'major_axis_length'])
    print(f"\n--- Análisis de Regiones (sobre Otsu+Opening) ---")
    print(f"Número de objetos encontrados: {num_features}")
    # print("Propiedades de los objetos:")
    # print(props) # Descomentar para ver la tabla de propiedades
    return label_image, props

def run_full_analysis(img_gray, base_filename):
    """Ejecuta todos los análisis y los muestra/guarda."""
    
    # Esta función ya no necesita 'out_analysis_plots'
    # Los 'print' irán directamente a la consola.
    
    print(f"--- Iniciando análisis para: {base_filename} ---")
    
    # Crear una figura para todos los análisis
    fig_analysis, axs = plt.subplots(3, 2, figsize=(10, 12))
    axs = axs.ravel() # Aplanar el array de ejes
    
    # a) Histograma (sobre la imagen float original)
    mostrar_histograma(axs[0], img_gray, 'Histograma Original')
    
    # b) CLAHE (creado como float)
    img_clahe = procesar_clahe(img_gray)
    axs[1].imshow(img_clahe, cmap='gray')
    axs[1].set_title('Ecualización (CLAHE)')
    axs[1].axis('off')
    save_fig(f"{base_filename}_01_clahe.png")

    # Convertir la imagen CLAHE a uint8 [0, 255]
    # Esta versión se usará para las funciones (sobel, canny, otsu)
    img_clahe_u8 = img_as_ubyte(img_clahe)

    # c) Sobel (usando la versión uint8)
    img_sobel = procesar_sobel(img_clahe_u8)
    axs[2].imshow(img_sobel, cmap='gray')
    axs[2].set_title('Filtro Sobel (sobre CLAHE)')
    axs[2].axis('off')
    save_fig(f"{base_filename}_02_sobel.png")

    # d) Canny (usando la versión uint8)
    img_canny = procesar_canny(img_clahe_u8, sigma=2.0)
    axs[3].imshow(img_canny, cmap='binary')
    axs[3].set_title('Bordes Canny (sigma=2)')
    axs[3].axis('off')
    save_fig(f"{base_filename}_03_canny.png")

    # e) Otsu (usando la versión uint8)
    img_otsu = procesar_otsu(img_clahe_u8)
    axs[4].imshow(img_otsu, cmap='gray')
    axs[4].set_title('Umbralización Otsu (sobre CLAHE)')
    axs[4].axis('off')
    save_fig(f"{base_filename}_04_otsu.png")

    # f) Morfología (Opening, sobre la imagen de Otsu que ya es booleana)
    img_opened = procesar_morfologia(img_otsu, op_type='opening', disk_size=2)
    axs[5].imshow(img_opened, cmap='gray')
    axs[5].set_title('Morfología (Opening, disk=2)')
    axs[5].axis('off')
    save_fig(f"{base_filename}_05_opening.png")
    
    plt.tight_layout()
    
    # plt.show() es bloqueante en un script.
    # Mostrará la figura y pausará el script hasta que la cierres.
    print("Mostrando gráfico de análisis... Cierra la ventana para continuar.")
    plt.show()
    
    # g) Etiquetado (se imprime en la consola)
    procesar_etiquetado(img_opened)
    
    print(f"--- Análisis para {base_filename} completado ---")


# --- Lógica Principal del Script ---

def main():
    """Función principal que ejecuta el script."""
    
    # Crear el directorio de resultados si no existe
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Buscar todas las imágenes individuales (jpg, png) en el directorio
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_files.extend(glob.glob(str(FILES_DIR / ext)))

    # Mapear el nombre del archivo (opción) a su ruta completa
    image_path_map = {Path(f).name: f for f in image_files}

    print(f"Script de Análisis de Fractografía")
    print("="*30)
    print(f"Directorio de imágenes: {FILES_DIR.resolve()}")
    print(f"Directorio de resultados: {RESULTS_DIR.resolve()}")

    if not image_path_map:
        print(f"\nADVERTENCIA: No se encontraron imágenes en '{FILES_DIR.resolve()}'.")
        print("Asegúrate de que tus archivos .jpg o .png estén en ese directorio.")
        print("Saliendo del script.")
        sys.exit(1) # Salir del script si no hay imágenes
    else:
        print(f"Imágenes encontradas: {list(image_path_map.keys())}\n")

    # --- Bucle de Procesamiento ---
    # Iterar sobre todas las imágenes encontradas y procesarlas
    
    for filename, file_path in image_path_map.items():
        
        base_filename = Path(filename).stem
        
        print(f"\n[{filename}] -> Cargando imagen...")
        
        # Cargar nueva imagen
        img_gray = cargar_imagen_gris(file_path)
        
        if img_gray is not None:
            # Ejecutar el análisis completo para la imagen
            run_full_analysis(img_gray, base_filename)
        else:
            print(f"ERROR: No se pudo cargar la imagen seleccionada: {filename}. Saltando...")

    print("\n" + "="*30)
    print("Proceso completado. Todas las imágenes han sido analizadas.")
    print(f"Los resultados están en: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()