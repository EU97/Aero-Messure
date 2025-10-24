#!/usr/bin/env python3
"""
# P01: Protección ocular frente a radiación láser

Script para cálculos de H0 y OD requerida, evaluación automática 
contra un catálogo de gafas (CSV), y generación de gráficos comparativos.

Este script es la versión .py equivalente del notebook P01_Proteccion_Laser.ipynb.
La funcionalidad de widgets interactivos ha sido omitida ya que 
requiere un entorno Jupyter.

Fuentes base (según práctica):
- Láser 1 (PIV): 532 nm, E = 0.2 J, τ = 8 ns, f = 15 Hz, a = 5 mm, HMPE = 5.0e-7 J/cm².
- Láser 2 (LDA, CW): 514.5 nm, P = 1.5 W, a = 1.2 mm, HMPE = 2.5e-3 W/cm² (0.25 s); para 10 s usar 1.0e-3 W/cm².
- Láser 3 (Alineación, CW): 635 nm, P = 4.5 mW, a = 3 mm, HMPE = 2.5e-3 W/cm² (0.25 s).

Catálogo de gafas (entrada): `../data/epo_lenses.csv`
"""

# --- IMPORTACIONES (de Celda 2) ---
import math, csv, sys, json
import os # Necesario para manejar rutas de archivos

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Nota: ipywidgets no se usará en este script.
try:
    import ipywidgets as widgets
    from ipywidgets import interact, fixed
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

# --- FUNCIONES DE CARGA (de Celda 3) ---

def load_lenses(path):
    """Carga de lentes desde CSV (soporta pandas o csv estándar)"""
    if HAS_PANDAS:
        try:
            df = pd.read_csv(path)
            # normaliza nombres de columnas
            cols = {c: c.strip().lower() for c in df.columns}
            df = df.rename(columns=cols)
            return df
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo CSV en {path}", file=sys.stderr)
            return None
    
    # Fallback sin pandas
    rows = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k.strip().lower(): v for k, v in r.items()})
        return rows
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV en {path}", file=sys.stderr)
        return None

# --- FUNCIONES DE CÁLCULO (de Celda 4) ---

def area_cm2_from_diameter_mm(a_mm: float) -> float:
    d_cm = float(a_mm) / 10.0
    return math.pi * (d_cm / 2.0) ** 2

def h0_exposure(laser: dict):
    A = area_cm2_from_diameter_mm(laser['a_mm'])
    if laser['mode'].lower() == 'cw':
        H0 = laser['P_W'] / A
        return H0, 'W/cm^2', A
    else:
        H0 = laser['E_J'] / A
        return H0, 'J/cm^2', A

def od_required(H0: float, HMPE: float) -> float:
    if HMPE <= 0:
        raise ValueError('HMPE debe ser positiva')
    if H0 <= HMPE:
        return 0.0
    return math.log10(H0 / HMPE)

def df_to_rows(df):
    """Convierte un DataFrame de pandas a lista de dicts, o la devuelve si ya lo es."""
    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        return df.to_dict(orient='records')
    return df # Ya es una lista de dicts

def available_od_for_lambda(lens_rows, lam_nm: float):
    ods = []
    for r in lens_rows:
        try:
            lo = float(r['band_lo_nm']); hi = float(r['band_hi_nm']); od = float(r['od_value'])
        except Exception:
            continue
        if lo <= lam_nm <= hi:
            ods.append(od)
    return max(ods) if ods else None

def evaluate_against_lenses(laser: dict, lenses_data):
    H0, units, A = h0_exposure(laser)
    od_req = od_required(H0, laser['HMPE'])
    rows = df_to_rows(lenses_data)
    
    # Agrupar filas por lens_id
    groups = {}
    for r in rows:
        lid = r.get('lens_id') # Usar .get() para seguridad
        if lid:
            groups.setdefault(lid, []).append(r)
            
    out = []
    for lid, rs in groups.items():
        od_av = available_od_for_lambda(rs, laser['lambda_nm'])
        brand = rs[0].get('brand',''); model = rs[0].get('model','');
        vlt = float(rs[0].get('vlt_pct', '0') or 0)
        safe = (od_av is not None) and (od_av >= od_req)
        margin = (od_av - od_req) if od_av is not None else None
        out.append({
            'lens_id': lid, 'brand': brand, 'model': model, 'vlt_pct': vlt,
            'lambda_nm': laser['lambda_nm'], 'mode': laser['mode'], 'a_mm': laser['a_mm'],
            'H0': H0, 'H0_units': units, 'HMPE': laser['HMPE'], 'OD_req': od_req,
            'OD_avail': od_av, 'safe': safe, 'margin': margin
        })
        
    if HAS_PANDAS:
        # Ordenar resultados
        return pd.DataFrame(out).sort_values(['safe','margin','vlt_pct'], ascending=[False, False, False])
    
    # Fallback sin pandas
    out.sort(key=lambda x: (not x['safe'], -(x['margin'] if x['margin'] is not None else -float('inf')), -x['vlt_pct']))
    return out

# --- FUNCIÓN DE EVALUACIÓN Y GRÁFICO (de Celda 5) ---

def evaluate_and_plot(laser, lenses_data, save_dir):
    """
    Función modificada de la Celda 5:
    Evalúa un láser, imprime resultados en consola y guarda el gráfico.
    """
    res = evaluate_against_lenses(laser, lenses_data)
    
    if HAS_PANDAS:
        # Imprime la tabla en la consola (en lugar de 'display')
        # Usamos .to_string() para un mejor formato en la terminal
        print(res.head(10).to_string()) 
        
        if HAS_MPL:
            # Top 10 por margen
            top = res.sort_values(['safe','margin','vlt_pct'], ascending=[False, False, False]).head(10)
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['tab:green' if s else 'tab:red' for s in top['safe']]
            ax.bar(top['lens_id'], top['margin'].fillna(-1), color=colors)
            ax.axhline(0, color='k', linestyle='--', linewidth=1)
            ax.set_title(f"Margen de OD para {laser['name']} @ {laser['lambda_nm']} nm")
            ax.set_ylabel('Margen de OD')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Limpiar el nombre del láser para usarlo como nombre de archivo
            clean_name = laser['name'].replace(' ', '_').replace('–', '_').replace('.', '_').replace('á','a').replace('ó','o')
            # Asegura que la ruta sea correcta
            save_path = os.path.join(save_dir, f"{clean_name}.png")
            
            try:
                plt.savefig(save_path)
                print(f"Gráfica guardada en: {save_path}\n")
            except Exception as e:
                print(f"Error al guardar gráfica: {e}\n")
            
            # Cierra la figura para liberar memoria (muy importante en scripts)
            plt.close(fig) 
        else:
            print('Matplotlib no disponible: se omiten gráficos\n')
    else:
        # Fallback sin pandas
        print('Resultados (sin pandas):')
        for r in res[:10]:
            print(r)
        print("\n")
        
    # Evaluación alternativa @10s si aplica
    if 'HMPE_long' in laser:
        laser2 = dict(laser)
        H0, units, _ = h0_exposure(laser2)
        od_req_long = od_required(H0, laser2['HMPE_long'])
        print(f"OD_req (exposición larga) @10 s: {od_req_long:.2f}\n")

# --- BLOQUE DE EJECUCIÓN PRINCIPAL ---

def main():
    """Función principal que ejecuta el análisis estático del notebook."""
    
    # --- Configuración de rutas ---
    # Asume que este script (.py) está en 'P01/src/'
    # y los datos están en 'P01/data/'
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directorio 'src'
        PROJECT_ROOT = os.path.dirname(BASE_DIR) # Directorio 'P01'
    except NameError:
        # Fallback si se ejecuta en un entorno donde __file__ no está definido
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))

    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'epo_lenses.csv')
    IMAGE_DIR = os.path.join(PROJECT_ROOT, 'data', 'images')

    # Crear directorio de imágenes si no existe
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # --- Definición de escenarios base (de Celda 5) ---
    LASERS = [
        {'name':'Láser 1 – PIV','lambda_nm':532.0,'mode':'pulsed','a_mm':5.0,'E_J':0.2,'HMPE':5.0e-7},
        {'name':'Láser 2 – LDA 514.5','lambda_nm':514.5,'mode':'CW','a_mm':1.2,'P_W':1.5,'HMPE':2.5e-3,'HMPE_long':1.0e-3},
        {'name':'Láser 3 – Alineación','lambda_nm':635.0,'mode':'CW','a_mm':3.0,'P_W':4.5e-3,'HMPE':2.5e-3},
    ]
    
    # --- Ejecución (de Celdas 2, 3 y 5) ---
    
    print(f'Pandas: {HAS_PANDAS}, Matplotlib: {HAS_MPL}, Widgets: {HAS_WIDGETS} (Funcionalidad interactiva omitida)')
    print(f'CSV esperado: {DATA_PATH}\n')
    
    # Carga de lentes (de Celda 3)
    lenses = load_lenses(DATA_PATH)
    if lenses is None:
        print("No se pudieron cargar los datos de lentes. Saliendo.", file=sys.stderr)
        return

    # Mostrar vista previa (de Celda 3)
    print("--- Vista previa de los datos de lentes ---")
    if HAS_PANDAS:
        # Usamos .to_string() para imprimir bien en la consola
        print(lenses.head(8).to_string())
    else:
        print('Primeras filas (sin pandas):')
        for r in lenses[:5]:
            print(r)
    print("\n" + "="*30 + "\n")
    
    # Ejecutar evaluación para cada láser base (de Celda 5)
    for L in LASERS:
        print(f"=== {L['name']} ===")
        evaluate_and_plot(L, lenses, IMAGE_DIR)
        
    print("--- Evaluación de escenarios base completada ---")
    
    # La funcionalidad de widgets (Celda 5 del notebook) se omite 
    # ya que requiere un entorno Jupyter interactivo.

# Este es el punto de entrada estándar para un script de Python
if __name__ == "__main__":
    main()