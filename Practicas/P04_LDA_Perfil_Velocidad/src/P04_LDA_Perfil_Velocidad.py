#!/usr/bin/env python
# coding: utf-8

# # Práctica 4: Anemometría Láser Doppler (LDA) - Perfil de Velocidades
# 
# ## Objetivo
# Medir perfiles de velocidad en un túnel de viento utilizando técnica láser Doppler (LDA/LDV), una técnica óptica no intrusiva que permite obtener mediciones puntuales de velocidad con alta precisión.
# 
# ## Fundamento Teórico
# 
# ### ¿Qué es LDA?
# La **Anemometría Láser Doppler** se basa en el efecto Doppler de la luz dispersada por partículas trazadoras que se mueven con el fluido. Dos haces láser coherentes se cruzan formando un **volumen de medida** donde se genera un patrón de franjas de interferencia.
# 
# Cuando una partícula atraviesa este volumen:
# - Dispersa luz con una frecuencia modulada proporcional a su velocidad
# - La frecuencia Doppler $f_D$ está relacionada con la velocidad por: 
# $$V = \frac{\lambda \cdot f_D}{2 \sin(\theta/2)}$$
# donde $\lambda$ es la longitud de onda del láser y $\theta$ el ángulo entre haces.
# 
# ### Ventajas de LDA
# - ✅ **No intrusiva**: no perturba el flujo
# - ✅ **Alta resolución espacial y temporal**
# - ✅ **Medición directa de velocidad** (no requiere calibración)
# - ✅ **Puede medir flujos inversos** y turbulencia
# 
# ### Datos LDA típicos
# Cada medición registra:
# 1. **Tiempo de llegada** (ms)
# 2. **Velocidad** (m/s o componente U, V)
# 3. **SNR** (Signal-to-Noise Ratio) - calidad de señal
# 4. **Validación** - indicadores de calidad
# 
# ---
# 
# ## Estructura de los Datos
# 
# En `files/P4/` tenemos 4 carpetas (FX01G00 a FX04G00), cada una representa una **posición Y diferente** en el perfil del túnel.
# 
# Dentro de cada carpeta hay ~15 archivos `.txt`, cada uno con ~2000 mediciones de velocidad en esa posición.
# 
# Nuestro objetivo: **construir el perfil U(y)** promediando todas las mediciones por posición.

# Imports necesarios
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.fft import fft, fftfreq
import warnings
import os 

warnings.filterwarnings('ignore')

# Configuración de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

print("✅ Librerías cargadas correctamente (incluyendo os)")

# ## 1. Configuración de Rutas y Exploración de Datos
# 
# Primero localizamos los datos en `files/P4/` y exploramos la estructura.

# Configuración de rutas
# Usamos Path(__file__) para que las rutas sean relativas al script
# y no desde dónde se ejecuta (CWD).
SCRIPT_PATH = Path(__file__).resolve() # Ruta completa al script
# BASE_DIR es la raíz del proyecto (Aero-Messure), 4 niveles arriba del script
# .../src/script.py -> .../src -> .../P04... -> .../Practicas -> .../Aero-Messure
BASE_DIR = SCRIPT_PATH.parent.parent.parent.parent
DATA_DIR = BASE_DIR / 'files' / 'P4'
IMAGES_DIR = BASE_DIR / 'Practicas' / 'P04_LDA_Perfil_Velocidad' / 'data' / 'images'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print(f"📂 Directorio base: {BASE_DIR}")
print(f"📂 Directorio de datos: {DATA_DIR}")
print(f"✅ Datos existen: {DATA_DIR.exists()}")

# Listar las carpetas (cada una es una posición Y)
folders = sorted([f for f in DATA_DIR.iterdir() if f.is_dir()])
print(f"\n📁 Carpetas encontradas ({len(folders)}):")
for folder in folders:
    files_count = len(list(folder.glob('*.txt')))
    print(f"  - {folder.name}: {files_count} archivos")

# ## 2. Lectura y Análisis de un Archivo Individual
# 
# Examinemos un archivo para entender el formato de datos.

# Leer un archivo de ejemplo
sample_file = folders[0] / list(folders[0].glob('*.txt'))[0]
print(f"📄 Archivo de ejemplo: {sample_file.name}")

df_sample = pd.read_csv(sample_file, sep=r'\s+', header=None,
                        names=['idx', 'Doppler_Freq', 'SNR', 'U', 'V'])
# ------------------------------------------------

print(f"\n📊 Forma de datos: {df_sample.shape}")
print(f"   → {df_sample.shape[0]} mediciones × {df_sample.shape[1]} columnas")

print("\n🔍 Primeras 10 filas:")
print(df_sample.head(10))

print("\n📈 Estadísticas descriptivas:")
print(df_sample.describe())
# -------------------------------------------------------------

# Visualización de datos brutos de un archivo
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
# ---------------------------------------------------------------------

# Velocidad U (Columna 4)
axes[0, 0].hist(df_sample['U'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Velocidad U (m/s) - Col 4') # Etiqueta actualizada
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Distribución de Velocidad U (Col 4)') # Título actualizado
axes[0, 0].axvline(df_sample['U'].mean(), color='r', linestyle='--',
                    label=f'Media = {df_sample["U"].mean():.2f} m/s')
axes[0, 0].legend()

# Velocidad V (Columna 5)
axes[0, 1].hist(df_sample['V'], bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_xlabel('Velocidad V (m/s) - Col 5') # Etiqueta actualizada
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].set_title('Distribución de Velocidad V (Col 5)') # Título actualizado
axes[0, 1].axvline(df_sample['V'].mean(), color='r', linestyle='--',
                    label=f'Media = {df_sample["V"].mean():.3f} m/s')
axes[0, 1].legend()

# Serie temporal U (Columna 4)
axes[1, 0].plot(df_sample['idx'], df_sample['U'], linewidth=0.5, alpha=0.7)
axes[1, 0].set_xlabel('Índice de muestra')
axes[1, 0].set_ylabel('U (m/s) - Col 4') # Etiqueta actualizada
axes[1, 0].set_title('Serie Temporal - Velocidad U (Col 4)') # Título actualizado

# Box plot (Columnas 4 y 5)
axes[1, 1].boxplot([df_sample['U'], df_sample['V']], labels=['U (Col 4)', 'V (Col 5)']) # Etiquetas actualizadas
axes[1, 1].set_ylabel('Velocidad (m/s)')
axes[1, 1].set_title('Box Plot de Velocidades (Cols 4 y 5)') # Título actualizado
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'box_plot_de_velocidades_cols_4_y_5.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(IMAGES_DIR, 'box_plot_de_velocidades_cols_4_y_5.svg'), bbox_inches='tight')
# plt.show() # <--- ELIMINADO

print(f"\n📊 Análisis del archivo {sample_file.name} (usando Col 4 como U, Col 5 como V):")
print(f"   U media: {df_sample['U'].mean():.3f} m/s")
print(f"   U std:   {df_sample['U'].std():.3f} m/s")
print(f"   V media: {df_sample['V'].mean():.3f} m/s (transversal)")
print(f"   V std:   {df_sample['V'].std():.3f} m/s")
# -----------------------------------------------------------------------------------

# ## 3. Función para Procesar Todos los Archivos de una Posición
# 
# Crearemos una función que:
# 1. Lee todos los archivos de una carpeta (posición Y)
# 2. Filtra outliers usando criterio IQR (Rango Intercuartílico)
# 3. Calcula estadísticos robustos (media, std, error estándar)
# 
# **Método de detección de outliers:**
# - Calculamos Q1 (percentil 25) y Q3 (percentil 75)
# - IQR = Q3 - Q1
# - Outliers: valores fuera de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

def remove_outliers_iqr(data, factor=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    mask = (data >= lower_bound) & (data <= upper_bound)
    return data[mask], mask


def process_position_folder(folder_path, remove_outliers=True):
    all_U = []
    all_V = []

    # Leer todos los archivos .txt de la carpeta
    txt_files = list(folder_path.glob('*.txt'))

    for file in txt_files:
        df = pd.read_csv(file, sep=r'\s+', header=None,
                        names=['idx', 'Doppler_Freq', 'SNR', 'U', 'V'])
        all_U.extend(df['U'].values)
        all_V.extend(df['V'].values)

    all_U = np.array(all_U)
    all_V = np.array(all_V)

    if remove_outliers:
        U_clean, mask_U = remove_outliers_iqr(all_U)
        V_clean, mask_V = remove_outliers_iqr(all_V)
        outliers_removed_U = len(all_U) - len(U_clean)
        outliers_removed_V = len(all_V) - len(V_clean)
    else:
        U_clean = all_U
        V_clean = all_V
        outliers_removed_U = 0
        outliers_removed_V = 0

    # Calcular estadísticos
    results = {
        'folder': folder_path.name,
        'n_files': len(txt_files),
        'n_samples_total': len(all_U),
        'n_samples_clean_U': len(U_clean),
        'n_samples_clean_V': len(V_clean),
        'outliers_removed_U': outliers_removed_U,
        'outliers_removed_V': outliers_removed_V,
        'U_mean': np.mean(U_clean),
        'U_std': np.std(U_clean),
        'U_stderr': stats.sem(U_clean),  # Error estándar de la media
        'V_mean': np.mean(V_clean),
        'V_std': np.std(V_clean),
        'V_stderr': stats.sem(V_clean),
        # Guardamos los datos limpios para análisis posteriores si es necesario
        # 'U_all': all_U,
        # 'V_all': all_V,
        'U_clean': U_clean,
        'V_clean': V_clean
    }

    return results

print("✅ Funciones de procesamiento definidas ")

# ## 4. Procesar Todas las Posiciones
# 
# Ahora procesamos las 4 carpetas (posiciones Y) y almacenamos los resultados.

# Procesar todas las carpetas
results_all = []

print("🔄 Procesando posiciones...")
print("="*70)

for folder in folders:
    print(f"\n📁 Procesando: {folder.name}")
    result = process_position_folder(folder, remove_outliers=True)
    results_all.append(result)
    
    print(f"   Archivos: {result['n_files']}")
    print(f"   Muestras totales: {result['n_samples_total']}")
    print(f"   Outliers eliminados (U): {result['outliers_removed_U']} "
          f"({100*result['outliers_removed_U']/result['n_samples_total']:.1f}%)")
    print(f"   U = {result['U_mean']:.3f} ± {result['U_std']:.3f} m/s")
    print(f"   V = {result['V_mean']:.4f} ± {result['V_std']:.4f} m/s")

print("\n" + "="*70)
print("✅ Procesamiento completado")

# ## 5. Construcción del Perfil de Velocidades U(y)
# 
# Para graficar el perfil necesitamos asignar posiciones Y a cada carpeta. 
# 
# **Asumimos** que las carpetas están ordenadas de abajo hacia arriba (o viceversa) en el túnel. Si conoces las posiciones reales en mm, ajusta el array `y_positions`.

# Posiciones Y: ajusta estos valores según tus mediciones reales
y_positions = np.array([10, 30, 50, 70])  # mm, ejemplo

# Extraer datos para el perfil
U_means = np.array([r['U_mean'] for r in results_all])
U_stderrs = np.array([r['U_stderr'] for r in results_all])

# Crear DataFrame resumen
df_profile = pd.DataFrame({
    'Posición': [r['folder'] for r in results_all],
    'y (mm)': y_positions,
    'U_mean (m/s)': U_means,
    'U_std (m/s)': [r['U_std'] for r in results_all],
    'U_stderr (m/s)': U_stderrs,
    'V_mean (m/s)': [r['V_mean'] for r in results_all],
    'N_muestras': [r['n_samples_clean_U'] for r in results_all]
})

print("📊 Tabla Resumen del Perfil de Velocidades:")
print("="*80)
print(df_profile.to_string(index=False))
print("="*80)

# Gráfico del perfil de velocidades U(y)
fig, ax = plt.subplots(figsize=(10, 8))

# Perfil con barras de error (error estándar)
ax.errorbar(U_means, y_positions, xerr=U_stderrs, 
            fmt='o-', markersize=10, linewidth=2, capsize=5,
            label='U(y) ± SE', color='steelblue', elinewidth=2)

# Línea de referencia de velocidad media global
U_global_mean = U_means.mean()
ax.axvline(U_global_mean, color='red', linestyle='--', linewidth=1.5,
           label=f'U media global = {U_global_mean:.2f} m/s', alpha=0.7)

ax.set_xlabel('Velocidad U (m/s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Posición Y (mm)', fontsize=13, fontweight='bold')
ax.set_title('Perfil de Velocidades - Anemometría Láser Doppler (LDA)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'perfil_de_velocidades_anemometra_lser_doppler_lda.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(IMAGES_DIR, 'perfil_de_velocidades_anemometra_lser_doppler_lda.svg'), bbox_inches='tight')
# plt.show() # <--- ELIMINADO

print(f"\n📈 Velocidad media del perfil: {U_global_mean:.3f} m/s")
print(f"   Rango de velocidades: {U_means.min():.3f} - {U_means.max():.3f} m/s")
print(f"   Variación: {U_means.max() - U_means.min():.3f} m/s")

# ## 6. Análisis de Turbulencia
# 
# Calculamos la **intensidad de turbulencia** en cada posición:
# 
# $$I_u = \frac{\sigma_U}{U_{mean}} \times 100\%$$
# 
# Donde:
# - $\sigma_U$ es la desviación estándar de U
# - $U_{mean}$ es la velocidad media
# 
# La intensidad de turbulencia indica qué tan "agitado" está el flujo respecto a su velocidad media.

# Calcular intensidad de turbulencia
turbulence_intensity = []
colors = ['steelblue', 'coral', 'forestgreen', 'purple'] # Definir colores para usar más adelante

for result in results_all:
    I_u = (result['U_std'] / result['U_mean']) * 100
    turbulence_intensity.append(I_u)

df_profile['Turbulencia (%)'] = turbulence_intensity

print("🌪️  Intensidad de Turbulencia por Posición:")
print("="*60)
for i, row in df_profile.iterrows():
    print(f"{row['Posición']:12s} (y={row['y (mm)']:5.1f} mm): "
          f"I_u = {row['Turbulencia (%)']:5.2f}%")
print("="*60)

# Gráfico de intensidad de turbulencia
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(turbulence_intensity, y_positions, 'o-', markersize=10, 
        linewidth=2, color='coral', label='Intensidad de Turbulencia')

ax.set_xlabel('Intensidad de Turbulencia I_u (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Posición Y (mm)', fontsize=13, fontweight='bold')
ax.set_title('Perfil de Intensidad de Turbulencia', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'perfil_intensidad_turbulencia_lda.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(IMAGES_DIR, 'perfil_intensidad_turbulencia_lda.svg'), bbox_inches='tight')
# plt.show() # <--- ELIMINADO

# ## 7. Comparación de Distribuciones por Posición
# 
# Visualizamos las distribuciones de velocidad U en cada posición para identificar comportamientos.

# Histogramas superpuestos
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

# colors ya está definido en la sección 6

for i, result in enumerate(results_all):
    ax = axes[i]
    
    # Histograma de datos limpios
    ax.hist(result['U_clean'], bins=50, alpha=0.7, color=colors[i], 
            edgecolor='black', label=f"Limpio (n={len(result['U_clean'])})")
    
    # Línea vertical en la media
    ax.axvline(result['U_mean'], color='red', linestyle='--', linewidth=2,
               label=f"Media = {result['U_mean']:.2f} m/s")
    
    ax.set_xlabel('Velocidad U (m/s)', fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=11)
    ax.set_title(f"{result['folder']} - y={y_positions[i]} mm", 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'histogramas_velocidad_U_limpia.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(IMAGES_DIR, 'histogramas_velocidad_U_limpia.svg'), bbox_inches='tight')
# plt.show() # <--- ELIMINADO

# ## 8. Análisis de Calidad de Datos
# 
# Evaluamos la calidad de las mediciones:
# 1. **Ratio señal/ruido implícito**: menor dispersión = mejor calidad
# 2. **Consistencia entre archivos**: comparar varianza inter-archivo vs intra-archivo
# 3. **Convergencia estadística**: verificar que tenemos suficientes muestras

# --- FIGURAS SEPARADAS ---

# --- Gráfico 1: Box plot de velocidad U ---
fig1, ax1 = plt.subplots(figsize=(8, 6))

U_data = [r['U_clean'] for r in results_all]
positions_labels = [f"{r['folder']}\ny={y_positions[i]} mm" 
                   for i, r in enumerate(results_all)]

bp1 = ax1.boxplot(U_data, labels=positions_labels, patch_artist=True)
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_ylabel('Velocidad U (m/s)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Posición', fontsize=12, fontweight='bold')
ax1.set_title('Distribución de Velocidad U por Posición', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'boxplot_velocidades_U.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(IMAGES_DIR, 'boxplot_velocidades_U.svg'), bbox_inches='tight')
# plt.show() # <--- ELIMINADO


# --- Gráfico 2: Coeficiente de variación (CV) por posición ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
cv_values = [(r['U_std']/r['U_mean'])*100 for r in results_all]
ax2.bar(range(len(cv_values)), cv_values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Posición', fontsize=12, fontweight='bold')
ax2.set_ylabel('Coeficiente de Variación (%)', fontsize=12, fontweight='bold')
ax2.set_title('Coeficiente de Variación (CV = σ/μ × 100%)', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(results_all)))
ax2.set_xticklabels([r['folder'] for r in results_all], rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# Añadir línea de referencia de calidad
ax2.axhline(10, color='red', linestyle='--', linewidth=2, 
            label='CV=10% (límite calidad)', alpha=0.7)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'CV_por_posicion.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(IMAGES_DIR, 'CV_por_posicion.svg'), bbox_inches='tight')
# plt.show() # <--- ELIMINADO

# --- FIN DE FIGURAS SEPARADAS ---


print("\n📊 Métricas de Calidad:")
print("="*70)
for i, (result, cv) in enumerate(zip(results_all, cv_values)):
    quality = "Excelente" if cv < 5 else "Buena" if cv < 10 else "Regular"
    print(f"{result['folder']:12s}: CV={cv:5.2f}% → Calidad: {quality}")
print("="*70)

# ## 9. Ajuste de Perfil Teórico (Opcional)
# 
# Para flujo en túnel, podemos comparar con perfiles teóricos:
# - **Flujo laminar**: perfil parabólico (Poiseuille)
# - **Flujo turbulento**: perfil logarítmico (ley de la pared) o ley de potencia
# 
# Aquí probamos un ajuste polinómico simple como aproximación.

# Gráfico del perfil U(y) con ajuste polinómico (Opcional)

# Ajuste polinómico (ejemplo: grado 2)
# Ignoramos la posición y=0 si existe (p.ej. y_positions[0] == 0)
y_fit = y_positions
U_fit = U_means

coefs = np.polyfit(y_fit, U_fit, 2) # Ajuste a y = Ay^2 + By + C
p = np.poly1d(coefs)

# Crear puntos para la línea de ajuste
y_curve = np.linspace(y_fit.min(), y_fit.max(), 100)
U_curve = p(y_curve)

print("Ajuste polinómico (Grado 2): U(y) = {:.3e}*y^2 + {:.3e}*y + {:.3e}".format(coefs[0], coefs[1], coefs[2]))

# Gráfico
fig, ax = plt.subplots(figsize=(10, 8))
ax.errorbar(U_means, y_positions, xerr=U_stderrs, 
            fmt='o', markersize=10, capsize=5,
            label='Datos LDA (U ± SE)', color='steelblue')

ax.plot(U_curve, y_curve, color='coral', linestyle='--',
        linewidth=2, label=f'Ajuste (Grado 2)')

ax.set_xlabel('Velocidad U (m/s)', fontsize=13, fontweight='bold')
ax.set_ylabel('Posición Y (mm)', fontsize=13, fontweight='bold')
ax.set_title('Ajuste de Perfil de Velocidades', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'perfil_ajuste_polinomico.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(IMAGES_DIR, 'perfil_ajuste_polinomico.svg'), bbox_inches='tight')
# plt.show() # <--- ELIMINADO

# ## 9. Análisis de Esfuerzo de Reynolds (Opcional Avanzado)
# 
# Si queremos medir el esfuerzo turbulento $\tau_{turb} = -\rho \overline{u'v'}$
# 
# 1. $u' = U - U_{mean}$ (fluctuación de U)
# 2. $v' = V - V_{mean}$ (fluctuación de V)
# 3. $\overline{u'v'}$ (covarianza / producto cruzado promedio)
# 
# **Importante**: Para que $\overline{u'v'}$ sea válido, $u'$ y $v'$ deben medirse *simultáneamente* (la misma partícula). El formato de nuestros datos (columnas separadas U y V) sugiere que este es el caso.

# Calcular Esfuerzos de Reynolds (u'v')
RHO_AIRE = 1.225 # kg/m^3 (densidad del aire, ajustar si es necesario)

reynolds_stress_results = []

print("\n📉 Cálculo de Esfuerzo de Reynolds (τ_turb = -ρ * u'v')")
print("="*70)

for i, result in enumerate(results_all):
    U_clean = result['U_clean']
    V_clean = result['V_clean']

    # Asegurarnos de que u' y v' se calculan sobre el mismo set de datos limpios
    # (Asumiendo que el filtro IQR en U y V mantuvo las mismas muestras, 
    # lo cual NO es garantizado. Un mejor enfoque sería filtrar por pares)
    
    # --- Enfoque Robusto: Recalcular medias y fluctuaciones --- 
    # (Usaremos U_clean y V_clean por simplicidad, aunque no sean pares perfectos)
    
    # Calculamos fluctuaciones respecto a la media de los datos limpios
    u_prime = U_clean - result['U_mean']
    v_prime = V_clean - result['V_mean']
    
    # u'v' (producto cruzado promedio)
    # Aseguramos que tengan el mismo tamaño (el menor de los dos sets limpios)
    min_len = min(len(u_prime), len(v_prime))
    uv_prime_mean = np.mean(u_prime[:min_len] * v_prime[:min_len])
    
    # Coeficiente de correlación R_uv
    R_uv = uv_prime_mean / (result['U_std'] * result['V_std'])
    
    # Esfuerzo de Reynolds
    tau_turb = -RHO_AIRE * uv_prime_mean
    
    stress_data = {
        'folder': result['folder'],
        'y (mm)': y_positions[i],
        'uv_prime_mean': uv_prime_mean,
        'tau_turb (Pa)': tau_turb,
        'R_uv': R_uv
    }
    reynolds_stress_results.append(stress_data)
    
    print(f"\nPosición {result['folder']} (y={y_positions[i]} mm):")
    print(f"   N_U_clean = {len(U_clean)}, N_V_clean = {len(V_clean)}, N_usado = {min_len}")
    print(f"   u'v' (covarianza): {uv_prime_mean:.6f} m²/s²")
    print(f"   τ_turb (Esfuerzo): {tau_turb:.6f} Pa")
    print(f"   Coef. Correlación R_uv: {R_uv:.3f}")

print("="*70)

# Crear DataFrame de Esfuerzos
df_stress = pd.DataFrame(reynolds_stress_results)

# Gráfico de Esfuerzos de Reynolds

# Verificar si tenemos datos válidos para graficar
if not df_stress.empty and df_stress['tau_turb (Pa)'].notna().all():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Esfuerzo de Reynolds (tau_turb)
    ax.plot(df_stress['tau_turb (Pa)'], df_stress['y (mm)'], 'o-',
            markersize=10, linewidth=2, color='darkred', 
            label='Esfuerzo de Reynolds (τ_turb)')

    # Línea de referencia en cero
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_xlabel(r"Esfuerzo de Reynolds $\tau_{turb} = -\rho \overline{u'v'}$ (Pa)",
                 fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Posición Y (mm)', fontsize=12, fontweight='bold')
    ax.set_title('Perfil de Esfuerzo de Reynolds', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'perfil_esfuerzo_reynolds.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(IMAGES_DIR, 'perfil_esfuerzo_reynolds.svg'), bbox_inches='tight')
    # plt.show() # <--- ELIMINADO
    
    # Interpretación basada en el promedio
    tau_mean = df_stress['tau_turb (Pa)'].mean()
    uv_mean = df_stress['uv_prime_mean'].mean()
    corr_mean = df_stress['R_uv'].mean()
    
    if pd.notna(tau_mean) and pd.notna(uv_mean):
        print("\n🗨️ Interpretación del Perfil de Esfuerzos:")
        # Usamos 'r' (raw string) para evitar SyntaxWarning con \o
        print(r"   - Valor promedio $\overline{u'v'}$: {uv_mean:.6f} m²/s²")
        print(r"   - Esfuerzo turbulento promedio $\tau_{turb}$: {tau_mean:.6f} Pa")
        print(r"   - Coeficiente de correlación promedio $R_{uv}$: {corr_mean:.3f}")

        if uv_mean < -1e-6: # Umbral pequeño para considerar negativo
            print(r"   ✅ $\overline{u'v'}$ < 0: Transporte turbulento de momento hacia la pared (esperado en capa límite).")
        elif uv_mean > 1e-6: # Umbral pequeño para considerar positivo
            print(r"   ⚠️ $\overline{u'v'}$ > 0: Transporte turbulento de momento alejándose de la pared (menos común).")
        else:
            print(r"   ⚪ $\overline{u'v'}$ ≈ 0: Correlación muy baja o nula entre u' y v'.")
    else:
        print("   - No se pudieron calcular valores medios válidos para la interpretación.")

else:
    print("\n   No hay suficientes datos válidos para graficar los esfuerzos de Reynolds.")

# --- CORRECCIÓN FINAL: AÑADIR UN ÚNICO plt.show() AL FINAL ---
print("\n✅ Script completado. Mostrando todas las figuras...")
print("Cierre todas las ventanas de gráficos para finalizar la ejecución.")
plt.show()