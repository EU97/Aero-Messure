# P08 - Análisis de Productos de Combustión

Análisis completo de productos de combustión con cálculos estequiométricos, temperatura adiabática, estadísticos y visualizaciones avanzadas.

## Contenido

### Notebook Principal
`notebooks/P08_Productos_Combustion.ipynb`

### Datos
Coloca tus archivos (CSV/TXT/TSV) en `data/` con columnas como:
- `time` o `tiempo`: marca temporal
- `O2`: oxígeno [%]
- `CO2`: dióxido de carbono [%]
- `CO`: monóxido de carbono [ppm]
- `NOx`: óxidos de nitrógeno [ppm]
- `T` o `temperature`: temperatura [°C]
- `P` o `pressure`: presión [kPa]

Si no hay archivos, el notebook genera datos sintéticos de demostración.

## Análisis Implementados

### 1. Cálculos de Combustión
- **Estequiometría**: AFR (Air-Fuel Ratio), ν_O2 para CxHy
- **CO2 teórico seco**: Fracción molar y porcentaje
- **Lambda (λ)**: Exceso de aire desde O2/CO2 medidos
- **Phi (φ)**: Equivalence ratio = 1/λ
- **Temperatura adiabática**: Estimación simplificada con Cp promedio para diferentes valores de λ

### 2. Análisis Estadístico
- Estadísticos descriptivos (media, desv. estándar, percentiles 1%, 5%, 95%, 99%)
- Detección de outliers (método IQR)
- Estadísticos móviles (rolling mean/std con ventana de 30 muestras)
- Detección de picos en CO, NOx y temperatura
- Reporte de máximos y mínimos clave

### 3. Visualizaciones
**Series temporales**:
- O2/CO2 vs tiempo
- CO/NOx vs tiempo
- Lambda (λ) vs tiempo

**Distribuciones**:
- Histogramas con KDE
- Box plots para identificar outliers
- Violin plots para visualizar densidades
- Distribución por cuartiles e IQR

**Relaciones**:
- Matriz de correlación (heatmap)
- Scatter matrix (pairplot) para relaciones bivariadas
- Lambda vs Temperatura con densidad de color

### 4. Exportación
Todas las tablas se guardan en `data/`:
- `combustion_metrics.csv`: Datos con métricas calculadas
- `temperatura_adiabatica.csv`: T_ad para diferentes λ
- `stats_*.csv`: Estadísticos descriptivos, outliers, rolling
- `peaks_*.csv`: Detección de picos por variable
- `summary_maximos.json`: Resumen de valores máximos

Todas las figuras se guardan en `data/figures/` en formato **PNG y SVG**:
- `series_temporales`: Evolución temporal de variables clave
- `histogramas`: Distribuciones individuales
- `correlacion`: Matriz de correlación
- `boxplots`: Distribuciones con outliers
- `violinplots`: Densidades de probabilidad
- `pairplot`: Scatter matrix de relaciones
- `cuartiles_iqr`: Análisis por cuartiles
- `lambda_vs_temperatura`: Relación entre exceso de aire y temperatura

## Referencias

- **NIST Chemistry WebBook**: Propiedades termoquímicas (https://webbook.nist.gov/chemistry/)
- **NASA CEA**: Chemical Equilibrium with Applications (McBride et al., 1993)
- **Turns, S. R.** (2012): An Introduction to Combustion: Concepts and Applications. McGraw-Hill
- **ISO 12039**: Determinación de CO, CO2 y O2 en gases de combustión
- Hojas técnicas de analizadores (Testo 350/340, Bacharach, Dräger)

## Notas

- Se asume reporte en **base seca** (común en analizadores). Si tu equipo reporta en húmedo, ajusta el preprocesado.
- El combustible por defecto es **Propano C3H8**. Puedes cambiar a Metano CH4, Octano C8H18 u otro CxHy editando la variable `FUEL` en el notebook.
- La temperatura adiabática es una **estimación simplificada** con Cp promedio. Para cálculos precisos usa NASA CEA o iteración con Cp(T).
