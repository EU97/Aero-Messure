# P04 - Anemometría Láser Doppler (LDA): Perfil de Velocidades

## Objetivo
Medir perfiles de velocidad U(y) en túnel de viento mediante técnica láser Doppler (LDA/LDV), una técnica óptica **no intrusiva** que permite obtener mediciones puntuales de alta precisión sin perturbar el flujo.

## Fundamento
La Anemometría Láser Doppler se basa en el efecto Doppler de la luz dispersada por partículas trazadoras. Dos haces láser coherentes se cruzan formando un volumen de medida donde se generan franjas de interferencia. La frecuencia Doppler de la luz dispersada es proporcional a la velocidad de la partícula.

**Ventajas:**
- ✅ No intrusiva (no perturba el flujo)
- ✅ Alta resolución espacial y temporal
- ✅ Medición directa (no requiere calibración)
- ✅ Puede medir flujos inversos y turbulencia

## Estructura de Datos

**Ubicación:** `files/P4/`

Contiene 4 carpetas (FX01G00 a FX04G00), cada una representa una **posición Y diferente** en el perfil del túnel. Dentro de cada carpeta hay ~15 archivos `.txt` con ~2000 mediciones cada uno.

**Formato de archivos** (5 columnas sin encabezado):
1. Índice de muestra
2. **Velocidad U (m/s)** - componente principal del flujo
3. Dato auxiliar (posiblemente tiempo o frecuencia)
4. **Velocidad V (m/s)** - componente transversal
5. Dato auxiliar (validación o calidad)

## Análisis Implementado

El notebook `notebooks/P04_LDA_Perfil_Velocidad.ipynb` realiza:

1. **Exploración de datos**: lectura y visualización de archivos individuales
2. **Filtrado de outliers**: método IQR (Rango Intercuartílico) para eliminar mediciones espurias
3. **Procesamiento por posición**: agregar todos los archivos de cada carpeta y calcular estadísticos
4. **Perfil U(y)**: construcción del perfil de velocidades con barras de error
5. **Análisis de turbulencia**: cálculo de intensidad de turbulencia $I_u = \sigma_U / U_{mean} \times 100\%$
6. **Calidad de datos**: coeficiente de variación y box plots comparativos
7. **Ajuste teórico**: comparación con perfil parabólico (flujo laminar) o logarítmico (turbulento)
8. **Exportación**: resultados en CSV para análisis posterior

## Ejecutar el Análisis

### Opción 1: Jupyter Notebook (recomendado)
```powershell
# Desde la raíz del repositorio
jupyter notebook
# Abrir: Practicas/P04_LDA_Perfil_Velocidad/notebooks/P04_LDA_Perfil_Velocidad.ipynb
```

### Opción 2: Script Python
Si prefieres ejecutar como script, los módulos en `src/` están disponibles.

## Resultados Esperados

- **Perfil U(y)**: gráfico de velocidad vs posición vertical con barras de error
- **Intensidad de turbulencia**: valores típicos 5-15% en túneles de viento
- **Ajuste de modelo**: R² > 0.9 indica buen ajuste a perfil teórico
- **Archivos CSV**: 
  - `data/perfil_velocidades_LDA.csv` - tabla resumen
  - `data/datos_completos_*.csv` - datos completos por posición

## Interpretación

- **Perfil parabólico**: sugiere flujo laminar o en desarrollo
- **Perfil logarítmico**: característico de flujo turbulento desarrollado con capa límite
- **Turbulencia alta en centro**: indica mezcla activa
- **Turbulencia baja**: flujo más estable y predecible

## Pasos de Análisis Manual

Si deseas analizar manualmente:
1. Ajustar `y_positions` en el notebook con las posiciones Y reales (mm)
2. Ejecutar todas las celdas secuencialmente
3. Revisar gráficos y métricas de calidad
4. Exportar resultados y comparar con teoría

## Referencias
- Durst, F. et al. (1981). *Principles and Practice of Laser-Doppler Anemometry*
- Adrian, R.J. & Westerweel, J. (2011). *Particle Image Velocimetry*
- Pope, S.B. (2000). *Turbulent Flows* - Cap. 7: Mediciones
