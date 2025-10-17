# Prácticas de Técnicas de Medida (Resumen y Metodologías)

Este documento resume las 9 prácticas incluidas en la carpeta `Practicas/` y propone metodologías concisas para analizarlas. Donde aplique, se sugieren técnicas de adquisición, procesamiento (incluidos filtros para imágenes y señales) y validación de resultados.

> Nota: Los PDFs en `Practicas/` contienen el detalle completo. Aquí se listan objetivos y guías prácticas para ejecutar y analizar datos/imágenes.

---

## 1) Protección ocular frente a radiación láser
- Objetivo: Comprender riesgos, normativas y selección de gafas/OD adecuados para distintas longitudes de onda y potencias.
- Metodología de análisis:
  - Identificar λ del láser, potencia/salida (CW vs. pulsado) y camino óptico.
  - Determinar OD requerido por norma (EN 207/208, ANSI) y comparar con especificaciones de gafas.
  - Verificar ancho de banda de atenuación y compatibilidad con el láser utilizado.
  - Evaluar riesgos residuales (reflejos especulares/difractivos) y controles administrativos.
  - Entregable: Tabla de láseres vs. OD y una matriz de riesgos con medidas de mitigación.

## 2) Calibración del Túnel de Viento
- Objetivo: Establecer la relación entre ajustes de túnel (RPM/compuertas) y velocidad/caudal.
- Metodología de análisis:
  - Adquisición: Medir presión diferencial (Pitot-estático) y temperatura/ρ para convertir a V.
  - Procesado: Promedio temporal; eliminar outliers (IQR) y correcciones por densidad/temperatura.
  - Curva de calibración: Ajuste V = f(RPM) con regresión (lineal o polinómica de bajo orden).
  - Validación cruzada: Comparar con anemómetro independiente (hilo caliente o referencia).
  - Incertidumbre: Propagación de errores (manómetro, alineación, temperatura).

## 3) Anemometría de Hilo Caliente
- Objetivo: Medir perfiles de velocidad y fluctuaciones (turbulencia) en el túnel.
- Metodología de análisis:
  - Calibración: Relación voltaje-velocidad (ley de King); curva V(E) con flujo de referencia.
  - Adquisición: Alta frecuencia (≥10× fcontenidos) y duración suficiente para estadísticos.
  - Filtrado de señal: Paso-bajo/antialiasing; eliminación de drift con detrending.
  - Cálculos: Umean, u' RMS, intensidad de turbulencia, espectros (FFT) si aplica.
  - Incertidumbre: Sensibilidad a temperatura y orientación del sensor.

## 4) Anemometría Láser Doppler (LDA/LDV) – Perfil de Velocidades
- Objetivo: Obtener perfiles de velocidad puntual sin intrusión.
- Metodología de análisis:
  - Configuración óptica: Cruce de haces, ángulo y factor de escala (franja de interferencia).
  - Preprocesado: Validar calidad de señales (SNR) y rechazo de puntos espurios.
  - Estadística: Promedios y desviaciones por posición; densidad de muestras por bin.
  - Perfil: Trazar U(y) y comparar contra teoría/canales o datos de túnel calibrado.
  - Notas: Corrección por índice del medio, alineación y seeding adecuado.

## 5) Análisis de Imagen
- Objetivo: Extraer métricas de imágenes (e.g., bordes, áreas, intensidades) para casos experimentales.
- Metodología de análisis (pipeline típico):
  - Preprocesado: Conversión a escala de grises, corrección de iluminación (homomorphic o rolling ball), normalización.
  - Filtros: Suavizado (Gaussian/median) para ruido; realce (unsharp mask) si requiere.
  - Segmentación: Umbral global (Otsu) o adaptativo; morfología (erode/dilate/open/close) para limpieza.
  - Medidas: Detección de bordes (Canny/Sobel), contornos, áreas, perímetros, centroides.
  - Validación: GT manual en subconjunto; sensibilidad a umbrales y parámetros de filtro.
  - Herramientas sugeridas: OpenCV, scikit-image, MATLAB Image Processing.

## 6) Velocimetría de Partículas por Imagen (PIV)
- Objetivo: Medir campos de velocidad a partir de imágenes con partículas trazadoras.
- Metodología de análisis:
  - Preprocesado: Sustracción de fondo; ecualización de histograma; filtro pasa-banda para realzar seeding.
  - PIV: Enfoque por ventanas de interrogación (multi-pass, decreasing window size), subpixel peak fitting.
  - Postprocesado: Detección/corrección de outliers (universal median filter); suavizado vectorial.
  - Derivados: Vorticidad, líneas de corriente; validación con balances y condiciones de contorno.
  - Parámetros clave: Δt entre pares, densidad de partículas, tamaño de ventana, overlap.
  - Herramientas: OpenPIV, PIVlab (MATLAB), OpenCV + librerías PIV.

## 7) Termografía
- Objetivo: Medir distribución de temperatura superficial mediante cámara IR.
- Metodología de análisis:
  - Calibración: Emisividad del material, temperatura ambiente y reflejada; corrección de distancia y humedad.
  - Preprocesado: Correcciones radiométricas; suavizado espacial leve si hay ruido.
  - Segmentación/medidas: Identificar hotspots, isoterma, gradientes; extracción de perfiles T(x,y).
  - Validación: Termopar/punto negro como referencia; evaluación de incertidumbre.
  - Resultados: Mapas térmicos con escalas claras y barras de color calibradas.

## 8) Análisis de Productos de Combustión
- Objetivo: Caracterizar gases (CO2, CO, NOx, O2, hidrocarburos) y eficiencia de combustión.
- Metodología de análisis:
  - Muestreo: Puntos representativos; corrección por dilución y condiciones estándar (STP).
  - Instrumentación: Analizadores específicos; calibración con gases patrón.
  - Cálculos: Razón aire-combustible (AFR), λ, rendimiento térmico; balance elemental si aplica.
  - Calidad de datos: Deriva del sensor, tiempo de respuesta, repetibilidad.
  - Reporte: Tablas comparativas por condición y gráfico de emisiones vs. carga/φ.

## 9) Promedio Temporal
- Objetivo: Derivar promedios y estadísticos robustos de señales no estacionarias o ruidosas.
- Metodología de análisis:
  - Estacionariedad: Tests (ADF, KPSS) o inspección de segmentos; elegir ventanas adecuadas.
  - Filtrado: Pasa-bajo como antirruido; evitar desfases (filtros cero-fase) para análisis temporal.
  - Promedios: Móvil, por bloques, sincronizado por evento (phase-averaging) si hay periodicidad.
  - Incertidumbre: Intervalos de confianza (bootstrapping) y estimación de error de la media.
  - Visualización: Envelopes, bandas de confianza; comparación multi-condición.

---

## Recomendaciones generales de análisis
- Control de versión de datos: Mantener metadatos (fecha, condiciones, sensores, calibración).
- Propagación de incertidumbre: Documentar fuentes y estimar su contribución.
- Reproducibilidad: Scripts/notebooks con parámetros versionados.
- Visualización efectiva: Etiquetas claras, unidades SI, barras de error cuando corresponda.

## Referencias de software sugeridas
- Python: NumPy, SciPy, Pandas, Matplotlib/Seaborn, scikit-image, OpenCV, OpenPIV.
- MATLAB: Signal Processing Toolbox, Image Processing Toolbox, PIVlab.

Si deseas, puedo convertir estas metodologías en checklists operativas o plantillas de notebook/código para procesar tus datos/imágenes.
