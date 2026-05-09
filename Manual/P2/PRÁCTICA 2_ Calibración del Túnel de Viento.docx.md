## **PRÁCTICA 2: Calibración del Túnel de Viento**

**I. Introducción**

Los túneles de viento son instalaciones experimentales cruciales en la ingeniería aeronáutica. Permiten simular el flujo de aire sobre aeronaves o sus componentes (perfiles alares, fuselajes, etc.) en un entorno controlado, facilitando el estudio de fenómenos aerodinámicos, la medición de fuerzas y momentos, y la validación de diseños computacionales (CFD). Para que los resultados obtenidos en un túnel de viento sean fiables y extrapolables a condiciones de vuelo reales, es imperativo conocer con precisión la velocidad del flujo en la sección de pruebas.

La calibración de un túnel de viento consiste en establecer una relación funcional entre un parámetro de control del túnel (generalmente relacionado con la potencia o velocidad del ventilador/motor) y la velocidad del aire resultante en la sección de pruebas. Esta relación, conocida como curva de calibración, es fundamental para operar el túnel a las velocidades deseadas.

En esta práctica, se analizarán conjuntos de datos experimentales para determinar la curva de calibración de un túnel de viento. Los datos consisten en mediciones de velocidad del aire, obtenidas mediante Anemometría Láser Doppler (LDA), para diferentes frecuencias de operación del motor del ventilador del túnel. El objetivo es procesar estos datos, generar la curva de calibración y obtener la ecuación matemática que la describe.

**II. Objetivos de Aprendizaje Específicos**

Al finalizar esta práctica, el estudiante será capaz de:

* Comprender el propósito y la metodología general de la calibración de un túnel de viento.  
* Identificar los parámetros clave involucrados en la calibración de un túnel de viento.  
* Procesar datos crudos de velocidad obtenidos mediante LDA para determinar velocidades medias representativas para diferentes condiciones de operación del túnel.  
* Elaborar una tabla de datos que relacione el parámetro de control del túnel (frecuencia del motor) con la velocidad media del flujo.  
* Graficar la curva de calibración del túnel de viento (Velocidad Media vs. Frecuencia del Motor).  
* Aplicar técnicas de regresión lineal para obtener la ecuación matemática de la curva de calibración (V=m⋅f+b).  
* Evaluar la calidad del ajuste lineal mediante el coeficiente de determinación (R2).  
* Utilizar la ecuación de calibración obtenida para predecir la velocidad del flujo o la frecuencia del motor requerida.  
* Discutir las posibles fuentes de error e incertidumbre asociadas al proceso de calibración.

**III. Fundamentos Teóricos**

**A. Túneles de Viento en Aeronáutica**

Un túnel de viento es un dispositivo que genera un flujo de aire controlado y uniforme a través de una sección de pruebas donde se colocan los modelos a estudiar. Son herramientas indispensables para:

* Medir fuerzas aerodinámicas (sustentación, arrastre, momentos).  
* Visualizar patrones de flujo (líneas de corriente, desprendimiento de capa límite).  
* Estudiar la distribución de presiones sobre superficies.  
* Investigar fenómenos de aeroelasticidad y aeroacústica.

Existen diversos tipos de túneles (subsónicos, transónicos, supersónicos, hipersónicos; de circuito abierto o cerrado). Sus componentes principales suelen incluir un sistema de propulsión (motor y ventilador), una cámara de aquietamiento (con mallas y panales para reducir turbulencia), una contracción (para acelerar el flujo y uniformizarlo), la sección de pruebas, y un difusor \[1, 2\].

**B. Calibración de Túneles de Viento**

La calibración es el proceso de establecer una relación precisa y repetible entre el ajuste de control del túnel (ej. frecuencia del variador del motor del ventilador, RPM del ventilador, potencia eléctrica suministrada) y una o más características del flujo en la sección de pruebas, primordialmente la velocidad del aire. Esta relación no siempre es lineal y puede variar con el tiempo debido al desgaste o cambios en los componentes del túnel.

Para la calibración, se utiliza un instrumento de referencia o patrón para medir la velocidad del aire con alta precisión. Los instrumentos comúnmente utilizados incluyen:

* **Tubo de Pitot-Static:** Mide la presión total y estática del flujo, a partir de las cuales se calcula la velocidad utilizando la ecuación de Bernoulli. Es un método robusto para flujos incompresibles y subsónicos \[1\].  
* **Anemometría Láser Doppler (LDA):** Técnica óptica no intrusiva que mide la velocidad de partículas pequeñas que se asume se mueven con el flujo. Es ideal por no perturbar el flujo que mide \[3, 4\].  
* **Anemometría de Hilo Caliente (HWA):** Utiliza un filamento calentado eléctricamente cuya resistencia cambia con la velocidad del flujo debido a la transferencia de calor por convección. Ofrece alta respuesta en frecuencia \[1\].

La calibración generalmente se realiza midiendo la velocidad en el centro de la sección de pruebas (o en varios puntos para evaluar la uniformidad del flujo) para una serie de ajustes del parámetro de control del túnel.

**C. Anemometría Láser Doppler (LDA) – Principio Básico**

La LDA es una técnica óptica que permite medir la velocidad de un fluido de manera no intrusiva. Su funcionamiento se basa en el efecto Doppler: cuando la luz láser es dispersada por partículas pequeñas (naturalmente presentes o sembradas intencionalmente) que se mueven con el flujo, la frecuencia de la luz dispersada cambia en proporción a la velocidad de la partícula.

En una configuración común de LDA (modo de franjas o diferencial), dos haces láser coherentes de la misma longitud de onda se cruzan en un punto, creando un patrón de interferencia de franjas claras y oscuras en el volumen de medición. Cuando una partícula atraviesa estas franjas, dispersa luz con una intensidad que fluctúa a una frecuencia (frecuencia Doppler, fD​) directamente proporcional a la componente de velocidad de la partícula perpendicular a las franjas (U⊥​) y al espaciado de las franjas (δf​):

U⊥​=fD​⋅δf​

El espaciado de las franjas δf​ depende de la longitud de onda del láser (λ) y del semiángulo de cruce de los haces (κ):

δf​=2sin(κ)λ​

El sistema LDA detecta la frecuencia Doppler fD​ y, conociendo δf​, calcula la velocidad. Los datos proporcionados en esta práctica ("LDA1 \[m/s\]") son mediciones de velocidad obtenidas mediante esta técnica \[3, 4\].

**D. Procesamiento Estadístico de Datos de Velocidad LDA**

El paso de partículas a través del volumen de medición LDA es un proceso aleatorio. Por lo tanto, para una condición de flujo estable, el sistema LDA registrará una serie de mediciones de velocidad individuales a lo largo del tiempo. Para obtener un valor representativo de la velocidad del flujo para un ajuste de frecuencia del túnel (fmotor​), es necesario calcular el promedio estadístico (media) de estas mediciones individuales:

Uˉfmotor=N1​∑i=1N​Ui​

Donde:

* Uˉfmotor es la velocidad media del aire para una frecuencia de motor dada.  
* Ui​ es una medición individual de velocidad LDA.  
* N es el número total de mediciones de velocidad tomadas para esa frecuencia de motor.

En los datos proporcionados, las velocidades LDA pueden ser negativas. Esto indica la dirección del flujo relativa al sistema de coordenadas del LDA. Para la calibración de la magnitud de la velocidad (rapidez), se utilizará el valor absoluto de las mediciones:  
Uˉfmotor=N1​∑i=1N​∣(LDA1)i​∣  
**E. Regresión Lineal y Coeficiente de Determinación**

Una vez obtenidas las velocidades medias (Uˉ) para cada frecuencia del motor (fmotor​), se busca una relación matemática entre estas dos variables. Frecuentemente, esta relación es aproximadamente lineal en un rango de operación significativo del túnel:

Uˉ=m⋅fmotor​+b

Donde:

* m es la pendiente de la línea de calibración (sensibilidad de la velocidad a la frecuencia).  
* b es la ordenada al origen (velocidad teórica a fmotor​=0 Hz).

El método de mínimos cuadrados se utiliza para encontrar los valores de m y b que mejor ajustan la línea a los datos experimentales.

El **coeficiente de determinación (**R2**)** es una medida estadística que indica qué tan bien la línea de regresión se ajusta a los datos. Varía entre 0 y 1\. Un valor de R2 cercano a 1 indica que el modelo lineal explica una gran proporción de la variabilidad en la velocidad del aire y que el ajuste es bueno. Un valor cercano a 0 indica un mal ajuste \[5\].

**IV. Metodología Experimental y de Análisis de Datos**

**A. Descripción del Montaje Experimental Típico (Conceptual)**

La calibración de un túnel de viento como el que generó los datos para esta práctica involucraría:

1. **Configuración del Túnel:** Asegurar que la sección de pruebas esté despejada y que la configuración sea la estándar para la cual se desea la calibración.  
2. **Sistema de Control del Ventilador:** Un variador de frecuencia ajusta la frecuencia eléctrica suministrada al motor del ventilador, controlando así su velocidad de rotación.  
3. **Sistema LDA:** La óptica de transmisión del LDA se alinea para que los haces láser se crucen en el punto de interés dentro de la sección de pruebas (usualmente el centro o un punto de referencia). La óptica de recepción se alinea para capturar la luz dispersada por las partículas.  
4. **Siembra de Partículas (si es necesario):** Si el aire ambiente no contiene suficientes partículas naturales, se introduce un generador de aerosol (ej. con aceite DEHS o similar) para sembrar el flujo con partículas trazadoras adecuadas para LDA.  
5. **Adquisición de Datos:**  
   * Se establece una frecuencia del motor (fmotor​) y se espera a que el flujo en el túnel se estabilice.  
   * Se adquieren datos de velocidad con el sistema LDA durante un tiempo determinado para obtener un número suficiente de muestras (validaciones de velocidad).  
   * Se repite este proceso para un rango de frecuencias del motor, cubriendo el rango operativo del túnel.

**Nota de Transición:** Para esta práctica, se proporcionarán archivos de datos (.txt) que contienen las mediciones de velocidad ("LDA1 \[m/s\]") obtenidas a diferentes frecuencias de operación del ventilador del túnel de viento (ej. 5 Hz, 10 Hz, ..., 55 Hz). El enfoque principal será el procesamiento y análisis de estos datos para derivar la curva de calibración.

**B. Descripción de los Conjuntos de Datos (Datasets)**

Se proporcionará un conjunto de archivos de texto (.txt). Cada archivo corresponde a una frecuencia de operación específica del motor del ventilador del túnel de viento. Por ejemplo:

* 5Hz.000001.txt (Datos para fmotor​=5 Hz)  
* 10Hz.000001.txt (Datos para fmotor​=10 Hz)  
* ... y así sucesivamente para las frecuencias 15, 20, 25, 35, (y se espera tener 40, 45, 50, 55 Hz).

El formato de cada archivo es el siguiente:

* Líneas de cabecera que pueden incluir información del software de adquisición (DXEX v3), ruta del archivo original, fecha, hora y coordenadas de medición.  
* Una línea indicando la región de medición.  
* Una línea de encabezado de columnas: "Row\#" (número de muestra), "AT \[ms\]" (tiempo absoluto o relativo de la muestra en milisegundos), "LDA1 \[m/s\]" (medición de velocidad en metros por segundo).  
* Las líneas subsiguientes contienen los datos tabulados.

**Importante:** La columna "LDA1 \[m/s\]" contiene las mediciones de velocidad instantánea. Observe que los valores pueden ser negativos. Para la calibración de la rapidez del túnel, se utilizará el **valor absoluto** de estas mediciones.

**C. Procedimiento de Tratamiento y Análisis de Datos**

Paso 1: Carga e Inspección Inicial de Datos.  
Para cada frecuencia de motor para la cual se dispone de un archivo .txt:

1. **Cargar los Datos:** Utilice una herramienta de software (como Python con la biblioteca Pandas, MATLAB, o una hoja de cálculo como Excel/Google Sheets) para cargar los datos de cada archivo. Ignore las líneas de cabecera que no contienen datos numéricos.  
2. **Extraer Velocidades:** Aísle la columna de datos correspondiente a "LDA1 \[m/s\]".  
3. **Manejo del Signo:** Tome el valor absoluto de cada medición de velocidad, ya que nos interesa la magnitud (rapidez) para la calibración.  
   * **Ejemplo en Python (usando Pandas):**  
     import pandas as pd  
     import numpy as np

     \# Suponiendo que el archivo '5Hz.000001.txt' está en el mismo directorio  
     \# Se omiten las primeras 5 líneas de cabecera y se usa tabulador como delimitador  
     try:  
         data\_5Hz \= pd.read\_csv('5Hz.000001.txt', skiprows=5, delimiter='\\t')  
         \# Asegurarse que la columna de velocidad se interpreta como numérica  
         \# y manejar posibles errores de conversión reemplazándolos con NaN  
         velocities\_5Hz \= pd.to\_numeric(data\_5Hz\['LDA1 \[m/s\]'\], errors='coerce').dropna()  
         \# Tomar el valor absoluto  
         velocities\_5Hz\_abs \= np.abs(velocities\_5Hz)  
         print(f"Datos para 5Hz cargados. Número de muestras de velocidad: {len(velocities\_5Hz\_abs)}")  
         \# print(velocities\_5Hz\_abs.head()) \# Descomentar para ver las primeras velocidades  
     except FileNotFoundError:  
         print("Archivo 5Hz.000001.txt no encontrado.")  
     except KeyError:  
         print("La columna 'LDA1 \[m/s\]' no se encontró o tiene un nombre diferente en el archivo 5Hz.")

**Paso 2: Cálculo de la Velocidad Media y Desviación Estándar.**

1. Para cada frecuencia del motor (fmotor​):  
   * Calcule la velocidad media (Uˉ) de las mediciones de velocidad (absolutas) obtenidas en el Paso 1\.  
   * (Opcional, pero recomendado) Calcule la desviación estándar (σU​) de estas mediciones para tener una idea de la dispersión o turbulencia del flujo.  
   * **Ejemplo en Python (continuación):**  
     \# Suponiendo que velocities\_5Hz\_abs ya está definido  
     if 'velocities\_5Hz\_abs' in locals() and not velocities\_5Hz\_abs.empty:  
         mean\_velocity\_5Hz \= velocities\_5Hz\_abs.mean()  
         std\_velocity\_5Hz \= velocities\_5Hz\_abs.std()  
         print(f"Para 5Hz: Velocidad Media \= {mean\_velocity\_5Hz:.3f} m/s, Desv. Estándar \= {std\_velocity\_5Hz:.3f} m/s")  
     else:  
         print("No se pudieron calcular las estadísticas para 5Hz (datos no cargados o vacíos).")

2. Manejo de Múltiples Archivos por Frecuencia (ej. 35Hz):  
   Para la frecuencia de 35Hz, se proporcionan varios archivos (35Hz.000001.txt, 35Hz.000002.txt, etc.). Calcule la velocidad media para cada archivo individualmente. Luego, calcule el promedio de estas velocidades medias para obtener un valor único y más robusto para 35Hz.  
   * Uˉ35Hz,total=k1​∑j=1kUˉ35Hz,archivoj​​ (donde k es el número de archivos para 35Hz).

Paso 3: Tabulación de Datos para Calibración.  
Cree una tabla que resuma la frecuencia del motor y la velocidad media del aire calculada correspondiente:

| Frecuencia del Motor (Hz) | Velocidad Media Uˉ (m/s) | (Opcional) Desv. Estándar σU​ (m/s) |
| :---- | :---- | :---- |
| 5 | (valor calculado) | (valor calculado) |
| 10 | (valor calculado) | (valor calculado) |
| 15 | (valor calculado) | (valor calculado) |
| 20 | (valor calculado) | (valor calculado) |
| 25 | (valor calculado) | (valor calculado) |
| 35 | (valor promediado) | (promedio de desv. o desv. de medias) |
| (40, 45, 50, 55 Hz...) | (valores calculados) | (valores calculados) |

**Paso 4: Graficación de la Curva de Calibración.**

1. Genere una gráfica de dispersión con la Frecuencia del Motor en el eje horizontal (x) y la Velocidad Media Uˉ en el eje vertical (y).  
2. Asegúrese de etiquetar los ejes correctamente (incluyendo unidades) y titular la gráfica (ej. "Curva de Calibración del Túnel de Viento").  
   * **Ejemplo en Python (usando Matplotlib y Seaborn):**  
     import matplotlib.pyplot as plt  
     import seaborn as sns

     \# Suponiendo que se tiene una lista o DataFrame de frecuencias y velocidades medias  
     \# Ejemplo de datos (reemplazar con los valores calculados)  
     \# frecuencias \= np.array(\[5, 10, 15, 20, 25, 35, 40, 45, 50, 55\]) \# Hz  
     \# velocidades\_medias \= np.array(\[...valores correspondientes...\]) \# m/s

     \# Crear un DataFrame para facilitar el ploteo con Seaborn (opcional pero recomendado)  
     \# df\_calibracion \= pd.DataFrame({'Frecuencia\_Motor\_Hz': frecuencias, 'Velocidad\_Media\_ms': velocidades\_medias})

     \# plt.figure(figsize=(10, 6))  
     \# sns.scatterplot(x='Frecuencia\_Motor\_Hz', y='Velocidad\_Media\_ms', data=df\_calibracion, s=100, label='Datos Medidos')  
     \# plt.title('Curva de Calibración del Túnel de Viento')  
     \# plt.xlabel('Frecuencia del Motor (Hz)')  
     \# plt.ylabel('Velocidad Media del Aire (m/s)')  
     \# plt.grid(True)  
     \# plt.legend()  
     \# plt.show()

     *(Se requerirá que el estudiante complete las listas frecuencias y velocidades\_medias con sus cálculos).*

**Paso 5: Regresión Lineal para Obtener la Ecuación de Calibración.**

1. Aplique un ajuste de regresión lineal a los puntos de datos (fmotor​, Uˉ).  
2. Determine la pendiente (m) y la ordenada al origen (b) de la ecuación Uˉ=m⋅fmotor​+b.  
3. Calcule el coeficiente de determinación (R2).  
   * **Herramientas de Cálculo:**  
     * **Python:**  
       from scipy import stats

       \# Suponiendo que 'frecuencias' y 'velocidades\_medias' son arrays de NumPy  
       \# slope, intercept, r\_value, p\_value, std\_err \= stats.linregress(frecuencias, velocidades\_medias)  
       \# r\_squared \= r\_value\*\*2

       \# print(f"\\n--- Resultados de la Regresión Lineal \---")  
       \# print(f"Ecuación de calibración: V \[m/s\] \= {slope:.4f} \* f \[Hz\] \+ {intercept:.4f}")  
       \# print(f"Pendiente (m): {slope:.4f} (m/s)/Hz")  
       \# print(f"Ordenada al origen (b): {intercept:.4f} m/s")  
       \# print(f"Coeficiente de determinación (R^2): {r\_squared:.4f}")

       \# Para graficar la línea de regresión:  
       \# velocidades\_ajustadas \= slope \* frecuencias \+ intercept  
       \# plt.plot(frecuencias, velocidades\_ajustadas, color='red', label=f'Ajuste lineal: V \= {slope:.2f}f \+ {intercept:.2f}\\nR$^2$ \= {r\_squared:.3f}')  
       \# plt.legend() \# Actualizar la leyenda si se añade la línea al scatter plot anterior  
       \# plt.show() \# Mostrar la gráfica actualizada

     * **Hoja de Cálculo (Excel/Google Sheets):**  
       * Pendiente: \=SLOPE(rango\_velocidades, rango\_frecuencias)  
       * Ordenada al origen: \=INTERCEPT(rango\_velocidades, rango\_frecuencias)  
       * R2: \=RSQ(rango\_velocidades, rango\_frecuencias)  
       * Para graficar, seleccionar los datos, insertar gráfico de dispersión y agregar línea de tendencia lineal mostrando la ecuación y R2.

**Paso 6: Análisis e Interpretación de la Calibración.**

1. **Interpretación de la Ecuación:** Explique el significado físico de la pendiente m (cuántos m/s aumenta la velocidad por cada Hz de incremento en la frecuencia del motor) y de la ordenada al origen b (¿Es la velocidad cero a frecuencia cero? ¿Es físicamente realista?).  
2. **Calidad del Ajuste:** Discuta el valor de R2. Un valor cercano a 1 indica un buen ajuste lineal. Si es bajo, podría indicar no linealidades o una gran dispersión en los datos.  
3. **Uso de la Ecuación:** Explique cómo se utilizaría esta ecuación para configurar el túnel a una velocidad deseada o para conocer la velocidad a una frecuencia dada.

**VI. Actividades a Realizar y Resultados Esperados**

1. **Procesamiento de Datos:** Para cada archivo de datos proporcionado (5Hz, 10Hz, 15Hz, 20Hz, 25Hz, y los cinco archivos de 35Hz, más los de 40, 45, 50, 55Hz si se proporcionan):  
   * Calcular la velocidad media (∣Uˉ∣) y la desviación estándar (σU​).  
   * Para 35Hz, calcular la media de las medias de los cinco archivos.  
2. **Tabla de Calibración:** Presentar una tabla completa con las frecuencias del motor y sus correspondientes velocidades medias y desviaciones estándar.  
3. **Gráfica de Calibración:**  
   * Graficar Velocidad Media ∣Uˉ∣ vs. Frecuencia del Motor.  
   * Incluir la línea de regresión lineal en la misma gráfica.  
   * Mostrar la ecuación de la línea y el valor de R2 en la gráfica.  
4. **Ecuación de Calibración:** Indicar claramente la ecuación de calibración obtenida (∣Uˉ∣=m⋅fmotor​+b) con los valores numéricos de m y b y sus unidades.  
5. **Coeficiente de Determinación:** Indicar el valor de R2.  
6. **Respuestas al Caso de Estudio y al Cuestionario/Preguntas de Reflexión.**

**VII. Caso de Estudio / Aplicación Práctica**

1. Utilizando la ecuación de calibración obtenida:  
   * Determine la frecuencia del motor (en Hz) necesaria para obtener una velocidad del aire de 12 m/s en la sección de pruebas.  
   * Si el motor se ajusta a una frecuencia de 28 Hz, ¿qué velocidad del aire se esperaría en la sección de pruebas?  
2. Suponga que se va a realizar un ensayo con un modelo de automóvil a escala 1:10. Si el automóvil real viaja a 108 km/h (30 m/s), y se desea mantener el mismo número de Reynolds en el túnel de viento (asumiendo que la temperatura del aire y, por lo tanto, su viscosidad cinemática, son las mismas que en condiciones reales, y que la longitud característica del modelo es 1/10 de la del automóvil real):  
   * ¿A qué velocidad (Utunel​) se debería operar el túnel de viento? (Recordar: Re=νUL​. Si Remodelo​=Rereal​, entonces νUmodelo​Lmodelo​​=νUreal​Lreal​​).  
   * ¿Qué frecuencia del motor se necesitaría para alcanzar esta velocidad en el túnel, según su calibración?

**VIII. Discusión en el Contexto Aeronáutico**

* Discutir por qué una calibración precisa del túnel es fundamental antes de realizar cualquier prueba aerodinámica para el diseño o certificación de componentes de aeronaves.  
* ¿Qué factores en un laboratorio aeronáutico podrían hacer que la calibración de un túnel de viento cambie con el tiempo, necesitando recalibraciones periódicas?  
* La calibración se realizó en un punto (o se asume que los datos LDA son representativos del flujo principal). ¿Por qué podría ser importante mapear la velocidad en varios puntos de la sección de prueba (estudio de uniformidad de flujo)?  
* Comparar brevemente la técnica LDA (usada como referencia aquí) con el uso de un Tubo de Pitot para la calibración. ¿Cuáles son las ventajas y desventajas relativas en este contexto?

**IX. Cuestionario / Preguntas de Reflexión**

1. ¿Por qué es importante calcular la velocidad *media* a partir de múltiples mediciones LDA para cada frecuencia del motor, en lugar de usar una única medición?  
2. Si el valor de R2 obtenido en la regresión lineal fuera de 0.85, ¿qué indicaría sobre la relación entre la frecuencia del motor y la velocidad del aire? ¿Sería aceptable esta calibración?  
3. La ordenada al origen (b) en su ecuación de calibración, ¿qué valor tiene? ¿Es cercano a cero? Discuta el significado físico de este valor. ¿Esperaría que un túnel de viento produzca flujo si la frecuencia del motor es cero?  
4. Si la temperatura del aire en el laboratorio aumenta significativamente un día de verano, ¿cómo podría esto afectar la densidad del aire? ¿Y cómo podría, a su vez, afectar la velocidad real del aire para una misma frecuencia del motor, comparado con su calibración original (asumiendo que la potencia del ventilador no cambia)?  
5. Mencione dos tipos específicos de ensayos aeronáuticos que se benefician directamente de una calibración precisa del túnel de viento y explique brevemente por qué.

**X. Requisitos del Reporte (Formato AIAA \- Adaptado)**

Elabore un reporte técnico que incluya:

* **Título, Autores, Afiliación, Fecha.**  
* **Abstract (Resumen):** Objetivo de la práctica (calibración del túnel mediante análisis de datos Frecuencia-Velocidad LDA), metodología resumida (procesamiento de datos, regresión lineal), principales resultados (tabla de velocidades medias, gráfica de calibración, ecuación obtenida, valor de R2), y una conclusión sobre la relación encontrada y la importancia de la calibración.  
* **1\. Introducción:** Breve descripción de la importancia de los túneles de viento en aeronáutica y el propósito de su calibración. Objetivos de la práctica.  
* **2\. Marco Teórico:** Resumen de los principios de funcionamiento de túneles de viento, el concepto de calibración, una breve descripción de LDA, y los fundamentos de la regresión lineal y el R2.  
* **3\. Metodología de Análisis de Datos:**  
  * Descripción de los datos proporcionados (formato de archivos, variables).  
  * Pasos detallados del procesamiento de datos: cómo se cargaron los datos, cómo se calcularon las velocidades medias (y desviaciones estándar si se incluyen), y cómo se manejaron los datos de 35Hz.  
  * Descripción del método de regresión lineal utilizado.  
* **4\. Resultados:**  
  * Tabla completa de Frecuencia del Motor vs. Velocidad Media (y σU​ opcional).  
  * Gráfica de la curva de calibración, mostrando los puntos experimentales (medias) y la línea de regresión ajustada. La gráfica debe estar correctamente etiquetada, con la ecuación de la línea y el valor de R2 mostrados.  
  * Presentación clara de la ecuación de calibración final: Uˉ=m⋅fmotor​+b, con los valores de m y b y sus unidades.  
  * Valor del coeficiente de determinación R2.  
  * Resultados y respuestas detalladas del Caso de Estudio.  
* **5\. Discusión:**  
  * Interpretación de los parámetros de la ecuación de calibración (m,b).  
  * Análisis de la calidad del ajuste (R2).  
  * Aplicabilidad de la curva de calibración y sus limitaciones.  
  * Posibles fuentes de error e incertidumbre en un proceso real de calibración de túnel de viento.  
  * Relevancia de la calibración para la ingeniería aeronáutica.  
* **6\. Conclusiones:** Resumen de los hallazgos principales, la ecuación de calibración obtenida y su validez (basada en R2). Importancia de la práctica.  
* **7\. Referencias:** (Si se consultaron fuentes adicionales).  
* **Apéndices (Opcional):** Por ejemplo, scripts de Python utilizados, o tablas detalladas de todas las lecturas LDA si fuera necesario (generalmente no se requiere para este tipo de reporte, la tabla de medias es suficiente).

**XI. Referencias Sugeridas**

1. Barlow, J. B., Rae, W. H., & Pope, A. (1999). *Low-Speed Wind Tunnel Testing* (3rd ed.). John Wiley & Sons.  
2. Pope, A., & Goin, K. L. (1978). *High-Speed Wind Tunnel Testing*. Robert E. Krieger Publishing Company.  
3. Albrecht, H.-E., Borys, M., Damaschke, N., & Tropea, C. (2003). *Laser Doppler and Phase Doppler Measurement Techniques*. Springer-Verlag.  
4. Durst, F., Melling, A., & Whitelaw, J. H. (1981). *Principles and Practice of Laser-Doppler Anemometry* (2nd ed.). Academic Press.  
5. Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). *Introduction to Linear Regression Analysis* (5th ed.). John Wiley & Sons.  
6. Figliola, R. S., & Beasley, D. E. (2011). *Theory and Design for Mechanical Measurements* (5th ed.). John Wiley & Sons. (Contiene capítulos sobre análisis de incertidumbre y planificación de experimentos).