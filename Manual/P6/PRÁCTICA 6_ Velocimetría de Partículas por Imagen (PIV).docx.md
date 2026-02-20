# **PRÁCTICA 6: Velocimetría de Partículas por Imagen (PIV)**

**I. Introducción**

La Velocimetría por Imágenes de Partículas (PIV, por sus siglas en inglés) es una técnica óptica no intrusiva de gran relevancia en la ingeniería aeronáutica y otras disciplinas de la mecánica de fluidos. Permite obtener campos de velocidad instantáneos en una sección de un flujo mediante el análisis del movimiento de partículas trazadoras. Estas mediciones son cruciales para la comprensión y caracterización de fenómenos fluidodinámicos complejos, tales como la formación de vórtices, el desprendimiento de la capa límite, las interacciones de estelas y la turbulencia. A diferencia de las técnicas de punto único (como LDA o HWA), PIV proporciona una visión global del campo de flujo en un plano o volumen.

En esta práctica, se utilizarán pares de imágenes sintéticas PIV. Estas imágenes, aunque no provienen directamente de un túnel de viento, simulan el tipo de datos que se obtendrían y son ideales para familiarizarse con los algoritmos de procesamiento y análisis fundamentales de la técnica PIV. El enfoque principal será la estimación del campo de velocidad a partir de estos pares de imágenes utilizando algoritmos de flujo óptico y la posterior derivación de cantidades fluidodinámicas de interés.

**II. Objetivos de Aprendizaje Específicos**

Al finalizar esta práctica, el estudiante será capaz de:

* Comprender los principios fundamentales de la técnica de Velocimetría de Partículas por Imagen (PIV) y su aplicabilidad en la ingeniería aeronáutica.  
* Identificar los componentes básicos de un sistema PIV y el propósito de cada uno.  
* Procesar pares de imágenes PIV (sintéticas en este caso) para estimar campos de velocidad bidimensionales utilizando algoritmos de flujo óptico.  
* Visualizar los campos de velocidad resultantes mediante mapas de vectores (quiver plots) y líneas de corriente.  
* Calcular magnitudes derivadas del campo de velocidad, como la vorticidad, e interpretar su significado físico.  
* Interpretar las estructuras y la física del flujo a partir de los campos vectoriales y escalares obtenidos.  
* Reconocer las ventajas y limitaciones de PIV y las técnicas de flujo óptico en el análisis de flujos.

**III. Fundamentos Teóricos**

**A. Principio de Funcionamiento de PIV**

La PIV es una técnica de medición de campo completo que determina la velocidad de un fluido iluminando partículas trazadoras suspendidas en él. Los pasos básicos son:

1. **Siembra del Flujo**: Se introducen pequeñas partículas trazadoras en el fluido, que se asume siguen fielmente el movimiento del flujo sin alterarlo significativamente.  
2. **Iluminación**: Una lámina láser (generalmente de un láser pulsado Nd:YAG) ilumina un plano delgado del flujo dos veces en un corto intervalo de tiempo conocido (Δt).  
3. **Captura de Imágenes**: Una cámara digital (CCD o CMOS), sincronizada con los pulsos láser, captura dos imágenes consecutivas de las partículas iluminadas. La primera imagen se toma con el primer pulso láser y la segunda con el segundo pulso.  
4. **Análisis de Imágenes**: Las dos imágenes se dividen en pequeñas subregiones llamadas "ventanas de interrogación". Dentro de cada ventana, se busca el desplazamiento promedio de los grupos de partículas entre la primera y la segunda imagen. Este desplazamiento (Δx) se determina típicamente mediante algoritmos de correlación cruzada estadística.  
5. Cálculo de la Velocidad: Conocido el desplazamiento promedio (Δx) y el intervalo de tiempo entre pulsos (Δt), el vector de velocidad (V) para esa ventana de interrogación se calcula como:  
   V=ΔtΔx​

   Este proceso se repite para todas las ventanas de interrogación, generando un mapa vectorial del campo de velocidad en el plano iluminado.

**B. Flujo Óptico como Alternativa Computacional para PIV**

El flujo óptico es un concepto de visión por computadora que describe el movimiento aparente de los objetos, superficies y bordes en una secuencia de imágenes causado por el movimiento relativo entre un observador (la cámara) y la escena. En el contexto de PIV, las "superficies" son los patrones de partículas. Los algoritmos de flujo óptico estiman el movimiento entre dos imágenes consecutivas.

Existen varios algoritmos para calcular el flujo óptico. Uno comúnmente utilizado es el **algoritmo de Farnebäck**, que aproxima el vecindario de cada píxel con un polinomio y luego estima el desplazamiento de estos polinomios mediante un método basado en la expansión polinómica. Otro método clásico es el de **Lucas-Kanade**, que asume que el flujo es esencialmente constante en un vecindario local del píxel bajo consideración y resuelve la ecuación básica del flujo óptico para todos los píxeles en ese vecindario, mediante el método de mínimos cuadrados.

Estos métodos pueden ser computacionalmente menos intensivos que la correlación cruzada tradicional para campos densos y pueden ser una alternativa útil, especialmente con imágenes sintéticas o cuando se requiere una estimación densa del vector de velocidad (un vector por píxel o casi por píxel, en lugar de por ventana de interrogación).

**C. Derivadas del Campo de Velocidad**

Una vez que se ha obtenido el campo de velocidad bidimensional V(x,y)=u(x,y)i^+v(x,y)j^​, donde u es la componente de velocidad en x y v es la componente en y, se pueden calcular varias magnitudes derivadas importantes para el análisis del flujo:

1. Vorticidad (ωz​): La vorticidad es una medida de la rotación local del fluido. Para un flujo bidimensional en el plano xy, la única componente no nula de la vorticidad es la perpendicular a este plano (ωz​):  
   ωz​=∂x∂v​−∂y∂u​

   Una vorticidad positiva indica rotación antihoraria, mientras que una negativa indica rotación horaria. La vorticidad es fundamental para identificar vórtices y regiones de alta cizalladura.  
2. Magnitud de Velocidad (∣V∣): Es la rapidez del flujo en cada punto:  
   ∣V∣=u2+v2​

   Permite identificar regiones de alta y baja velocidad.  
3. Líneas de Corriente (ψ): Son curvas que son tangentes instantáneamente al vector de velocidad en cada punto. Para un flujo bidimensional, se pueden obtener a partir de la función de corriente ψ, donde:  
   u=∂y∂ψ​yv=−∂x∂ψ​

   Las líneas de corriente son útiles para visualizar la dirección y la forma del flujo.

**IV. Metodología de Análisis de Datos**

**A. Descripción del Conjunto de Datos (Datasets)**

Se utilizarán pares de imágenes sintéticas que simulan partículas trazadoras en un flujo. En esta práctica, se proporcionan archivos de imagen en formato JPG (aunque en aplicaciones reales, formatos sin pérdida como TIFF son preferibles para mantener la integridad de los datos de intensidad de las partículas). Cada par de imágenes representa dos instantes de tiempo sucesivos. Por ejemplo:

* rankine\_vortex03\_0.jpg (imagen en tiempo t0​)  
* rankine\_vortex03\_1.jpg (imagen en tiempo t0​+Δt)

Se espera que el estudiante procese al menos un par de estas imágenes. Las imágenes proporcionadas son:

* rankine\_vortex02\_1.jpg  
* rankine\_vortex03\_0 (2).jpg  
* rankine\_vortex03\_0.jpg  
* rankine\_vortex03\_1.jpg  
* rankine\_vortex05\_1.jpg

El estudiante deberá seleccionar un par coherente para el análisis (e.g., rankine\_vortex03\_0.jpg y rankine\_vortex03\_1.jpg).

**B. Procedimiento de Tratamiento y Análisis de Datos**

**Paso 1: Carga e Inspección Inicial de Imágenes.**

1. **Cargar las Imágenes**: Utilice una herramienta de software (Python con OpenCV) para cargar el par de imágenes PIV seleccionadas. Es crucial cargarlas en escala de grises, ya que la información de color no es relevante para los algoritmos de flujo óptico basados en intensidad y puede añadir complejidad innecesaria.  
2. **Verificación**: Asegúrese de que ambas imágenes tengan las mismas dimensiones y tipo de datos. Inspeccione visualmente las imágenes para identificar las partículas y tener una idea preliminar de la dirección del movimiento.

**Paso 2: Estimación del Campo de Velocidad (Flujo Óptico).**

1. **Aplicar Algoritmo de Flujo Óptico**: Utilice la función cv2.calcOpticalFlowFarneback de OpenCV para calcular el campo de flujo óptico denso entre las dos imágenes. Este algoritmo devuelve un arreglo 2D donde cada elemento contiene las componentes (u,v) del vector de desplazamiento para el píxel correspondiente.  
2. **Ajuste de Parámetros**: El algoritmo de Farnebäck tiene varios parámetros (e.g., pyr\_scale, levels, winsize, iterations, poly\_n, poly\_sigma). Experimentar con estos parámetros puede ser necesario para obtener resultados óptimos, aunque los valores por defecto suelen ser un buen punto de partida.

**Paso 3: Cálculo de Magnitudes Derivadas.**

1. **Separar Componentes de Velocidad**: Extraiga las componentes u(x,y) y v(x,y) del campo de flujo óptico resultante.  
2. **Calcular Vorticidad**: Calcule la componente z de la vorticidad (ωz​) utilizando diferencias finitas para aproximar las derivadas parciales (∂x∂v​ y ∂y∂u​). Se puede usar np.gradient para esto.

**Paso 4: Visualización de Resultados.**

1. **Campo Vectorial (Quiver Plot)**: Genere una gráfica de dispersión de vectores (quiver plot) para visualizar el campo de velocidad. Es recomendable submuestrear el campo vectorial para la visualización, ya que graficar un vector en cada píxel resultaría en una imagen demasiado densa e ilegible.  
2. **Mapa de Vorticidad**: Represente el campo de vorticidad como una imagen o un mapa de contorno, utilizando un mapa de colores divergente (e.g., 'bwr' \- azul-blanco-rojo) para distinguir fácilmente entre rotación horaria y antihoraria.  
3. **(Opcional) Líneas de Corriente**: Grafique líneas de corriente sobre el campo de velocidad para ayudar a visualizar la trayectoria de las partículas.

Paso 5: Interpretación Física.  
Analice las visualizaciones para identificar características importantes del flujo, como la ubicación y la intensidad de los vórtices, regiones de flujo uniforme, zonas de aceleración o deceleración, y cualquier otra estructura de flujo aparente.  
**V. Actividades a Realizar y Resultados Esperados**

1. **Configuración del Entorno**: Asegúrese de que Python y las bibliotecas necesarias (OpenCV, NumPy, Matplotlib, SciPy) estén instaladas.  
2. **Selección y Carga de Imágenes**:  
   * Seleccione un par de imágenes PIV sintéticas de las proporcionadas (e.g., rankine\_vortex03\_0.jpg y rankine\_vortex03\_1.jpg).  
   * Modifique el código Python (ver Apéndice A) para cargar estas dos imágenes en escala de grises.  
3. **Procesamiento del Flujo Óptico**:  
   * Ejecute el código para calcular el flujo óptico entre el par de imágenes utilizando el método de Farnebäck.  
   * Observe los campos de componentes de velocidad u y v generados.  
4. **Cálculo de Vorticidad**:  
   * Calcule el campo de vorticidad a partir de las componentes de velocidad u y v.  
5. **Visualización y Documentación**:  
   * Genere y guarde las siguientes visualizaciones:  
     * El campo de velocidad superpuesto a una de las imágenes originales (usando plt.quiver). Ajuste el parámetro step y scale para una visualización clara.  
     * Un mapa de colores de la magnitud de la velocidad.  
     * Un mapa de colores del campo de vorticidad. Use un mapa de colores divergente.  
   * (Opcional) Genere una visualización de líneas de corriente.  
6. **Análisis e Interpretación**:  
   * Para el par de imágenes analizado, describa las principales características del flujo observadas.  
   * Identifique regiones de alta y baja velocidad.  
   * Localice e interprete las regiones de alta vorticidad (positiva y negativa). ¿Corresponden a estructuras de vórtice esperadas (por ejemplo, si el nombre del archivo sugiere un tipo de flujo como "rankine\_vortex")?  
   * Documente cualquier observación física relevante sobre la estructura del flujo simulado.

**Resultados Esperados:**

* Un script de Python funcional que procesa el par de imágenes PIV.  
* Figuras claras y bien etiquetadas del campo de velocidad (vectores), magnitud de velocidad y campo de vorticidad.  
* Una breve descripción escrita de las estructuras de flujo identificadas y su interpretación física para el caso analizado.

**VI. Discusión en el Contexto Aeronáutico**

* Considerando las estructuras de flujo identificadas (e.g., vórtices, regiones de cizalladura, estelas), discuta cuál podría ser su relevancia si este flujo ocurriera sobre o alrededor de un componente aeronáutico (e.g., un perfil alar, un flap, un fuselaje).  
* ¿Cómo podría extenderse esta técnica PIV (o PIV estereoscópica/tomográfica) para estudiar flujos tridimensionales más complejos, como las estelas detrás de alas completas, rotores de helicóptero o dentro de cámaras de combustión de turbinas?  
* ¿Cuáles son las principales limitaciones de utilizar algoritmos de flujo óptico como sustituto de las técnicas de correlación cruzada PIV tradicionales, especialmente al analizar datos experimentales ruidosos de túneles de viento reales en lugar de imágenes sintéticas limpias?  
* Piense en un problema específico en aeronáutica (e.g., optimización de la eficiencia de un ala, reducción del ruido de un tren de aterrizaje). ¿Cómo podría utilizarse PIV para abordar experimentalmente este problema?

**VII. Cuestionario / Preguntas de Reflexión**

1. ¿Por qué es preferible utilizar formatos de imagen sin pérdida (como TIFF) para PIV en lugar de formatos con pérdida (como JPG) en un entorno experimental riguroso?  
2. El algoritmo de Farnebäck calcula un campo de flujo "denso". ¿Qué significa esto en comparación con los métodos PIV tradicionales basados en ventanas de interrogación que producen un campo de vectores más espaciado?  
3. ¿Qué representa físicamente el parámetro Δt en un experimento PIV real? ¿Cómo se elegiría un Δt apropiado?  
4. Si las partículas trazadoras utilizadas en un experimento PIV fueran significativamente más densas que el fluido, ¿cómo afectaría esto la precisión de las mediciones de velocidad del fluido?  
5. Mencione dos fuentes de error comunes en las mediciones PIV experimentales y cómo podrían mitigarse.

**VIII. Requisitos del Reporte (Formato similar al de la Práctica 2\)**

Elabore un reporte técnico que incluya:

* **Título, Autores, Afiliación, Fecha.**  
* **Abstract (Resumen):** Objetivo de la práctica, metodología resumida, principales resultados y una conclusión sobre la aplicabilidad de la técnica.  
* **1\. Introducción:** Importancia de PIV en aeronáutica, propósito de la práctica y objetivos específicos.  
* **2\. Marco Teórico:** Principios de PIV, flujo óptico (Farnebäck) y definición de magnitudes derivadas (vorticidad).  
* **3\. Metodología de Análisis de Datos:**  
  * Descripción de los datos proporcionados.  
  * Pasos detallados del procesamiento: carga de imágenes, selección de par, parámetros clave del algoritmo de flujo óptico y cálculo de magnitudes derivadas.  
  * Software y bibliotecas utilizadas.  
* **4\. Resultados:**  
  * Par de imágenes seleccionado.  
  * Figuras de alta calidad (campo de vectores de velocidad, mapa de magnitud de velocidad, mapa de vorticidad).  
  * Descripción detallada de las estructuras de flujo observadas.  
* **5\. Discusión:**  
  * Interpretación de los resultados en el contexto de la mecánica de fluidos.  
  * Respuestas a las preguntas de la sección "VI. Discusión en el Contexto Aeronáutico".  
  * Limitaciones del análisis realizado.  
* **6\. Conclusiones:** Resumen de hallazgos, utilidad de PIV y flujo óptico, y relevancia de la práctica.  
* **7\. Referencias:** (Si se consultaron fuentes adicionales).  
  * SPID: Synthetic Particle Image Dataset. Zenodo. DOI: 10.5281/zenodo.7935215  
  * Raffel, M., Willert, C. E., Scarano, F., Kähler, C. J., Wereley, S. T., & Kompenhans, J. (2018). *Particle Image Velocimetry: A Practical Guide* (3rd ed.). Springer.  
  * OpenCV Documentation: [https://docs.opencv.org](https://docs.opencv.org)  
* **Apéndices:** Incluir el código Python utilizado (referenciar al Apéndice A de este manual).

**IX. Referencias Sugeridas (Adicionales a las del reporte)**

1. Adrian, R. J. (1991). Particle-imaging techniques for experimental fluid mechanics. *Annual Review of Fluid Mechanics, 23*(1), 261-304.  
2. Westerweel, J. (1997). Fundamentals of digital particle image velocimetry. *Measurement Science and Technology, 8*(12), 1379\.  
3. Barlow, J. B., Rae, W. H., & Pope, A. (1999). *Low-Speed Wind Tunnel Testing* (3rd ed.). John Wiley & Sons.

## **Apéndice A: Código de Ejemplo en Python** 

\# Importación de bibliotecas necesarias  
import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.ndimage import gaussian\_filter \# Aunque no se usa explícitamente abajo, es útil para suavizar

\# \--- Paso 1: Carga e Inspección Inicial de Imágenes \---  
\# Modifica los nombres de archivo según el par que estés analizando.  
\# Se recomienda usar imágenes que representen una secuencia, por ejemplo:  
\# 'rankine\_vortex03\_0.jpg' y 'rankine\_vortex03\_1.jpg'

\# Ruta a la primera imagen (tiempo t0)  
img1\_path \= 'rankine\_vortex03\_0.jpg'  
\# Ruta a la segunda imagen (tiempo t0 \+ delta\_t)  
img2\_path \= 'rankine\_vortex03\_1.jpg'

\# Carga de imágenes en escala de grises  
img1 \= cv2.imread(img1\_path, cv2.IMREAD\_GRAYSCALE)  
img2 \= cv2.imread(img2\_path, cv2.IMREAD\_GRAYSCALE)

\# Validación básica de la carga de imágenes  
if img1 is None:  
    print(f"Error: No se pudo cargar la imagen 1\. Verifica la ruta: {img1\_path}")  
    exit()  
if img2 is None:  
    print(f"Error: No se pudo cargar la imagen 2\. Verifica la ruta: {img2\_path}")  
    exit()

\# Verificación de que las imágenes tengan el mismo tamaño  
if img1.shape \!= img2.shape:  
    print("Error: Las imágenes deben tener el mismo tamaño para el análisis.")  
    exit()

print(f"Imágenes cargadas exitosamente:")  
print(f"  Imagen 1: {img1\_path} (Forma: {img1.shape})")  
print(f"  Imagen 2: {img2\_path} (Forma: {img2.shape})")

\# \--- Paso 2: Estimación del Campo de Velocidad (Flujo Óptico Farnebäck) \---  
\# Parámetros para cv2.calcOpticalFlowFarneback:  
\# prev: primera imagen de 8-bit en escala de grises.  
\# next: segunda imagen de 8-bit en escala de grises del mismo tamaño que prev.  
\# flow: campo de flujo óptico calculado que tiene el mismo tamaño que prev y tipo CV\_32FC2.  
\# pyr\_scale: parámetro, especificando la reducción de la imagen para construir la pirámide de imágenes;  
\#            pyr\_scale=0.5 significa una pirámide clásica, donde cada capa siguiente es la mitad de grande que la anterior.  
\# levels: número de niveles de la pirámide incluyendo la imagen inicial; levels=1 significa que no se usa pirámide.  
\# winsize: tamaño de la ventana de promediado; valores mayores aumentan la robustez del algoritmo al ruido  
\#          y permiten detectar movimientos más rápidos, pero también producen un desenfoque del campo de movimiento.  
\# iterations: número de iteraciones del algoritmo en cada nivel de la pirámide.  
\# poly\_n: tamaño del vecindario de píxeles usado para encontrar la expansión polinómica en cada píxel;  
\#         valores mayores significan que la imagen se aproxima con polinomios más suaves,  
\#         produce resultados más robustos y precisos, pero también requiere más tiempo.  
\# poly\_sigma: desviación estándar del gaussiano usado para suavizar las derivadas usadas como base para  
\#             la expansión polinómica; para poly\_n=5, se puede usar poly\_sigma=1.1, para poly\_n=7, poly\_sigma=1.5.  
\# flags: banderas de operación:  
\#        0: Sin banderas.  
\#        cv2.OPTFLOW\_USE\_INITIAL\_FLOW: Usa el flujo de entrada como una estimación inicial del flujo.  
\#        cv2.OPTFLOW\_FARNEBACK\_GAUSSIAN: Usa un filtro Gaussiano en lugar de un filtro de caja para winsize \> 1\.

flow \= cv2.calcOpticalFlowFarneback(prev=img1,  
                                    next=img2,  
                                    flow=None, \# Se calculará y devolverá  
                                    pyr\_scale=0.5,  
                                    levels=3,  
                                    winsize=15,  
                                    iterations=3,  
                                    poly\_n=5,  
                                    poly\_sigma=1.2,  
                                    flags=0)

\# 'flow' es un array con forma (altura, anchura, 2\)  
\# flow\[..., 0\] es la componente u (desplazamiento en x)  
\# flow\[..., 1\] es la componente v (desplazamiento en y)  
u \= flow\[..., 0\] \# Componente horizontal del flujo óptico  
v \= flow\[..., 1\] \# Componente vertical del flujo óptico

print(f"Campo de flujo óptico calculado. Forma de u: {u.shape}, Forma de v: {v.shape}")

\# \--- Paso 3: Cálculo de Magnitudes Derivadas \---

\# Magnitud de la velocidad (escala en píxeles por intervalo de tiempo entre imágenes)  
magnitude \= np.sqrt(u\*\*2 \+ v\*\*2)

\# Vorticidad: dv/dx \- du/dy  
\# Se utiliza np.gradient para calcular las derivadas parciales.  
\# Para un array 2D f, np.gradient(f) devuelve dos arrays: df/dy y df/dx.  
\# El primer array es la derivada con respecto al eje 0 (filas, y).  
\# El segundo array es la derivada con respecto al eje 1 (columnas, x).

grad\_v \= np.gradient(v) \# (dv/dy, dv/dx)  
dv\_dx \= grad\_v\[1\]       \# Derivada parcial de v con respecto a x

grad\_u \= np.gradient(u) \# (du/dy, du/dx)  
du\_dy \= grad\_u\[0\]       \# Derivada parcial de u con respecto a y

vorticity \= dv\_dx \- du\_dy \# Componente z de la vorticidad  
print(f"Vorticidad calculada. Forma: {vorticity.shape}")

\# \--- Paso 4: Visualización de Resultados \---

\# Submuestreo para la visualización del quiver plot (para evitar saturación de vectores)  
\# 'step' determina que se tomará un vector cada 'step' píxeles en ambas direcciones.  
step \= 20  
\# Creación de una malla de coordenadas para el submuestreo  
y\_coords, x\_coords \= np.mgrid\[step//2:img1.shape\[0\]:step, step//2:img1.shape\[1\]:step\]

\# Componentes de velocidad submuestreadas  
u\_subsampled \= u\[::step, ::step\]  
v\_subsampled \= v\[::step, ::step\]

\# 1\. Visualización de Vectores de Velocidad (Quiver Plot)  
plt.figure(figsize=(12, 9)) \# Tamaño de la figura  
\# Mostrar la primera imagen como fondo con cierta transparencia  
plt.imshow(img1, cmap='gray', alpha=0.7)  
\# Dibujar los vectores de velocidad (quiver)  
plt.quiver(x\_coords, y\_coords, u\_subsampled, v\_subsampled,  
           color='lime',          \# Color de las flechas  
           scale\_units='xy',      \# Las unidades de la escala son las mismas que los datos  
           angles='xy',           \# Los ángulos se calculan a partir de (u,v)  
           scale=0.2,             \# Factor de escala para la longitud de las flechas.  
                                  \# Un valor más pequeño hace las flechas más largas. Ajustar según necesidad.  
           headwidth=3,           \# Ancho de la cabeza de la flecha  
           headlength=5,          \# Longitud de la cabeza de la flecha  
           width=0.003            \# Ancho del cuerpo de la flecha (relativo a las unidades del gráfico)  
           )  
plt.title(f'Campo de Velocidad Estimado (Método Farnebäck)\\nImágenes: {img1\_path} y {img2\_path}')  
plt.xlabel('X (píxeles)')  
plt.ylabel('Y (píxeles)')  
plt.grid(False) \# Desactivar la cuadrícula para no interferir con la imagen  
plt.gca().invert\_yaxis() \# Invertir el eje Y para que el origen (0,0) esté arriba a la izquierda  
plt.tight\_layout() \# Ajustar el layout para que todo encaje bien  
plt.show()

\# 2\. Visualización de la Magnitud de la Velocidad  
plt.figure(figsize=(10, 7))  
plt.imshow(magnitude, cmap='viridis') \# 'viridis', 'plasma', 'inferno', 'magma' son buenos mapas de color  
plt.colorbar(label='Magnitud de Velocidad (píxeles/Δt)')  
plt.title('Magnitud del Campo de Velocidad')  
plt.xlabel('X (píxeles)')  
plt.ylabel('Y (píxeles)')  
plt.gca().invert\_yaxis()  
plt.tight\_layout()  
plt.show()

\# 3\. Visualización de Vorticidad  
plt.figure(figsize=(10, 7))  
\# 'bwr' (Blue-White-Red) es un mapa de color divergente adecuado para vorticidad.  
\# El centro (cero) será blanco, valores positivos serán rojos, y negativos azules.  
\# Se determina el límite absoluto máximo para centrar la barra de color en cero.  
max\_abs\_vorticity \= np.max(np.abs(vorticity))  
plt.imshow(vorticity, cmap='bwr', vmin=-max\_abs\_vorticity, vmax=max\_abs\_vorticity)  
plt.colorbar(label='Vorticidad (unidades de 1/Δt)')  
plt.title('Campo de Vorticidad')  
plt.xlabel('X (píxeles)')  
plt.ylabel('Y (píxeles)')  
plt.gca().invert\_yaxis()  
plt.tight\_layout()  
plt.show()

\# 4\. (Opcional) Visualización con Líneas de Corriente (Streamplot)  
\# plt.figure(figsize=(12, 9))  
\# plt.imshow(img1, cmap='gray', alpha=0.5) \# Imagen de fondo opcional  
\# \# Crear una malla para streamplot (necesita coordenadas x, y para cada punto de u, v)  
\# y\_stream\_coords, x\_stream\_coords \= np.mgrid\[0:img1.shape\[0\], 0:img1.shape\[1\]\]  
\# plt.streamplot(x\_stream\_coords, y\_stream\_coords, u, v,  
\#                color='cyan',          \# Color de las líneas de corriente  
\#                linewidth=1,           \# Ancho de las líneas  
\#                density=1.5,           \# Densidad de las líneas (mayor valor, más líneas)  
\#                arrowstyle='-\>',       \# Estilo de las flechas en las líneas  
\#                arrowsize=1.5)         \# Tamaño de las flechas  
\# plt.title('Campo de Velocidad con Líneas de Corriente')  
\# plt.xlabel('X (píxeles)')  
\# plt.ylabel('Y (píxeles)')  
\# plt.xlim(0, img1.shape\[1\]) \# Establecer límites del eje x  
\# plt.ylim(0, img1.shape\[0\]) \# Establecer límites del eje y  
\# plt.gca().invert\_yaxis()  
\# plt.tight\_layout()  
\# plt.show()

print("Análisis y visualización completados.")  
