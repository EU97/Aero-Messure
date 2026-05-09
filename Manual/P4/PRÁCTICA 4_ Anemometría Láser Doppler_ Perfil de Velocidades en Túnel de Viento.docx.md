# **PRÁCTICA 4: Anemometría Láser Doppler: Perfil de Velocidades en Túnel de Viento**

**Laboratorio de Técnicas de Medida en Aeronáutica**

## **I. Introducción**

Bienvenido a la práctica de Anemometría Láser Doppler (LDA). La LDA es una técnica óptica sofisticada y no intrusiva, fundamental en la ingeniería aeronáutica para la medición precisa de la velocidad de flujos. Su aplicación es crucial para caracterizar flujos complejos, tanto internos (toberas, difusores) como externos, siendo especialmente valiosa en túneles de viento para el estudio de la aerodinámica de cuerpos, perfiles alares, y la estructura de estelas.

En esta práctica, analizarás conjuntos de datos de velocidad obtenidos mediante LDA en la sección de pruebas de un túnel de viento. Aunque no realizarás el montaje experimental, te enfocarás en el tratamiento, análisis e interpretación de estos datos. El objetivo es que comprendas el funcionamiento del sistema LDA y proceses los datos para construir perfiles de velocidad media e intensidad de turbulencia, comparando diferentes condiciones de flujo y posiciones dentro del túnel.

## **II. Objetivos de Aprendizaje**

Al finalizar esta práctica, serás capaz de:

* **Comprender** los principios físicos fundamentales del funcionamiento de un sistema de Anemometría Láser Doppler (LDA).  
* **Procesar** datos crudos de velocidad instantánea obtenidos por LDA para calcular parámetros estadísticos clave como la velocidad media y las fluctuaciones de velocidad.  
* **Calcular** la intensidad de turbulencia a partir de las mediciones de velocidad.  
* **Construir y graficar** perfiles de velocidad media e intensidad de turbulencia para diferentes condiciones de flujo y posiciones.  
* **Analizar y comparar** dichos perfiles, identificando el efecto de variar parámetros como la velocidad del flujo libre en el túnel y la posición de medición (axial y transversal).  
* **Interpretar** los perfiles en el contexto de la calidad del flujo en un túnel de viento y el desarrollo de capas límite o estelas.  
* **Discutir** la relevancia de estos análisis en aplicaciones aeronáuticas y experimentación en túneles de viento.

## **III. Fundamentos Teóricos**

Antes de analizar los datos, es crucial que entiendas los conceptos subyacentes.

### **A. Anemometría Láser Doppler (LDA)**

La Anemometría Láser Doppler (LDA) es una técnica de medición óptica que permite determinar la velocidad de partículas muy pequeñas (trazadores o "semillas") suspendidas en un fluido (en este caso, aire). Se asume que estas partículas son lo suficientemente pequeñas como para seguir fielmente el movimiento del fluido sin alterarlo significativamente.

El principio se basa en el **efecto Doppler**: cuando la luz láser es dispersada por una partícula en movimiento, la frecuencia de la luz dispersada cambia. Este cambio de frecuencia es directamente proporcional a la velocidad de la partícula.

En la configuración más común, conocida como **modo de franjas (o modo diferencial)**, dos haces láser coherentes (de la misma longitud de onda, λ) se hacen cruzar en un punto específico del flujo. En la región de cruce, llamada **volumen de medición**, los dos haces interfieren creando un patrón de franjas de interferencia brillantes y oscuras, con un espaciado constante conocido como δf​.

Cuando una partícula sembrada en el flujo atraviesa estas franjas, dispersa luz cada vez que cruza una franja brillante. Un fotodetector recoge esta luz dispersada, que aparece modulada en intensidad con una frecuencia, fD​ (frecuencia Doppler), que es directamente proporcional a la componente de la velocidad de la partícula, U⊥​, perpendicular al plano de las franjas:

U⊥​=fD​⋅δf​

El espaciado de las franjas, δf​, se determina por la longitud de onda del láser (λ) y el semiángulo de cruce de los haces (k):

δf​=2sin(k)λ​

En esta práctica, la componente de velocidad medida es la axial, es decir, la componente principal del flujo en el túnel de viento.

### **B. Flujo en un Túnel de Viento**

Un túnel de viento está diseñado para generar un flujo de aire controlado y uniforme en su **sección de pruebas**, donde se colocan los modelos para estudio aerodinámico.

* **Calidad del Flujo:** Es crucial que el flujo en la sección de pruebas sea lo más uniforme posible (velocidad constante en toda la sección transversal, excepto en las capas límite de las paredes) y con bajos niveles de turbulencia. Las mediciones LDA pueden ayudar a caracterizar esta uniformidad y los niveles de turbulencia del flujo libre.  
* **Capas Límite:** Sobre las paredes del túnel de viento, así como sobre la superficie de cualquier modelo introducido, se desarrollan capas límite. Dentro de estas capas, la velocidad del fluido varía desde cero en la superficie sólida hasta la velocidad del flujo libre a una cierta distancia de la pared (espesor de la capa límite, δ). El perfil de velocidad dentro de la capa límite depende de si esta es laminar o turbulenta.  
* **Estelas:** Detrás de un objeto sumergido en el flujo (ej. un perfil alar, un cilindro), se forma una región de flujo perturbado conocida como estela, caracterizada por velocidades menores que el flujo libre y, a menudo, por una mayor turbulencia.  
* **Coordenadas:** En esta práctica, las mediciones se realizan en diferentes posiciones. Asumiremos una coordenada axial z (a lo largo de la dirección principal del flujo en el túnel) y una coordenada transversal r o y (perpendicular a la dirección principal del flujo, desde una línea central o una pared). La notación r/R o y/H (donde R o H son dimensiones características de la sección de pruebas o del objeto de estudio) se usa para adimensionalizar la posición transversal. Para esta práctica, se asume que D (utilizado en z/D) es un diámetro o dimensión característica relevante para la configuración experimental, y R es un radio o semialtura característica para la posición transversal. El radio de referencia para r/R es de 26 mm.

### **C. Parámetros Estadísticos del Flujo**

Dado que el flujo, especialmente si es turbulento o si se miden fluctuaciones inherentes al túnel, presenta variaciones de velocidad, se recurre a parámetros estadísticos para caracterizarlo. A partir de una serie de N mediciones de velocidad instantánea (ui​) tomadas en un punto fijo del espacio:

* Velocidad Media (uˉ): Es el promedio temporal de las velocidades instantáneas.  
  uˉ=N1​∑i=1N​ui​  
* Intensidad de Turbulencia (e2): Este parámetro cuantifica la magnitud de las fluctuaciones de velocidad. Para los fines de esta práctica, la intensidad de turbulencia (e2) se define mediante la siguiente expresión:  
  e2=maxi​(∣uˉ−ui​∣)∑i=1N​∣(uˉ)2−(ui​)2∣​​  
  Donde la suma (∑) se realiza sobre todas las N mediciones de velocidad instantánea ui​ en el punto de medición, y maxi​(∣uˉ−ui​∣) es el valor máximo de la diferencia absoluta entre la velocidad media y cada velocidad instantánea en ese conjunto de N mediciones. Un valor más alto de e2 indica mayores fluctuaciones relativas. En el flujo libre de un túnel de viento, se busca que e2 sea bajo. En capas límite o estelas, puede ser significativamente mayor.

## **IV. Descripción del Experimento (Conceptual)**

No realizarás el experimento físico, pero es importante que entiendas cómo se obtuvieron los datos.

### **A. Arreglo Experimental Típico en Túnel de Viento**

Se utiliza un sistema LDA para medir la velocidad del aire en la sección de pruebas de un túnel de viento. El sistema LDA típicamente incluye:

* Una **fuente láser**.  
* **Ópticas de transmisión:** Incluyen un divisor de haz y una lente de enfoque para formar el volumen de medición.  
* **Volumen de medición:** La pequeña región donde los haces se cruzan.  
* **Ópticas de recepción:** Para colectar la luz dispersada y enfocarla en un fotodetector.  
* **Fotodetector:** Convierte la señal óptica en eléctrica.  
* **Procesador de señales LDA:** Analiza la señal eléctrica para obtener la velocidad.  
* **Sistema de siembra de partículas:** El aire en el túnel debe ser sembrado con partículas trazadoras adecuadas (ej. gotas de aceite, humo) si no hay suficientes aerosoles naturales.  
* **Sistema de posicionamiento (traverse):** Permite mover con precisión el volumen de medición a diferentes puntos (z,r) dentro de la sección de pruebas del túnel.

### **B. Condiciones Experimentales y Sets de Datos**

Para esta práctica, analizarás datos obtenidos en dos posiciones axiales de referencia, z/D=10 y z/D=50 (donde D es una longitud característica de referencia), y bajo dos condiciones de velocidad de flujo libre en el túnel (Velocidad 1 y Velocidad 2, siendo Velocidad 2 mayor que Velocidad 1). Esto resulta en los siguientes **4 sets de datos distintos**, contenidos en las carpetas FX01G00, FX02G00, FX03G00, FX04G00. Cada set de datos corresponde a un perfil transversal de velocidades (mediciones en varias posiciones r/R a una z/D fija) bajo estas condiciones específicas:

* **Set 1 (Carpeta FX01G00):** Posición axial de referencia z/D=10, Velocidad de túnel 1 (más baja).  
* **Set 2 (Carpeta FX02G00):** Posición axial de referencia z/D=10, Velocidad de túnel 2 (más alta).  
* **Set 3 (Carpeta FX03G00):** Posición axial de referencia z/D=50, Velocidad de túnel 1 (más baja).  
* **Set 4 (Carpeta FX04G00):** Posición axial de referencia z/D=50, Velocidad de túnel 2 (más alta).

*(**Nota Importante:** Esta es la asignación de condiciones para cada set de datos. Confirma con tu instructor si esta es la configuración exacta o si existe una descripción diferente para identificar cada set de datos. Los nombres descriptivos en el script de Python (NOMBRES\_DESCRIPTIVOS\_SETS) deberán ajustarse si esta asignación no es correcta).*

Cada set de datos (cada carpeta FXnnG00) contiene:

* Múltiples archivos de datos de velocidad instantánea (ej. L01G00\_001.txt, L01G00\_00a.txt, etc.). Cada uno de estos archivos corresponde a las mediciones de velocidad instantánea tomadas en **una única posición transversal** r/R.  
* Un archivo llamado Posiciones.txt. Este archivo es crucial, ya que mapea cada uno de los nombres de archivo de datos de velocidad (columna 'Archivo') con su correspondiente posición transversal adimensionalizada (r/R) (columna 'Posición r/R'). El radio de referencia para r/R es R=26 mm.

## **V. Materiales Necesarios para el Análisis de Datos**

Para realizar el análisis de los datos, necesitarás:

* **Computadora personal.**  
* **Software:**  
  * **Python:** Versión 3.x (se recomienda la más reciente estable).  
  * **Bibliotecas de Python:**  
    * numpy: Para cálculo numérico eficiente, especialmente con arrays.  
    * pandas: Para la manipulación y análisis de datos tabulares.  
    * matplotlib: Para la generación de gráficas.  
    * glob: Para encontrar archivos y directorios.  
    * os: Para interactuar con el sistema operativo.  
  * **Entorno de Desarrollo para Python (IDE):** Cualquiera con el que te sientas cómodo (VS Code, Spyder, Jupyter Notebook/Lab).  
* **Archivos de Datos:** Los 4 sets de datos, cada uno en su respectiva carpeta (FX01G00 a FX04G00), conteniendo los archivos .txt de velocidad y el archivo Posiciones.txt.  
* **Script de Python:** El archivo analizador\_lda\_p4.py, cuyo contenido se detalla en el Apéndice.

## **VI. Procedimiento de Adquisición de Datos (Conceptual)**

Para que entiendas el origen de los datos, este es el proceso conceptual que se siguió para obtener cada uno de los 4 sets:

1. **Establecer Condiciones de Túnel:** Se ajusta la velocidad del túnel de viento para obtener la velocidad de flujo libre deseada (Velocidad 1 o Velocidad 2). El sistema de posicionamiento del LDA se ajusta para la posición axial de referencia de interés (z/D=10 o z/D=50).  
2. **Posicionamiento Transversal Inicial:** El volumen de medición del LDA se mueve a la primera posición transversal r/R indicada en el archivo Posiciones.txt para ese set.  
3. **Adquisición de Muestras:** En esa posición (z,r), el sistema LDA adquiere un gran número de mediciones de velocidad instantánea. Estos datos se guardan en un archivo de texto. **La penúltima columna (-2) de estos archivos de datos contiene la componente de velocidad axial relevante.**  
4. **Iteración Transversal:** Se repiten los pasos 2 y 3 para cada una de las posiciones r/R listadas en el archivo Posiciones.txt (típicamente unas 21 posiciones para cubrir la región de interés transversalmente).

Este proceso se repite para cada una de las 4 combinaciones de z/D y velocidad del túnel, generando los 4 sets de datos.

## **VII. Procedimiento de Tratamiento y Análisis de Datos**

Esta sección detalla los pasos que debes seguir para procesar y analizar los datos.

### **A. Preparación del Entorno de Trabajo**

1. **Verificar Python:** Asegúrate de tener Python 3.x (python \--version).  
2. **Instalar Bibliotecas:** Si es necesario, instala numpy, pandas, y matplotlib vía pip:  
   pip install numpy pandas matplotlib

3. **Organizar Archivos:**  
   * Crea una carpeta principal (ej. Practica4\_LDA\_TunelViento).  
   * Dentro, copia las carpetas de los 4 sets de datos (FX01G00, etc.).  
   * Copia el script analizador\_lda\_p4.py (del Apéndice de este manual) en la carpeta principal.

Tu estructura de carpetas debería verse así:Practica4\_LDA\_TunelViento/  
├── FX01G00/  
│   ├── L01G00\_001.txt  
│   ├── ...  
│   └── Posiciones.txt  
├── FX02G00/  
│   └── ...  
├── FX03G00/  
│   └── ...  
├── FX04G00/  
│   └── ...  
└── analizador\_lda\_p4.py

### **B. Uso del Script de Python analizador\_lda\_p4.py**

1. **Revisar Constantes y Nombres Descriptivos en el Script:**  
   * Abre analizador\_lda\_p4.py con tu editor de Python.  
   * Localiza la sección de "CONSTANTES Y CONFIGURACIÓN" al inicio del script.  
   * Asegúrate de que PREFIJO\_CARPETAS\_SETS (ej. "FX"), PREFIJO\_REAL\_ARCHIVOS (ej. "L01G00\_"), NOMBRE\_ARCHIVO\_POSICIONES (ej. "Posiciones.txt") y RADIO\_TUBERIA\_MM (que es 26.0, usado como radio de referencia R para r/R) coincidan con tu estructura de archivos y los datos de la práctica.  
   * **MUY IMPORTANTE:** Verifica que la lista NOMBRES\_DESCRIPTIVOS\_SETS en el script corresponda al orden y descripción de las condiciones experimentales de cada carpeta FXnnG00 (como se indica en la Sección IV.B):  
     NOMBRES\_DESCRIPTIVOS\_SETS \= \[  
         "Set 1: z/D=10, Velocidad Túnel 1 (Baja)",  
         "Set 2: z/D=10, Velocidad Túnel 2 (Alta)",  
         "Set 3: z/D=50, Velocidad Túnel 1 (Baja)",  
         "Set 4: z/D=50, Velocidad Túnel 2 (Alta)"  
     \]

     Si tienes dudas sobre la correspondencia, consulta a tu instructor.  
2. **Ejecutar el Script:**  
   * Abre una terminal o símbolo del sistema.  
   * Navega usando comandos (cd) hasta tu carpeta Practica4\_LDA\_TunelViento.  
   * Ejecuta el script con el comando: python analizador\_lda\_p4.py (o python3 analizador\_lda\_p4.py si es necesario).  
3. **Funcionamiento del Script:**  
   * El script buscará automáticamente las carpetas de los sets (ej. FX01G00, FX02G00, etc.) en el directorio donde se ejecuta.  
   * Para cada set de datos:  
     * Leerá el archivo Posiciones.txt para obtener el mapeo de archivos de datos a posiciones r/R.  
     * Para cada archivo de datos listado, cargará las velocidades de la penúltima columna, calculará la velocidad media (uˉ) y la intensidad de turbulencia (e2).  
     * Almacenará los resultados (r/R, uˉ, e2) para ese set.  
     * Imprimirá en la consola una tabla con los resultados del set.  
     * Generará y guardará como archivos .png las gráficas individuales de uˉ vs r/R y e2 vs r/R.  
   * Finalmente, generará y guardará gráficas comparativas mostrando los 4 perfiles de velocidad media superpuestos, y los 4 perfiles de intensidad de turbulencia superpuestos.  
   * Todas las gráficas generadas se guardarán como archivos .png en la misma carpeta donde ejecutaste el script.

### **C. Tabulación de Resultados**

El script imprimirá en la consola tablas con los resultados calculados para cada set. Deberás copiar estas tablas (o los datos relevantes) y presentarlas de forma clara y ordenada en tu reporte de laboratorio. Asegúrate de que cada tabla esté claramente identificada con el set de datos al que corresponde (utilizando los nombres descriptivos).

Un ejemplo del formato de la tabla para **un set** es:

| r/R | Archivo Original (Posiciones.txt) | Archivo de Datos Procesado | Velocidad Media (uˉ) (m/s) | Intensidad de Turbulencia (e2) (-) |
| :---- | :---- | :---- | :---- | :---- |
| 0.000 | XG00\_00f | L01G00\_00f.txt | ... | ... |
| 0.115 | XG00\_00e | L01G00\_00e.txt | ... | ... |
| ... | ... | ... | ... | ... |
| 0.981 | XG00\_001 | L01G00\_001.txt | ... | ... |

### **D. Generación y Presentación de Gráficas**

El script generará automáticamente las siguientes gráficas:

1. **Para cada uno de los 4 sets de datos (8 gráficas en total):**  
   * Un gráfico del perfil de Velocidad Media (uˉ) vs. Posición Transversal Adimensional (r/R).  
   * Un gráfico del perfil de Intensidad de Turbulencia (e2) vs. Posición Transversal Adimensional (r/R).  
2. **Gráficas Comparativas (2 gráficas en total):**  
   * Un único gráfico que muestre los **4 perfiles de velocidad media** superpuestos.  
   * Un único gráfico que muestre los **4 perfiles de intensidad de turbulencia** superpuestos.

**Requisitos para las gráficas en tu reporte:** Todas las gráficas deben incluirse, tener títulos claros y descriptivos, ejes correctamente etiquetados con unidades (ej. "Velocidad Media (u) \[m/s\]", "Posición Transversal Adimensional, r/R \[-\]"), y leyendas claras en las gráficas comparativas para distinguir cada set. Asegúrate de que sean legibles y de buena calidad.

## **VIII. Análisis e Interpretación de Resultados**

Una vez que hayas obtenido las tablas y gráficas, el siguiente paso es analizarlas e interpretarlas. Utiliza las siguientes preguntas como base para la sección de Discusión en tu reporte:

### **A. Análisis Individual de Perfiles por Set de Datos**

Para cada uno de los 4 sets de datos, examina sus dos gráficas (velocidad media y turbulencia) y responde:

* **Perfil de Velocidad Media (**uˉ **vs** r/R**):**  
  * Describe la forma general del perfil. ¿Es simétrico respecto a r/R=0 (el centro de la región de medida)?  
  * ¿En qué posición r/R se observa la velocidad máxima? ¿Corresponde esta velocidad a la del flujo libre esperado en el túnel?  
  * ¿Cómo se comporta la velocidad cerca de los extremos del barrido transversal (r/R≈±1)? ¿Sugiere la presencia de las paredes del túnel de viento o los bordes de una capa límite o estela?  
* **Perfil de Intensidad de Turbulencia (**e2 **vs** r/R**):**  
  * Describe la forma general del perfil.  
  * ¿En qué regiones la intensidad de turbulencia es mayor? ¿Y menor? ¿Es el nivel de turbulencia bajo en la región de flujo libre? ¿Aumenta cerca de las paredes o en regiones con fuertes gradientes de velocidad?

### **B. Comparación entre Diferentes Velocidades del Túnel (manteniendo z/D constante)**

Utiliza las gráficas comparativas.

* Compara el **Set 1** (z/D=10, Velocidad Túnel 1\) con el **Set 2** (z/D=10, Velocidad Túnel 2).  
* Compara el **Set 3** (z/D=50, Velocidad Túnel 1\) con el **Set 4** (z/D=50, Velocidad Túnel 2).

Para cada comparación:

* ¿Cómo afecta el aumento de la velocidad del túnel a la **magnitud** de la velocidad media en toda la sección transversal?  
* Si normalizaras los perfiles de velocidad media (dividiendo todas las velocidades de un perfil por la velocidad del flujo libre del túnel, U∞​), ¿cómo afectaría el aumento de la velocidad del túnel a la **forma** de este perfil de velocidad normalizado?  
* ¿Cómo afecta el incremento de la velocidad del túnel a la magnitud y forma del perfil de **intensidad de turbulencia** e2?

### **C. Comparación entre Posiciones Axiales (manteniendo la velocidad del túnel constante)**

Utiliza las gráficas comparativas.

* Compara el **Set 1** (z/D=10, Velocidad Túnel 1\) con el **Set 3** (z/D=50, Velocidad Túnel 1).  
* Compara el **Set 2** (z/D=10, Velocidad Túnel 2\) con el **Set 4** (z/D=50, Velocidad Túnel 2).

Para cada comparación:

* ¿Observas diferencias significativas en la **forma** de los perfiles de velocidad media entre la posición z/D=10 y z/D=50 para la misma velocidad del túnel? ¿Podrían estas diferencias indicar el desarrollo de la capa límite en las paredes del túnel o la evolución de una estela si hubiera un objeto de estudio aguas arriba de las mediciones?  
* ¿Cómo cambian los perfiles de intensidad de turbulencia e2 con la posición axial para una misma velocidad del túnel?

### **D. Uniformidad y Simetría del Flujo**

* Evalúa la uniformidad de la velocidad media uˉ en la región central del barrido (r/R≈0) para cada set. ¿Es el perfil de velocidad razonablemente plano en esta región, como se esperaría en un flujo libre de buena calidad?  
* Evalúa visualmente la simetría de los perfiles obtenidos. Si observas pequeñas asimetrías, ¿a qué podrían deberse en un experimento real (considera la alineación del modelo, no uniformidades en el flujo del túnel, etc.)?

### **E. Discusión Detallada de la Intensidad de Turbulencia**

* ¿Cómo se relaciona, en general, el perfil de e2 con el perfil de uˉ? Considera las regiones de alto gradiente de velocidad. ¿Tiende e2 a ser mayor donde los gradientes de uˉ son altos?  
* ¿Qué indican los valores de e2 en la región de flujo libre (centro del barrido) y cerca de los bordes del barrido transversal?

### **F. Concordancia con Características Esperadas en Túneles de Viento**

* ¿Se asemejan los perfiles de velocidad media obtenidos en la región central (r/R≈0) a un flujo uniforme, característico de la corriente libre en un túnel de viento?  
* Si los perfiles transversales se tomaron atravesando una capa límite (cerca de una pared) o una estela (detrás de un objeto), ¿la forma de los perfiles es consistente con la teoría (ej. un perfil más lleno para una capa límite turbulenta, un déficit de velocidad en una estela)?  
* ¿Son los niveles de intensidad de turbulencia e2 en la región de flujo libre suficientemente bajos como para indicar una buena calidad de flujo en el túnel?

## **IX. Cuestionario**

Responde de manera concisa y clara a las siguientes preguntas:

1. Explica brevemente el principio físico de la Anemometría Láser Doppler (LDA) y qué magnitud física mide directamente el sistema.  
2. En el contexto de LDA, ¿qué es el "volumen de medición" y cómo se forma?  
3. Describe las diferencias clave en la forma de un perfil de velocidad entre una capa límite laminar y una capa límite turbulenta que se desarrolla sobre una placa plana en un túnel de viento.  
4. En esta práctica, se indica que se utiliza la penúltima columna de los archivos de datos. ¿Qué información crees que contiene típicamente esta columna en un archivo de salida de LDA y por qué es la relevante aquí?  
5. Explica el significado físico de la intensidad de turbulencia e2 (según la fórmula dada). ¿Qué indica un valor alto de e2 en comparación con un valor bajo, específicamente en el contexto de mediciones en un túnel de viento?  
6. Menciona al menos dos ventajas de la técnica LDA en comparación con otras técnicas de medición de velocidad de fluidos como los tubos de Pitot o la Anemometría de Hilo Caliente (HWA) para mediciones en túneles de viento. Menciona también al menos una desventaja o limitación de la LDA.  
7. Si la velocidad del flujo libre en un túnel de viento se duplica, ¿cómo esperarías que cambiara cualitativamente el espesor de la capa límite en un punto fijo sobre una placa plana colocada en la sección de pruebas? Justifica tu respuesta.  
8. Menciona tres posibles fuentes de error o incertidumbre en un experimento real de mediciones con LDA en un túnel de viento.  
9. ¿Por qué es importante asegurar la uniformidad del flujo y un bajo nivel de turbulencia en la sección de pruebas de un túnel de viento para la experimentación aeronáutica?  
10. ¿Cómo podría utilizarse la LDA para estudiar la estructura de la estela detrás de un perfil alar montado en un túnel de viento? ¿Qué características específicas buscarías en los perfiles de uˉ y e2 al atravesar la estela?

## **X. Requisitos del Reporte de Laboratorio**

Elabora un reporte técnico formal que incluya las siguientes secciones:

1. **Portada:** Título de la práctica, tu(s) nombre(s), asignatura, grupo (si aplica), fecha de entrega.  
2. **Resumen (Abstract):** Una breve descripción de la práctica (no más de 250 palabras), la metodología de análisis de datos empleada, los principales resultados obtenidos y las conclusiones más relevantes.  
3. **Introducción:** Presentación de los objetivos de la práctica, la importancia de la técnica LDA y la relevancia del estudio de perfiles de flujo en túneles de viento para la ingeniería aeronáutica.  
4. **Marco Teórico:** Explicación concisa de los principios de funcionamiento de la LDA, características del flujo en túneles de viento (flujo libre, capas límite, estelas), y definición de los parámetros estadísticos utilizados (uˉ y e2, incluyendo sus fórmulas).  
5. **Metodología de Análisis de Datos:**  
   * Descripción de los datos proporcionados: estructura de los archivos, organización en sets, y las condiciones experimentales asumidas para cada set (según la Sección IV.B).  
   * Descripción detallada del procedimiento seguido para el tratamiento y análisis de los datos, incluyendo cómo se utilizó el script de Python y las fórmulas clave para uˉ y e2.  
6. **Resultados:**  
   * **Tablas:** Presentación clara y ordenada de las tablas de resultados (r/R, uˉ, e2) para cada uno de los 4 sets de datos.  
   * **Gráficas:** Inclusión de todas las gráficas generadas, debidamente etiquetadas y con buena calidad:  
     * Perfiles individuales de uˉ vs r/R para cada set.  
     * Perfiles individuales de e2 vs r/R para cada set.  
     * Gráfica comparativa de los 4 perfiles de uˉ.  
     * Gráfica comparativa de los 4 perfiles de e2.  
7. **Discusión de Resultados:**  
   * Análisis e interpretación detallada de las tablas y gráficas, abordando todas las cuestiones planteadas en la Sección VIII.  
   * Interpretación física de los perfiles observados y de las diferencias encontradas entre los distintos sets de datos.  
   * Discusión sobre posibles fuentes de error o incertidumbre en los datos o en el análisis, y limitaciones del estudio.  
8. **Conclusiones:** Resumen de los hallazgos más importantes de la práctica y del aprendizaje obtenido en relación con los objetivos planteados.  
9. **Respuestas al Cuestionario:** Respuestas claras y concisas a todas las preguntas de la Sección IX.  
10. **Referencias:** Listado de cualquier fuente bibliográfica adicional que hayas consultado (libros, artículos, etc.), utilizando un formato de citación consistente.  
11. **Apéndice:** Incluye el código Python utilizado para el análisis (ver Sección Apéndice: Script de Python para Análisis de Datos). Si realizaste modificaciones significativas al script, indícalo y explica los cambios.

## **XI. Referencias Sugeridas**

* Albrecht, H.-E., Borys, M., Damaschke, N., & Tropea, C. (2003). *Laser Doppler and Phase Doppler Measurement Techniques*. Springer.  
* Barlow, J. B., Rae, W. H., & Pope, A. (1999). *Low-Speed Wind Tunnel Testing* (3rd ed.). Wiley.  
* White, F. M. (2016). *Fluid Mechanics* (8th ed.). McGraw-Hill. (O cualquier libro de texto estándar de Mecánica de Fluidos que cubra flujo en túneles de viento, capas límite y estelas).  
* Consulta el documento "Lab\_TM\_P4.pdf" (si te fue proporcionado con ese nombre) y cualquier material de clase adicional.

## **Apéndice: Script de Python para Análisis de Datos**

A continuación, se presenta el código del script analizador\_lda\_p4.py que utilizarás para procesar los datos.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import glob \# Para encontrar las carpetas de los sets

\# \--- CONSTANTES Y CONFIGURACIÓN \---

\# Asegúrate de que estos nombres coincidan con tu estructura de archivos.

\# PREFIJO\_CARPETAS\_SETS: Parte inicial del nombre de las carpetas de cada set de datos.

\# EJEMPLO: Si tus carpetas son FX01G00, FX02G00, etc., el prefijo es "FX"

\# Si son Set1\_zD10, Set2\_zD10\_alto, etc., podrías usar "Set" o un patrón más específico.

PREFIJO\_CARPETAS\_SETS \= "FX" \# Modificar según el nombre de tus carpetas (FX01G00, FX02G00...)

\# PREFIJO\_REAL\_ARCHIVOS: Prefijo de los archivos de datos de velocidad dentro de cada set.

PREFIJO\_REAL\_ARCHIVOS \= "L01G00\_" \# Ej: L01G00\_001.txt, L01G00\_00a.txt

NOMBRE\_ARCHIVO\_POSICIONES \= "Posiciones.txt"

EXTENSION\_ARCHIVOS\_DATOS \= ".txt"

\# Para flujo en túnel de viento, este es un radio o semialtura de referencia para r/R.

\# Si los datos provienen de una tubería, es el radio de la tubería.

RADIO\_REFERENCIA\_MM \= 26.0 \# Radio de referencia para r/R en mm.

\# Nombres descriptivos para los sets (para leyendas en gráficas).

\# Deben coincidir en orden con el orden alfabético/numérico de tus carpetas de sets.

\# Ejemplo: si las carpetas son FX01G00, FX02G00, FX03G00, FX04G00

\# y corresponden a las condiciones de la Sección IV.B:

NOMBRES\_DESCRIPTIVOS\_SETS \= \[

    "Set 1: z/D=10, Velocidad Túnel 1 (Baja)",

    "Set 2: z/D=10, Velocidad Túnel 2 (Alta)",

    "Set 3: z/D=50, Velocidad Túnel 1 (Baja)",

    "Set 4: z/D=50, Velocidad Túnel 2 (Alta)"

\]

\# Si no estás seguro de la correspondencia, usa nombres genéricos como "Set 1", "Set 2", etc.

\# y luego ajústalos cuando sepas qué representa cada carpeta FX0\*G00.

\# \--- FUNCIÓN DE ANÁLISIS DE ARCHIVO INDIVIDUAL \---

def analizar\_archivo\_lda(filepath):

    """

    Analiza un archivo de datos LDA para calcular la velocidad media y la

    intensidad de turbulencia.

    """

    velocidades \= \[\]

    try:

        with open(filepath, 'r') as f:

            for linea in f:

                columnas \= linea.strip().split()

                if len(columnas) \>= 4: \# Asegurar que hay al menos 4 columnas

                    try:

                        \# La penúltima columna contiene la velocidad

                        velocidad\_instantanea \= float(columnas\[-2\])

                        velocidades.append(velocidad\_instantanea)

                    except ValueError:

                        \# print(f"Advertencia: Dato no numérico en {filepath}, línea: {linea.strip()}")

                        continue

        

        if not velocidades:

            \# print(f"Advertencia: No se encontraron datos de velocidad válidos en {filepath}")

            return None, None

        velocidades\_array \= np.array(velocidades)

        u\_bar \= np.mean(velocidades\_array)

        

        \# e^2 \= sqrt( sum( |u\_bar^2 \- u\_i^2| ) ) / MAX( |u\_bar \- u\_i| )

        \# Asegurarse de que u\_bar no es cero para evitar división por cero si todas las u\_i son cero.

        if u\_bar \== 0 and np.all(velocidades\_array \== 0): \# Si u\_bar es 0 y todas las u\_i son 0

             numerador \= 0

             max\_abs\_diff\_u \= 0 \# o 1 para evitar división por cero si se quiere e^2 \= 0

        elif np.all(velocidades\_array \== u\_bar): \# Todas las velocidades son iguales a la media (flujo perfectamente estable)

            numerador \= 0

            max\_abs\_diff\_u \= 0 \# o un valor pequeño si se espera e^2 \= 0

        else:

            sum\_diff\_sq\_abs \= np.sum(np.abs(u\_bar\*\*2 \- velocidades\_array\*\*2))

            numerador \= np.sqrt(sum\_diff\_sq\_abs)

            max\_abs\_diff\_u \= np.max(np.abs(u\_bar \- velocidades\_array))

        if max\_abs\_diff\_u \== 0:

            \# Si todas las velocidades son idénticas, la turbulencia es cero.

            \# O si u\_bar es 0 y todas las u\_i son 0\.

            intensidad\_turbulencia \= 0 

        else:

            intensidad\_turbulencia \= numerador / max\_abs\_diff\_u

            

        return u\_bar, intensidad\_turbulencia

    except FileNotFoundError:

        print(f"Error Crítico: El archivo de datos {filepath} no fue encontrado.")

        return None, None

    except Exception as e:

        print(f"Error procesando el archivo {filepath}: {e}")

        return None, None

\# \--- FUNCIÓN PARA PROCESAR UN SET DE DATOS COMPLETO \---

def procesar\_set\_de\_datos(directorio\_set, nombre\_set\_descriptivo):

    """

    Procesa un conjunto de datos LDA (un set completo) basado en su Posiciones.txt.

    """

    print(f"\\n--- Procesando: {nombre\_set\_descriptivo} (desde carpeta: {directorio\_set}) \---")

    ruta\_posiciones \= os.path.join(directorio\_set, NOMBRE\_ARCHIVO\_POSICIONES)

    

    try:

        \# Intentar leer con diferentes delimitadores comunes si falla el tabulador

        try:

            df\_posiciones \= pd.read\_csv(ruta\_posiciones, sep='\\t')

        except pd.errors.ParserError:

            try:

                df\_posiciones \= pd.read\_csv(ruta\_posiciones, sep=r'\\s+', comment='\#') \# Espacios múltiples, ignorar comentarios

            except Exception as e\_parser:

                 print(f"Error Crítico: No se pudo parsear '{ruta\_posiciones}'. Error: {e\_parser}")

                 return None

        df\_posiciones.columns \= \[col.strip() for col in df\_posiciones.columns\] \# Limpiar nombres de columnas

        

        \# Adaptar a nombres de columna comunes si los esperados no están

        col\_archivo\_nombre \= 'Archivo'

        col\_posicion\_nombre \= 'Posición r/R'

        if col\_archivo\_nombre not in df\_posiciones.columns:

            \# Intentar encontrar columnas con nombres similares o por posición

            if len(df\_posiciones.columns) \>= 2:

                print(f"Advertencia: Columna '{col\_archivo\_nombre}' no encontrada. Usando columna 0 como nombre de archivo.")

                col\_archivo\_nombre \= df\_posiciones.columns\[0\] \# Asumir primera columna

            else:

                print(f"Error: '{ruta\_posiciones}' no tiene suficientes columnas para 'Archivo'.")

                return None

        

        if col\_posicion\_nombre not in df\_posiciones.columns:

            if len(df\_posiciones.columns) \>= 2:

                print(f"Advertencia: Columna '{col\_posicion\_nombre}' no encontrada. Usando columna 1 como posición r/R.")

                col\_posicion\_nombre \= df\_posiciones.columns\[1\] \# Asumir segunda columna

            else:

                print(f"Error: '{ruta\_posiciones}' no tiene suficientes columnas para 'Posición r/R'.")

                return None

    except FileNotFoundError:

        print(f"Error Crítico: El archivo '{ruta\_posiciones}' no fue encontrado.")

        return None

    except Exception as e:

        print(f"Error leyendo o procesando '{ruta\_posiciones}': {e}")

        return None

    if df\_posiciones.empty:

        print(f"Advertencia: '{ruta\_posiciones}' está vacío o no se pudo leer correctamente.")

        return None

    resultados\_set\_actual \= \[\]

    for index, fila in df\_posiciones.iterrows():

        try:

            \# Convertir la posición a flotante, manejando comas como decimales si es necesario

            pos\_str \= str(fila\[col\_posicion\_nombre\]).strip().replace(',', '.')

            rr\_posicion \= float(pos\_str)

            nombre\_archivo\_base\_pos \= str(fila\[col\_archivo\_nombre\]).strip() \# Ej: XG00\_001

        except ValueError:

            print(f"Advertencia: Valor no numérico para '{col\_posicion\_nombre}' ('{fila\[col\_posicion\_nombre\]}') en {ruta\_posiciones}, fila {index+1}. Saltando fila.")

            continue

        except KeyError:

            print(f"Advertencia: Columna '{col\_posicion\_nombre}' o '{col\_archivo\_nombre}' no encontrada en {ruta\_posiciones}, fila {index+1} (después del intento de adaptación). Saltando fila.")

            continue

        sufijo\_archivo \= nombre\_archivo\_base\_pos.split('\_')\[-1\]

        nombre\_archivo\_real \= PREFIJO\_REAL\_ARCHIVOS \+ sufijo\_archivo \+ EXTENSION\_ARCHIVOS\_DATOS

        ruta\_archivo\_dato \= os.path.join(directorio\_set, nombre\_archivo\_real)

        

        u\_media, e\_cuadrado \= analizar\_archivo\_lda(ruta\_archivo\_dato)

        

        if u\_media is not None and e\_cuadrado is not None:

            resultados\_set\_actual.append({

                "r/R": rr\_posicion,

                "Archivo Original (Posiciones.txt)": nombre\_archivo\_base\_pos,

                "Archivo de Datos Procesado": nombre\_archivo\_real,

                "Velocidad Media (u\_bar)": u\_media,

                "Intensidad de Turbulencia (e^2)": e\_cuadrado,

                "Set": nombre\_set\_descriptivo 

            })

        else:

            print(f"  No se pudieron procesar datos para archivo base: {nombre\_archivo\_base\_pos} (esperado como {nombre\_archivo\_real})")

    if not resultados\_set\_actual:

        print(f"No se procesaron datos exitosamente para {nombre\_set\_descriptivo}.")

        return None

        

    df\_resultados\_completos \= pd.DataFrame(resultados\_set\_actual)

    df\_resultados\_completos \= df\_resultados\_completos.sort\_values(by='r/R').reset\_index(drop=True)

    

    print(f"Resultados para {nombre\_set\_descriptivo}:")

    \# Mostrar solo las columnas relevantes en la impresión de la consola

    print(df\_resultados\_completos\[\['r/R', 'Velocidad Media (u\_bar)', 'Intensidad de Turbulencia (e^2)'\]\].to\_string())

    return df\_resultados\_completos

\# \--- FUNCIÓN PARA GRAFICAR PERFILES INDIVIDUALES \---

def graficar\_perfil\_individual(df\_resultados, titulo\_grafica\_base):

    if df\_resultados is None or df\_resultados.empty:

        print(f"No hay datos para graficar para {titulo\_grafica\_base}.")

        return

    \# Limpiar el nombre del archivo para guardar la gráfica

    nombre\_archivo\_limpio \= "".join(c if c.isalnum() or c in (' ', '\_', '-') else '\_' for c in titulo\_grafica\_base).rstrip()

    nombre\_archivo\_limpio \= nombre\_archivo\_limpio.replace(' ', '\_').replace('/', '-').replace(':', '')

    \# Gráfica de Velocidad Media

    plt.figure(figsize=(10, 6))

    plt.plot(df\_resultados\['r/R'\], df\_resultados\['Velocidad Media (u\_bar)'\], marker='o', linestyle='-')

    plt.xlabel('Posición Transversal Adimensional, r/R \[-\]')

    plt.ylabel('Velocidad Media ($\\overline{u}$) \[m/s\]')

    plt.title(f'Perfil de Velocidad Media \- {titulo\_grafica\_base}')

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.axvline(0, color='k', linestyle=':', linewidth=0.8) \# Línea en r/R \= 0

    plt.savefig(f"Perfil\_Velocidad\_{nombre\_archivo\_limpio}.png")

    plt.show()

    \# Gráfica de Intensidad de Turbulencia

    plt.figure(figsize=(10, 6))

    plt.plot(df\_resultados\['r/R'\], df\_resultados\['Intensidad de Turbulencia (e^2)'\], marker='s', linestyle='--', color='red')

    plt.xlabel('Posición Transversal Adimensional, r/R \[-\]')

    plt.ylabel('Intensidad de Turbulencia ($e^2$) \[-\]')

    plt.title(f'Perfil de Intensidad de Turbulencia \- {titulo\_grafica\_base}')

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.axvline(0, color='k', linestyle=':', linewidth=0.8) \# Línea en r/R \= 0

    plt.savefig(f"Perfil\_Turbulencia\_{nombre\_archivo\_limpio}.png")

    plt.show()

\# \--- FUNCIÓN PARA GRAFICAR PERFILES COMPARATIVOS \---

def graficar\_perfiles\_comparativos(lista\_dfs\_resultados, nombres\_sets\_descriptivos):

    if not lista\_dfs\_resultados:

        print("No hay datos para gráficas comparativas.")

        return

    \# Comparativa de Velocidad Media

    plt.figure(figsize=(12, 7))

    for i, df in enumerate(lista\_dfs\_resultados):

        if df is not None and not df.empty:

            \# Usar el nombre descriptivo correspondiente al df actual

            nombre\_legenda \= nombres\_sets\_descriptivos\[i\] if i \< len(nombres\_sets\_descriptivos) else f"Set {i+1}"

            plt.plot(df\['r/R'\], df\['Velocidad Media (u\_bar)'\], marker='o', linestyle='-', label=nombre\_legenda)

    plt.xlabel('Posición Transversal Adimensional, r/R \[-\]')

    plt.ylabel('Velocidad Media ($\\overline{u}$) \[m/s\]')

    plt.title('Comparación de Perfiles Transversales de Velocidad Media')

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.axvline(0, color='k', linestyle=':', linewidth=0.8) \# Línea en r/R \= 0

    plt.legend()

    plt.savefig("Comparacion\_Perfiles\_Velocidad.png")

    plt.show()

    \# Comparativa de Intensidad de Turbulencia

    plt.figure(figsize=(12, 7))

    colores \= \['blue', 'red', 'green', 'purple', 'orange', 'brown'\] \# Para distinguir los sets

    for i, df in enumerate(lista\_dfs\_resultados):

        if df is not None and not df.empty:

            nombre\_legenda \= nombres\_sets\_descriptivos\[i\] if i \< len(nombres\_sets\_descriptivos) else f"Set {i+1}"

            plt.plot(df\['r/R'\], df\['Intensidad de Turbulencia (e^2)'\], marker='s', linestyle='--', label=nombre\_legenda, color=colores\[i % len(colores)\])

    plt.xlabel('Posición Transversal Adimensional, r/R \[-\]')

    plt.ylabel('Intensidad de Turbulencia ($e^2$) \[-\]')

    plt.title('Comparación de Perfiles Transversales de Intensidad de Turbulencia')

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.axvline(0, color='k', linestyle=':', linewidth=0.8) \# Línea en r/R \= 0

    plt.legend()

    plt.savefig("Comparacion\_Perfiles\_Turbulencia.png")

    plt.show()

\# \--- EJECUCIÓN PRINCIPAL DEL SCRIPT \---

if \_\_name\_\_ \== "\_\_main\_\_":

    \# Directorio base donde se encuentra este script y las carpetas de los sets (FX01G00, etc.)

    directorio\_base \= "." \# Directorio actual

    \# Encontrar las carpetas de los sets automáticamente

    patron\_carpetas \= os.path.join(directorio\_base, PREFIJO\_CARPETAS\_SETS \+ "\*")

    \# sorted() para un orden consistente (ej. FX01, FX02, ...)

    \# Es importante que este orden coincida con NOMBRES\_DESCRIPTIVOS\_SETS

    directorios\_sets \= sorted(\[d for d in glob.glob(patron\_carpetas) if os.path.isdir(d)\]) 

    if not directorios\_sets:

        print(f"Error: No se encontraron carpetas de sets con el patrón '{patron\_carpetas}' en '{directorio\_base}'.")

        print("Asegúrate de que las carpetas (ej. FX01G00, FX02G00) estén en el mismo directorio que el script,")

        print("y que PREFIJO\_CARPETAS\_SETS esté correctamente definido en el script.")

    else:

        print(f"Carpetas de sets encontradas y ordenadas: {directorios\_sets}")

    todos\_los\_dataframes\_resultados \= \[\]

    nombres\_reales\_sets\_procesados \= \[\] 

    if len(directorios\_sets) \!= len(NOMBRES\_DESCRIPTIVOS\_SETS) and directorios\_sets:

        print(f"Advertencia: Se encontraron {len(directorios\_sets)} carpetas de set, pero se definieron {len(NOMBRES\_DESCRIPTIVOS\_SETS)} nombres descriptivos.")

        print("Se usarán nombres genéricos para los sets si la cantidad no coincide o si hay más carpetas que nombres.")

        usar\_nombres\_genericos\_por\_discrepancia \= True

    else:

        usar\_nombres\_genericos\_por\_discrepancia \= False

    for i, dir\_set in enumerate(directorios\_sets):

        \# dir\_set ya es un directorio por el filtro en glob.glob y os.path.isdir

        

        if usar\_nombres\_genericos\_por\_discrepancia or i \>= len(NOMBRES\_DESCRIPTIVOS\_SETS):

            \# Usar nombre genérico si hay discrepancia o si nos quedamos sin nombres descriptivos

            nombre\_actual\_set \= f"Set {i+1} ({os.path.basename(dir\_set)})"

        else:

            \# Usar el nombre descriptivo provisto

            nombre\_actual\_set \= NOMBRES\_DESCRIPTIVOS\_SETS\[i\]

        

        df\_set\_actual \= procesar\_set\_de\_datos(dir\_set, nombre\_actual\_set)

        if df\_set\_actual is not None and not df\_set\_actual.empty:

            todos\_los\_dataframes\_resultados.append(df\_set\_actual)

            nombres\_reales\_sets\_procesados.append(nombre\_actual\_set) \# Guardar el nombre usado

            graficar\_perfil\_individual(df\_set\_actual, nombre\_actual\_set)

        else:

            print(f"  No se generó DataFrame para el set en '{dir\_set}' o estaba vacío.")

    if todos\_los\_dataframes\_resultados:

        \# Pasar los nombres que realmente se usaron para los sets procesados

        graficar\_perfiles\_comparativos(todos\_los\_dataframes\_resultados, nombres\_reales\_sets\_procesados)

        print("\\nAnálisis completado. Se generaron tablas en consola y gráficas.")

        print("Las gráficas también se han guardado como archivos .png en el directorio del script.")

    else:

        print("\\nNo se procesaron datos de ningún set o todos los DataFrames resultaron vacíos. Verifica los errores anteriores.")

