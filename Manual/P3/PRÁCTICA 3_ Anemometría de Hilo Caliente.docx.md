# **PRÁCTICA 3: Anemometría de Hilo Caliente**

## **I. Introducción**

Se introducirán los principios de la anemometría de hilo caliente (HWA, por sus siglas en inglés *Hot-Wire Anemometry*) para la medición de velocidades de flujo y turbulencia, destacando su aplicación en la investigación aerodinámica y estudios de capa límite. La anemometría de hilo caliente es una técnica de medición indirecta de la velocidad de fluidos, de gran relevancia en la ingeniería aeronáutica, especialmente para el estudio de flujos gaseosos complejos, la caracterización detallada de la turbulencia y el análisis de capas límite. El principio físico subyacente es la transferencia de calor por convección desde un elemento sensor (un hilo metálico muy delgado), calentado eléctricamente, hacia el fluido gaseoso que lo rodea. La cantidad de calor disipado por el sensor está directamente relacionada con la velocidad del fluido que incide sobre él.

A diferencia de la anemometría de película caliente (HFA), que es más robusta y se emplea a menudo en líquidos o flujos con partículas, la anemometría de hilo caliente se utiliza primordialmente en gases limpios debido a la fragilidad inherente del sensor de hilo desnudo. Los sensores de HWA, al no poseer una capa protectora, ofrecen una inercia térmica muy baja, lo que resulta en una respuesta en frecuencia significativamente más alta, permitiendo la medición de fluctuaciones rápidas de velocidad y, por ende, el estudio detallado de la turbulencia. Sin embargo, esta misma característica los hace susceptibles al daño mecánico por partículas o al ensuciamiento en flujos no acondicionados.

Para que las mediciones obtenidas mediante un sensor de hilo caliente sean cuantitativamente útiles y precisas, es indispensable realizar un proceso de calibración. La calibración tiene como objetivo establecer una relación funcional unívoca entre la señal eléctrica de salida del anemómetro (generalmente un voltaje, E) y la velocidad del fluido (U) a la que está expuesto el sensor. Esta relación, conocida como curva de calibración, es específica para cada sensor y puede variar con el tiempo debido al envejecimiento del mismo o a la acumulación de impurezas en su superficie.

La práctica se enfocará en el análisis de datos de voltaje obtenidos de un sensor HWA en diferentes condiciones de flujo. Los estudiantes aplicarán la ley de King (o un modelo similar) a partir de datos de calibración provistos, para convertir las lecturas de voltaje en velocidades, calcular promedios, fluctuaciones de velocidad (intensidad de turbulencia) y discutir la respuesta del sensor basándose en los datos.

## **II. Objetivos de Aprendizaje Específicos**

Al finalizar esta práctica, el estudiante será capaz de:

* Comprender el principio de funcionamiento de un anemómetro de hilo caliente y sus diferencias fundamentales con otras técnicas de medición de velocidad.  
* Identificar los componentes principales de un sistema de anemometría de hilo caliente operando en modo de temperatura constante (CTA).  
* Entender la importancia y la metodología general para la calibración de un sensor de hilo caliente.  
* Reconocer la relevancia de la limpieza (o la prevención de contaminación) del sensor y cómo la contaminación puede afectar la precisión de las mediciones.  
* Procesar series temporales de datos de voltaje crudos, provenientes de mediciones experimentales con un sensor HWA, para extraer la información relevante.  
* Aplicar criterios para seleccionar la porción útil de una señal y calcular voltajes promedio representativos para diferentes condiciones de flujo conocidas.  
* Relacionar un parámetro de control del sistema de flujo (porcentaje de velocidad del accionador, SD) con la velocidad del fluido (Uref​) utilizando una curva de calibración preexistente del sistema de flujo.  
* Construir una tabla de datos que relacione el voltaje promedio del sensor (ESD​) con la velocidad de referencia del fluido (Uref​) para un rango de condiciones.  
* Graficar la curva de calibración del sensor de hilo caliente (Uref​ vs. ESD​).  
* Aplicar técnicas de ajuste de curvas mediante regresión lineal sobre datos transformados para obtener el modelo matemático de calibración del sensor, según la forma A⋅E2+B=U1/n (Ley de King modificada), determinando las constantes A y B para un valor de n dado.  
* Evaluar la calidad del ajuste del modelo de calibración obtenido mediante el coeficiente de determinación (R2).  
* Interpretar las señales de voltaje para estimar fluctuaciones de velocidad y comprender el concepto de intensidad de turbulencia (aunque el cálculo detallado de esta última puede requerir pasos adicionales no cubiertos por el script básico).  
* Discutir las posibles fuentes de error e incertidumbre inherentes al proceso de calibración de un sensor de hilo caliente en un entorno experimental.  
* Discutir la respuesta del sensor basándose en los datos analizados.

## **III. Fundamentos Teóricos**

### **A. Principio de Funcionamiento del Anemómetro de Hilo Caliente**

El funcionamiento de un anemómetro de hilo caliente se basa en el fenómeno de transferencia de calor por convección forzada. El sensor consiste en un hilo metálico muy delgado (típicamente de tungsteno, platino o aleaciones de platino-iridio), con diámetros del orden de unos pocos micrómetros (ej. 5 µm) y longitudes de unos pocos milímetros (ej. 1-3 mm). Este hilo está soportado entre dos agujas o puntas metálicas que, a su vez, lo conectan eléctricamente al resto del sistema. El hilo metálico se calienta eléctricamente mediante el paso de una corriente, alcanzando una temperatura significativamente superior (ej. 200-300 °C) a la del fluido gaseoso circundante. Cuando el gas fluye sobre el hilo, este cede calor al gas, enfriándose. La tasa de pérdida de calor depende, entre otros factores, de la velocidad del gas.

Existen dos modos principales de operación para los anemómetros térmicos:

1. **Anemómetro de Corriente Constante (CCA \- Constant Current Anemometer):** En este modo, la corriente eléctrica que atraviesa el sensor se mantiene constante. A medida que la velocidad del fluido varía, la temperatura del hilo (y, por lo tanto, su resistencia eléctrica) cambia debido al enfriamiento convectivo. La variación de la resistencia del sensor se mide como una variación de voltaje, la cual se relaciona con la velocidad del fluido. Este modo es menos común hoy en día para mediciones de turbulencia debido a su limitada respuesta en frecuencia comparada con el CTA.  
2. **Anemómetro de Temperatura Constante (CTA \- Constant Temperature Anemometer):** Este es el modo de operación más extendido y el que se considera en esta práctica. En un CTA, la temperatura del hilo (y, por ende, su resistencia eléctrica) se mantiene constante a un valor predefinido, independientemente de la velocidad del fluido. Esto se logra mediante un circuito de retroalimentación, típicamente un puente de Wheatstone en el que el sensor es uno de los brazos, y un amplificador servo-controlado de alta ganancia. Si la velocidad del fluido aumenta, el hilo tiende a enfriarse; el puente detecta este cambio de resistencia y el servo-amplificador incrementa la corriente que circula por el hilo para restaurar su temperatura (y resistencia) original. De forma análoga, si la velocidad disminuye, la corriente se reduce. El voltaje (E) necesario para mantener constante la temperatura del hilo es la señal de salida del anemómetro y está directamente relacionado con la velocidad del fluido (U). A mayor velocidad del fluido, mayor es el enfriamiento, y por lo tanto, mayor es el voltaje (y la potencia eléctrica disipada) que el sistema debe suministrar. Los CTA ofrecen una excelente respuesta en frecuencia, esencial para mediciones de turbulencia.

### **B. Ley de King**

La relación entre la potencia eléctrica disipada por el sensor y la velocidad del fluido fue estudiada inicialmente por L.V. King (1914) para un hilo cilíndrico calentado en un flujo perpendicular. La Ley de King establece que la potencia calorífica (Q) disipada por el hilo es:

Q=(Ts​−Tf​)(k0​+k1​U​)

Donde:

* Ts​ es la temperatura de la superficie del hilo sensor.  
* Tf​ es la temperatura del fluido (gas).  
* k0​ y k1​ son constantes que dependen de las propiedades del fluido (conductividad térmica, densidad, viscosidad), las dimensiones del hilo y, en el caso de k0​, también de la conducción de calor a los soportes del hilo y la convección natural. El término k1​U​ representa la pérdida de calor por convección forzada.

Para un anemómetro de temperatura constante (CTA), la temperatura del hilo Ts​ (y su resistencia Rs​) se mantienen constantes. La potencia eléctrica disipada por el sensor es Q=E2/Rs​, donde E es el voltaje aplicado al puente del anemómetro (proporcional al voltaje sobre el hilo). Sustituyendo esto en la Ley de King y agrupando términos constantes, se obtiene una relación de la forma:

E2=A′+B′⋅Un

Donde A′ y B′ son nuevas constantes de calibración y el exponente n teóricamente es 0.5 para un hilo ideal en flujo cruzado. En la práctica, n puede variar (típicamente entre 0.4 y 0.5 para hilos) debido a efectos de conducción en los extremos del hilo, convección natural a bajas velocidades, y la dependencia de las propiedades del fluido con la temperatura.

Para facilitar la linealización y el ajuste de curvas, es común utilizar una forma modificada de la Ley de King, como la propuesta en el documento de referencia "Lab\_PeliculaCaliente.pdf" (adaptada aquí para HWA):

A⋅E2+B=U1/n

Donde A y B son las constantes de calibración a determinar, y n es un exponente que se fija (en esta práctica, se asumirá n=3 según la referencia original, aunque para HWA un valor de n≈2 que corresponde al n≈0.5 de la forma E2=A′+B′Un sería más típico; se mantendrá n=3 si así lo requiere la práctica específica para mantener consistencia con materiales previos, o se ajustará si se indica). Esta forma permite transformar la ecuación en una relación lineal si se consideran las variables X=E2 e Y=U1/n, resultando en Y=A⋅X+B. Una vez obtenidas A y B, la velocidad U se puede calcular como:

U=(A⋅E2+B)n

### **C. Calibración del Sensor de Hilo Caliente**

La calibración es el proceso experimental mediante el cual se determinan las constantes (A, B, y a veces n) de la ecuación de la Ley de King que mejor describen la respuesta del sensor específico que se está utilizando. Este proceso es fundamental porque la relación E−U varía de un sensor a otro (debido a pequeñas diferencias en longitud, diámetro, material) e incluso para un mismo sensor a lo largo del tiempo debido al envejecimiento, deformación o contaminación.

El procedimiento general de calibración implica:

1. Exponer el sensor a una serie de flujos de gas con velocidades conocidas y estables (Ui​).  
2. Registrar el voltaje de salida promedio del anemómetro (Ei​) para cada una de estas velocidades.  
3. Utilizar los pares de datos (Ui​,Ei​) para ajustar la ecuación de la Ley de King y determinar las constantes de calibración.

La velocidad del fluido de referencia (Ui​) se obtiene normalmente a partir de un instrumento de medición de velocidad más preciso (ej. Tubo de Pitot, Anemómetro Láser Doppler \- LDA) o, como en el caso de esta práctica, a partir de una calibración previa del propio sistema generador de flujo (ej. un túnel de viento). En nuestro caso, se utiliza un parámetro de control del sistema de flujo, denominado SD (Speed Drive percentage), que se relaciona con la velocidad del fluido (Uref​) mediante una ecuación de calibración del sistema de flujo:

Uref​,\[mm/s\]=7.6243⋅SD,\[

### **D. Importancia de la Limpieza y Cuidado del Sensor de Hilo Caliente**

Los sensores de hilo caliente son extremadamente sensibles a la contaminación y muy frágiles. La acumulación de polvo, aceite u otras partículas diminutas sobre el hilo sensor puede alterar significativamente sus características de transferencia de calor y su respuesta dinámica.

**Efectos de la Contaminación:**

* **Aislamiento Térmico Parcial:** Una capa de contaminante puede actuar como un aislante térmico, reduciendo la eficiencia con la que el calor se transfiere del hilo al fluido. Esto provoca que, para una misma velocidad real del fluido, el hilo se enfríe menos, y el sistema CTA requiera un voltaje E menor para mantener la temperatura Ts​. Como resultado, la velocidad indicada por el anemómetro (usando la calibración original) será inferior a la real.  
* **Cambio en la Geometría y Masa:** La acumulación de material puede cambiar el diámetro efectivo y la masa del hilo, afectando las características del flujo local, la convección y, crucialmente, la respuesta en frecuencia del sensor (aumentando su inercia térmica).  
* **Deriva de Calibración:** La contaminación es una de las principales causas de la deriva de la calibración con el tiempo.

Cuidado y Prevención (Limpieza Conceptual):  
La limpieza de un hilo caliente es un proceso extremadamente delicado y, a menudo, se intenta evitar mediante el uso en ambientes limpios. Si la limpieza es necesaria:

1. **Inspección Visual:** Antes y después de cada uso, inspeccionar el hilo (preferiblemente con una lupa o microscopio de bajo aumento) para detectar signos de contaminación o daño (como deformaciones o rotura).  
2. **Prevención:** Utilizar siempre aire filtrado en el túnel de viento. Evitar tocar el hilo o exponerlo a corrientes de aire repentinas cuando no está en uso.  
3. **Limpieza (con extrema precaución y según fabricante):**  
   * *Solventes:* Algunos fabricantes pueden recomendar sumergir cuidadosamente la punta de la sonda en un solvente volátil y limpio (ej. alcohol isopropílico de alta pureza) y permitir que se seque por evaporación. No se debe agitar bruscamente.  
   * *Corriente de Aire Suave:* Una corriente muy suave de aire limpio y seco puede usarse a veces para desalojar partículas sueltas, pero debe hacerse con sumo cuidado.  
   * *Nunca* se debe intentar limpiar un hilo caliente mecánicamente (ej. con un cepillo o paño), ya que esto casi con seguridad lo romperá.

*Es imperativo consultar siempre el manual del fabricante del sensor y del anemómetro para conocer los procedimientos de manejo, las limitaciones y, si aplica, los métodos de limpieza específicos y recomendados.* Un manejo cuidadoso y la operación en un entorno limpio son cruciales para mantener la precisión de las mediciones y prolongar la vida útil del frágil sensor. La necesidad de recalibración frecuente puede ser un indicador de contaminación del hilo.

## **IV. Materiales y Equipos**

### **A. Para un Montaje Experimental Típico (Contextual)**

Para llevar a cabo la fase experimental de adquisición de datos (que en esta práctica se omite para centrarse en el análisis), se requerirían los siguientes elementos:

1. **Sensor de Hilo Caliente:** Con su sonda y cableado correspondiente (ej. Dantec Dynamics, TSI).  
2. **Anemómetro CTA:** Unidad electrónica que alimenta el sensor, implementa el puente de Wheatstone y el servo-amplificador, y proporciona la señal de voltaje de salida.  
3. **Sistema de Adquisición de Datos (DAQ):** Tarjeta de adquisición de datos y una computadora con software adecuado para registrar las series temporales de voltaje del anemómetro (con alta frecuencia de muestreo para estudios de turbulencia).  
4. **Fuente de Flujo Calibrada (Túnel de Viento):**  
   * Túnel de viento con sección de pruebas que proporcione un flujo de aire uniforme, estable y de baja turbulencia (para calibración).  
   * Sistema de control de velocidad del flujo (ej. variador de frecuencia para el motor del ventilador), cuya configuración (ej. SD%) esté previamente calibrada contra una referencia de velocidad (ej. Tubo de Pitot, LDA).  
   * Filtros de aire para asegurar la limpieza del flujo.  
5. **Material de Cuidado del Sensor:**  
   * Solventes de alta pureza (ej. alcohol isopropílico), si el fabricante lo recomienda para limpieza muy ocasional.  
   * Recipientes limpios.  
   * Fuente de aire limpio y seco para secado (opcional y con cuidado).  
6. **Termómetro y Barómetro:** Para medir y monitorizar la temperatura y presión del aire, ya que las propiedades del gas (y por tanto las constantes de la Ley de King) dependen de ellas. Es crucial que la temperatura del fluido se mantenga constante durante la calibración y las mediciones posteriores, o que se apliquen correcciones.

### **B. Para el Análisis de Datos en esta Práctica**

Para la realización de esta práctica, se necesitarán:

1. **Computadora Personal:** Con sistema operativo Windows, macOS o Linux.  
2. **Software Python:** Versión 3.x. Se recomienda la distribución Anaconda que incluye la mayoría de las bibliotecas necesarias.  
   * **Bibliotecas de Python:**  
     * NumPy: Para cálculo numérico eficiente.  
     * Pandas: Para manipulación y análisis de datos tabulares.  
     * Matplotlib: Para la generación de gráficas.  
     * SciPy: Para funciones científicas, incluyendo la regresión lineal.  
3. **Archivos de Datos Experimentales:** Un conjunto de archivos de texto (.txt) proporcionados por el instructor, conteniendo las series temporales de voltaje del sensor HWA para diferentes valores de SD (ej., 15\_1.txt, 20\_3.txt, etc.). Estos archivos deben estar organizados en una carpeta accesible por el script de Python.  
4. **Script de Python:** El script (ver Apéndice A) para procesar los datos, realizar los cálculos y generar los resultados.  
5. **Editor de Texto o Entorno de Desarrollo Integrado (IDE):** Para visualizar y ejecutar el script de Python (ej. VS Code, Spyder, Jupyter Notebook).

## **V. Procedimiento Experimental y Adquisición de Datos**

Dado el enfoque analítico de este laboratorio, donde el énfasis recae en el tratamiento e interpretación de datos, **en esta práctica se proporcionarán directamente los conjuntos de datos de voltaje (**E**) del sensor HWA.** Estos datos han sido previamente adquiridos para diferentes condiciones de flujo, identificadas por un parámetro de control del sistema de flujo denominado SD (*Speed Drive percentage*).

Los datos se suministran en archivos de texto (.txt). Cada archivo representa una serie temporal de mediciones de voltaje para un valor de SD específico. La nomenclatura de los archivos sigue el formato SD\_iteracion.txt. Se dispone de múltiples archivos (iteraciones) para cada valor de SD, cubriendo un rango desde SD=15% hasta SD=35%.

A continuación, se describe conceptualmente cómo se llevaría a cabo la preparación del sensor y la adquisición de estos datos en un entorno experimental, seguido de la metodología estructurada para el tratamiento de los datos proporcionados.

### **A. Preparación del Sensor y Montaje Experimental (Conceptual)**

1. **Inspección del Sensor:** Antes de iniciar cualquier medición, se inspeccionaría visualmente el sensor de hilo caliente bajo una lupa o microscopio para detectar cualquier signo de daño físico (hilo roto, doblado) o contaminación evidente.  
2. **Montaje del Sensor:** El sensor se montaría cuidadosamente en la sección de pruebas del túnel de viento, asegurando su correcta orientación (generalmente perpendicular a la dirección principal del flujo para un sensor de un solo hilo). Se debe tener extremo cuidado para no dañar el hilo durante el montaje.  
3. **Conexión al Sistema CTA:** El sensor se conectaría al anemómetro CTA. Se encendería el equipo y se permitiría un tiempo de calentamiento y estabilización adecuado (generalmente 15-30 minutos) antes de proceder con la calibración. Durante este tiempo, el hilo alcanza su temperatura de operación.  
4. **Ajuste de Parámetros del CTA:** Se configuraría la relación de sobrecalentamiento (o la temperatura de operación del hilo) en el anemómetro CTA. Este valor es crucial y debe mantenerse constante durante toda la calibración y las mediciones posteriores. También se ajustaría la compensación del cable y otros parámetros del puente según el manual del equipo.  
5. **Verificación de la Temperatura del Fluido:** Se mediría y registraría la temperatura del aire en el túnel. Es fundamental que esta temperatura permanezca estable durante todo el proceso.

### **B. Adquisición de Datos de Calibración (Conceptual)**

1. **Flujo Cero (o Mínimo):** Con el sensor estabilizado, se ajustaría el sistema de flujo para obtener una condición de velocidad cero (si es posible y seguro para el hilo) o la mínima velocidad estable posible. Se registraría el voltaje de salida del anemómetro (E0​) correspondiente.  
2. **Incremento Gradual de la Velocidad:** Se incrementaría la velocidad del flujo de manera escalonada, estableciendo diferentes valores del parámetro de control SD (desde 15% hasta 35%, en los incrementos para los que se dispone de datos).  
3. **Estabilización y Registro:** Para cada valor de SD:  
   * Se esperaría un tiempo suficiente para que el flujo en la sección de pruebas y la señal del anemómetro se estabilicen completamente.  
   * Se adquiriría una serie temporal de datos de voltaje del anemómetro (E(t)) durante un período de tiempo adecuado (ej. varios segundos o decenas de segundos) y a una frecuencia de muestreo suficientemente alta (ej. varios kHz o superior, dependiendo de la turbulencia esperada) para capturar las características de la señal, incluyendo las fluctuaciones.  
   * Se repetiría la adquisición varias veces (múltiples iteraciones) para cada valor de SD. Esto permite evaluar la repetibilidad de las mediciones y obtener un promedio más robusto. Los archivos proporcionados (SD\_iteracion.txt) son el resultado de este proceso.  
4. **Registro de Datos:** Cada serie temporal de datos se guardaría en un archivo individual, con una nomenclatura clara que identifique la condición de SD y el número de iteración.

### **C. Metodología Estructurada para el Procesamiento de los Datos Proporcionados**

Los datos para esta práctica consisten en múltiples archivos de texto. Cada archivo contiene una columna de valores de voltaje instantáneo (VDC) correspondientes a una medición específica (iteración) para un determinado valor de SD.

El procesamiento seguirá los siguientes pasos, implementados mediante el script de Python (Apéndice A):

1. **Configuración Inicial:** Definir en el script la ruta a los datos, rango de SD, exponente n de King, coeficientes de calibración del túnel, y porcentajes de descarte de señal.  
2. **Organización y Carga de Archivos:** El script identificará y agrupará los archivos por SD.  
3. **Procesamiento por Iteración Individual:**  
   * Carga de la serie temporal E(t).  
   * Selección de la porción útil (descarte de transitorios).  
   * Cálculo del voltaje promedio de la iteración (Eiter​).  
   * *(Nota: Para análisis de turbulencia, aquí también se calcularía la desviación estándar o RMS de los voltajes de la porción útil,* erms′​*, pero el script actual se enfoca en* Eiter​ *para la calibración de velocidad media).*  
4. **Consolidación por Condición de SD:**  
   * Cálculo del voltaje promedio para SD (ESD​) promediando los Eiter​ de ese SD.  
   * Cálculo de la desviación estándar de los Eiter​ para ese SD (como medida de repetibilidad entre iteraciones).  
5. **Cálculo de Velocidades de Referencia (**Uref​**):** Usando la ecuación de calibración del sistema de flujo.  
6. **Construcción de la Tabla de Datos para Calibración:** Columnas: SD, Niter​, ESD​, σESD​​, Uref​, ESD2​, Uref1/n​.  
7. **Ajuste de Curva (Regresión Lineal):** Ajustar Y=A⋅X+B donde X=ESD2​ e Y=Uref1/n​. Obtener A,B,R2.  
8. **Obtención del Modelo de Calibración Final del Sensor:** U=(A⋅E2+B)n.  
9. **Generación de Gráficas:** Uref1/n​ vs ESD2​ con ajuste lineal, y Uref​ vs ESD​ con la curva de King ajustada.

## **VI. Procesamiento y Análisis de Datos (Mediante Software)**

El procesamiento detallado de los datos se realizará utilizando Python y las bibliotecas científicas (NumPy, Pandas, Matplotlib, SciPy), según el script del **Apéndice A**. Se espera que el estudiante:

1. Revise y comprenda el código.  
2. Configure los parámetros iniciales (ruta de datos).  
3. Ejecute el script.  
4. Analice los resultados numéricos y gráficos.  
5. Utilice estos resultados para el informe.

El script realizará los pasos descritos en la Sección V.C.

## **VII. Presentación de Resultados**

El informe de la práctica deberá incluir, como mínimo:

1. **Tabla de Datos de Calibración:**  
   * SD (%), Número de iteraciones, ESD​ (V), σESD​​ (V), Uref​ (mm/s), ESD2​ (V2), Uref1/3​ ((mm/s)1/3).  
2. **Resultados del Ajuste Lineal:**  
   * Constantes A y B (con unidades).  
   * Coeficiente de determinación R2.  
   * Ecuación final del modelo de calibración: U=(A⋅E2+B)3.  
3. **Gráficas (generadas por el script):**  
   * **Gráfica de Datos Transformados y Ajuste Lineal:** Uref1/3​ vs. ESD2​, mostrando puntos experimentales y recta de ajuste (con ecuación y R2).  
   * **Curva de Calibración Final del Sensor:** U vs. ESD​, mostrando puntos de calibración y curva del modelo de King ajustado.

Todos los ejes de las gráficas deben estar claramente etiquetados.

## **VIII. Cuestionario**

1. Explique con sus propias palabras el principio de funcionamiento de un anemómetro de hilo caliente operando en modo de temperatura constante (CTA). ¿Cuál es la magnitud que se mide directamente y cómo se relaciona con la velocidad del fluido?  
2. ¿Cuáles son las principales ventajas y desventajas de la anemometría de hilo caliente (HWA)? ¿En qué tipo de fluidos y aplicaciones se prefiere usar HWA? Considere compararlo brevemente con otras técnicas si lo desea (ej. tubos de Pitot, LDA).  
3. Describa la Ley de King. ¿Qué relación fundamental establece? Explique la forma de la ecuación utilizada en esta práctica (A⋅E2+B=U1/n) y el significado de cada término, incluyendo las unidades esperadas para A y B si U está en mm/s, E en Volts y n=3.  
4. ¿Por qué es indispensable calibrar un sensor de hilo caliente? ¿Qué factores pueden hacer que la calibración de un sensor cambie con el tiempo?  
5. Detalle la importancia del cuidado y la prevención de la contaminación del sensor de hilo caliente. ¿Cómo puede la contaminación del sensor (por ejemplo, por polvo o acumulación de aceite) afectar las mediciones de velocidad? Sea específico sobre los efectos en la transferencia de calor y la señal del anemómetro.  
6. En el procesamiento de las series temporales de voltaje, ¿cuál es el propósito de seleccionar una "porción útil" de la señal antes de calcular el promedio? ¿Qué tipo de fenómenos o problemas se intentan evitar con este paso?  
7. ¿Qué representa el coeficiente de determinación (R2) en el contexto del ajuste lineal de los datos transformados? Si obtiene un R2=0.995, ¿cómo interpretaría este resultado? ¿Y si obtuviera R2=0.75?  
8. Si la temperatura del fluido (Tf​) aumentara significativamente durante una serie de mediciones realizadas *después* de haber calibrado el sensor a una temperatura Tf,cal​, ¿cómo esperaría que este cambio afecte la precisión de las velocidades medidas utilizando la curva de calibración original? Justifique su respuesta basándose en la Ley de King (Q=(Ts​−Tf​)(k0​+k1​U​)).  
9. Mencione y describa brevemente al menos tres posibles fuentes de error o incertidumbre que pueden surgir durante un proceso *experimental real* de calibración de un sensor de hilo caliente.  
10. Proponga y describa una aplicación específica en el campo de la ingeniería aeronáutica donde la anemometría de hilo caliente sería una técnica de medición adecuada y valiosa. Justifique su elección.  
11. ¿Cómo se podrían utilizar los datos de voltaje de un sensor HWA para obtener información sobre las fluctuaciones de velocidad (turbulencia) en el flujo? Describa conceptualmente qué se calcularía a partir de la señal E(t) y cómo se relacionaría con la intensidad de turbulencia.

## **IX. Conclusiones**

El estudiante deberá redactar un apartado de conclusiones basado en el análisis realizado. Estas conclusiones deben abordar, como mínimo:

* El proceso seguido para obtener la curva de calibración del sensor de hilo caliente.  
* La presentación de la ecuación de calibración final obtenida.  
* Una discusión sobre la validez y la calidad del modelo de calibración, basándose en R2 y las gráficas.  
* La comprensión adquirida sobre la importancia de la calibración y los factores que afectan la medición con HWA (limpieza, temperatura del fluido, fragilidad del sensor).  
* Reflexiones sobre las posibles limitaciones del análisis o de la técnica.  
* La relevancia de la HWA en ingeniería aeronáutica, incluyendo su capacidad para medir turbulencia (aunque no se haya implementado su cálculo completo en el script).  
* Una breve discusión sobre cómo se podría extender el análisis para cuantificar la intensidad de turbulencia.

## **X. Referencias**

* Bruun, H. H. (1995). *Hot-Wire Anemometry: Principles and Signal Analysis*. Oxford University Press.  
* Comte-Bellot, G. (1976). Hot-Wire Anemometry. *Annual Review of Fluid Mechanics, 8*, 209-231.  
* Goldstein, R. J. (Ed.). (1983). *Fluid Mechanics Measurements* (2nd ed.). Hemisphere Publishing Corporation.  
* Ligrani, P. (2013). *Fluid Mechanics: An Intermediate Approach*. Cambridge University Press.  
* Documento "Lab\_PeliculaCaliente.pdf" (o similar adaptado para HWA) proporcionado como material de referencia.  
* Manuales de fabricantes de equipos de anemometría (ej. Dantec Dynamics, TSI Inc.).

## **XI. Apéndices**

### **Apéndice A: Código Python para el Análisis de Datos**

\# \-----------------------------------------------------------------------------  
\# PRÁCTICA 3: ANEMOMETRÍA DE HILO CALIENTE (HWA)  
\# Laboratorio de Técnicas de Medida \- Ingeniería Aeronáutica  
\#  
\# Nombre: \[Nombre del Estudiante\]  
\# Fecha: \[Fecha de Realización\]  
\#  
\# Descripción:  
\# Este script procesa los datos de voltaje de un sensor de hilo caliente (HWA)  
\# para diferentes condiciones de flujo (identificadas por SD \- Speed Drive percentage),  
\# calcula las velocidades de referencia correspondientes, y ajusta los datos a  
\# una forma modificada de la Ley de King (U^(1/n) \= A\*E^2 \+ B) para obtener  
\# la curva de calibración del sensor para velocidad media.  
\# \-----------------------------------------------------------------------------

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from scipy.stats import linregress  
import glob \# Para buscar archivos por patrón  
import os   \# Para operaciones del sistema operativo como unir rutas

\# \--- Parámetros Configurables por el Estudiante \---  
\# \--------------------------------------------------

\# 1\. Directorio donde se encuentran los archivos de datos .txt  
\#    Asegúrese de que esta ruta sea correcta. Los archivos deben seguir la  
\#    nomenclatura SD\_Iteracion.txt (ej. 15\_1.txt, 20\_3.txt).  
directorio\_datos \= "./datos\_hilo\_caliente/" \# MODIFICAR SEGÚN SEA NECESARIO

\# 2\. Rango de valores de SD (Speed Drive percentage) a procesar.  
SD\_min \= 15  
SD\_max \= 35  
SD\_VALORES\_A\_PROCESAR \= list(range(SD\_min, SD\_max \+ 1))

\# 3\. Exponente 'n' para la Ley de King en la forma U^(1/n) \= A\*E^2 \+ B.  
\#    Para HWA, n=2 (correspondiente a U^0.5) es más común, pero se usa n=3  
\#    si es especificado por la práctica para mantener consistencia.  
N\_KING \= 3.0 

\# 4\. Coeficientes para la ecuación de calibración del sistema de flujo (túnel/canal),  
\#    Ecuación: U\_ref \[mm/s\] \= COEF\_M\_SISTEMA \* SD\[%\] \+ COEF\_B\_SISTEMA  
COEF\_M\_SISTEMA \= 7.6243  
COEF\_B\_SISTEMA \= \-1.8926

\# 5\. Parámetros para la selección de la "porción útil" de la señal de voltaje.  
DESCARTAR\_INICIO\_PORC \= 0.10  
DESCARTAR\_FIN\_PORC \= 0.10

\# \--- Funciones Auxiliares \---  
\# \----------------------------

def calcular\_velocidad\_referencia\_sistema(sd\_percent):  
    velocidad \= COEF\_M\_SISTEMA \* sd\_percent \+ COEF\_B\_SISTEMA  
    return max(0, velocidad)

def cargar\_y\_procesar\_archivo\_iteracion(filepath):  
    try:  
        datos\_voltaje \= np.loadtxt(filepath)  
        if datos\_voltaje.ndim \== 0:   
            datos\_voltaje \= np.array(\[datos\_voltaje\])   
        if len(datos\_voltaje) \== 0: return np.nan  
        n\_puntos\_total \= len(datos\_voltaje)  
        inicio\_idx \= int(DESCARTAR\_INICIO\_PORC \* n\_puntos\_total)  
        fin\_idx \= int(n\_puntos\_total \* (1.0 \- DESCARTAR\_FIN\_PORC))  
        if inicio\_idx \>= fin\_idx:  
            porcion\_util \= datos\_voltaje if n\_puntos\_total \> 0 else np.array(\[\])  
        else:  
            porcion\_util \= datos\_voltaje\[inicio\_idx:fin\_idx\]  
            if len(porcion\_util) \== 0 and n\_puntos\_total \> 0:  
                 porcion\_util \= datos\_voltaje  
        return np.mean(porcion\_util) if len(porcion\_util) \> 0 else np.nan  
    except Exception: return np.nan

\# \--- Procesamiento Principal \---  
print("Iniciando procesamiento de datos de anemometría de HILO CALIENTE (HWA)...\\n")  
if not os.path.isdir(directorio\_datos):  
    print(f"ERROR: Directorio de datos NO EXISTE: '{directorio\_datos}'")  
    exit()

resultados\_calibracion \= \[\]  
for sd\_valor in SD\_VALORES\_A\_PROCESAR:  
    print(f"Procesando SD \= {sd\_valor}%...")  
    patron\_archivos \= os.path.join(directorio\_datos, f"{sd\_valor}\_\*.txt")  
    archivos\_iteracion \= glob.glob(patron\_archivos)  
    if not archivos\_iteracion:  
        print(f"  No se encontraron archivos para SD \= {sd\_valor}%. Saltando.")  
        continue  
    voltajes\_iteraciones\_sd \= \[\]  
    num\_iteraciones\_validas \= 0  
    for filepath in archivos\_iteracion:  
        voltaje\_prom\_iter \= cargar\_y\_procesar\_archivo\_iteracion(filepath)  
        if not np.isnan(voltaje\_prom\_iter):  
            voltajes\_iteraciones\_sd.append(voltaje\_prom\_iter)  
            num\_iteraciones\_validas \+= 1  
    if num\_iteraciones\_validas \== 0:  
        print(f"  No se procesaron datos válidos para SD \= {sd\_valor}%.")  
        continue  
    E\_sd\_promedio \= np.mean(voltajes\_iteraciones\_sd)  
    E\_sd\_std\_dev \= np.std(voltajes\_iteraciones\_sd, ddof=1) if num\_iteraciones\_validas \> 1 else 0.0  
    U\_ref\_sd \= calcular\_velocidad\_referencia\_sistema(sd\_valor)  
    E\_sd\_promedio\_sq \= E\_sd\_promedio\*\*2  
    if U\_ref\_sd \<= 1e-6: \# Evitar log de cero o negativo, y problemas con potencias  
        U\_ref\_sd\_pow\_inv\_n \= 0.0  
        \# print(f"  Advertencia: U\_ref cercana a cero ({U\_ref\_sd:.3f} mm/s) para SD={sd\_valor}%. U\_ref^(1/n) se establece a 0.")  
    else:  
        U\_ref\_sd\_pow\_inv\_n \= U\_ref\_sd\*\*(1.0/N\_KING)  
          
    resultados\_calibracion.append({  
        "SD": sd\_valor, "Num\_Iteraciones": num\_iteraciones\_validas,  
        "E\_sd\_V": E\_sd\_promedio, "E\_sd\_std\_V": E\_sd\_std\_dev,  
        "U\_ref\_mm\_s": U\_ref\_sd, "E\_sd\_sq\_V2": E\_sd\_promedio\_sq,  
        f"U\_ref\_pow\_1\_div\_{int(N\_KING)}\_mms\_pow": U\_ref\_sd\_pow\_inv\_n  
    })  
    print(f"  Resultados SD \= {sd\_valor}%: E\_prom \= {E\_sd\_promedio:.4f} V, U\_ref \= {U\_ref\_sd:.2f} mm/s ({num\_iteraciones\_validas} iter.)")

df\_calibracion \= pd.DataFrame(resultados\_calibracion)  
df\_calibracion.dropna(subset=\['E\_sd\_sq\_V2', f"U\_ref\_pow\_1\_div\_{int(N\_KING)}\_mms\_pow"\], inplace=True)

\# Filtrar puntos donde U\_ref es muy bajo para la regresión, si es necesario  
df\_calibracion\_filtrada \= df\_calibracion\[df\_calibracion\['U\_ref\_mm\_s'\] \> 1e-6\] \# Umbral pequeño

if df\_calibracion\_filtrada.empty or len(df\_calibracion\_filtrada) \< 2:  
    print("\\nNo hay suficientes datos válidos (U\_ref \> 0\) para regresión lineal.")  
else:  
    print("\\n--- Tabla de Datos de Calibración (Filtrada para Regresión U\_ref \> 0\) \---")  
    print(df\_calibracion\_filtrada.to\_string())  
    X\_reg \= df\_calibracion\_filtrada\['E\_sd\_sq\_V2'\]  
    Y\_reg \= df\_calibracion\_filtrada\[f"U\_ref\_pow\_1\_div\_{int(N\_KING)}\_mms\_pow"\]  
    slope\_A, intercept\_B, r\_value, p\_value, std\_err \= linregress(X\_reg, Y\_reg)  
    R\_cuadrado \= r\_value\*\*2  
    print("\\n--- Resultados del Ajuste Lineal (Ley de King Modificada para HWA) \---")  
    print(f"Forma: U^(1/n) \= A \* E^2 \+ B  (n={N\_KING})")  
    print(f"A (pendiente): {slope\_A:.4f}")  
    print(f"B (ordenada): {intercept\_B:.4f}")  
    print(f"R^2: {R\_cuadrado:.6f}")  
    print("\\nEcuación de calibración del sensor HWA:")  
    print(f"U \[mm/s\] \= ({slope\_A:.4f} \* E^2 \[V^2\] \+ {intercept\_B:.4f})^{int(N\_KING)}")

    plt.style.use('seaborn-v0\_8-whitegrid')  
    plt.figure(figsize=(10, 6))  
    plt.scatter(X\_reg, Y\_reg, label='Datos experimentales transformados (HWA)', color='blue', marker='o')  
    Y\_ajustado \= slope\_A \* X\_reg \+ intercept\_B  
    plt.plot(X\_reg, Y\_ajustado, color='red', label=f'Ajuste lineal: Y \= {slope\_A:.3f}X \+ {intercept\_B:.3f}\\n$R^2 \= {R\_cuadrado:.4f}$')  
    plt.xlabel('$E\_{SD}^2 \\, \[V^2\]$', fontsize=12)  
    plt.ylabel(f'$U\_{{ref}}^{{1/{int(N\_KING)}}} \\, \[(mm/s)^{{1/{int(N\_KING)}}}\]$', fontsize=12)  
    plt.title('Ajuste Lineal de Datos Transformados (HWA \- Ley de King)', fontsize=14)  
    plt.legend(fontsize=10); plt.grid(True); plt.tight\_layout()  
    plt.savefig("grafica\_ajuste\_lineal\_transformados\_HWA.png")  
    print("\\nGráfica de ajuste lineal (HWA) guardada como 'grafica\_ajuste\_lineal\_transformados\_HWA.png'")

    E\_sd\_original\_filtrado \= df\_calibracion\_filtrada\['E\_sd\_V'\]  
    U\_ref\_original\_filtrado \= df\_calibracion\_filtrada\['U\_ref\_mm\_s'\]  
      
    \# Usar todos los puntos para la gráfica final, no solo los filtrados para regresión,  
    \# pero la curva se basa en A y B de la regresión con datos filtrados.  
    E\_plot\_puntos \= df\_calibracion\['E\_sd\_V'\]  
    U\_plot\_puntos \= df\_calibracion\['U\_ref\_mm\_s'\]

    if not E\_plot\_puntos.empty:  
        E\_teorico\_curva \= np.linspace(min(E\_plot\_puntos) \* 0.95, max(E\_plot\_puntos) \* 1.05, 200\)  
        U\_calculado\_curva \= (slope\_A \* E\_teorico\_curva\*\*2 \+ intercept\_B)\*\*N\_KING  
        U\_calculado\_curva \= np.maximum(U\_calculado\_curva, 0\) \# Evitar velocidades negativas  
        plt.figure(figsize=(10, 6))  
        plt.scatter(E\_plot\_puntos, U\_plot\_puntos, label='Puntos de calibración ($U\_{ref}$ vs $E\_{SD}$)', color='green', marker='x', s=100)  
        plt.plot(E\_teorico\_curva, U\_calculado\_curva, color='purple',   
                 label=f'Modelo HWA: $U \= ({slope\_A:.3f}E^2 \+ {intercept\_B:.3f})^{int(N\_KING)}$')  
        plt.xlabel('$E\_{SD}$ (Voltaje promedio del sensor HWA) \[V\]', fontsize=12)  
        plt.ylabel('$U$ (Velocidad del fluido) \[mm/s\]', fontsize=12)  
        plt.title('Curva de Calibración del Sensor de Hilo Caliente (HWA)', fontsize=14)  
        plt.legend(fontsize=10); plt.grid(True); plt.ylim(bottom=0)  
        plt.xlim(left=min(E\_plot\_puntos) \* 0.9 if not E\_plot\_puntos.empty else 0\)  
        plt.tight\_layout()  
        plt.savefig("grafica\_curva\_calibracion\_final\_HWA.png")  
        print("Gráfica de curva de calibración final (HWA) guardada como 'grafica\_curva\_calibracion\_final\_HWA.png'")  
    else:  
        print("No hay datos para graficar la curva de calibración final.")

print("\\nProcesamiento HWA completado.")  
