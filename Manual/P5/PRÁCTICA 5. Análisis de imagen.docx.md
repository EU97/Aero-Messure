**PRÁCTICA 5\. Análisis de imagen.** 

**I. Introducción**

El análisis de superficies de fractura, o fractografía, mediante microscopía electrónica de barrido (SEM) es una herramienta indispensable en la ingeniería aeronáutica para comprender los mecanismos de falla de los materiales, evaluar la integridad de componentes y guiar el diseño de materiales más resistentes. Las micrografías SEM proporcionan imágenes de alta resolución y gran profundidad de campo, revelando detalles topográficos cruciales de la superficie de fractura que permiten identificar el origen de la falla, el modo de propagación de la grieta y las características microestructurales involucradas.

En esta práctica, usted analizará imágenes SEM de superficies de fractura de especímenes utilizando técnicas de análisis de imagen para extraer información cualitativa (identificación de modos de fractura, características topográficas) y cuantitativa (medición de tamaños de características microestructurales utilizando las escalas provistas en las imágenes). Esta práctica simula el proceso que un ingeniero de materiales o de análisis de fallas realizaría para diagnosticar el comportamiento de un material bajo carga.

**II. Objetivos de Aprendizaje**

Al finalizar esta práctica, usted podrá:

* Comprender la importancia del análisis de micrografías SEM de superficies de fractura en la ingeniería aeronáutica.  
* Identificar las características principales de las micrografías SEM (escalas, información de adquisición).  
* Aplicar técnicas básicas de pre-procesamiento de imágenes para mejorar la visualización de detalles en las micrografías, utilizando el script de Python proporcionado.  
* Identificar y describir cualitativamente diferentes características y modos de fractura visibles en las imágenes SEM proporcionadas.  
* Realizar mediciones dimensionales de características microestructurales utilizando las barras de escala presentes en las imágenes, con la ayuda del script de Python.  
* Interpretar la información obtenida del análisis de imagen en el contexto de la caracterización del comportamiento a fractura del material.  
* Valorar la importancia de la calidad de la imagen y la correcta interpretación de las escalas para la fiabilidad del análisis fractográfico.

**III. Fundamentos Teóricos**

A. Fractografía y Microscopía Electrónica de Barrido (SEM) en Aeronáutica  
La fractografía es el estudio de las superficies de fractura de los materiales. Su objetivo es determinar la causa de la falla y relacionar las características de la fractura con la microestructura del material y las condiciones de servicio. El SEM es una herramienta poderosa para la fractografía debido a:

* **Alta Resolución:** Permite observar detalles muy finos.  
* **Gran Profundidad de Campo:** Produce imágenes con apariencia tridimensional de superficies rugosas.  
* **Información Adicional:** A menudo se combina con espectroscopía de rayos X de energía dispersiva (EDS/EDX) para análisis de composición química.

En aeronáutica, la fractografía es crucial para investigar fallas en componentes de motores, estructuras de aeronaves, trenes de aterrizaje, etc., ayudando a prevenir futuros incidentes.

**B. Características Comunes en Superficies de Fractura (visibles en SEM)**

* **Fractura Dúctil:** Caracterizada por una deformación plástica significativa. A nivel microscópico, a menudo muestra **hoyuelos (dimples)**, que son cavidades formadas por la nucleación, crecimiento y coalescencia de microvacíos.  
* **Fractura Frágil:** Ocurre con poca o ninguna deformación plástica. Las superficies suelen ser relativamente planas y pueden mostrar:  
  * **Facetas de Clivaje:** Superficies planas y cristalinas. A menudo muestran "patrones de río".  
  * **Fractura Intergranular:** La grieta se propaga a lo largo de los límites de grano.  
* **Fatiga:** Falla progresiva bajo cargas cíclicas. Las superficies de fractura por fatiga a menudo muestran **estrías de fatiga**, marcas finas y paralelas.

**C. Conceptos Básicos de Imagen Digital y Metrología en SEM**

* **Píxeles, Resolución, Contraste:** Conceptos estándar de imagen digital.  
* **Barra de Escala (Scale Bar):** Presente en las micrografías SEM (ej. "10 µm"), indica la relación entre una longitud en la imagen y la longitud real. Es fundamental para mediciones cuantitativas.  
* **Información de Adquisición:** Datos como voltaje (EHT), distancia de trabajo (WD), magnificación (Mag), detector (Signal A), y fecha/hora.

D. Pre-procesamiento Básico de Imágenes  
Incluye ajuste de brillo/contraste y aplicación de filtros (suavizado o realce). El script de Python proporcionado incluye una función básica para ajuste de brillo/contraste.  
**IV. Metodología de Análisis de Datos**

A. Descripción de los Conjuntos de Datos (Datasets) a Analizar  
Se utilizarán dos archivos de imagen principales, que usted deberá tener disponibles en el mismo directorio donde ejecute el script de Python:

* fatigue.jpg: Contiene 4 micrografías SEM individuales, organizadas en una cuadrícula de 2x2.  
* fatrigue2.jpg: Contiene 4 micrografías SEM individuales, organizadas en una cuadrícula de 2x2.

Cada una de estas 8 micrografías (especímenes) representa una vista de una superficie de fractura. Preste atención a la barra de escala y a la información de adquisición en cada micrografía.

**B. Herramientas de Software**

* **Script de Python (analisis\_sem.py):** Se proporciona un script de Python para facilitar la carga de imágenes, selección de especímenes, calibración de escala, medición de características y ajuste básico de brillo/contraste. Siga las instrucciones que el script le proporcionará en la consola.  
* **ImageJ (o Fiji) (Opcional):** Si desea explorar herramientas adicionales, ImageJ es un software de dominio público potente para el análisis de imágenes.

**C. Procedimiento de Tratamiento y Análisis de Datos con el Script de Python**

Para cada una de las 8 micrografías individuales, siga los pasos guiados por el script analisis\_sem.py:

**Paso 1: Carga, Selección de Espécimen y Calibración de Escala.**

1. **Ejecutar el Script:** Inicie el script de Python.  
2. **Seleccionar Imagen Principal:** El script le pedirá que elija entre fatigue.jpg o fatrigue2.jpg.  
3. **Seleccionar Espécimen:** Indique cuál de los 4 especímenes de la imagen principal desea analizar (0: superior izquierda, 1: superior derecha, 2: inferior izquierda, 3: inferior derecha). El script mostrará el espécimen seleccionado.  
4. **Calibrar la Escala:**  
   * El script le pedirá que haga clic en los dos extremos de la barra de escala visible en la imagen del espécimen.  
   * Luego, deberá ingresar la longitud real de esa barra de escala en micrómetros (µm) (ej. 10, según lo indique la imagen).  
   * El script calculará y almacenará la relación micrómetros/píxel para ese espécimen.

**Paso 2: Pre-procesamiento de la Imagen (Opcional).**

1. **Ajuste de Brillo/Contraste:** El script ofrecerá la opción de ajustar el brillo y el contraste del espécimen actual. Siga las instrucciones para ingresar los factores de ajuste. La imagen se actualizará. Este paso es opcional.

Paso 3: Análisis Cualitativo – Identificación de Características Fractográficas.  
Este paso lo realizará usted visualmente sobre la imagen del espécimen mostrada por el script, y lo documentará en su reporte.

1. **Observación Detallada:** Examine la morfología de la superficie.  
2. **Identificar Modos de Fractura:**  
   * ¿Predominan hoyuelos? Describa su forma.  
   * ¿Se observan facetas planas, patrones de río?  
   * ¿Hay evidencia de fractura intergranular?  
   * Busque estrías de fatiga (pueden ser sutiles).  
3. **Describir otras Características:** Inclusiones, microfisuras secundarias, etc.

Paso 4: Análisis Cuantitativo – Mediciones Dimensionales.  
Utilice la funcionalidad de medición del script de Python:

1. **Seleccionar Características a Medir:** Basado en su análisis cualitativo (ej. diámetro de hoyuelos, tamaño de facetas).  
2. **Realizar Mediciones:**  
   * El script le pedirá que haga clic en dos puntos sobre la imagen para definir la característica a medir.  
   * El script calculará y mostrará la longitud de esa característica en micrómetros (µm).  
   * Puede realizar múltiples mediciones. Anote estos valores en su reporte.  
3. **Registrar Mediciones:** Anote sistemáticamente las mediciones para cada espécimen. Para características como hoyuelos, mida varios (10-20) para obtener un promedio representativo.

**V. Actividades a Realizar y Entregables del Reporte**

Para cada uno de los 8 especímenes analizados con el script:

1. **Documentación Inicial:**  
   * Indique el nombre del archivo original y la etiqueta del espécimen (ej. fatigue.jpg \- Espécimen 0).  
   * Registre la información de adquisición visible en la imagen (Mag, EHT, WD, valor de la barra de escala).  
   * Incluya en su reporte una captura de pantalla del espécimen analizado (puede ser la imagen original o la pre-procesada si aplicó ajustes), con la barra de escala claramente visible.  
2. **Pre-procesamiento (si se realizó):**  
   * Describa brevemente cualquier ajuste de brillo/contraste realizado y justifique su uso.  
3. **Análisis Cualitativo:**  
   * Describa detalladamente la morfología de la superficie de fractura.  
   * Identifique y justifique los posibles modos de fractura predominantes.  
   * En su captura de pantalla del espécimen, señale (mediante flechas o recuadros, puede hacerlo editando la captura) ejemplos claros de las características identificadas.  
4. **Análisis Cuantitativo:**  
   * Presente una tabla con las mediciones dimensionales realizadas (ej. diámetro de hoyuelos, tamaño de facetas) obtenidas con el script. Indique el número de mediciones realizadas por característica y el promedio/rango si aplica.  
5. **Comparación (Opcional y Exploratoria):**  
   * Comente brevemente las similitudes o diferencias observadas en la morfología y mediciones entre los diferentes especímenes, si considera que pueden representar diferentes condiciones o magnificaciones.

**Entregables Globales del Reporte:**

* Un informe estructurado que presente el análisis individual de los 8 especímenes.  
* Demostración de la correcta calibración de escalas y la realización de mediciones utilizando el script.  
* Interpretación coherente de las características fractográficas observadas.

**VI. Caso de Estudio / Aplicación Práctica**

Suponga que los 8 especímenes analizados provienen de la superficie de fractura de un componente crítico de una aeronave (ej. un perno del tren de aterrizaje, un álabe de turbina) que falló en servicio.

1. **Diagnóstico del Modo de Falla:** Basándose en su análisis cualitativo y cuantitativo:  
   * ¿Cuál sería su diagnóstico preliminar sobre el modo o modos de fractura predominantes? Justifique.  
   * ¿Hay evidencia que sugiera fatiga como mecanismo principal o contribuyente?  
   * ¿La fractura parece ser predominantemente dúctil o frágil? ¿Qué características lo indican?  
2. **Implicaciones para la Investigación de la Falla:**  
   * ¿Qué información adicional sería útil para confirmar su diagnóstico?  
   * Si tuviera que seleccionar una o dos micrografías como las más representativas, ¿cuáles elegiría y por qué?  
3. **Recomendaciones (Conceptuales):**  
   * Si la falla fuera por fatiga, ¿qué acciones se podrían considerar para prevenir fallas similares?  
   * Si la fractura muestra fragilidad inesperada, ¿qué podría implicar sobre el material?

**VII. Puntos para la Discusión en su Reporte**

* Discuta la importancia crítica del análisis fractográfico SEM para determinar la causa raíz de fallas en componentes aeronáuticos.  
* ¿Cómo se relacionan las características observadas con las propiedades mecánicas del material?  
* Explique cómo la magnificación afecta la observación de características fractográficas.  
* Considere las limitaciones del análisis basado en imágenes 2D de una superficie compleja.

**VIII. Cuestionario / Preguntas de Reflexión (A incluir en su reporte)**

1. Explique por qué la calibración precisa de la escala es fundamental. ¿Qué errores surgirían sin una calibración correcta?  
2. Si observa hoyuelos muy pequeños y equiaxiales, ¿qué infiere sobre la ductilidad del material comparado con una región con hoyuelos grandes y alargados?  
3. ¿Por qué es importante observar la fractura a diferentes magnificaciones?  
4. Si sospecha fatiga pero no identifica estrías claras, ¿qué otras características buscaría o qué imágenes adicionales solicitaría?  
5. Además del tamaño, ¿qué otra información cuantitativa podría extraerse con herramientas de análisis más avanzadas?

**IX. Requisitos del Reporte (Formato según indique su instructor)**

Elabore un reporte técnico que incluya:

* **Título, Autores, Afiliación, Fecha.**  
* **Abstract (Resumen):** Objetivo, metodología (uso del script, análisis de 8 especímenes), principales resultados (modos de fractura, mediciones) y conclusión.  
* **1\. Introducción:** Importancia de la fractografía SEM, objetivos de la práctica.  
* **2\. Marco Teórico (breve):** Principios SEM, tipos de fractura, importancia de la escala.  
* **3\. Metodología de Análisis de Datos:**  
  * Descripción de las imágenes y los 8 especímenes.  
  * Mención del uso del script analisis\_sem.py y descripción general de su aplicación para calibración y medición.  
* **4\. Resultados:**  
  * Análisis detallado de cada uno de los 8 especímenes (documentación inicial, pre-procesamiento si aplica, análisis cualitativo con imágenes señaladas, tabla de análisis cuantitativo).  
  * Respuestas detalladas al Caso de Estudio.  
* **5\. Discusión:**  
  * Interpretación global de resultados, análisis comparativo (si es relevante), limitaciones, relevancia aeronáutica.  
* **6\. Conclusiones:** Principales hallazgos y aprendizajes.  
* **7\. Referencias:** (Si consultó fuentes adicionales).  
* **8\. Respuestas al Cuestionario / Preguntas de Reflexión.**  
* **XI. Apéndice: Código Python para Análisis de Imágenes** (Ver sección XI).

**X. Referencias Sugeridas**

1. ASM Handbook, Volume 12: *Fractography*. ASM International.  
2. Hull, D. *Fractography: Observing, Measuring and Interpreting Fracture Surface Topography*. Cambridge University Press.  
3. Mills, K., & Gagliano, J. (Eds.). *Failure Analysis of Engineering Materials*. McGraw-Hill.  
4. Artículos de revistas especializadas como *Engineering Failure Analysis*, *Materials Science and Engineering: A*, *Acta Materialia*.

**XI. Apéndice: Código Python para Análisis de Imágenes**

import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance

import numpy as np

import os

\# \--- Configuración de Matplotlib para mejor interacción \---

\# Permite cerrar ventanas de Matplotlib con 'q' o 'ctrl+w'

plt.rcParams\['keymap.quit'\] \= \['ctrl+w', 'cmd+w', 'q'\] 

\# \--- Funciones Auxiliares \---

def cargar\_imagen\_principal(nombre\_archivo):

    """

    Carga la imagen principal (fatigue.jpg o fatrigue2.jpg).

    Retorna el objeto Imagen de Pillow o None si hay error.

    """

    try:

        imagen \= Image.open(nombre\_archivo)

        print(f"Imagen '{nombre\_archivo}' cargada exitosamente.")

        return imagen

    except FileNotFoundError:

        print(f"Error: El archivo '{nombre\_archivo}' no fue encontrado. Asegúrese de que esté en el mismo directorio que el script.")

        return None

    except Exception as e:

        print(f"Error al cargar la imagen '{nombre\_archivo}': {e}")

        return None

def seleccionar\_especimen(imagen\_principal, indice\_especimen):

    """

    Recorta y retorna uno de los 4 especímenes de la imagen principal.

    Indice\_especimen: 0 (sup-izq), 1 (sup-der), 2 (inf-izq), 3 (inf-der).

    """

    ancho\_total, alto\_total \= imagen\_principal.size

    ancho\_especimen \= ancho\_total // 2

    alto\_especimen \= alto\_total // 2

    coordenadas \= \[

        (0, 0, ancho\_especimen, alto\_especimen),                           \# 0: Superior Izquierda

        (ancho\_especimen, 0, ancho\_total, alto\_especimen),                 \# 1: Superior Derecha

        (0, alto\_especimen, ancho\_especimen, alto\_total),                  \# 2: Inferior Izquierda

        (ancho\_especimen, alto\_especimen, ancho\_total, alto\_total)         \# 3: Inferior Derecha

    \]

    if 0 \<= indice\_especimen \< 4:

        caja \= coordenadas\[indice\_especimen\]

        especimen \= imagen\_principal.crop(caja)

        print(f"Espécimen {indice\_especimen} seleccionado.")

        return especimen

    else:

        print("Error: Índice de espécimen no válido. Debe ser 0, 1, 2 o 3.")

        return None

def mostrar\_imagen\_con\_instrucciones(imagen\_especimen, titulo\_ventana, instruccion\_extra=""):

    """Muestra la imagen y espera a que el usuario la cierre para continuar con el script, o interactúa si es para ginput."""

    fig, ax \= plt.subplots()

    ax.imshow(np.array(imagen\_especimen)) \# Convertir a numpy array para imshow

    ax.set\_title(titulo\_ventana)

    

    instrucciones\_base \= "Cierre esta ventana para continuar con el script."

    if "ginput" in instruccion\_extra.lower() or "haga clic" in instruccion\_extra.lower() : \# Adaptar si es para ginput

        instrucciones\_base \= "Realice los clics solicitados en la imagen. La ventana se cerrará automáticamente."

    if instruccion\_extra:

        plt.text(0.01, 0.01, instruccion\_extra, transform=fig.transFigure, fontsize=9, color='red',

                 bbox=dict(facecolor='white', alpha=0.7))

    

    plt.text(0.5, \-0.05, instrucciones\_base, transform=ax.transAxes, ha='center', fontsize=9)

    plt.figtext(0.01, 0.95, "Zoom: Rueda del mouse o herramientas. Pan: Botón central o herramientas.", fontsize=8, color='blue')

    

    print(f"\\n--- {titulo\_ventana} \---")

    if instruccion\_extra:

        print(instruccion\_extra)

    \# No imprimir "Cierre esta ventana..." si es para ginput, ya que se cierra sola.

    if not ("ginput" in instruccion\_extra.lower() or "haga clic" in instruccion\_extra.lower()):

        print(instrucciones\_base)

    

    if "ginput" in instruccion\_extra.lower() or "haga clic" in instruccion\_extra.lower():

        return fig, ax \# Retornar fig y ax para que ginput funcione en esta figura

    else:

        plt.show() \# Bloquea hasta que se cierre la ventana

        return None, None

def calibrar\_escala(imagen\_especimen):

    """

    Permite al usuario calibrar la escala haciendo clic en la barra de escala.

    Retorna la relación micrometros\_por\_pixel o None si la calibración falla.

    """

    print("\\n--- Calibración de Escala \---")

    instruccion\_calibracion \= "Identifique la barra de escala. Haga clic en un extremo y luego en el otro extremo de la barra."

    

    fig, ax \= mostrar\_imagen\_con\_instrucciones(imagen\_especimen, 

                                               "Calibración: Clic en los 2 extremos de la barra de escala", 

                                               instruccion\_calibracion)

    if fig is None: \# Si mostrar\_imagen\_con\_instrucciones no retornó fig (error o no interactivo)

        print("No se pudo mostrar la imagen para calibración.")

        return None

    print("Esperando 2 clics en la imagen para la barra de escala...")

    puntos \= fig.ginput(2, timeout=-1) \# Espera 2 clics, sin timeout

    if len(puntos) \< 2:

        print("Calibración cancelada o no se seleccionaron suficientes puntos.")

        plt.close(fig)

        return None

    (x1, y1), (x2, y2) \= puntos

    distancia\_pixeles \= np.sqrt((x2 \- x1)\*\*2 \+ (y2 \- y1)\*\*2)

    print(f"Distancia en píxeles de la barra de escala: {distancia\_pixeles:.2f} píxeles.")

    plt.close(fig) 

    while True:

        try:

            longitud\_real\_um\_str \= input("Ingrese la longitud real de la barra de escala en micrómetros (µm) (ej. 10): ")

            longitud\_real\_um \= float(longitud\_real\_um\_str)

            if longitud\_real\_um \<= 0:

                print("La longitud debe ser un número positivo.")

                continue

            break

        except ValueError:

            print("Entrada no válida. Por favor, ingrese un número.")

        except Exception as e:

            print(f"Un error inesperado ocurrió: {e}")

            return None

    micrometros\_por\_pixel \= longitud\_real\_um / distancia\_pixeles

    print(f"Calibración completada: {micrometros\_por\_pixel:.4f} µm/píxel.")

    return micrometros\_por\_pixel

def medir\_caracteristica(imagen\_especimen, micrometros\_por\_pixel):

    """

    Permite al usuario medir una característica en la imagen.

    Imprime la medición en micrómetros.

    """

    if micrometros\_por\_pixel is None or micrometros\_por\_pixel \== 0:

        print("Error: La escala no ha sido calibrada o es inválida. No se pueden realizar mediciones.")

        return

    print("\\n--- Medición de Característica \---")

    instruccion\_medicion \= "Haga clic en dos puntos para definir la longitud de la característica a medir."

    fig, ax \= mostrar\_imagen\_con\_instrucciones(imagen\_especimen, 

                                               "Medición: Clic en 2 puntos para medir", 

                                               instruccion\_medicion)

    if fig is None:

        print("No se pudo mostrar la imagen para medición.")

        return

    print("Esperando 2 clics en la imagen para medir la característica...")

    puntos \= fig.ginput(2, timeout=-1) 

    if len(puntos) \< 2:

        print("Medición cancelada o no se seleccionaron suficientes puntos.")

        plt.close(fig)

        return

    (x1, y1), (x2, y2) \= puntos

    distancia\_pixeles \= np.sqrt((x2 \- x1)\*\*2 \+ (y2 \- y1)\*\*2)

    distancia\_um \= distancia\_pixeles \* micrometros\_por\_pixel

    print(f"Distancia en píxeles de la característica: {distancia\_pixeles:.2f} píxeles.")

    print(f"Longitud de la característica medida: {distancia\_um:.2f} µm.")

    plt.close(fig)

def ajustar\_brillo\_contraste(imagen\_pil, factor\_brillo=1.0, factor\_contraste=1.0):

    """

    Ajusta el brillo y contraste de una imagen PIL.

    factor\_brillo: 1.0 \= original, \>1.0 más brillante, \<1.0 más oscuro.

    factor\_contraste: 1.0 \= original, \>1.0 más contraste, \<1.0 menos contraste.

    """

    try:

        enhancer\_brillo \= ImageEnhance.Brightness(imagen\_pil)

        imagen\_ajustada \= enhancer\_brillo.enhance(factor\_brillo)

        

        enhancer\_contraste \= ImageEnhance.Contrast(imagen\_ajustada)

        imagen\_ajustada \= enhancer\_contraste.enhance(factor\_contraste)

        

        print(f"Brillo ajustado por factor {factor\_brillo}, contraste por factor {factor\_contraste}.")

        return imagen\_ajustada

    except Exception as e:

        print(f"Error al ajustar brillo/contraste: {e}")

        return imagen\_pil \# Retornar original en caso de error

\# \--- Flujo Principal del Script \---

def main():

    print("Bienvenido al Asistente de Análisis de Imágenes SEM para la Práctica 5.")

    

    imagen\_principal\_actual \= None

    micrometros\_por\_pixel\_actual \= None

    especimen\_actual\_para\_analisis \= None \# Almacena la imagen PIL del espécimen

    nombre\_archivo\_principal\_seleccionado \= ""

    while True:

        print("\\n--- Menú Principal \---")

        if nombre\_archivo\_principal\_seleccionado:

            print(f"Archivo principal actual: {nombre\_archivo\_principal\_seleccionado}")

        if especimen\_actual\_para\_analisis:

            print(f"Espécimen actual: Seleccionado (Índice recordado internamente)")

        if micrometros\_por\_pixel\_actual:

            print(f"Calibración actual: {micrometros\_por\_pixel\_actual:.4f} µm/píxel")

        else:

            print("Calibración actual: No calibrado")

            

        print("\\n1. Seleccionar imagen principal (fatigue.jpg o fatrigue2.jpg)")

        print("2. Seleccionar espécimen de la imagen actual")

        print("3. Ajustar brillo/contraste del espécimen actual (Opcional)")

        print("4. Calibrar escala del espécimen actual")

        print("5. Medir característica(s) en el espécimen actual")

        print("6. Mostrar espécimen actual (para inspección visual)")

        print("0. Salir")

        opcion\_menu \= input("Seleccione una opción: ")

        if opcion\_menu \== '1':

            while True:

                nombre\_archivo \= input("Ingrese el nombre del archivo principal (fatigue.jpg o fatrigue2.jpg): ").strip()

                if nombre\_archivo.lower() in \["fatigue.jpg", "fatrigue2.jpg"\]:

                    img\_cargada \= cargar\_imagen\_principal(nombre\_archivo)

                    if img\_cargada:

                        imagen\_principal\_actual \= img\_cargada

                        nombre\_archivo\_principal\_seleccionado \= nombre\_archivo

                        especimen\_actual\_para\_analisis \= None 

                        micrometros\_por\_pixel\_actual \= None 

                        print(f"Imagen principal '{nombre\_archivo\_principal\_seleccionado}' cargada.")

                    break

                else:

                    print("Nombre de archivo no válido. Use 'fatigue.jpg' o 'fatrigue2.jpg'.")

        

        elif opcion\_menu \== '2':

            if imagen\_principal\_actual is None:

                print("Primero debe seleccionar una imagen principal (Opción 1).")

                continue

            

            while True:

                try:

                    idx\_str \= input("Ingrese el índice del espécimen (0: Sup-Izq, 1: Sup-Der, 2: Inf-Izq, 3: Inf-Der): ")

                    idx \= int(idx\_str)

                    if 0 \<= idx \<= 3:

                        esp\_seleccionado \= seleccionar\_especimen(imagen\_principal\_actual, idx)

                        if esp\_seleccionado:

                             especimen\_actual\_para\_analisis \= esp\_seleccionado

                             micrometros\_por\_pixel\_actual \= None 

                             mostrar\_imagen\_con\_instrucciones(especimen\_actual\_para\_analisis, 

                                                             f"Espécimen {idx} de '{nombre\_archivo\_principal\_seleccionado}'")

                        break

                    else:

                        print("Índice fuera de rango. Debe ser 0, 1, 2 o 3.")

                except ValueError:

                    print("Entrada no válida. Ingrese un número entero.")

        

        elif opcion\_menu \== '3':

            if especimen\_actual\_para\_analisis is None:

                print("Primero debe seleccionar un espécimen (Opción 2).")

                continue

            try:

                brillo\_str \= input("Ingrese factor de brillo (ej. 1.0 para original, 1.2 para más brillante): ")

                brillo \= float(brillo\_str)

                contraste\_str \= input("Ingrese factor de contraste (ej. 1.0 para original, 1.2 para más contraste): ")

                contraste \= float(contraste\_str)

                

                especimen\_actual\_para\_analisis \= ajustar\_brillo\_contraste(especimen\_actual\_para\_analisis, brillo, contraste)

                mostrar\_imagen\_con\_instrucciones(especimen\_actual\_para\_analisis, "Espécimen con Brillo/Contraste Ajustado")

            except ValueError:

                print("Factores no válidos. Deben ser números.")

            except Exception as e:

                print(f"Error durante el ajuste: {e}")

        elif opcion\_menu \== '4':

            if especimen\_actual\_para\_analisis is None:

                print("Primero debe seleccionar un espécimen (Opción 2).")

                continue

            micrometros\_por\_pixel\_actual \= calibrar\_escala(especimen\_actual\_para\_analisis)

        elif opcion\_menu \== '5':

            if especimen\_actual\_para\_analisis is None:

                print("Primero debe seleccionar un espécimen (Opción 2).")

                continue

            if micrometros\_por\_pixel\_actual is None:

                print("Primero debe calibrar la escala para el espécimen actual (Opción 4).")

                continue

            

            while True:

                medir\_caracteristica(especimen\_actual\_para\_analisis, micrometros\_por\_pixel\_actual)

                continuar\_midiendo \= input("¿Desea medir otra característica en este espécimen? (s/n): ").lower()

                if continuar\_midiendo \!= 's':

                    break

        

        elif opcion\_menu \== '6':

            if especimen\_actual\_para\_analisis is None:

                print("No hay un espécimen seleccionado actualmente. Use la Opción 2.")

                continue

            mostrar\_imagen\_con\_instrucciones(especimen\_actual\_para\_analisis, "Visualización del Espécimen Actual")

        elif opcion\_menu \== '0':

            print("Saliendo del programa de análisis de imágenes SEM. ¡Hasta luego\!")

            break

        else:

            print("Opción no válida. Por favor, intente de nuevo.")

if \_\_name\_\_ \== "\_\_main\_\_":

    main()

