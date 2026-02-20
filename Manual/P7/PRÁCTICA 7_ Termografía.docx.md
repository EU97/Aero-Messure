## **PRÁCTICA 7: Termografía**

Departamento de Ingeniería Aeroespacial  
Laboratorio de Técnicas de Medida  
Título de la Práctica: Termografía  
Autores: (Espacio para el nombre del estudiante)  
Afiliación: (Espacio para la afiliación del estudiante)  
Fecha: (Espacio para la fecha)

### **Resumen**

Esta práctica introduce el uso de la termografía infrarroja como herramienta de diagnóstico y medición de temperatura sin contacto en aplicaciones aeronáuticas. Se analizarán imágenes térmicas (termogramas) de componentes aeronáuticos y otros elementos bajo diversas condiciones. Los estudiantes interpretarán patrones de temperatura, identificarán anomalías térmicas (puntos calientes o fríos) y relacionarán estas observaciones con posibles fallos, eficiencias o fenómenos térmicos relevantes, utilizando datos de temperatura inferidos de las imágenes y sus escalas.

### **I. Introducción**

La termografía infrarroja es una técnica esencial en la ingeniería aeronáutica que permite visualizar y cuantificar patrones de temperatura superficial sin contacto. Esta capacidad es crucial para la inspección de motores, la evaluación de la integridad de estructuras de materiales compuestos, la supervisión de sistemas electrónicos y el estudio de fenómenos de transferencia de calor. La correcta interpretación de los termogramas facilita la detección temprana de fallos, la optimización del rendimiento y la validación de diseños. Esta práctica se enfoca en el análisis de termogramas para desarrollar habilidades en la interpretación de datos térmicos y su aplicación en la resolución de problemas aeronáuticos.

### **II. Objetivos de Aprendizaje Específicos**

Al finalizar esta práctica, el estudiante será capaz de:

* Comprender los principios físicos de la termografía infrarroja y su aplicación en aeronáutica.  
* Identificar los componentes clave de un sistema termográfico.  
* Interpretar termogramas, incluyendo el uso de paletas de colores y escalas de temperatura.  
* Identificar anomalías térmicas y relacionarlas con posibles causas o fenómenos.  
* Extraer datos de temperatura aparentes a partir de las escalas en los termogramas.  
* Comprender la importancia de la emisividad y la temperatura reflejada.  
* Analizar la distribución térmica mediante histogramas de intensidad.  
* Discutir las ventajas y limitaciones de la termografía.

### **III. Fundamentos Teóricos**

#### **A. Termografía Infrarroja en Aeronáutica**

La termografía es una técnica de Ensayo No Destructivo (END) que detecta la radiación infrarroja emitida por los objetos para crear termogramas que representan su distribución de temperatura. Sus aplicaciones en aeronáutica incluyen:

* **Motores**: Detección de sobrecalentamientos, patrones de escape.  
* **Estructuras de Composites**: Identificación de delaminaciones, humedad.  
* **Sistemas Electrónicos (Aviónica)**: Localización de componentes sobrecalentados.  
* **Estudios de Transferencia de Calor**: Visualización de capa límite, eficiencia de sistemas de deshielo.

#### **B. Principios de la Termografía**

Todo objeto por encima del cero absoluto (0 K) emite radiación infrarroja, dependiente de su temperatura y emisividad (ϵ).

* **Radiación Infrarroja**: Parte del espectro electromagnético (MWIR: 3-5 µm; LWIR: 8-14 µm).  
* **Emisividad (**ϵ**)**: Eficiencia de un objeto para emitir radiación infrarroja (0\<ϵ\<1). Los metales pulidos suelen tener baja emisividad.  
* **Radiación Detectada**: La cámara capta la radiación emitida por el objeto, la radiación reflejada del entorno y, en menor medida, la transmitida. Para objetos opacos: Wtotal​=(ϵ⋅Wobj​)+((1−ϵ)⋅Wref​).  
* **Cámara Termográfica**: Consta de óptica (lentes de germanio), detector (convierte IR en señal eléctrica) y sistema de procesamiento y visualización.

#### **C. Interpretación de Termogramas**

* **Paleta de Colores**: Asocia colores a temperaturas.  
* **Escala de Temperatura**: Rango de temperaturas visualizado.  
* **Anomalías Térmicas**: Puntos calientes (fricción, fallos eléctricos) o fríos (pérdidas, humedad).  
* **Factores de Influencia**: Emisividad incorrecta, reflejos, condiciones atmosféricas, ángulo de visión, convección.

#### **D. Análisis Cualitativo y Cuantitativo**

* **Cualitativo**: Identificación de patrones, comparación relativa de temperaturas.  
* **Cuantitativo**: Medición de temperaturas específicas. Requiere ajustar emisividad, temperatura reflejada, etc. Herramientas como puntos, líneas o áreas (ROIs) permiten extraer estadísticas. Los histogramas muestran la distribución de temperaturas en un ROI.

### **IV. Metodología Experimental y de Análisis de Datos**

#### **A. Descripción del Montaje Experimental Típico (Conceptual)**

1. **Configuración de la Cámara**: Encendido, estabilización, ajuste de parámetros (emisividad, temperatura ambiente, distancia), selección de paleta y rango, enfoque.  
2. **Preparación del Objeto**: Superficie limpia, considerar estado operativo, minimizar reflejos.  
3. **Adquisición de Termogramas**: Captura de imágenes, idealmente en formato radiométrico. Para esta práctica, se usarán imágenes JPG.  
4. **Análisis de Datos**: Transferencia a software, ajuste de visualización, identificación de ROIs, extracción de datos (aparentes de la escala), interpretación.

#### **B. Descripción de los Conjuntos de Datos (Termogramas)**

Se proporcionarán seis archivos de imagen (.jpg). El análisis cuantitativo directo es limitado con JPGs, pero se podrán interpretar patrones, leer temperaturas aparentes de escalas visibles y analizar distribuciones de intensidad.

* **Imagen1.jpg**: Aeronave con hélice.  
* **Imagen2.jpg**: Mano y araña (Escala: 21.9 °C a 33.8 °C).  
* **Imagen3.jpg**: Edificio (dos vistas), posible pérdida de calor.  
* **Imagen4.jpg**: Edificio (visual y térmica), posible aislamiento.  
* **Imagen5.jpg**: Tazas con líquido caliente (escalas variables).  
* **Imagen6.jpg**: Imagen con cruceta de medición (Escala: 25.2 °C a 40.7 °C).

#### **C. Procedimiento de Tratamiento y Análisis de Datos**

1. **Carga e Inspección Inicial**: Cargar cada imagen. Observar paleta, escala (si visible), características térmicas principales. (Ver Apéndice para código de ejemplo).  
2. **Análisis Cualitativo y ROIs**: Identificar anomalías. Definir ROIs para análisis.  
3. **Extracción de Datos de Temperatura (de escalas visibles)**: Anotar rangos de escala. Estimar temperaturas de ROIs.  
4. **Generación de Histogramas (Distribución de Intensidad)**: Para ROIs, generar histogramas de valores de píxeles (convertidos a escala de grises o por canal). (Ver Apéndice para código de ejemplo).  
5. **Interpretación Detallada por Imagen**:  
   * **Imagen1.jpg (Aeronave FLIR)**: Motor y hélice calientes. Sin escala explícita. Patrón esperado para motor en funcionamiento/recientemente apagado.  
   * **Imagen2.jpg (Mano y Araña)**: Mano (\>30 °C) más cálida que araña (\~23.4 °C). Demuestra distinción de temperaturas en seres vivos.  
   * **Imagen3.jpg (Casa, dos vistas)**: Zonas cálidas en ventanas/paredes/techo indican pérdida de calor.  
   * **Imagen4.jpg (Casa, visual y térmica)**: Comparación visual/térmica. Zonas claras en termograma (escala de grises) indican mayor temperatura/pérdida de calor.  
   * **Imagen5.jpg (Tazas)**: Líquido caliente, gradientes de temperatura visibles. Diferentes escalas permiten distintos niveles de detalle.  
   * **Imagen6.jpg (FLIR, cruceta)**: Punto medido 33.6 °C. Banda horizontal más cálida. Aplicable a detección de sobrecalentamiento.

### **V. Actividades a Realizar y Resultados Esperados**

1. **Procesamiento de Datos**: Para cada imagen JPG:  
   * Cargar y visualizar. Describir observaciones.  
   * Si hay escala: anotar rango, estimar temperaturas de dos ROIs (caliente/frío), anotar lecturas puntuales.  
   * Identificar anomalías o patrones destacables.  
   * (Opcional avanzado) Para un ROI, generar histograma de intensidad. Interpretarlo.  
2. Tabla de Análisis de Termogramas:  
   | Imagen No. | Descripción Breve | Escala Temp. (Sí/No, Rango) | ROI 1 Desc. y Temp. Est. | ROI 2 Desc. y Temp. Est. | Observaciones/Anomalías | Aplicación Aeronáutica Potencial |  
   | :--------: | :---------------- | :------------------------: | :-----------------------: | :-----------------------: | :-----------------------: | :-------------------------------: |  
   | Imagen1.jpg| Aeronave/motor | No | Motor (Caliente) | Fuselaje (Frío) | Motor y hélice calientes | Inspección motor, huella térmica |  
   | Imagen2.jpg| Mano y araña | Sí, 21.9−33.8 °C | Mano (\~30−33 °C) | Araña (\~23.4 °C) | Diferencia temp. seres vivos | Factor humano, entorno |  
   | Imagen3.jpg| Edificio | No (inferible por paleta) | Ventanas (Cálido) | Pared ext. (Frío) | Pérdidas de calor visibles | Aislamiento cabina/bodega |  
   | Imagen4.jpg| Edificio (vis/term)| No (escala de grises) | Ventanas (Claro/Cálido) | Tejado nieve (Oscuro/Frío)| Contraste térmico | Detección defectos ocultos |  
   | Imagen5.jpg| Tazas calientes | Sí (variables) | Líquido (Muy caliente) | Aire amb. (Frío) | Gradientes, disipación calor| Sistemas fluidos calientes |  
   | Imagen6.jpg| Componente | Sí, 25.2−40.7 °C | Punto medido (33.6 °C)| Fondo (Más frío) | Punto caliente específico | Inspección cableado/tuberías |  
3. **Informe de Análisis Individual por Imagen**: Para al menos tres imágenes (recomendadas: Imagen1, Imagen5, Imagen6):  
   * Presentar imagen. Discutir patrones térmicos. Explicar interpretación de escala. Describir ROIs y temperaturas. Incluir histograma (si se generó) y su explicación. Proponer aplicación aeronáutica.

### **VI. Caso de Estudio / Aplicación Práctica**

1. **Inspección de Materiales Compuestos**: Un termograma de un ala de composite muestra una zona circular térmicamente anómala.  
   * ¿Posible defecto? (Ej. delaminación, inclusión de agua).  
   * ¿Principio físico? (Alteración de la conductividad/capacidad térmica local).  
   * ¿Información adicional necesaria? (Ej. termografía activa, ultrasonido).  
2. **Monitoreo de Aviónica**: Componente a 85 °C, adyacentes a 50 °C (Temp. máx. segura: 75 °C).  
   * ¿Indicación? (Sobrecalentamiento crítico).  
   * ¿Causas? (Fallo interno, mala refrigeración, sobrecarga).  
   * ¿Consecuencias? (Fallo del componente, daño a sistemas adyacentes, riesgo de incendio).  
3. **Evaluación de Sistema de Deshielo**: Prueba de sistema eléctrico en borde de ataque.  
   * ¿Uso de termografía? (Verificar uniformidad y temperatura alcanzada en superficie).  
   * ¿Patrones óptimos? (Calentamiento uniforme a la temperatura de diseño).  
   * ¿Patrones deficientes? (Zonas frías, calentamiento irregular, puntos sobrecalentados).

### **VII. Discusión en el Contexto Aeronáutico**

* Ventajas de la termografía (no destructiva, sin contacto) para componentes críticos y materiales sensibles.  
* Factores ambientales en inspecciones (luz solar, viento, reflejos) y su mitigación (sombrear, proteger del viento, usar pantallas, medir emisividad).  
* Impacto de la emisividad variable en superficies aeronáuticas (pinturas, metales) y estrategias para manejarla (cintas de alta emisividad, tablas de emisividad, ajuste en cámara).  
* Comparación de termografía con otros END (ej. ultrasonido para delaminaciones) destacando ventajas/desventajas relativas.

### **VIII. Cuestionario / Preguntas de Reflexión**

1. ¿Qué es la emisividad y por qué es crucial para mediciones cuantitativas precisas?  
2. En el carenado de un motor, ¿qué podría indicar una sección inesperadamente mucho más caliente o fría que el resto?  
3. En termografía activa para una reparación de composite, ¿qué firma térmica indicaría mala adhesión o vacíos?  
4. ¿Cómo se usaría la termografía en diseño y prueba de nuevos componentes aeronáuticos?  
5. Limitaciones de imágenes JPG vs. archivos radiométricos para análisis termográfico.  
6. En Imagen1.jpg, ¿cómo afectaría una pintura de baja emisividad y alta reflectividad al termograma en un día soleado?

### **IX. Requisitos del Reporte (Formato AIAA \- Adaptado)**

* **Título, Autores, Afiliación, Fecha.**  
* **Abstract (Resumen)**: Objetivos, metodología resumida (análisis visual de termogramas JPG, extracción de datos de escalas), principales resultados, conclusión sobre utilidad y limitaciones.  
* **1\. Introducción**: Importancia de termografía en aeronáutica, propósito de la práctica.  
* **2\. Marco Teórico**: Principios de radiación IR, termografía, cámaras, emisividad, interpretación.  
* **3\. Metodología de Análisis de Datos**: Descripción de datos JPG, pasos de análisis (visualización, ROIs, lectura de escalas, histogramas si aplica).  
* **4\. Resultados**: Tabla de Análisis de Termogramas. Informe de Análisis Individual (3 imágenes). Respuestas al Caso de Estudio.  
* **5\. Discusión**: Interpretación general. Utilidad de herramientas de análisis. Limitaciones (JPG, falta de contexto). Fuentes de error. Relevancia para ingeniería aeronáutica.  
* **6\. Conclusiones**: Principales aprendizajes. Validez y limitaciones. Importancia de la termografía.  
* **7\. Referencias**: (Si se consultaron).  
* **Apéndices (Opcional)**: Código Python utilizado.

### **X. Referencias Sugeridas**

1. Vollmer, M., & Möllmann, K. P. (2017). *Infrared Thermal Imaging: Fundamentals, Research and Applications* (2nd ed.). Wiley-VCH.  
2. Kaplan, H. (2007). *Practical Applications of Infrared Thermal Sensing and Imaging Equipment* (3rd ed.). SPIE Press.  
3. Meola, C. (Ed.). (2012). *Infrared Thermography in the Evaluation of Aerospace Composite Materials*. Woodhead Publishing.  
4. ASTM E1933: *Standard Practice for Measuring and Compensating for Emissivity Using Infrared Imaging Radiometers*.

### **Apéndice: Código Python de Ejemplo**

El siguiente código utiliza las bibliotecas Pillow para la manipulación de imágenes y Matplotlib para la visualización y generación de histogramas. Asegúrate de tener estas bibliotecas instaladas (pip install Pillow matplotlib numpy).

import matplotlib.pyplot as plt  
from PIL import Image  
import numpy as np

def load\_and\_display\_image(image\_path, title=''):  
    """  
    Carga y muestra una imagen.

    Args:  
        image\_path (str): Ruta al archivo de imagen.  
        title (str, optional): Título para la gráfica. Por defecto ''.  
      
    Returns:  
        PIL.Image.Image or None: Objeto de imagen si se carga correctamente, None si no.  
    """  
    try:  
        img \= Image.open(image\_path)  
        plt.figure(figsize=(8, 6))  
        plt.imshow(np.asarray(img)) \# Convertir a array para imshow  
        plt.title(title if title else image\_path)  
        plt.axis('off') \# Desactivar ejes para visualización limpia de termogramas  
        plt.show()  
        return img  
    except FileNotFoundError:  
        print(f"Error: Archivo no encontrado en la ruta: {image\_path}")  
        return None  
    except Exception as e:  
        print(f"Error al cargar o mostrar la imagen {image\_path}: {e}")  
        return None

def plot\_histogram\_roi(image\_pil, roi\_coords=None, channel='L', title\_prefix=''):  
    """  
    Muestra un ROI de una imagen y su histograma de intensidad.

    Args:  
        image\_pil (PIL.Image.Image): Objeto de imagen de Pillow.  
        roi\_coords (tuple, optional): Coordenadas (left, upper, right, lower) para recortar el ROI.  
                                     Si es None, se usa la imagen completa. Por defecto None.  
        channel (str or int, optional): Canal a analizar. 'L' para escala de grises,  
                                     0 para Rojo, 1 para Verde, 2 para Azul. Por defecto 'L'.  
        title\_prefix (str, optional): Prefijo para el título de la figura.  
    """  
    if image\_pil is None:  
        print("Error: No se proporcionó una imagen válida para el histograma.")  
        return

    if roi\_coords:  
        try:  
            img\_roi \= image\_pil.crop(roi\_coords)  
        except Exception as e:  
            print(f"Error al recortar ROI con coordenadas {roi\_coords}: {e}")  
            img\_roi \= image\_pil \# Usar imagen completa si el recorte falla  
    else:  
        img\_roi \= image\_pil

    img\_array\_processed \= None  
    cmap\_hist \= 'gray' \# Color por defecto para el histograma  
    plot\_title\_suffix \= ''

    try:  
        if channel \== 'L':  
            \# Convertir a escala de grises si no lo está ya  
            if img\_roi.mode \!= 'L':  
                img\_array\_processed \= np.array(img\_roi.convert('L'))  
            else:  
                img\_array\_processed \= np.array(img\_roi)  
            plot\_title\_suffix \= ' (Escala de Grises)'  
            cmap\_hist \= 'gray'  
        elif channel in \[0, 1, 2\] and img\_roi.mode in \['RGB', 'RGBA'\]:  
            img\_array \= np.array(img\_roi)  
            img\_array\_processed \= img\_array\[:, :, channel\] \# Seleccionar canal  
            channel\_names \= \["Rojo", "Verde", "Azul"\]  
            cmap\_hist \= \['red', 'green', 'blue'\]\[channel\]  
            plot\_title\_suffix \= f' (Canal {channel\_names\[channel\]})'  
        elif img\_roi.mode \== 'L': \# Si la imagen ya es monocromática y se pide un canal de color  
            print(f"Advertencia: La imagen ya está en escala de grises. Mostrando histograma de luminancia.")  
            img\_array\_processed \= np.array(img\_roi)  
            plot\_title\_suffix \= ' (Escala de Grises \- Original)'  
            cmap\_hist \= 'gray'  
        else:  
            print(f"Advertencia: Modo de imagen '{img\_roi.mode}' o canal '{channel}' no soportado directamente para histograma a color. Convirtiendo a escala de grises.")  
            img\_array\_processed \= np.array(img\_roi.convert('L'))  
            plot\_title\_suffix \= ' (Escala de Grises \- Conversión)'  
            cmap\_hist \= 'gray'

    except Exception as e:  
        print(f"Error procesando canales de imagen: {e}. Intentando conversión a escala de grises.")  
        try:  
            img\_array\_processed \= np.array(img\_roi.convert('L'))  
            plot\_title\_suffix \= ' (Escala de Grises \- Fallback)'  
            cmap\_hist \= 'gray'  
        except Exception as e\_fallback:  
            print(f"Error crítico al convertir a escala de grises: {e\_fallback}")  
            return \# No se puede continuar

    if img\_array\_processed is None:  
        print("Error: No se pudo procesar la imagen para el histograma.")  
        return

    plt.figure(figsize=(12, 5))  
    plt.suptitle(f'{title\_prefix}Análisis de ROI e Histograma', fontsize=14)

    \# Subplot para la imagen ROI  
    plt.subplot(1, 2, 1\)  
    \# Mostrar la imagen ROI tal como se procesó para el histograma si es monocromática  
    \# o la original si es a color (el histograma será de un canal)  
    if img\_array\_processed.ndim \== 2: \# Monocromática  
        plt.imshow(img\_array\_processed, cmap='gray')  
    else: \# Si img\_array\_processed sigue siendo 3D (no debería ocurrir aquí) o para mostrar el ROI original  
        plt.imshow(img\_roi)  
    plt.title('ROI Seleccionado')  
    plt.axis('off')

    \# Subplot para el histograma  
    plt.subplot(1, 2, 2\)  
    if img\_array\_processed.ndim \== 2: \# Asegurar que es 2D para flatten  
        flat\_array \= img\_array\_processed.flatten()  
        plt.hist(flat\_array, bins=256, range=(0, 255), color=cmap\_hist, density=True)  
        plt.title(f'Histograma de Intensidad{plot\_title\_suffix}')  
        plt.xlabel('Intensidad del Píxel (0-255)')  
        plt.ylabel('Frecuencia Normalizada')  
        plt.grid(True, linestyle='--', alpha=0.7)  
    else:  
        plt.text(0.5, 0.5, "Datos no aptos para histograma (verificar dimensiones)", ha='center', va='center')  
        plt.title(f'Error en Histograma{plot\_title\_suffix}')

    plt.tight\_layout(rect=\[0, 0, 1, 0.96\]) \# Ajustar para el suptitle  
    plt.show()

\# \--- Ejemplo de Uso \---  
\# Primero, define las rutas a tus imágenes  
\# path\_imagen1 \= 'ruta/a/Imagen1.jpg'  
\# path\_imagen5 \= 'ruta/a/Imagen5.jpg'

\# Cargar y mostrar una imagen  
\# print("Mostrando Imagen 1:")  
\# img1\_pil \= load\_and\_display\_image(path\_imagen1, title='Imagen 1: Aeronave')

\# Cargar otra imagen y analizar un ROI con histograma  
\# print("\\nAnalizando ROI de Imagen 5 (ejemplo):")  
\# img5\_pil \= Image.open(path\_imagen5) \# Cargar sin mostrar inmediatamente si se va a usar en plot\_histogram\_roi

\# if img5\_pil:  
    \# Definir coordenadas aproximadas para una taza en Imagen5.jpg (ajustar según la imagen)  
    \# Estas coordenadas son (left, upper, right, lower)  
    \# roi\_taza\_coords\_img5 \= (50, 100, 250, 400\) \# Ejemplo, ajustar estas\!  
      
    \# Analizar el ROI en escala de grises  
    \# plot\_histogram\_roi(img5\_pil,   
    \#                    roi\_coords=roi\_taza\_coords\_img5,   
    \#                    channel='L',   
    \#                    title\_prefix='Imagen 5 \- Taza 1 ')  
      
    \# Analizar el canal rojo del mismo ROI (suponiendo que la paleta usa rojo para calor)  
    \# plot\_histogram\_roi(img5\_pil,   
    \#                    roi\_coords=roi\_taza\_coords\_img5,   
    \#                    channel=0, \# 0 para Rojo  
    \#                    title\_prefix='Imagen 5 \- Taza 1 ')  
\# else:  
\#    print(f"No se pudo cargar {path\_imagen5} para análisis de ROI.")

\# Para usar este código:  
\# 1\. Reemplaza 'ruta/a/ImagenX.jpg' con las rutas correctas a tus archivos de imagen.  
\# 2\. Descomenta las líneas de "Ejemplo de Uso".  
\# 3\. Ajusta las coordenadas \`roi\_taza\_coords\_img5\` para que realmente seleccionen una región de interés  
\#    en tu \`Imagen5.jpg\`. Puedes obtener estas coordenadas usando un visor de imágenes que muestre  
\#    las coordenadas del cursor.  
\# 4\. Ejecuta el script en un entorno Python donde tengas las bibliotecas instaladas.

