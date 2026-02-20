## **PRÁCTICA 8: Análisis de Productos de Combustión**

**I. Introducción**

El análisis de los gases de escape en motores de reacción y otros sistemas de combustión aeronáuticos es fundamental para evaluar la eficiencia, las emisiones y detectar posibles problemas de funcionamiento. La composición de estos gases revela información crucial sobre la calidad del proceso de combustión dentro del motor. En esta práctica, se abordará la importancia de este análisis. Los estudiantes trabajarán con datos representativos de la concentración de diversas especies químicas (ej. CO2​, CO, O2​, NOx​, hidrocarburos no quemados) en los gases de escape, medidos bajo diferentes condiciones de operación de un motor aeronáutico típico. El objetivo principal es que los estudiantes analicen estos datos para calcular parámetros clave como la relación aire/combustible y la eficiencia de combustión. Finalmente, se discutirán las implicaciones de los resultados en términos de rendimiento del motor, su estado operativo y el cumplimiento de las normativas ambientales vigentes en la industria aeronáutica.

**II. Objetivos de Aprendizaje Específicos**

Al finalizar esta práctica, el estudiante será capaz de:

* Comprender la importancia y el propósito del análisis de productos de combustión en motores aeronáuticos.  
* Identificar las principales especies químicas presentes en los gases de escape y entender su significado en relación con el proceso de combustión.  
* Procesar datos crudos de concentración de gases de escape para diferentes condiciones operativas del motor.  
* Asumir un combustible aeronáutico típico y calcular la relación aire/combustible estequiométrica.  
* Calcular la relación aire/combustible real y el coeficiente de exceso de aire (λ) para cada condición operativa a partir de la composición de los gases de escape.  
* Estimar la eficiencia de combustión utilizando las concentraciones de productos de combustión incompleta.  
* Elaborar tablas y gráficas para presentar los resultados del análisis.  
* Interpretar los parámetros calculados (AFR, λ, ηc​) y las tendencias de las emisiones (CO, NOx, UHC) en función de las condiciones de operación del motor.  
* Discutir las implicaciones de los resultados en términos de rendimiento del motor, diagnóstico de fallas y cumplimiento de normativas ambientales.  
* Identificar posibles fuentes de error en la medición y análisis de los gases de escape.

**III. Fundamentos Teóricos**

**A. Combustión en Motores de Reacción Aeronáuticos**

La combustión es una reacción química exotérmica de oxidación rápida entre un combustible y un comburente (generalmente oxígeno del aire), liberando energía en forma de calor. En los motores de reacción, esta energía térmica se convierte en energía cinética para producir empuje. Un combustible aeronáutico típico es el queroseno, que puede representarse por una fórmula promedio como C12​H23​.

La combustión ideal y completa (estequiométrica) con aire se describe como:  
Cx​Hy​+(x+y/4)(O2​+3.76N2​)→xCO2​+(y/2)H2​O+(x+y/4)3.76N2​  
En la práctica, la combustión puede ser incompleta o realizarse con exceso de aire (mezclas pobres, λ\>1) o defecto de aire (mezclas ricas, λ\<1), lo que afecta la composición de los gases de escape.  
**B. Principales Productos de la Combustión y su Significado**

* **Dióxido de Carbono (**CO2​**)**: Principal producto de la combustión completa del carbono. Su concentración es un indicador de cuán completa ha sido la reacción.  
* **Agua (**H2​O**)**: Principal producto de la combustión completa del hidrógeno. Generalmente no se mide en análisis de gases secos, pero su formación es crucial.  
* **Oxígeno (**O2​**)**: El oxígeno residual en los gases de escape indica combustión con exceso de aire (mezcla pobre), común en muchas fases de operación de los motores de turbina.  
* **Monóxido de Carbono (**CO**)**: Producto de la combustión incompleta debido a falta de oxígeno local, bajas temperaturas o tiempo de residencia insuficiente. Es un gas tóxico y representa una pérdida de eficiencia energética.  
* **Óxidos de Nitrógeno (**NOx​**)**: Incluyen principalmente NO y NO2​. Se forman a altas temperaturas en la cámara de combustión por la reacción del nitrógeno y oxígeno del aire. Son contaminantes atmosféricos significativos.  
* **Hidrocarburos No Quemados (UHC o HC)**: Restos de combustible que no participaron en la combustión o lo hicieron parcialmente. Representan ineficiencia y son contaminantes.  
* **Hollín (Partículas)**: Partículas de carbono sólido formadas en zonas ricas en combustible y altas temperaturas. Afectan la durabilidad del motor y son un contaminante.

**C. Parámetros Clave del Análisis de Gases**

1. **Relación Aire/Combustible (AFR)**:  
   * AFR Estequiométrica (AFRstoich​): Proporción másica de aire a combustible para una combustión teóricamente completa. Para C12​H23​, Mcomb​=167.316 g/mol. La reacción estequiométrica es C12​H23​+17.75O2​→12CO2​+11.5H2​O.  
     AFRstoich​=1 mol C12​H23​×MC12​H23​​17.75 moles O2​×(MO2​+3.76MN2​)​=167.31617.75×(32+3.76×28.013)​≈14.57  
   * **AFR Real (**AFRactual​**)**: Relación real en la que opera el motor, calculada a partir de la composición de los gases de escape.  
   * Coeficiente de Exceso de Aire (λ): λ=AFRstoich​AFRactual​​.  
     λ\>1: Mezcla pobre (exceso de aire).  
     λ\<1: Mezcla rica (defecto de aire).  
     λ=1: Mezcla estequiométrica.  
2. Eficiencia de Combustión (ηc​):  
   Indica qué tan efectivamente se libera la energía química del combustible. Las pérdidas se deben principalmente a la presencia de CO y UHC en los gases de escape.  
   ηc​(

D. Sistemas de Monitoreo de Gases (Breve Mención)  
Los motores de aviación pueden ser equipados con sensores para monitorear parámetros de combustión. El análisis detallado de gases se realiza con equipos especializados como:

* Analizadores NDIR (Infrarrojo No Dispersivo) para CO,CO2​.  
* Detectores de Ionización de Llama (FID) para UHC.  
* Sensores electroquímicos o paramagnéticos para O2​.  
* Analizadores de Quimioluminiscencia para NOx​.

**IV. Metodología Experimental y de Análisis de Datos**

A. Descripción del Montaje Experimental Típico (Conceptual)  
Para obtener los datos de esta práctica, conceptualmente se utilizaría un motor de reacción montado en un banco de pruebas. Se tomarían muestras de los gases de escape mediante una sonda introducida en el flujo de escape. Estas muestras serían conducidas a un sistema analizador de gases que mide las concentraciones de las especies de interés bajo diversas condiciones de operación del motor (ej. ralentí, crucero, despegue).  
B. Descripción de los Conjuntos de Datos (Datasets)  
Para esta práctica, se proporcionarán datos representativos de la composición de gases de escape de un motor aeronáutico que utiliza queroseno (C12​H23​) como combustible. Los datos se presentan en la siguiente tabla:  
Tabla 1: Datos de Concentración de Gases de Escape  
| Condición Operativa | Potencia Relativa (%) | CO2​ (%) | O2​ (%) | CO (ppm) | NOx​ (ppm) | UHC (ppm C1​) |  
|-----------------------------|-----------------------|------------|-----------|------------|--------------|-----------------|  
| Ralentí (Idle) | 25 | 7.5 | 11.5 | 2500 | 60 | 350 |  
| Aproximación (Low Power) | 40 | 9.0 | 9.0 | 1000 | 150 | 150 |  
| Crucero (Mid Power) | 75 | 12.0 | 5.0 | 400 | 700 | 40 |  
| Máxima Continua (High Power)| 90 | 13.0 | 3.0 | 600 | 1300 | 60 |  
| Despegue (Take-off) | 100 | 13.8 | 1.8 | 700 | 1800 | 70 |  
*Nota: ppm* C1​ *significa partes por millón referidas a metano (un átomo de carbono).*

**C. Procedimiento de Tratamiento y Análisis de Datos**

**Paso 1: Constantes del Combustible y Aire.**

* Combustible: Queroseno, C12​H23​.  
* Masa molar del combustible (Mfuel​): 12×12.011+23×1.008=167.316 g/mol.  
* Masa molar del aire (Mair​): 28.97 g/mol.  
* AFRstoich​ para C12​H23​≈14.57.

Paso 2: Cálculo de N2​ en base seca.  
Para cada condición operativa, calcular el porcentaje de N2​ en los gases de escape secos, asumiendo que el resto son las especies medidas:  
N2​(  
Nota: Convertir ppm a % dividiendo por 10000\.  
Paso 3: Cálculo de la Relación Aire/Combustible Real (AFRactual​).  
Se convierten las concentraciones de CO2​, CO y UHC a fracción molar (Yi​):  
YCO2​=CO2​(  
YCO​=COppm​/106  
YUHC​=UHCppm​/106  
Utilizar la siguiente fórmula basada en el balance de carbono y nitrógeno:  
AFR\_{actual} \= \\frac{M\_{air}}{M\_{fuel}} \\times \\frac{12 \\times (N\_2 (%) / 79.05)}{Y\_{CO2} \+ Y\_{CO} \+ Y\_{UHC}}  
(donde 12 es el número de átomos de carbono en C12​H23​, y 79.05 es el porcentaje de N2​ en el aire).  
Paso 4: Cálculo del Coeficiente de Exceso de Aire (λ).  
λ=AFRstoich​AFRactual​​  
Paso 5: Cálculo de la Eficiencia de Combustión (ηc​).  
Utilizar la siguiente fórmula aproximada basada en las pérdidas por CO y UHC:  
ηc​(  
Donde 0.427 es la relación aproximada del poder calorífico del CO al del carbono del combustible (Hu,CO​/Hu,fuel,C​). El término CO2​( convierte el porcentaje de CO2​ a ppm para que todas las concentraciones en el denominador estén en la misma unidad (ppm).  
Paso 6: Tabulación de Resultados.  
Crear una tabla que incluya para cada condición operativa: Potencia Relativa (%), CO2​ (%), O2​ (%), CO (ppm), NOx​ (ppm), UHC (ppm C1​), N2​ (%), AFRactual​, λ, y ηc​ (%).  
**Paso 7: Graficación.**

1. Graficar AFRactual​ y λ vs. Potencia Relativa (%).  
2. Graficar ηc​ (%) vs. Potencia Relativa (%).  
3. Graficar las emisiones (CO ppm, NOx​ ppm, UHC ppm C1​) vs. Potencia Relativa (%). (Puede usar ejes Y separados si las escalas son muy diferentes o una gráfica con múltiples ejes Y).

**V. Actividades a Realizar y Resultados Esperados**

1. **Cálculos Preliminares:**  
   * Anote la fórmula del combustible (C12​H23​) y su masa molar.  
   * Anote el valor de AFRstoich​.  
2. **Procesamiento de Datos (para cada condición operativa de la Tabla 1):**  
   * Calcular N2​ (%) usando la fórmula del Paso 2 (Metodología).  
   * Calcular AFRactual​ usando la fórmula del Paso 3 (Metodología).  
   * Calcular λ usando la fórmula del Paso 4 (Metodología).  
   * Calcular ηc​ (%) usando la fórmula del Paso 5 (Metodología).  
3. **Tabla de Resultados:**  
   * Presentar una tabla completa con todas las especies medidas y todos los parámetros calculados para cada condición operativa.  
4. **Gráficas:**  
   * Elaborar las gráficas indicadas en el Paso 7 (Metodología), asegurándose de que los ejes estén correctamente etiquetados (con unidades) y las gráficas tengan títulos descriptivos.  
5. **Análisis e Interpretación:**  
   * Analizar las tendencias observadas en las gráficas. ¿Cómo varían λ, ηc​, CO, NOx​, y UHC con la potencia del motor?  
   * Relacionar los valores de λ con las concentraciones de O2​, CO, y UHC.  
   * Discutir los niveles de NOx​ en función de la carga del motor y las posibles temperaturas de combustión.

**VI. Caso de Estudio / Aplicación Práctica**

1. **Análisis de** NOx​**:**  
   * Explique por qué las emisiones de NOx​ son generalmente más altas a condiciones de alta potencia del motor.  
   * Si una nueva normativa ambiental limita las emisiones de NOx​ a 800 ppm en condición de crucero, ¿cumpliría el motor analizado según sus datos?  
2. **Diagnóstico de Motor:**  
   * Suponga que en una revisión posterior, el motor en condición de "Crucero" presenta los siguientes datos: CO2​=11.5, O2​=5.5, CO=1500 ppm, NOx​=650 ppm, UHC \=200 ppm C1​. Calcule la nueva ηc​ y compare con la original. ¿Qué posibles problemas de funcionamiento del motor podrían inferirse de este cambio en las emisiones?  
3. **Optimización de Combustión:**  
   * ¿Qué estrategias generales podrían implementarse en el diseño o control de un motor de reacción para reducir simultáneamente las emisiones de CO, UHC y NOx​ manteniendo una alta eficiencia?

**VII. Discusión en el Contexto Aeronáutico**

* Discutir la importancia del análisis de productos de combustión para el mantenimiento predictivo y la optimización del rendimiento de los motores aeronáuticos.  
* Analizar cómo la eficiencia de combustión impacta directamente el consumo de combustible y, por ende, los costos operativos y la autonomía de la aeronave.  
* Relacionar las emisiones de CO, NOx​, UHC y hollín con las regulaciones ambientales internacionales (ej. OACI) y el esfuerzo de la industria aeronáutica por desarrollar tecnologías más limpias.  
* Explicar cómo las diferentes fases de un vuelo (ralentí, despegue, crucero, aproximación) imponen distintas demandas al motor que se reflejan en la composición de los gases de escape.  
* Mencionar brevemente cómo los avances en sensores y sistemas de diagnóstico a bordo contribuyen a un mejor seguimiento de la salud del motor y la combustión.

**VIII. Cuestionario / Preguntas de Reflexión**

1. ¿Por qué es crucial medir múltiples especies de gases (CO2​,O2​,CO,NOx,UHC) en lugar de solo una o dos para evaluar completamente el proceso de combustión?  
2. ¿Qué indica un valor de λ significativamente mayor que 1 sobre el régimen de combustión? ¿Y cómo se relaciona esto con la eficiencia y las emisiones de CO y UHC?  
3. ¿Cuáles son los principales factores que influyen en la formación de NOx​ en una cámara de combustión de turbina de gas?  
4. Si la eficiencia de combustión calculada (ηc​) es del 98.5%, ¿qué significa el 1.5% restante en términos de energía y emisiones?  
5. ¿Cuáles son los principales desafíos en la optimización de la combustión en motores de aviación para mejorar la eficiencia y reducir emisiones simultáneamente?  
6. ¿Cómo se monitorea y mantiene la salud de la combustión en los motores de aviación durante los vuelos o mediante inspecciones periódicas?  
7. Mencione dos métodos o tecnologías que se utilicen para reducir las emisiones contaminantes de los motores de aviación.

**IX. Requisitos del Reporte**

Elabore un reporte técnico que incluya:

* **Título, Autores, Afiliación, Fecha.**  
* **Abstract (Resumen):** Objetivo de la práctica (análisis de productos de combustión), metodología resumida (procesamiento de datos de emisiones, cálculo de AFR, λ, ηc​), principales resultados (tablas, gráficas clave, tendencias observadas), y una conclusión sobre la relación entre condiciones operativas, rendimiento y emisiones del motor.  
* **1\. Introducción:** Breve descripción de la importancia del análisis de la combustión en la ingeniería aeronáutica y el propósito específico de esta práctica. Objetivos de la práctica.  
* **2\. Marco Teórico:** Resumen de los principios de combustión en motores de reacción, las principales especies químicas en los gases de escape y su significado, la definición y cálculo de AFR, λ, y ηc​.  
* **3\. Metodología de Análisis de Datos:**  
  * Descripción de los datos de entrada proporcionados (Tabla 1, tipo de combustible).  
  * Pasos detallados del procesamiento de datos: cómo se calculó N2​, AFRactual​, λ, y ηc​, incluyendo las fórmulas utilizadas.  
* **4\. Resultados:**  
  * Tabla completa con los datos originales y todos los parámetros calculados para cada condición operativa.  
  * Gráficas claramente etiquetadas de:  
    * AFRactual​ y λ vs. Potencia Relativa (%).  
    * ηc​ (%) vs. Potencia Relativa (%).  
    * Concentraciones de CO, NOx​, y UHC vs. Potencia Relativa (%).  
  * Resultados y respuestas detalladas del Caso de Estudio.  
* **5\. Discusión:**  
  * Interpretación de las tendencias observadas en las gráficas (cómo λ, ηc​, y las emisiones varían con la carga del motor).  
  * Análisis de la relación entre λ y la eficiencia de combustión.  
  * Análisis de la formación de NOx​ en relación con otros parámetros.  
  * Limitaciones del análisis realizado (ej. supuestos, precisión de fórmulas).  
  * Relevancia de este tipo de análisis para la ingeniería aeronáutica (eficiencia, mantenimiento, medio ambiente).  
* **6\. Conclusiones:** Resumen de los hallazgos principales, cómo las condiciones de operación afectan los productos de combustión, la eficiencia y las emisiones. Importancia de la práctica.  
* **7\. Referencias:** Listado de las fuentes citadas (incluyendo el manual de laboratorio y las referencias sugeridas si se consultaron).  
* **8\. Apéndices:** Incluir el script de Python utilizado para el análisis (si se usa) o un ejemplo de cálculo manual detallado.

**X. Referencias Sugeridas**

1. Lieuwen, T. C., & Yang, V. (2006). *Combustion Instabilities in Gas Turbine Engines*. AIAA.  
2. Lefebvre, A. H., & Ballal, D. R. (2010). *Gas turbine combustion alternative fuels and emissions* (3rd ed.). Taylor & Francis.  
3. Heywood, J. B. (1988). *Internal Combustion Engine Fundamentals*. McGraw-Hill.  
4. Mattingly, J. D. (2006). *Elements of Propulsion: Gas Turbines and Rockets*. AIAA Education Series.  
5. Instructivo de Laboratorio de Técnicas de Medida, FIME, UANL.

**XI. Apéndice: Herramientas para el Análisis de Datos (Ejemplo con Python)**

Para realizar los cálculos y la graficación de manera eficiente, se puede utilizar Python con las bibliotecas pandas para el manejo de datos y matplotlib para las gráficas.

**Instalación de bibliotecas (si no las tienes):**

pip install pandas matplotlib

**Script de Ejemplo en Python:**

import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np

\# \--- Constantes \---  
M\_fuel \= 167.316  \# g/mol (C12H23)  
M\_air \= 28.97    \# g/mol  
AFR\_stoich \= 14.57  
C\_atoms\_in\_fuel \= 12 \# Átomos de Carbono en C12H23  
N2\_in\_air\_percent \= 79.05 \# % de N2 en el aire

\# \--- Datos de Entrada (Tabla 1\) \---  
data \= {  
    'Condición Operativa': \['Ralentí (Idle)', 'Aproximación (Low Power)', 'Crucero (Mid Power)', 'Máxima Continua (High Power)', 'Despegue (Take-off)'\],  
    'Potencia Relativa (%)': \[25, 40, 75, 90, 100\],  
    'CO2 (%)': \[7.5, 9.0, 12.0, 13.0, 13.8\],  
    'O2 (%)': \[11.5, 9.0, 5.0, 3.0, 1.8\],  
    'CO (ppm)': \[2500, 1000, 400, 600, 700\],  
    'NOx (ppm)': \[60, 150, 700, 1300, 1800\],  
    'UHC (ppm C1)': \[350, 150, 40, 60, 70\]  
}  
df \= pd.DataFrame(data)

\# \--- Cálculos \---

\# Paso 2: N2 (%)  
df\['CO (%)'\] \= df\['CO (ppm)'\] / 10000  
df\['UHC (%)'\] \= df\['UHC (ppm C1)'\] / 10000  
df\['N2 (%)'\] \= 100 \- df\['CO2 (%)'\] \- df\['O2 (%)'\] \- df\['CO (%)'\] \- df\['UHC (%)'\]

\# Paso 3: AFR\_actual  
\# Convertir a fracción molar para la fórmula de AFR\_actual  
Y\_CO2 \= df\['CO2 (%)'\] / 100  
Y\_CO \= df\['CO (ppm)'\] / 1e6  
Y\_UHC \= df\['UHC (ppm C1)'\] / 1e6

\# Denominador para AFR\_actual (suma de fracciones molares de C en productos)  
C\_in\_products\_molar\_fraction\_sum \= Y\_CO2 \+ Y\_CO \+ Y\_UHC

df\['AFR\_actual'\] \= (M\_air / M\_fuel) \* (C\_atoms\_in\_fuel \* (df\['N2 (%)'\] / N2\_in\_air\_percent)) / C\_in\_products\_molar\_fraction\_sum

\# Paso 4: Lambda  
df\['Lambda'\] \= df\['AFR\_actual'\] / AFR\_stoich

\# Paso 5: Eficiencia de Combustión (eta\_c)  
\# Denominador para eta\_c (concentraciones en ppm)  
denominator\_eta\_c \= df\['CO2 (%)'\] \* 10000 \+ df\['CO (ppm)'\] \+ df\['UHC (ppm C1)'\]  
\# Evitar división por cero si el denominador es cero (aunque improbable con estos datos)  
denominator\_eta\_c \= np.where(denominator\_eta\_c \== 0, 1e-9, denominator\_eta\_c) \# Evita división por cero

df\['eta\_c (%)'\] \= (1 \- (0.427 \* df\['CO (ppm)'\] \+ df\['UHC (ppm C1)'\]) / denominator\_eta\_c) \* 100

\# \--- Mostrar Tabla de Resultados \---  
print("--- Tabla de Resultados del Análisis de Combustión \---")  
results\_table \= df\[\['Condición Operativa', 'Potencia Relativa (%)', 'CO2 (%)', 'O2 (%)',  
                    'CO (ppm)', 'NOx (ppm)', 'UHC (ppm C1)', 'N2 (%)',  
                    'AFR\_actual', 'Lambda', 'eta\_c (%)'\]\]  
print(results\_table.to\_string()) \# .to\_string() para imprimir todo el dataframe

\# \--- Graficación \---  
potencia\_relativa \= df\['Potencia Relativa (%)'\]

\# Gráfica 1: AFR\_actual y Lambda  
fig1, ax1 \= plt.subplots(figsize=(10, 6))  
color \= 'tab:red'  
ax1.set\_xlabel('Potencia Relativa (%)')  
ax1.set\_ylabel('AFR\_actual', color=color)  
ax1.plot(potencia\_relativa, df\['AFR\_actual'\], color=color, marker='o', label='AFR\_actual')  
ax1.tick\_params(axis='y', labelcolor=color)  
ax1.grid(True)

ax2 \= ax1.twinx() \# Instanciar un segundo eje que comparte el mismo eje x  
color \= 'tab:blue'  
ax2.set\_ylabel('Lambda ($\\lambda$)', color=color)  
ax2.plot(potencia\_relativa, df\['Lambda'\], color=color, marker='s', linestyle='--', label='Lambda')  
ax2.tick\_params(axis='y', labelcolor=color)

fig1.tight\_layout() \# Para que no se solapen los ylabel  
plt.title('AFR\_actual y Lambda vs. Potencia Relativa')  
\# Para leyendas de múltiples ejes:  
lines, labels \= ax1.get\_legend\_handles\_labels()  
lines2, labels2 \= ax2.get\_legend\_handles\_labels()  
ax2.legend(lines \+ lines2, labels \+ labels2, loc='best')  
plt.show()

\# Gráfica 2: Eficiencia de Combustión  
plt.figure(figsize=(10, 6))  
plt.plot(potencia\_relativa, df\['eta\_c (%)'\], marker='o', linestyle='-', color='green')  
plt.title('Eficiencia de Combustión ($\\eta\_c$) vs. Potencia Relativa')  
plt.xlabel('Potencia Relativa (%)')  
plt.ylabel('$\\eta\_c$ (%)')  
plt.grid(True)  
plt.ylim(min(90, df\['eta\_c (%)'\].min() \- 1\) , 100.5) \# Ajustar límites Y para mejor visualización  
plt.show()

\# Gráfica 3: Emisiones  
fig3, ax\_emissions \= plt.subplots(figsize=(12, 7))  
ax\_emissions.plot(potencia\_relativa, df\['CO (ppm)'\], marker='o', label='CO (ppm)', color='purple')  
ax\_emissions.plot(potencia\_relativa, df\['UHC (ppm C1)'\], marker='s', label='UHC (ppm C1)', color='brown')  
\# Eje Y secundario para NOx debido a posible diferencia de escala  
ax\_nox \= ax\_emissions.twinx()  
ax\_nox.plot(potencia\_relativa, df\['NOx (ppm)'\], marker='^', label='NOx (ppm)', color='orange')

ax\_emissions.set\_xlabel('Potencia Relativa (%)')  
ax\_emissions.set\_ylabel('CO (ppm) / UHC (ppm C1)')  
ax\_nox.set\_ylabel('NOx (ppm)')  
ax\_emissions.set\_title('Emisiones vs. Potencia Relativa')  
ax\_emissions.grid(True)

\# Leyendas para gráfica con doble eje Y  
lines\_em, labels\_em \= ax\_emissions.get\_legend\_handles\_labels()  
lines\_nox, labels\_nox \= ax\_nox.get\_legend\_handles\_labels()  
ax\_nox.legend(lines\_em \+ lines\_nox, labels\_em \+ labels\_nox, loc='best')

fig3.tight\_layout()  
plt.show()

print("\\nNota: El script anterior es un ejemplo. Asegúrate de entender cada paso y fórmula.")  
print("Puede ser necesario ajustar las rutas de archivos si los datos se cargan desde un archivo externo.")

**Fin del Apéndice**