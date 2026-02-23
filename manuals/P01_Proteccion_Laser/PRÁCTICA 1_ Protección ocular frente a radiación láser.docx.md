## **PRÁCTICA 1: Protección ocular frente a radiación láser**

**I. Introducción**

Los láseres (Amplificación de Luz por Emisión Estimulada de Radiación) son herramientas fundamentales en numerosos campos de la ingeniería aeronáutica, especialmente en laboratorios de investigación y desarrollo para técnicas de medición avanzadas como la Anemometría Láser Doppler (LDA), la Velocimetría por Imagen de Partículas (PIV), la metrología de precisión, y ensayos no destructivos. La naturaleza altamente concentrada y coherente de la radiación láser, si bien es ventajosa para estas aplicaciones, representa un riesgo significativo para la seguridad, particularmente para la visión. Una exposición ocular inadecuada, incluso de corta duración o a través de reflexiones especulares, puede ocasionar daños irreversibles.

Esta práctica introduce los principios fundamentales de la seguridad láser, con un enfoque en la comprensión analítica de los riesgos asociados a diferentes tipos de láseres comúnmente empleados en el contexto aeronáutico. Se abordará la selección y especificación del Equipo de Protección Ocular (EPO) adecuado, no mediante la manipulación física de equipos, sino a través del análisis de datos técnicos de fuentes láser y equipos de protección, el cálculo de parámetros de seguridad críticos y la toma de decisiones informada basada en dichos cálculos. El objetivo es capacitar al estudiante para evaluar cuantitativamente los riesgos y determinar los requisitos de protección en escenarios aeronáuticos simulados.

**II. Objetivos de Aprendizaje Específicos**

Al finalizar esta práctica, el estudiante será capaz de:

* Identificar y describir los peligros asociados a la radiación láser en el contexto de técnicas de medida y aplicaciones aeronáuticas.  
* Comprender y aplicar la clasificación de seguridad láser según estándares internacionales (ANSI Z136.1 / IEC 60825-1) y sus implicaciones.  
* Analizar cualitativamente la interacción de la radiación láser con el tejido ocular en función de la longitud de onda.  
* Interpretar las especificaciones técnicas de fuentes láser (potencia, energía, longitud de onda, modo de operación) y de equipos de protección ocular (Densidad Óptica, Transmisión de Luz Visible).  
* Calcular la Exposición Potencial (H0​) a partir de los parámetros de una fuente láser.  
* Calcular la Densidad Óptica (OD) mínima requerida para protección ocular, dados los parámetros del láser y la Exposición Máxima Permisible (MPE).  
* Seleccionar el equipo de protección ocular apropiado de un conjunto de opciones, justificando la elección mediante cálculos y análisis de datos.  
* Describir los elementos fundamentales de un programa de seguridad láser y los protocolos básicos en un entorno de laboratorio aeronáutico.

**III. Fundamentos Teóricos**

**A. Principios de la Radiación Láser**

La radiación láser se distingue de la luz convencional por cuatro propiedades fundamentales \[1, 2\]:

1. **Monocromaticidad:** La luz láser es de una única longitud de onda (o un rango espectral muy estrecho). Esto es crucial porque la interacción con los tejidos y la efectividad de los filtros de protección son altamente dependientes de la longitud de onda.  
2. **Coherencia:** Las ondas de luz emitidas por un láser están en fase tanto espacial como temporalmente. Esto permite que el haz mantenga su integridad a largas distancias.  
3. **Direccionalidad (Colimación):** El haz láser es altamente colimado, lo que significa que se propaga con muy poca divergencia. Esta propiedad permite que la irradiancia (potencia por unidad de área) se mantenga elevada incluso a grandes distancias de la fuente.  
4. **Alta Irradiancia/Radiancia:** Debido a la baja divergencia y la capacidad de enfocar el haz a un punto muy pequeño, los láseres pueden alcanzar irradiancias extremadamente altas, concentrando una gran cantidad de energía en áreas reducidas.

**B. Clasificación de Seguridad Láser**

Los estándares internacionales, como ANSI Z136.1 (EE.UU.) \[3\] e IEC 60825-1 (Internacional) \[4\], clasifican los láseres según su potencial de causar daño. Esta clasificación determina las medidas de control y protección necesarias. Las clases principales son:

* **Clase 1:** Seguros bajo condiciones razonablemente previsibles de operación. Pueden incluir láseres de alta potencia totalmente encapsulados donde la radiación peligrosa no es accesible.  
* **Clase 1M:** Como la Clase 1, a menos que se observe el haz con instrumentos ópticos de aumento (ej. lupas, telescopios).  
* **Clase 2:** Láseres visibles (400-700 nm) cuya potencia es inferior a 1 mW (CW). La protección natural del ojo (reflejo de aversión o parpadeo, \~0.25 s) es suficiente para prevenir daños.  
* **Clase 2M:** Como la Clase 2, a menos que se observe el haz con instrumentos ópticos de aumento.  
* **Clase 3R:** Láseres (visibles o invisibles) con riesgo potencial bajo visión directa, aunque el riesgo es bajo. Potencia entre 1 mW y 5 mW para CW visibles. Requieren precauciones y señalización.  
* **Clase 3B:** Peligrosos bajo exposición directa del haz. La reflexión difusa (desde una superficie mate) suele ser segura. Potencia entre 5 mW y 500 mW (CW) para láseres visibles. Requieren controles significativos y EPO es frecuentemente necesario.  
* **Clase 4:** Los láseres más peligrosos. Peligrosos bajo exposición directa, especular (desde superficies pulidas) y, a menudo, difusa. Pueden causar daños en la piel e incendios. Potencia superior a 500 mW (CW). Requieren los controles más estrictos, y el EPO es obligatorio. Muchos láseres utilizados en PIV y LDA en aeronáutica caen en esta categoría.

**C. Interacción Láser-Tejido Ocular**

El tipo y extensión del daño ocular dependen críticamente de la longitud de onda (λ) de la radiación láser, así como de la potencia/energía, duración de la exposición y el tamaño del punto en la retina \[1, 5\]:

* **Ultravioleta (UV: 100-400 nm):** Absorbido principalmente por la córnea y el cristalino. Puede causar fotoqueratitis (similar a una quemadura solar en la córnea) y, con exposiciones crónicas, cataratas.  
* **Visible y Infrarrojo Cercano (IR-A: 400-1400 nm):** Esta radiación es transmitida a través de las estructuras anteriores del ojo y enfocada eficientemente por el cristalino sobre la retina. Es la región de mayor riesgo para daño retiniano (térmico o fotoquímico). El ojo puede concentrar la irradiancia en la retina por un factor de hasta 105, lo que significa que un haz aparentemente de baja potencia puede volverse extremadamente peligroso.  
* **Infrarrojo Medio y Lejano (IR-B: 1400-3000 nm; IR-C: 3000 nm \- 1 mm):** Absorbido principalmente por la córnea y el humor acuoso. Puede causar quemaduras corneales y cataratas.

**D. Parámetros Clave de Seguridad Láser** \[2, 3\]

* **Longitud de Onda (λ):** Generalmente en nanómetros (nm) o micrómetros (µm). Determina cómo y dónde se absorbe la energía en el ojo y, por tanto, el tipo de daño y la MPE aplicable.  
* **Potencia (P):** Para láseres de onda continua (CW), en Watts (W) o miliwatts (mW).  
* **Energía por Pulso (E):** Para láseres pulsados, en Joules (J) o milijoules (mJ).  
* **Modo de Operación:**  
  * **Onda Continua (CW):** Emisión láser constante en el tiempo.  
  * **Pulsado:** Emisión en forma de pulsos cortos. Se caracteriza por:  
    * **Duración del Pulso (τ):** Tiempo que dura un pulso individual (ej. nanosegundos, ns; microsegundos, µs).  
    * **Frecuencia de Repetición de Pulsos (f) o Tasa de Repetición de Pulsos (PRF):** Número de pulsos por segundo (Hz).  
* **Diámetro del Haz (a):** Diámetro del haz láser en un punto específico, usualmente en la salida del láser (mm o cm).  
* **Divergencia del Haz (Φ):** Ángulo de expansión del haz a medida que se propaga (mrad). Para cálculos de seguridad, a menudo se considera el peor caso (haz colimado o a la salida).

**E. Exposición Máxima Permisible (MPE)**

La MPE es el nivel de radiación láser al cual, bajo condiciones normales, una persona puede estar expuesta sin sufrir efectos adversos para la salud (oculares o cutáneos). Se expresa en unidades de irradiancia (W/cm²) para exposiciones prolongadas o láseres CW, o en unidades de fluencia (J/cm²) para exposiciones pulsadas. Los valores de MPE son definidos por estándares como ANSI Z136.1 \[3\] y dependen de:

* Longitud de onda (λ).  
* Duración de la exposición (para CW) o duración del pulso (para láseres pulsados).  
* Si la exposición es única o repetitiva (para láseres pulsados).  
* El tejido expuesto (ojo o piel).

Para esta práctica, **se proporcionarán los valores de MPE relevantes** o se referirá a tablas simplificadas. No se requerirá el cálculo de MPE a partir de las complejas tablas de los estándares. (Ver Apéndice A para tablas de MPE simplificadas de referencia).

**F. Densidad Óptica (OD)**

La Densidad Óptica es una medida logarítmica de la atenuación de la luz proporcionada por un filtro óptico, como el material de unas gafas de protección láser. Es el parámetro fundamental para seleccionar EPO \[1, 6\]. Se define como:

OD=log10​(HMPE​H0​​)

Donde:

* H0​ es la exposición potencial (irradiancia o fluencia) a la que se espera estar expuesto en la córnea sin protección.  
* HMPE​ es la Exposición Máxima Permisible para la longitud de onda y condiciones de exposición dadas.

Una OD de N significa que el filtro atenúa la luz incidente por un factor de 10N. Por ejemplo, OD 4 significa que solo 1/10,000 de la luz incidente atraviesa el filtro. **La OD de las gafas debe ser específica para la longitud o longitudes de onda del láser contra el que se protege.**

**G. Transmisión de Luz Visible (VLT)**

La VLT es el porcentaje de luz visible (típicamente en el rango de 400-700 nm) que atraviesa las gafas de protección láser. Una VLT alta permite una mejor visión del entorno, lo cual es importante para la seguridad general en el laboratorio. Sin embargo, a menudo existe un compromiso: filtros con OD muy alta para ciertas longitudes de onda pueden tener una VLT baja \[6\].

**H. Normativas de Seguridad Láser**

* **ANSI Z136.1 (EE.UU.):** "American National Standard for Safe Use of Lasers" \[3\]. Es el estándar de referencia en EE.UU. y ampliamente reconocido internacionalmente.  
* **IEC 60825-1 (Internacional):** "Safety of laser products – Part 1: Equipment classification and requirements" \[4\]. Estándar global para la seguridad de productos láser.  
* **NOM (México):** En México, normativas como la NOM-013-STPS (relativa a radiaciones no ionizantes) pueden ser aplicables, y es importante verificar la legislación local vigente.

**IV. Metodología Experimental y de Análisis de Datos**

**A. Descripción del Entorno de Trabajo con Láser (Conceptual)**

Un laboratorio aeronáutico que utiliza láseres de Clase 3B o Clase 4 debe implementar estrictas medidas de control. Estas incluyen:

* **Área de Control Láser (LCA):** Una zona designada donde se opera el láser, con acceso restringido.  
* **Señalización:** Advertencias claras y visibles en las entradas de la LCA, indicando el tipo de láser, clase, y si el EPO es requerido.  
* **Oficial de Seguridad Láser (LSO):** Persona responsable de supervisar y hacer cumplir el programa de seguridad láser.  
* **Procedimientos Operativos Estándar (SOPs):** Instrucciones detalladas para la operación segura de cada sistema láser.  
* **Controles de Ingeniería:** Carcasas protectoras, barreras, cortinas láser, interlocks en las puertas de la LCA que desactivan el láser si se abren inesperadamente.  
* **Controles Administrativos:** Entrenamiento del personal, limitación del acceso, supervisión.  
* **Equipo de Protección Personal (EPP):** Principalmente gafas de protección láser (EPO) adecuadas para la radiación emitida.

**Nota de Transición:** Para esta práctica, no se operarán láseres físicamente. En su lugar, se analizarán escenarios y datos técnicos para determinar los requisitos de protección ocular, simulando una parte crucial de la planificación de seguridad en un entorno de investigación aeronáutica.

**B. Descripción de los Datos y Escenarios Proporcionados**

Se analizarán los siguientes escenarios con fuentes láser teóricas y un conjunto de gafas de protección ocular disponibles:

**Fuentes Láser Teóricas (Escenarios Aeronáuticos):**

1. **Láser 1: Sistema PIV (Velocimetría por Imagen de Partículas)**  
   * Tipo: Nd:YAG, doble pulso, frecuencia duplicada.  
   * Longitud de Onda (λ): 532 nm (verde)  
   * Energía por Pulso (E): 200 mJ/pulso  
   * Duración del Pulso (τ): 8 ns  
   * Frecuencia de Repetición (f): 15 Hz  
   * Diámetro del Haz en salida (a): 5 mm  
   * Clase: 4 (Asumida)  
2. **Láser 2: Sistema LDA (Anemometría Láser Doppler)**  
   * Tipo: Argón Ion, onda continua (CW).  
   * Longitud de Onda (λ): 488 nm (azul) y 514.5 nm (verde) \- *Analizar para la línea más potente o la que requiera mayor protección si se usan ambas simultáneamente y las gafas deben cubrir ambas.* Para este ejercicio, se analizará para **514.5 nm**.  
   * Potencia (P): 1.5 W (en la línea de 514.5 nm)  
   * Modo: CW  
   * Diámetro del Haz en salida (a): 1.2 mm  
   * Clase: 4 (Asumida)  
3. **Láser 3: Sistema de Alineación y Metrología**  
   * Tipo: Diodo láser, onda continua (CW).  
   * Longitud de Onda (λ): 635 nm (rojo)  
   * Potencia (P): 4.5 mW  
   * Modo: CW  
   * Diámetro del Haz en salida (a): 3 mm  
   * Clase: 3R (Asumida)

**Gafas de Protección Ocular (EPO) Disponibles (Hipotéticas):**

| ID Gafa | Color Lente | OD Especificada | VLT (%) |
| :---- | :---- | :---- | :---- |
| EPO-001 | Naranja | OD 5+ @ 190-540 nm | 40% |
| EPO-002 | Verde | OD 2+ @ 630-670 nm; OD 4+ @ 800-1100 nm | 55% |
| EPO-003 | Azul Cobalto | OD 7+ @ 190-400 nm; OD 6+ @ 532 nm; OD 5+ @ 1064nm | 15% |
| EPO-004 | Transparente | OD 2+ @ 10600 nm (Láser CO2​) | 90% |
| EPO-005 | Multi-λ | OD 7 @ 190-534 nm; OD 6 @ 800-820nm; OD 5 @ 821-900nm; OD 6 @ 900-1070nm | 30% |

**Valores de Exposición Máxima Permisible (MPE) de Referencia (para exposición ocular):**

* Para Láser 1 (532 nm, 8 ns, pulsado): HMPE​=5.0×10−7J/cm2 (para un solo pulso, considerar el peor caso).  
* Para Láser 2 (514.5 nm, CW): HMPE​=2.5×10−3W/cm2 (para exposición de 0.25 s, reflejo de aversión). Para exposiciones más largas (intencionales o accidentales \> 0.25s), la MPE sería menor. Para este ejercicio, si se considera una exposición accidental prolongada (ej. 10s), asumir HMPE​=1.0×10−3W/cm2. **Usar la MPE de 0.25s para el cálculo inicial.**  
* Para Láser 3 (635 nm, CW): HMPE​=2.5×10−3W/cm2 (para exposición de 0.25 s).

**C. Procedimiento de Análisis de Riesgos y Selección de EPO**

Paso 1: Caracterización de Fuentes Láser y Riesgos.  
Para cada uno de los tres láseres descritos:

* Anote la longitud de onda (λ), potencia (P) o energía por pulso (E), modo de operación (CW o pulsado), duración del pulso (τ) si aplica, y frecuencia de repetición (f) si aplica.  
* Confirme la Clase de seguridad asumida y reflexione sobre los peligros asociados (directa, especular, difusa).

Paso 2: Determinación de la Exposición Potencial (H0​) en la Córnea.  
Calcule la irradiancia (para láseres CW) o la fluencia (para láseres pulsados) en el peor caso (visión directa del haz sin divergir, en la salida del láser).

* Área del haz (A): A=π(2a​)2=4πa2​ (asegúrese de que 'a' esté en cm para que el área esté en cm2).  
* Para Láseres CW (Láser 2 y 3):  
  H0​(Irradiancia)=AP​\[W/cm2\]  
* Para Láseres Pulsados (Láser 1):  
  H0​(Fluencia por pulso)=AE​\[J/cm2\]  
  * **Ejemplo de cálculo para Láser 1 (PIV):**  
    * a=5 mm=0.5 cm  
    * A=4π(0.5 cm)2​=4π×0.25 cm2​≈0.196 cm2  
    * E=200 mJ=0.2 J  
    * H0​=0.196 cm20.2 J​≈1.02 J/cm2

Paso 3: Cálculo de la Densidad Óptica (OD) Requerida.  
Utilice la fórmula ODreq​=log10​(HMPE​H0​​) para cada láser, utilizando la H0​ calculada y la HMPE​ proporcionada para la longitud de onda y condiciones correspondientes.

* **Ejemplo de cálculo para Láser 1 (PIV):**  
  * H0​≈1.02 J/cm2  
  * HMPE​=5.0×10−7 J/cm2  
  * ODreq​=log10​(5.0×10−71.02​)=log10​(2.04×106)≈6.31  
  * Por lo tanto, se requeriría una gafa con OD≥6.31 a 532 nm.

Herramienta de Cálculo de OD (Script Python):  
Para verificar sus cálculos manuales o explorar cómo la OD requerida cambia con diferentes parámetros de entrada, puede utilizar el siguiente script en un entorno Python. Este script define una función calcular\_od y luego la aplica a los escenarios láser de esta práctica.  
import math

def calcular\_od(H0, HMPE):  
  """  
  Calcula la Densidad Óptica (OD) requerida.

  Argumentos:  
    H0 (float): Exposición potencial (Irradiancia en W/cm^2 o Fluencia en J/cm^2).  
    HMPE (float): Exposición Máxima Permisible (en las mismas unidades que H0).

  Retorna:  
    float: La Densidad Óptica requerida, o un mensaje de error si HMPE es cero o negativo.  
           Retorna 0 si la exposición ya es segura (H0 \< HMPE).  
  """  
  if HMPE \<= 0:  
    return "Error: HMPE debe ser un valor positivo."  
  if H0 \< HMPE: \# Si la exposición ya es segura, no se requiere atenuación adicional.  
    return 0   
    
  \# Calcula la OD usando la fórmula estándar.  
  od\_requerida \= math.log10(H0 / HMPE)  
  return od\_requerida

\# \--- Aplicación a los Láseres de la Práctica \---

\# Láser 1 (PIV)  
a\_laser1\_cm \= 0.5  \# Diámetro del haz en cm  
A\_laser1\_cm2 \= math.pi \* (a\_laser1\_cm / 2)\*\*2  
E\_laser1\_J \= 0.2   \# Energía por pulso en J  
H0\_laser1 \= E\_laser1\_J / A\_laser1\_cm2  
HMPE\_laser1 \= 5.0e-7 \# J/cm^2  
od\_req\_laser1 \= calcular\_od(H0\_laser1, HMPE\_laser1)

print(f"--- Resultados para Láser 1 (PIV) \---")  
print(f"  Diámetro del haz (a): {a\_laser1\_cm\*10:.1f} mm")  
print(f"  Área del haz (A): {A\_laser1\_cm2:.4f} cm^2")  
print(f"  Energía por Pulso (E): {E\_laser1\_J:.3f} J")  
print(f"  Exposición Potencial (H0): {H0\_laser1:.2e} J/cm^2")  
print(f"  Exposición Máxima Permisible (HMPE): {HMPE\_laser1:.2e} J/cm^2")  
print(f"  Densidad Óptica Requerida (OD\_req): {od\_req\_laser1:.2f}")

\# Láser 2 (LDA) \- MPE para 0.25s  
a\_laser2\_cm \= 0.12 \# Diámetro del haz en cm  
A\_laser2\_cm2 \= math.pi \* (a\_laser2\_cm / 2)\*\*2  
P\_laser2\_W \= 1.5   \# Potencia en W  
H0\_laser2 \= P\_laser2\_W / A\_laser2\_cm2  
HMPE\_laser2\_0\_25s \= 2.5e-3 \# W/cm^2 (para 0.25s)  
od\_req\_laser2\_0\_25s \= calcular\_od(H0\_laser2, HMPE\_laser2\_0\_25s)

print(f"\\n--- Resultados para Láser 2 (LDA) \[MPE @ 0.25s\] \---")  
print(f"  Diámetro del haz (a): {a\_laser2\_cm\*10:.1f} mm")  
print(f"  Área del haz (A): {A\_laser2\_cm2:.4f} cm^2")  
print(f"  Potencia (P): {P\_laser2\_W:.2f} W")  
print(f"  Exposición Potencial (H0): {H0\_laser2:.2e} W/cm^2")  
print(f"  Exposición Máxima Permisible (HMPE @ 0.25s): {HMPE\_laser2\_0\_25s:.2e} W/cm^2")  
print(f"  Densidad Óptica Requerida (OD\_req @ 0.25s): {od\_req\_laser2\_0\_25s:.2f}")

\# Láser 3 (Alineación)  
a\_laser3\_cm \= 0.3 \# Diámetro del haz en cm  
A\_laser3\_cm2 \= math.pi \* (a\_laser3\_cm / 2)\*\*2  
P\_laser3\_W \= 4.5e-3 \# Potencia en W (4.5 mW)  
H0\_laser3 \= P\_laser3\_W / A\_laser3\_cm2  
HMPE\_laser3 \= 2.5e-3 \# W/cm^2 (para 0.25s)  
od\_req\_laser3 \= calcular\_od(H0\_laser3, HMPE\_laser3)

print(f"\\n--- Resultados para Láser 3 (Alineación) \---")  
print(f"  Diámetro del haz (a): {a\_laser3\_cm\*10:.1f} mm")  
print(f"  Área del haz (A): {A\_laser3\_cm2:.4f} cm^2")  
print(f"  Potencia (P): {P\_laser3\_W:.3e} W") \# Usar notación científica para mW  
print(f"  Exposición Potencial (H0): {H0\_laser3:.2e} W/cm^2")  
print(f"  Exposición Máxima Permisible (HMPE): {HMPE\_laser3:.2e} W/cm^2")  
print(f"  Densidad Óptica Requerida (OD\_req): {od\_req\_laser3:.2f}")

\# Ejemplo para un caso donde no se requiere OD (verificación de la lógica de la función)  
H0\_seguro \= 1e-4 \# W/cm^2  
HMPE\_visible\_cw \= 2.5e-3 \# W/cm^2  
od\_req\_seguro \= calcular\_od(H0\_seguro, HMPE\_visible\_cw)  
print(f"\\n--- Verificación de Escenario Seguro \---")  
print(f"  Exposición Potencial (H0): {H0\_seguro:.2e} W/cm^2")  
print(f"  Exposición Máxima Permisible (HMPE): {HMPE\_visible\_cw:.2e} W/cm^2")  
print(f"  Densidad Óptica Requerida (OD\_req): {od\_req\_seguro:.2f}")

**Paso 4: Selección y Justificación de EPO.**

* Para cada láser, revise la tabla de "Gafas de Protección Ocular (EPO) Disponibles".  
* Compare la ODreq​ calculada con la OD especificada por cada gafa **a la longitud de onda (λ) del láser en cuestión**.  
* La OD especificada por las gafas **DEBE SER MAYOR O IGUAL** a la ODreq​.  
* Si varias gafas cumplen el requisito de OD, considere la VLT. Generalmente, se prefiere la gafa que cumple con la OD y tiene la VLT más alta para una mejor visibilidad del entorno.  
* Justifique detalladamente su selección para cada escenario láser, explicando por qué las gafas elegidas son adecuadas y por qué otras no lo son.

**V. Actividades a Realizar y Resultados Esperados**

1. **Cálculos Completos:** Realice los cálculos de Área del haz (A), Exposición Potencial (H0​), y Densidad Óptica Requerida (ODreq​) para los tres escenarios láser (Láser 1, Láser 2, Láser 3), utilizando los valores de MPE para 0.25s en los láseres CW. Presente estos cálculos de forma clara y ordenada en su reporte. Puede usar el script Python proporcionado para verificar sus resultados.  
2. Tabla Resumen de Requisitos: Complete la siguiente tabla con sus resultados:  
   | Láser (Aplicación) | λ (nm) | H0​ (unidad) | HMPE​ (unidad) | ODreq​ (calculada) |  
   | :--------------------- | :----- | :-------------------- | :-------------------- | :--------------------- |  
   | Láser 1 (PIV) | 532 | (valor) J/cm2 | (valor) J/cm2 | (valor) |  
   | Láser 2 (LDA) | 514.5 | (valor) W/cm2 | (valor) W/cm2 | (valor) |  
   | Láser 3 (Alineación) | 635 | (valor) W/cm2 | (valor) W/cm2 | (valor) |  
3. **Selección de EPO:** Para cada escenario láser, seleccione la(s) gafa(s) más adecuada(s) de la lista EPO-001 a EPO-005. Justifique su elección basándose en la OD requerida, la OD ofrecida por la gafa a la λ específica, y la VLT. Indique si ninguna gafa es adecuada.  
4. **Análisis del Caso de Estudio (ver sección VI).**

**VI. Caso de Estudio: Sistema LDA de Alta Potencia con Exposición Prolongada**

Considere el **Láser 2 (Sistema LDA)**, pero ahora suponga que se requiere trabajar cerca del haz durante periodos prolongados donde el reflejo de aversión de 0.25s no es una suposición conservadora. Para una exposición accidental prolongada (ej. \>10 segundos) a 514.5 nm, la HMPE​ se reduce a 1.0×10−3W/cm2.

* **Tarea 1:** Recalcule la ODreq​ para el Láser 2 utilizando esta HMPE​ más restrictiva (1.0×10−3W/cm2).  
* **Tarea 2:** Evalúe nuevamente las gafas EPO-001 a EPO-005 contra este nuevo requisito de OD para el Láser 2\. ¿Cambia su selección? Justifique.  
* **Tarea 3:** Considerando la VLT, ¿cuál sería la mejor opción de gafa si varias cumplen el nuevo requisito de OD?  
* **Tarea 4:** Además de las gafas, mencione al menos tres medidas de control de ingeniería y dos medidas de control administrativo que serían cruciales para operar de forma segura este sistema LDA de Clase 4 en un laboratorio aeronáutico.

**VII. Discusión en el Contexto Aeronáutico**

* Analice la importancia de realizar estos cálculos de OD en entornos de investigación y desarrollo aeronáutico donde se utilizan láseres para LDA, PIV, holografía, fabricación aditiva con láser, etc.  
* Discuta las posibles consecuencias de una selección incorrecta de EPO o de no utilizar EPO cuando es necesario.  
* Reflexione sobre cómo factores como la presencia de múltiples longitudes de onda láser en un mismo experimento podrían complicar la selección de EPO.  
* Comente las limitaciones de este análisis (ej. no se consideran reflexiones especulares o difusas que podrían alterar la H0​, ni la divergencia del haz a distancia).  
* ¿Cómo influye la cultura de seguridad y el entrenamiento del personal en la prevención de accidentes láser en la industria aeroespacial?

**VIII. Cuestionario / Preguntas de Reflexión**

1. ¿Qué significa la marca "OD 5+ @ 190-540 nm" en unas gafas de seguridad láser? ¿En cuánto reduce la intensidad de la luz a 532 nm? ¿Y a 600 nm?  
2. ¿Por qué un láser Clase 4 es peligroso incluso si solo se observa una reflexión difusa en una superficie mate, mientras que para un Clase 3B la reflexión difusa suele ser segura?  
3. Mencione al menos tres parámetros esenciales que necesita conocer sobre una fuente láser antes de poder calcular la ODreq​ para protección ocular.  
4. Si dos pares de gafas ofrecen la misma protección OD adecuada para el láser que está utilizando, pero una tiene una VLT del 45% y la otra del 20%, ¿cuál elegiría generalmente y por qué? ¿Existe alguna situación en la que preferiría la de menor VLT?  
5. ¿Es seguro usar gafas diseñadas exclusivamente para un láser Nd:YAG (1064 nm, IR) para protegerse de un láser verde de frecuencia duplicada (532 nm) proveniente del mismo sistema Nd:YAG? Explique su razonamiento.  
6. ¿Qué es un Oficial de Seguridad Láser (LSO) y cuál es su rol en un entorno aeronáutico que utiliza láseres?

**IX. Requisitos del Reporte (Formato AIAA \- Adaptado)**

El reporte de esta práctica debe ser un documento técnico conciso y bien estructurado. Se sugiere el siguiente formato (adaptado del estilo AIAA):

* **Título, Autores, Afiliación (Universidad, Facultad), Fecha.**  
* **Abstract (Resumen):** (Máx. 250 palabras) Breve descripción de los objetivos de la práctica (evaluación de riesgos láser, cálculo de OD, selección de EPO), la metodología de análisis de datos utilizada (escenarios láser, parámetros de EPO, MPEs de referencia), los resultados clave (ODs requeridas calculadas para cada escenario, gafas seleccionadas con justificación) y la conclusión principal (importancia de la selección informada de EPO para la seguridad láser en aplicaciones aeronáuticas).  
* **Nomenclatura (Opcional):** Lista de acrónimos y símbolos utilizados (ej. LDA, PIV, MPE, OD, VLT, LSO, CW, λ, P, E, H₀, etc.).  
* **1\. Introducción:** Relevancia de la seguridad láser en el contexto de la ingeniería y técnicas de medida aeronáuticas. Objetivos específicos de la práctica de análisis.  
* **2\. Marco Teórico:** Resumen conciso de los principios de radiación láser, clasificación de seguridad, interacción láser-ojo, y los parámetros clave para la protección (MPE, OD, VLT). Citar las normativas relevantes.  
* **3\. Metodología de Análisis:**  
  * Descripción de los escenarios láser teóricos analizados (parámetros de cada láser).  
  * Presentación de la tabla de gafas de protección ocular (EPO) consideradas.  
  * Mención de los valores de MPE de referencia utilizados y su fuente (en este caso, "proporcionados por la práctica").  
  * Detalle del procedimiento de cálculo para H0​ y ODreq​ (incluyendo las fórmulas generales).  
* **4\. Resultados:**  
  * Presentación clara de los cálculos de A, H0​, y ODreq​ para cada escenario láser.  
  * Tabla resumen de requisitos (ver sección V.2).  
  * Selección justificada de EPO para cada escenario, discutiendo la adecuación de la OD y la VLT.  
  * Resultados del Caso de Estudio (Tarea 1 a Tarea 4), incluyendo cálculos y justificaciones.  
* **5\. Discusión:**  
  * Análisis de la importancia de cada parámetro láser (λ, P/E, modo) en la determinación del riesgo y la OD requerida.  
  * Discusión sobre la correcta interpretación de las especificaciones de las gafas (OD por rango de λ, VLT).  
  * Impacto de la VLT en la seguridad y usabilidad en un laboratorio.  
  * Relación con aplicaciones aeronáuticas específicas y la necesidad de controles rigurosos.  
  * Limitaciones del análisis realizado (simplificaciones, factores no considerados).  
  * Reflexión sobre la importancia de los estándares (ANSI/IEC) y los procedimientos locales de seguridad.  
* **6\. Conclusiones:** Resumen de los hallazgos clave sobre la evaluación de riesgos y la selección de EPO para los láseres analizados. Reafirmar la importancia crítica de la seguridad láser y el análisis cuantitativo para garantizarla.  
* **7\. Referencias:** Citar cualquier fuente consultada para MPEs (ej. si se usara una tabla de un libro específico), calculadoras online (si se permitiera su uso para verificación), datasheets (si se usaran hipotéticos), o recursos generales de seguridad láser. (Para esta práctica, principalmente las referencias sugeridas abajo).  
* **Apéndices (Opcional):** Si se realizan cálculos muy extensos o se consulta información adicional detallada (ej. tabla de MPE más completa).

**X. Referencias Sugeridas**

1. Sliney, D., & Wolbarsht, M. (1980). *Safety with Lasers and Other Optical Sources: A Comprehensive Handbook*. Plenum Press.  
2. Hecht, E. (2017). *Optics* (5th ed.). Pearson Education.  
3. ANSI Z136.1 \- *American National Standard for Safe Use of Lasers*. Laser Institute of America. (Consultar la edición más reciente).  
4. IEC 60825-1 \- *Safety of laser products – Part 1: Equipment classification and requirements*. International Electrotechnical Commission. (Consultar la edición más reciente).  
5. Niemz, M. H. (2007). *Laser-Tissue Interactions: Fundamentals and Applications* (3rd ed.). Springer.  
6. Laser Institute of America (LIA). *LIA Guide to Laser Safety*. (Consultar la edición más reciente y otros recursos educativos en www.lia.org).  
7. OSHA (Occupational Safety and Health Administration). *Technical Manual (OTM) \- Section III: Chapter 6 \- Laser Hazards*. Recuperado de www.osha.gov.  
8. Rockwell Laser Industries. *Laser Safety Training Materials and Resources*. Recuperado de [www.rli.com](https://www.rli.com). (Ejemplo de proveedor de capacitación y recursos).

Apéndice A: Tablas de MPE Simplificadas (Ejemplos Ilustrativos)  
(Nota: Estas son tablas muy simplificadas y solo para fines ilustrativos de esta práctica. Para aplicaciones reales, siempre se debe consultar la versión completa y actualizada de los estándares ANSI Z136.1 o IEC 60825-1).  
Tabla A1: MPE para Exposición Ocular a Láseres CW y Exposiciones Prolongadas (Adaptado de ANSI Z136.1)  
| Rango de Longitud de Onda (nm) | Duración Exposición (s) | MPE (W/cm²) | Notas |  
| :----------------------------- | :---------------------- | :------------------ | :-------------------------------------- |  
| 400 \- 700 (Visible) | 0.25 (reflejo aversión) | 2.5×10−3 | |  
| 400 \- 700 (Visible) | 10 | 1.0×10−3 | Para visión accidental prolongada |  
| 700 \- 1050 (IR-A) | 10 | CA​×10−3 | CA​=100.002(λ−700); ej. a 1000nm CA​≈4 |  
| 1050 \- 1400 (IR-A) | 10 | 5.0×10−3 | |  
Tabla A2: MPE para Exposición Ocular a Láseres Pulsados Únicos (Adaptado de ANSI Z136.1)  
| Rango de Longitud de Onda (nm) | Duración Pulso (s) | MPE (J/cm²) | Notas |  
| :----------------------------- | :-------------------- | :------------------ | :-------------------------------------- |  
| 400 \- 700 (Visible) | 10−9 a 10−7 | 5.0×10−7 | |  
| 700 \- 1050 (IR-A) | 10−9 a 10−7 | CA​×5.0×10−7 | CA​=100.002(λ−700) |  
| 1050 \- 1400 (IR-A) | 10−9 a 10−7 | 5.0×10−6 | |  
*(Fin del Apéndice A)*