# P01 - Protección ocular frente a radiación láser

Objetivo: seleccionar protección adecuada (OD) según λ/potencia y normativa.

Checklist:
- Identificar láser (λ, CW/pulsado, potencia/energía), trayectorias y reflejos.
- Calcular OD requerido por norma (EN 207/208, ANSI).
- Verificar ancho de banda de atenuación de gafas.
- Matriz de riesgos y controles.

Estructura:
- `data/`: tablas de láseres/gafas (no versionar datos sensibles).
- `src/`: utilidades de cálculo OD.
- `notebooks/`: informe y cálculos.

Notebook principal:
- `notebooks/P01_Proteccion_Laser.ipynb` → incluye:
	- Resumen de objetivos y escenarios de la práctica.
	- Cálculo de H0 y OD requerida para 3 láseres (PIV, LDA, alineación).
	- Verificación automática contra catálogo EPO (OD vs λ) y recomendación.
	- Celda para escenarios personalizados.
