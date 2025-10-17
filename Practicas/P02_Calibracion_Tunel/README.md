# P02 - Calibración del Túnel de Viento

Objetivo: curva V=f(RPM/ajustes) mediante Pitot y correcciones de densidad.

Datos esperados:
- `data/calibracion.csv`: columnas `rpm, dp(Pa), T(C), rho(kgm3 optional)`.

Pasos:
- Convertir dp a V: V = sqrt(2*dp/rho).
- Regresión V=f(rpm), lineal/polynomial bajo orden.
- Validar con instrumento independiente.
