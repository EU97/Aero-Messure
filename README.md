# Aero-Messure
Repository for aero messurments


## Practicas
Consulta el resumen y metodologías en `Practicas/README.md`.

## Quickstart (Windows, PowerShell)

1. Instalar Python 3.10+ desde Microsoft Store o python.org y asegúrate de que `python`/`pip` estén en PATH.
2. Crear entorno virtual e instalar dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

3. Abrir Jupyter y ejecutar los notebooks de las prácticas:

```powershell
jupyter notebook
```

Navega a `Practicas/Pxx_*/notebooks/` y abre el notebook correspondiente.

### Notas
- Los datos crudos van en `Practicas/Pxx_*/data/` y no se versionan por defecto. Añade tus archivos allí.
- Si usas PIV, puede tardar en instalar `openpiv`; si fuese un problema, avísame y ajustamos la dependencia.
