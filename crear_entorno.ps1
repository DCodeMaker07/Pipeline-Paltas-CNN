# Crea un entorno virtual aislado en .venv con Python 3.12 e instala las versiones fijadas.
# Requisito: Python 3.12 instalado (https://www.python.org/downloads/ o: winget install -e --id Python.Python.3.12)
# Ejecutar desde PowerShell en la carpeta pipeline-palta:  .\crear_entorno.ps1
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { Write-Error "No se encontró Python 3.12. Instálalo y vuelve a ejecutar."; exit 1 }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch, sklearn, pandas; print('OK torch', torch.__version__)"
