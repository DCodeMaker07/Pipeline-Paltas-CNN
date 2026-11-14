# Pipeline: CNN multietiqueta para roña y antracnosis en palta

Clasifica frutos de palta en tres casos: sano, roña (R), antracnosis (A). La salida es
multietiqueta: cada fruto tiene dos probabilidades independientes, así que la co-infección
(ambas enfermedades) se puede representar. El dataset actual no tiene co-infecciones.

## 1. Preparar el entorno (una sola vez por persona)

Requisito: **Python 3.12** instalado en Windows (la 3.12.10 con instalador).
Verifiquen con `py --list`; debe aparecer `-V:3.12`.

1. Abran PowerShell en la carpeta `pipeline-palta`.
2. Si Windows bloquea scripts, ejecuten una vez:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
3. Creen el entorno:
   `.\crear_entorno.ps1`
4. Debe terminar con `OK torch 2.15.0`.

El entorno queda en `.venv` y no se sube al repositorio.

## 2. Colocar el dataset

Dentro de `pipeline-palta` debe existir la carpeta `dataset` con las 669 imágenes.
El prefijo del nombre define la clase: `A.*` antracnosis, `R.*` roña, `S.*` sano.
La carpeta `dataset` no se sube al repositorio: se comparte por fuera del equipo.

## 3. Ejecutar

Prueba corta (verifica que todo funciona, unos minutos):

    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo simple --epochs 2

Validación cruzada de 5 folds (la corrida real):

    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt adam --epochs 15

Opciones de `--opt`: `sgd`, `adam`, `adamw`.

## 4. Comparar optimizadores (reparto del equipo)

Cada integrante corre **un** optimizador con el mismo comando, cambiando solo `--opt`:

    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt sgd --epochs 15
    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt adam --epochs 15
    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt adamw --epochs 15

Reglas para que la comparación sea válida:

- Todos usan la **misma versión** de `pipeline.py` y el **mismo dataset**.
- Todos usan **15 épocas**, sin cambiarlas.
- No cambien `SEED`, la partición ni `--lr`.

## 5. Qué compartir

Cada corrida crea en `resultados\`:

- `resumen_<opt>_<modo>.json`: métricas por fold, media y desviación estándar, y tiempo total.
- `historial_<opt>_fold<N>.csv`: pérdida de entrenamiento y validación por época.

Compartan **los dos tipos de archivo** de su corrida (no las imágenes) en la carpeta del equipo,
junto con la consola o las últimas líneas de la salida.

## Problemas frecuentes

- **`py` no se reconoce:** reinicien PowerShell; si sigue, reinstalen Python 3.12 marcando "Add python.exe to PATH".
- **Error al instalar `pyyaml` con Visual C++:** usaron una versión de Python distinta a 3.12; creen el entorno de nuevo.
- **`No se encontró dataset` o 0 imágenes:** revisen que la carpeta se llame `dataset` y que las imágenes empiecen con `A.`, `R.` o `S.`.
- **Muy lento:** es normal en CPU. Dejen la corrida terminar; el tiempo queda en `segundos` del JSON.

## Advertencia metodológica

Los nombres de archivo no tienen ID de fruto, así que la partición es por imagen y puede haber fuga
de datos entre entrenamiento y validación. Lo indica también la diapositiva 7 del avance.
