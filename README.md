# Pipeline: CNN multietiqueta para roña y antracnosis en palta

Clasifica frutos de palta en tres casos: sano, roña (R), antracnosis (A). La salida es
multietiqueta: cada fruto tiene dos probabilidades independientes, así que la co-infección
(ambas enfermedades) se puede representar. El dataset actual no tiene co-infecciones.

## 0. Configuración vigente

La arquitectura y los hiperparámetros se eligieron con la búsqueda de `experimentos.py buscar`.
Los valores por defecto de `pipeline.py` ya son esa configuración, así que **no hace falta pasar
`--lr`, `--batch` ni `--wd`**.

| Elemento | Valor |
|---|---|
| Bloques convolucionales | 4: Conv3×3 (pad 1) → BatchNorm → GELU → MaxPool2 |
| Filtros | 32, 64, 128, 256 |
| Clasificador | GlobalAvgPool → Dense 256→64 → GELU → Dropout 0.3 → Dense 64→2 (sigmoide) |
| Inicialización | la por defecto de PyTorch |
| Parámetros entrenables | 405,954 |
| Learning rate (`--lr`) | 0.00012349482735064392 |
| Batch (`--batch`) | 16 |
| Weight decay (`--wd`) | 0 |

Las corridas anteriores con 16 filtros y lr 1e-3 **no son comparables** con estas. Están en
`resultados_v1_16filtros\` solo como referencia.

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

Opciones de `--opt`: `sgd`, `adam`, `adamw`. Para AdamW con decaimiento distinto de cero use
`--wd 0.0001` y, para no sobrescribir resultados, `--out resultados_adamw_wd`.

## 4. Comparar optimizadores (reparto del equipo)

Cada integrante corre **un** optimizador con el mismo comando, cambiando solo `--opt`:

    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt sgd --epochs 15
    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt adam --epochs 15
    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt adamw --epochs 15

Reglas para que la comparación sea válida:

- Todos usan la **misma versión** de `pipeline.py` y el **mismo dataset**.
- Todos usan **15 épocas**, sin cambiarlas.
- No cambien `SEED`, la partición, `--lr`, `--batch` ni `--wd`.

Nota: con `--wd 0` (valor por defecto), AdamW es matemáticamente igual a Adam. Con
`--wd 0.0001` y el learning rate de la configuración, el decaimiento es prácticamente nulo, y los
resultados de Adam y AdamW coincidieron hasta la cuarta cifra decimal. Por eso el informe toma Adam
como representante del optimizador adaptativo.

## 5. Qué compartir

Cada corrida crea en `resultados\`:

- `resumen_<opt>_<modo>.json`: métricas por fold, media y desviación estándar, tiempo total y
  el número de parámetros entrenables.
- `historial_<opt>_fold<N>.csv`: pérdida de entrenamiento y validación por época.

Compartan **los dos tipos de archivo** de su corrida (no las imágenes) en la carpeta del equipo,
junto con la consola o las últimas líneas de la salida. Si usaron `--out` con otra carpeta,
compartan esa carpeta.

## Problemas frecuentes

- **`py` no se reconoce:** reinicien PowerShell; si sigue, reinstalen Python 3.12 marcando "Add python.exe to PATH".
- **Error al instalar `pyyaml` con Visual C++:** usaron una versión de Python distinta a 3.12; creen el entorno de nuevo.
- **`No se encontró dataset` o 0 imágenes:** revisen que la carpeta se llame `dataset` y que las imágenes empiecen con `A.`, `R.` o `S.`.
- **Muy lento:** es normal en CPU. Dejen la corrida terminar; el tiempo queda en `segundos` del JSON.
- **`SyntaxWarning: invalid escape sequence`:** es solo un aviso sobre texto de ayuda; no afecta los resultados.
