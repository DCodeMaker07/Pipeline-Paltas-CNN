# Pipeline: CNN multietiqueta para roña y antracnosis en palta

Clasifica frutos de palta en cuatro casos: sano (0,0), solo antracnosis (1,0), solo roña (0,1)
y co-infección (1,1). La salida tiene dos sigmoides independientes, así que una imagen puede tener
las dos enfermedades. El dataset actual no tiene co-infecciones, así que ese caso no se puede evaluar.

## 0. Modelo final

| Elemento | Valor |
|---|---|
| Bloques convolucionales | 5: Conv 3×3 → BatchNorm → GELU → MaxPool 2 |
| Filtros | 32, 64, 128, 256 y 512 (se duplican por bloque) |
| Clasificador | GlobalAvgPool → Dense 512→64 → GELU → Dropout 0.3 → Dense 64→2 (sigmoide) |
| Entrenamiento | 535 imágenes (train + validación), AdamW, weight decay 0, tasa 1.2349e-4, lote 16 |
| Aumento de datos | volteos horizontales y verticales, brillo y contraste ±10% |
| Épocas y tasa | 20 épocas, tasa con decaimiento coseno |

La configuración sale de dos búsquedas y dos análisis:
- `experimentos.py buscar` eligió la configuración base (4 bloques) con validación.
- La profundidad, las mejoras y la validación cruzada multi-etiqueta (`profundidad_cv.py`,
  `mejoras_rona.py`, `multi_cv.py`) justificaron 5 bloques y el aumento de datos.

La prueba final se evalúa **una sola vez** con `prueba_final.py`. Ya existe un resultado; no
vuelvan a ejecutarlo para elegir nada.

## 1. Preparar el entorno (una sola vez por persona)

Requisito: Python 3.12 instalado en Windows. Verifiquen con `py --list`; debe aparecer `-V:3.12`.

1. Abran PowerShell en la carpeta `pipeline-palta`.
2. Si Windows bloquea scripts, ejecuten una vez:
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
3. Creen el entorno:
   `.\crear_entorno.ps1`
4. Si tienen GPU NVIDIA, reemplacen torch por la versión con CUDA (ver sección 1.1).
   Si no, el entorno ya funciona en CPU.

El entorno queda en `.venv` y no se sube al repositorio.

### 1.1 Usar la GPU (NVIDIA)

Primero verifiquen que el driver ve la tarjeta: `nvidia-smi`.

    .\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
    .\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu130
    .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"

Si la última línea dice `True`, los scripts usan la GPU sola (el código elige `cuda` cuando está
disponible). Si el comando de instalación falla por el driver, revisen en pytorch.org qué índice `cu...` corresponde a su versión de driver.

**Importante:** los resultados en GPU pueden diferir ligeramente de los de CPU, en las últimas
decimales, y el entrenamiento en GPU no es completamente reproducible. Para las tablas que se
comparan entre sí, todo el equipo debe usar el mismo dispositivo y la misma variante.

**Si corren en CPU:** el entrenamiento tarda bastante más, sobre todo `multi_cv.py` y `prueba_final.py`.
Las cifras pueden no coincidir exactamente con las de la GPU. Eso no cambia las conclusiones,
pero indíquenlo al reportar el dispositivo que usaron.

## 2. Colocar el dataset

Dentro de `pipeline-palta` debe existir la carpeta `dataset` con las 669 imágenes.
El prefijo del nombre define la clase: `A.*` antracnosis, `R.*` roña, `S.*` sano.
La carpeta `dataset` no se sube al repositorio: se comparte por fuera del equipo.

## 3. Scripts del proyecto

| Archivo | Para qué sirve |
|---|---|
| `entrenamiento.py` | Configuración final, aumento de datos y bucle de entrenamiento. Lo usan los demás scripts de la CNN final. |
| `experimentos.py` | Búsqueda aleatoria (`buscar`), sensibilidad, arquitecturas, ablación, `final` y complejidad. |
| `pipeline.py` | Entrenamiento con validación cruzada para comparar optimizadores. |
| `profundidad_cv.py` | Número de bloques, con validación cruzada, para roña o antracnosis. |
| `mejoras_rona.py` | Compara las mejoras (aumento, 20 épocas, coseno) en una sola salida. |
| `multi_cv.py` | Validación cruzada multi-etiqueta: variantes base y mejorada, y los modelos preentrenados. |
| `comparar_rona.py` | Compara la CNN con MobileNetV2 y ResNet-18 para una enfermedad. |
| `comparar_multi.py` | Compara los tres modelos multi-etiqueta con la variante mejorada. |
| `prueba_final.py` | Entrena la CNN final con train + validación y evalúa una vez en prueba. |
| `graficos_profundidad.py` | Gráficas de profundidad y comparaciones pareadas. |
| `graficos_optimizadores.py` | Gráficas de la comparación de optimizadores. |
| `bitacora.py` | Encabezados y tiempos de ejecución que se muestran en consola y se guardan en un `.txt`. |

Los scripts de entrenamiento aceptan `--pregunta N` para que la consola muestre `Pregunta N°…`. `comparar_multi.py` y `graficos_*.py` no lo aceptan.

## 4. Ejecutar

Prueba corta (unos minutos, para verificar que todo funciona):

    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo simple --epochs 2

Validación cruzada de optimizadores (cada integrante corre **uno**, cambiando solo `--opt`):

    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt sgd --epochs 15
    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt adam --epochs 15
    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt adamw --epochs 15

Reglas para que la comparación sea válida:
- Todos usan la misma versión de `pipeline.py`, el mismo dataset y el mismo dispositivo.
- Todos usan 15 épocas y no cambian `SEED`, la partición, `--lr`, `--batch` ni `--wd`.

Validación cruzada de la CNN final y comparaciones (el tiempo queda en `ejecucion_*.txt`):

    .\.venv\Scripts\python.exe multi_cv.py --variante base --pregunta 7
    .\.venv\Scripts\python.exe multi_cv.py --variante mejorada --pregunta 7
    .\.venv\Scripts\python.exe multi_cv.py --variante mejorada --modelo mobilenet_v2 --pregunta 7
    .\.venv\Scripts\python.exe multi_cv.py --variante mejorada --modelo resnet18 --pregunta 7
    .\.venv\Scripts\python.exe comparar_multi.py

Prueba final (una sola vez en el proyecto):

    .\.venv\Scripts\python.exe prueba_final.py --variante mejorada --pregunta 10

## 5. Qué compartir

Cada corrida deja sus resultados en carpetas que no se suben al repositorio:
- `resultados\`: búsqueda y optimizadores (`resumen_<opt>_<modo>.json`, `historial_*.csv`).
- `resultados_profundidad\`: validación cruzada, comparaciones y gráficas.
- `resultados_prueba_final_mejorada\`: prueba final de la CNN.

Compartan la carpeta de su corrida y el archivo `ejecucion_*.txt` (tiene el tiempo total y el
dispositivo). Si usaron `--out`, compartan esa carpeta.

## 6. Carpetas de respaldo y versiones anteriores

Las carpetas `resultados_*` que no son las de arriba son versiones anteriores o corridas
alternas, y no se usan en el documento final. No las borren sin avisar al equipo.

## Problemas frecuentes

- **`py` no se reconoce:** reinicien PowerShell; si sigue, reinstalen Python 3.12 marcando "Add python.exe to PATH".
- **`torch.cuda.is_available()` da `False` con GPU:** falta la versión con CUDA (sección 1.1) o el driver es antiguo.
- **`No se encontró dataset` o 0 imágenes:** la carpeta debe llamarse `dataset` y las imágenes deben empezar con `A.`, `R.` o `S.`.
- **`Ya existe ... La prueba se evalúa una sola vez`:** es intencional. No se vuelve a evaluar la prueba.
- **Muy lento en CPU:** es normal. En GPU la misma corrida tarda una fracción del tiempo.
- **`SyntaxWarning: invalid escape sequence`:** es solo un aviso sobre texto de ayuda; no afecta los resultados.
