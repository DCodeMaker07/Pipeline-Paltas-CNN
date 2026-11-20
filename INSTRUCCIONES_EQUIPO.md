# Instrucciones para el equipo: Evaluación 3

Lean primero el README.md (entorno, GPU y dataset). Aquí está lo que hace cada integrante.

## Antes de empezar (todos)
1. Actualicen el código: `git pull` en la rama que indique Deivid.
2. Creen o actualicen el entorno con `.\crear_entorno.ps1`. Si tienen GPU NVIDIA, sigan la sección 1.1 del README.
3. Coloquen la carpeta `dataset` (669 imágenes) dentro de `pipeline-palta`. No viene en el repo.
4. Ejecuten **solo** los comandos de sus preguntas. Usen siempre `--pregunta N` para que la consola muestre el número.
5. Cada video se graba en su propio equipo y empieza mostrando CPU, RAM y GPU (disclaimer de 10 a 15 s). En ese disclaimer digan si la corrida fue en GPU o en CPU. Deivid, Harold y Giovanni tienen GPU; el resto declara el dispositivo que use.
6. Nombren cada video `preg_N_nombre_modelo` (por ejemplo, `preg_4_adam_leslie.mp4`).

## Reglas
- **No ejecuten** `prueba_final.py` ni `experimentos.py final`. El conjunto de prueba ya se usó una vez; volver a evaluarlo lo contamina.
- Para la pregunta 10 usen solo `cargar_final.py`, que carga el modelo guardado y no reentrena.
- Tablas que se comparan entre sí deben correr en el mismo dispositivo.
- No cambien la semilla, la partición, el learning rate ni el tamaño de lote, salvo que la pregunta lo pida.
- Al terminar, compartan con Deivid la carpeta de resultados y el archivo `ejecucion_*.txt` de cada corrida. No suban datasets ni carpetas de resultados al repo.

## Reparto

### Elio David (Deivid): preguntas 1, 2, 3 y 9
- P1 (arquitecturas): `.\.venv\Scripts\python.exe experimentos.py arquitecturas --pregunta 1`
- P2 y P3 (espacio y algoritmo de búsqueda): `.\.venv\Scripts\python.exe experimentos.py buscar --pregunta 3 --n 10`
- P9 (selección automática): la misma búsqueda, más el diagrama de flujo y el pseudocódigo en el documento.

### Leslie: preguntas 4 y 5 (optimizadores y validación)
Un optimizador por corrida, 15 épocas:

    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt sgd --epochs 15 --pregunta 4
    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt adam --epochs 15 --pregunta 4
    .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt adamw --epochs 15 --pregunta 4

Para la pregunta 5 usen las mismas corridas: la tabla de métricas por partición y la media con desviación salen de `resumen_*.json`.

### Angel: preguntas 7 y 8 (comparación y generalización)
Comparación de los tres modelos con la variante mejorada (validación cruzada en entrenamiento y validación):

    .\.venv\Scripts\python.exe multi_cv.py --variante mejorada --pregunta 7
    .\.venv\Scripts\python.exe multi_cv.py --variante mejorada --modelo mobilenet_v2 --pregunta 7
    .\.venv\Scripts\python.exe multi_cv.py --variante mejorada --modelo resnet18 --pregunta 7
    .\.venv\Scripts\python.exe comparar_multi.py

Para la pregunta 8 (con y sin regularización):

    .\.venv\Scripts\python.exe experimentos.py ablacion --pregunta 8

Este comando solo reporta validación (los resultados quedan en `resultados\ablacion.csv`). Usen las filas con `conjunto = validacion`. Las curvas de pérdida salen de los `historial_*.csv` de `pipeline.py`.

### Harold: preguntas 6 y 10 (sensibilidad y evaluación final)
- P6: `.\.venv\Scripts\python.exe experimentos.py sensibilidad --pregunta 6`. Varía bloques, filtros, dropout, learning rate, weight decay y tamaño de lote, y evalúa solo en validación. Resultados en `resultados\sensibilidad.csv` y el rango por hiperparámetro en la consola.
- P10: `.\.venv\Scripts\python.exe cargar_final.py --pregunta 10`. Muestra la carga del modelo y las predicciones sobre prueba. No reentrena.

### Giovanni: preguntas 11 y 12 (costo y recursos)
- P11: `.\.venv\Scripts\python.exe experimentos.py complejidad --pregunta 11`
- P12 en GPU (su PC): `.\.venv\Scripts\python.exe recursos.py --pregunta 12 --semillas 42 7 123 1 2026 --salida resultados_recursos_gpu`
- P12 en CPU (su PC), desde la misma ventana:

      $env:CUDA_VISIBLE_DEVICES="-1"
      .\.venv\Scripts\python.exe recursos.py --pregunta 12 --semillas 42 7 123 1 2026 --salida resultados_recursos_cpu
      Remove-Item Env:CUDA_VISIBLE_DEVICES

Revisen que la segunda línea diga `Dispositivo: cuda` en la corrida de GPU y `Dispositivo: cpu` en la de CPU. Las dos corridas las hace él en su propia PC: no mezclen tablas de otros equipos. P12 es la única pregunta donde el hardware cambia los tiempos y la memoria, así que los tiempos de Giovanni no se comparan con los de los demás integrantes.

## Qué entregar a Deivid
- Carpeta de resultados de cada comando.
- `ejecucion_*.txt` de cada corrida.
- Video nombrado según la regla de arriba.
- Una nota corta con el dispositivo y cualquier error o advertencia que haya aparecido.
