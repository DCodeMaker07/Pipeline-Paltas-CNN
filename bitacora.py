"""Encabezados y tiempos de ejecución, para seguir el proceso en consola y dejarlo guardado."""
import time
from datetime import datetime
from pathlib import Path


def encabezado(texto, numero=None):
    """Imprime 'Pregunta N°numero: texto' (o 'Pregunta: texto' si no se indica número)."""
    etiqueta = f"Pregunta N°{numero}" if numero is not None else "Pregunta"
    print("\n" + "=" * 70)
    print(f"{etiqueta}: {texto}")
    print("=" * 70, flush=True)


def cerrar(t0, carpeta, nombre, datos):
    """Imprime el resumen con el tiempo total y lo guarda en carpeta\\ejecucion_<nombre>.txt."""
    total = time.time() - t0
    minutos, segundos = divmod(total, 60)
    lineas = [f"Ejecutado: {datetime.now():%Y-%m-%d %H:%M}"]
    lineas += [f"{clave}: {valor}" for clave, valor in datos.items()]
    lineas.append(f"Tiempo total: {int(minutos)} min {segundos:.0f} s")
    texto = "\n".join(lineas)
    print("\n" + texto)
    carpeta = Path(carpeta)
    carpeta.mkdir(exist_ok=True)
    with open(carpeta / f"ejecucion_{nombre}.txt", "w", encoding="utf-8") as fh:
        fh.write(texto + "\n")
