"""Gráficas de la prueba de profundidad (validación cruzada) y comparación pareada entre bloques.

Lee resultados_profundidad\\cv_rona.csv y cv_antracnosis.csv (generados por profundidad_cv.py)
y guarda profundidad_rona.png y profundidad_antracnosis.png en la misma carpeta.
"""
import csv
import statistics
import time
from pathlib import Path

import bitacora

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CARPETA = Path("resultados_profundidad")
OBJETIVOS = {"rona": "roña", "antracnosis": "antracnosis"}


def leer(objetivo):
    with open(CARPETA / f"cv_{objetivo}.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def graficar(objetivo, titulo, filas):
    bloques = sorted({int(r["bloques"]) for r in filas})
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, clave, nombre in [(ejes[0], "F1_val", "F1"), (ejes[1], "AUC_val", "AUC")]:
        xs, medias, errores = [], [], []
        for b in bloques:
            vals = [float(r[clave]) for r in filas if int(r["bloques"]) == b]
            xs.append(b)
            medias.append(statistics.mean(vals))
            errores.append(statistics.stdev(vals))
            ax.scatter([b] * len(vals), vals, alpha=0.35, s=18, color="#7aa6c2")
        ax.errorbar(xs, medias, yerr=errores, fmt="o-", color="#1f4e79", capsize=4, lw=2)
        ax.set_xticks(bloques)
        ax.set_xticklabels([f"{b} bloques\n({4 * b} capas conv.)" for b in bloques])
        ax.set_ylabel(nombre)
        ax.set_title(f"{nombre} de {titulo} (validación cruzada)", fontsize=10)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CARPETA / f"profundidad_{objetivo}.png", dpi=200)
    plt.close(fig)


def pareadas(filas, b1, b2, clave):
    """Cuenta en cuántos pares (semilla, fold) el número b2 de bloques supera a b1."""
    valores = {(int(r["bloques"]), int(r["semilla"]), int(r["fold"])): float(r[clave]) for r in filas}
    pares = {(s, f) for (b, s, f) in valores if b == b1 and (b2, s, f) in valores}
    ganadas = sum(valores[(b2, s, f)] > valores[(b1, s, f)] for s, f in pares)
    return ganadas, len(pares)


def main():
    t0 = time.time()
    bitacora.encabezado("Gráficas de la prueba de profundidad y comparaciones pareadas")
    for objetivo, titulo in OBJETIVOS.items():
        filas = leer(objetivo)
        graficar(objetivo, titulo, filas)
        print(f"\n{titulo.capitalize()} (F1 de validación, pares semilla x fold):")
        for b1, b2 in [(3, 4), (4, 5)]:
            g, n = pareadas(filas, b1, b2, "F1_val")
            print(f"  {b2} bloques mejor que {b1} en {g} de {n} pares")
        print(f"  Gráfica guardada: {CARPETA / f'profundidad_{objetivo}.png'}")
    bitacora.cerrar(t0, CARPETA, "graficos_profundidad", {"Objetivos": list(OBJETIVOS)})


if __name__ == "__main__":
    main()
