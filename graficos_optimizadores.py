"""Gráfica de convergencia de los optimizadores (Pregunta 4, evidencia 4).

Lee historial_<opt>_fold<N>.csv de la carpeta de resultados y grafica la media de la
pérdida por época, con banda de ± desviación estándar entre folds.

Uso (PowerShell, dentro de pipeline-palta):
  .\.venv\Scripts\python.exe graficos_optimizadores.py --carpeta resultados
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OPTIMIZADORES = ["sgd", "adam", "adamw"]


def leer_historiales(carpeta, opt):
    rutas = sorted(carpeta.glob(f"historial_{opt}_fold*.csv"))
    if not rutas:
        return None
    curvas_tr, curvas_va = [], []
    for r in rutas:
        with open(r, encoding="utf-8") as fh:
            filas = list(csv.DictReader(fh))
        curvas_tr.append([float(f["loss_train"]) for f in filas])
        curvas_va.append([float(f["loss_val"]) for f in filas])
    return np.array(curvas_tr), np.array(curvas_va), len(rutas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", default="resultados")
    ap.add_argument("--salida", default="graficos/convergencia_optimizadores.png")
    args = ap.parse_args()
    carpeta = Path(args.carpeta)

    fig, (ax_tr, ax_va) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for opt in OPTIMIZADORES:
        datos = leer_historiales(carpeta, opt)
        if datos is None:
            print(f"Sin historiales para {opt} en {carpeta}; se omite.")
            continue
        tr, va, n = datos
        epocas = np.arange(1, tr.shape[1] + 1)
        for ax, curvas in [(ax_tr, tr), (ax_va, va)]:
            media = curvas.mean(axis=0)
            desv = curvas.std(axis=0, ddof=1) if n > 1 else np.zeros_like(media)
            ax.plot(epocas, media, label=f"{opt} (n={n})")
            ax.fill_between(epocas, media - desv, media + desv, alpha=0.2)

    for ax, titulo in [(ax_tr, "Entrenamiento"), (ax_va, "Validación")]:
        ax.set_title(titulo)
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
        ax.legend()
    ax_tr.set_ylabel("Pérdida (BCE)")
    fig.suptitle("Convergencia de optimizadores (media ± desv. entre folds)")
    fig.tight_layout()

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150)
    print(f"Gráfica guardada en {salida.resolve()}")


if __name__ == "__main__":
    main()
