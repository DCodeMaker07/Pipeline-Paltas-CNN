"""Validación cruzada multi-etiqueta (dos salidas: antracnosis y roña).

Variantes:
  base:     sin aumento de datos, 10 épocas, tasa fija.
  mejorada: con aumento de datos, 20 épocas y tasa con coseno.
Modelos: cnn (5 bloques propios), mobilenet_v2 y resnet18 (preentrenados).
Mismos folds y semillas para todos. Cada salida tiene su sigmoide y su pérdida con pos_weight.
Resultados en resultados_profundidad\\ (ver nombre_archivo).
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import bitacora
import comparar_rona as cr
import entrenamiento as en
import experimentos as ex
import pipeline as pl

OUT = Path("resultados_profundidad")
CONFIG_CNN = en.CONFIG_FINAL
VARIANTES = en.VARIANTES
METRICAS = ["F1_antracnosis", "AUC_antracnosis", "F1_roña", "AUC_roña", "subset_accuracy", "hamming_loss"]
MODELOS = {
    "cnn": ("cnn", CONFIG_CNN, False),
    "mobilenet_v2": ("mobilenet_v2", None, True),
    "resnet18": ("resnet18", None, True),
}


def nombre_archivo(modelo, variante):
    if modelo == "cnn":
        return OUT / f"multi_cv_{variante}.csv"
    return OUT / f"multi_cv_{modelo}_{variante}.csv"


def diferencia(filas, clave):
    """Compara mejorada frente a base. En Hamming, menor es mejor, así que se invierte el orden."""
    if clave == "hamming_loss":
        return cr.comparar_pareado(filas, "mejorada", "base", clave)
    return cr.comparar_pareado(filas, "base", "mejorada", clave)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", choices=list(VARIANTES), required=True)
    ap.add_argument("--modelo", choices=list(MODELOS), default="cnn")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--semillas", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--pregunta", type=int, default=None)
    a = ap.parse_args()
    v = VARIANTES[a.variante]
    arq, cfg, imnet = MODELOS[a.modelo]

    t0 = time.time()
    bitacora.encabezado(f"Validación cruzada multi-etiqueta: {a.modelo}, variante {a.variante}", a.pregunta)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", dev)
    D = ex.cargar_todo(a.data)
    X, Y = D["X"], D["Y"]
    trv = torch.cat([D["tr"], D["va"]])
    estrato = (Y[trv, 0] * 2 + Y[trv, 1]).numpy().astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(trv)), estrato))

    filas = []
    for semilla in a.semillas:
        for k, (i_tr, i_va) in enumerate(folds, start=1):
            itr = trv[torch.as_tensor(i_tr)]
            iva = trv[torch.as_tensor(i_va)]
            t_corrida = time.time()
            m = en.entrenar_multi(cfg, X[itr], Y[itr], v["epochs"], v["aug"], v["sched"], dev, semilla + k, arq=arq, imnet=imnet)
            seg = time.time() - t_corrida
            P = ex.predecir(m, X[iva], dev, imnet)
            met = pl.metricas(Y[iva].numpy(), P)
            fila = {"modelo": a.modelo, "variante": a.variante, "semilla": semilla, "fold": k}
            fila.update({clave: round(met[clave], 4) for clave in METRICAS})
            fila.update({"tiempo_s": round(seg, 1), "dispositivo": str(dev)})
            filas.append(fila)
            print(fila, flush=True)

    OUT.mkdir(exist_ok=True)
    with open(nombre_archivo(a.modelo, a.variante), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print("\nResumen (media ± desviación; semillas x folds):")
    for clave in METRICAS:
        vals = [r[clave] for r in filas]
        print(f"  {clave:18s}: {statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}")

    otra = "mejorada" if a.variante == "base" else "base"
    ruta_otra = nombre_archivo(a.modelo, otra)
    if ruta_otra.exists():
        with open(ruta_otra, encoding="utf-8") as fh:
            otras = list(csv.DictReader(fh))
        combinadas = [dict(r, modelo=r["variante"]) for r in otras + [dict(r, variante=r["variante"]) for r in filas]]
        print("\nMejorada frente a base (pares semilla x fold):")
        for clave in METRICAS:
            g, n, p = diferencia(combinadas, clave)
            print(f"  {clave:18s}: mejorada mejor en {g} de {n} | p = {p:.4f}")

    bitacora.cerrar(t0, OUT, f"multi_cv_{a.modelo}_{a.variante}", {"Modelo": a.modelo, "Dispositivo": dev,
                                                                  "Épocas": v["epochs"], "Aumento": v["aug"],
                                                                  "Coseno": v["sched"], "Corridas": len(filas)})


if __name__ == "__main__":
    main()
