"""Profundidad de la CNN con validación cruzada (5 folds) y varias semillas.

Usa la misma partición de prueba fija que experimentos.py (20%, sin tocar).
Con el 80% restante hace 5 folds estratificados y repite con varias semillas.
Solo la salida del objetivo elegido recibe pérdida. Resultados en resultados_profundidad\\.
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import bitacora
import experimentos as ex

COLUMNA = {"antracnosis": 0, "rona": 1}   # columna de cada enfermedad en Y
UMBRAL = 0.5                               # mismo umbral que pipeline.metricas
OUT = Path("resultados_profundidad")

# Configuración ganadora de la búsqueda, excepto el número de bloques (la variable)
BASE = dict(filtros=32, activ="gelu", dropout=0.3)
OPT, LR, WD, BATCH = "adamw", 1.2349482735064392e-4, 0.0, 16


def entrenar(col, cfg, Xtr, Ytr, epochs, dev, semilla):
    """Entrena la CNN con pérdida solo sobre la columna del objetivo."""
    ex.fijar(semilla)
    modelo = ex.crear_modelo("cnn", cfg).to(dev)
    pos_w = ((1 - Ytr[:, col]).sum() / Ytr[:, col].sum().clamp(min=1)).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = ex.optimizador(OPT, modelo.parameters(), LR, WD)
    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)
    modelo.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            salida = modelo(ex.prep(xb).to(dev))[:, col]
            loss = crit(salida, yb[:, col].to(dev))
            loss.backward()
            opt.step()
    return modelo


def evaluar(modelo, col, X, Y, dev):
    """F1 y AUC de la columna indicada."""
    prob = ex.predecir(modelo, X, dev)[:, col]
    y = Y[:, col].numpy()
    return f1_score(y, prob >= UMBRAL, zero_division=0), roc_auc_score(y, prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objetivo", choices=list(COLUMNA), required=True)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bloques", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--semillas", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--pregunta", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    bitacora.encabezado(f"Profundidad de la CNN para {a.objetivo}: validación cruzada de 5 folds", a.pregunta)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", dev, "| objetivo:", a.objetivo)
    col = COLUMNA[a.objetivo]
    D = ex.cargar_todo(a.data)
    X, Y = D["X"], D["Y"]
    trv = torch.cat([D["tr"], D["va"]])            # 80% no usado para prueba (535 imágenes)
    estrato = (Y[trv, 0] * 2 + Y[trv, 1]).numpy().astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(trv)), estrato))

    filas = []
    for semilla in a.semillas:
        for b in a.bloques:
            for k, (i_tr, i_va) in enumerate(folds, start=1):
                itr = trv[torch.as_tensor(i_tr)]
                iva = trv[torch.as_tensor(i_va)]
                cfg = dict(BASE, bloques=b)
                t_corrida = time.time()
                m = entrenar(col, cfg, X[itr], Y[itr], a.epochs, dev, semilla + k)
                seg = time.time() - t_corrida
                f1, auc = evaluar(m, col, X[iva], Y[iva], dev)
                fila = {
                    "objetivo": a.objetivo, "bloques": b, "semilla": semilla, "fold": k,
                    "parametros": sum(p.numel() for p in m.parameters() if p.requires_grad),
                    "F1_val": round(f1, 4), "AUC_val": round(auc, 4),
                    "tiempo_s": round(seg, 1), "dispositivo": str(dev),
                }
                filas.append(fila)
                print(fila, flush=True)

    OUT.mkdir(exist_ok=True)
    with open(OUT / f"cv_{a.objetivo}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print("\nResumen (media ± desviación; semillas x folds):")
    for b in a.bloques:
        f1s = [r["F1_val"] for r in filas if r["bloques"] == b]
        aucs = [r["AUC_val"] for r in filas if r["bloques"] == b]
        print(f"{b} bloques: F1 {statistics.mean(f1s):.3f} ± {statistics.stdev(f1s):.3f} | "
              f"AUC {statistics.mean(aucs):.3f} ± {statistics.stdev(aucs):.3f} (n={len(f1s)})")
    bitacora.cerrar(t0, OUT, f"profundidad_{a.objetivo}", {"Dispositivo": dev, "Bloques": a.bloques, "Semillas": a.semillas, "Corridas entrenadas": len(filas)})


if __name__ == "__main__":
    main()
