"""Mejoras para roña, aplicadas a los tres modelos con el mismo tratamiento:
aumento de datos (volteos y cambios de brillo y contraste), 20 épocas y tasa con coseno.

Reutiliza la línea base de resultados_profundidad\\comparacion_rona.csv (mismos folds y semillas)
y solo entrena la variante nueva. Cada modelo se compara contra su propia versión anterior.
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
import comparar_rona as cr
import experimentos as ex
from entrenamiento import aumentar, preparar  # noqa: F401

COL = 1                                   # columna de roña en Y
OUT = Path("resultados_profundidad")
BASE_CSV = OUT / "comparacion_rona.csv"
SUFIJO = "_mejorado"
MODELOS = {
    "cnn_5_bloques": ("cnn", dict(cr.BASE, bloques=5), False),
    "mobilenet_v2": ("mobilenet_v2", None, True),
    "resnet18": ("resnet18", None, True),
}


def entrenar(arq, cfg, imnet, Xtr, Ytr, epochs, dev, semilla):
    ex.fijar(semilla)
    modelo = ex.crear_modelo(arq, cfg).to(dev)
    pos_w = ((1 - Ytr[:, COL]).sum() / Ytr[:, COL].sum().clamp(min=1)).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = ex.optimizador("adamw", modelo.parameters(), cr.LR, cr.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=cr.BATCH, shuffle=True)
    modelo.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            salida = modelo(preparar(xb, imnet, dev, True))[:, COL]
            loss = crit(salida, yb[:, COL].to(dev))
            loss.backward()
            opt.step()
        sched.step()
    return modelo


def evaluar(modelo, imnet, X, Y, dev):
    prob = ex.predecir(modelo, X, dev, imnet)[:, COL]
    y = Y[:, COL].numpy()
    return f1_score(y, prob >= 0.5, zero_division=0), roc_auc_score(y, prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--semillas", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--pregunta", type=int, default=None)
    a = ap.parse_args()

    t0 = time.time()
    bitacora.encabezado("Mejoras para roña: aumento de datos, 20 épocas y tasa con coseno en los tres modelos", a.pregunta)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", dev)
    D = ex.cargar_todo(a.data)
    X, Y = D["X"], D["Y"]
    trv = torch.cat([D["tr"], D["va"]])
    estrato = (Y[trv, 0] * 2 + Y[trv, 1]).numpy().astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(trv)), estrato))

    nuevas = []
    for semilla in a.semillas:
        for k, (i_tr, i_va) in enumerate(folds, start=1):
            itr = trv[torch.as_tensor(i_tr)]
            iva = trv[torch.as_tensor(i_va)]
            for nombre, (arq, cfg, imnet) in MODELOS.items():
                t_corrida = time.time()
                m = entrenar(arq, cfg, imnet, X[itr], Y[itr], a.epochs, dev, semilla + k)
                seg = time.time() - t_corrida
                f1, auc = evaluar(m, imnet, X[iva], Y[iva], dev)
                fila = {"modelo": nombre + SUFIJO, "semilla": semilla, "fold": k,
                        "parametros": sum(p.numel() for p in m.parameters() if p.requires_grad),
                        "F1_val": round(f1, 4), "AUC_val": round(auc, 4),
                        "tiempo_s": round(seg, 1), "dispositivo": str(dev)}
                nuevas.append(fila)
                print(fila, flush=True)

    OUT.mkdir(exist_ok=True)
    with open(OUT / "mejoras_rona.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(nuevas[0].keys()))
        w.writeheader()
        w.writerows(nuevas)

    with open(BASE_CSV, encoding="utf-8") as fh:
        base = [r for r in csv.DictReader(fh) if r["modelo"] in MODELOS]
    todas = base + nuevas

    print("\nResumen (media ± desviación; semillas x folds):")
    for nombre in MODELOS:
        for etiqueta, clave_modelo in [("antes ", nombre), ("después", nombre + SUFIJO)]:
            rs = [r for r in todas if r["modelo"] == clave_modelo]
            f1s = [float(r["F1_val"]) for r in rs]
            aucs = [float(r["AUC_val"]) for r in rs]
            print(f"{nombre:15s} {etiqueta} | F1 {statistics.mean(f1s):.3f} ± {statistics.stdev(f1s):.3f} "
                  f"| AUC {statistics.mean(aucs):.3f} ± {statistics.stdev(aucs):.3f} (n={len(rs)})")

    print("\nMejora de cada modelo frente a su versión anterior (pares semilla x fold):")
    for nombre in MODELOS:
        for clave in ["F1_val", "AUC_val"]:
            g, n, p = cr.comparar_pareado(todas, nombre, nombre + SUFIJO, clave)
            print(f"  {nombre:15s} {clave:8s}: {g} de {n} | p = {p:.4f}")

    print("\nComparación con las mejoras aplicadas (pares semilla x fold):")
    ref = "cnn_5_bloques" + SUFIJO
    for otro in ["mobilenet_v2" + SUFIJO, "resnet18" + SUFIJO]:
        for clave in ["F1_val", "AUC_val"]:
            g, n, p = cr.comparar_pareado(todas, ref, otro, clave)
            print(f"  {otro:20s} frente a CNN 5 bloques, {clave}: {g} de {n} | p = {p:.4f}")

    bitacora.cerrar(t0, OUT, "mejoras_rona", {"Dispositivo": dev, "Épocas": a.epochs, "Semillas": a.semillas,
                                              "Corridas nuevas": len(nuevas)})


if __name__ == "__main__":
    main()
