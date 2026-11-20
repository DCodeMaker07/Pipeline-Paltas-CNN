"""Comparación de arquitecturas para roña con la misma validación cruzada (5 folds x 3 semillas).

Mismo protocolo para todos los modelos: AdamW con weight decay 0, tasa 1.2349e-4, lote 16 y
10 épocas. Solo cambia el modelo. Los preentrenados usan la normalización de ImageNet.
Resultados en resultados_profundidad\\comparacion_rona.csv.
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import bitacora
import experimentos as ex

COL = 1                                   # columna de roña en Y
UMBRAL = 0.5
COLUMNA = {"antracnosis": 0, "rona": 1}   # columna de cada enfermedad en Y
OUT = Path("resultados_profundidad")
BASE = dict(filtros=32, activ="gelu", dropout=0.3)
OPT, LR, WD, BATCH = "adamw", 1.2349482735064392e-4, 0.0, 16

# nombre: (arquitectura para crear_modelo, configuración, usa normalización de ImageNet)
MODELOS = {
    "cnn_4_bloques": ("cnn", dict(BASE, bloques=4), False),
    "cnn_5_bloques": ("cnn", dict(BASE, bloques=5), False),
    "mobilenet_v2": ("mobilenet_v2", None, True),
    "resnet18": ("resnet18", None, True),
}


def entrenar(nombre, cfg, imnet, Xtr, Ytr, epochs, dev, semilla):
    ex.fijar(semilla)
    modelo = ex.crear_modelo(nombre, cfg).to(dev)
    pos_w = ((1 - Ytr[:, COL]).sum() / Ytr[:, COL].sum().clamp(min=1)).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = ex.optimizador(OPT, modelo.parameters(), LR, WD)
    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)
    modelo.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            salida = modelo(ex.prep(xb, imnet).to(dev))[:, COL]
            loss = crit(salida, yb[:, COL].to(dev))
            loss.backward()
            opt.step()
    return modelo


def evaluar(modelo, imnet, X, Y, dev):
    prob = ex.predecir(modelo, X, dev, imnet)[:, COL]
    y = Y[:, COL].numpy()
    return f1_score(y, prob >= UMBRAL, zero_division=0), roc_auc_score(y, prob)


def comparar_pareado(filas, ref, otro, clave):
    """Victorias de 'otro' sobre 'ref' en pares (semilla, fold) y p de Wilcoxon."""
    v = {(r["modelo"], int(r["semilla"]), int(r["fold"])): float(r[clave]) for r in filas}
    pares = sorted({(s, f) for (m, s, f) in v if m == ref and (otro, s, f) in v})
    dif = [v[(otro, s, f)] - v[(ref, s, f)] for s, f in pares]
    victorias = sum(d > 0 for d in dif)
    try:
        p = wilcoxon(dif).pvalue
    except ValueError:
        p = float("nan")
    return victorias, len(pares), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objetivo", choices=list(COLUMNA), default="rona")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--semillas", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--referencia", default="cnn_5_bloques", choices=list(MODELOS))
    ap.add_argument("--pregunta", type=int, default=None)
    a = ap.parse_args()
    global COL
    COL = COLUMNA[a.objetivo]
    t0 = time.time()
    bitacora.encabezado(f"Comparación de arquitecturas para {a.objetivo}: CNN 4 y 5 bloques, MobileNetV2 y ResNet-18", a.pregunta)

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
            for nombre, (arq, cfg, imnet) in MODELOS.items():
                t_corrida = time.time()
                m = entrenar(arq, cfg, imnet, X[itr], Y[itr], a.epochs, dev, semilla + k)
                seg = time.time() - t_corrida
                f1, auc = evaluar(m, imnet, X[iva], Y[iva], dev)
                fila = {"modelo": nombre, "semilla": semilla, "fold": k,
                        "parametros": sum(p.numel() for p in m.parameters() if p.requires_grad),
                        "F1_val": round(f1, 4), "AUC_val": round(auc, 4),
                        "tiempo_s": round(seg, 1), "dispositivo": str(dev)}
                filas.append(fila)
                print(fila, flush=True)

    OUT.mkdir(exist_ok=True)
    with open(OUT / f"comparacion_{a.objetivo}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print("\nResumen (media ± desviación; semillas x folds):")
    for nombre in MODELOS:
        rs = [r for r in filas if r["modelo"] == nombre]
        f1s = [float(r["F1_val"]) for r in rs]
        aucs = [float(r["AUC_val"]) for r in rs]
        print(f"{nombre:15s} params {rs[0]['parametros']:>10,} | F1 {statistics.mean(f1s):.3f} ± "
              f"{statistics.stdev(f1s):.3f} | AUC {statistics.mean(aucs):.3f} ± {statistics.stdev(aucs):.3f}"
              f" | tiempo medio {statistics.mean(float(r['tiempo_s']) for r in rs):.1f} s")

    print(f"\nComparación contra {a.referencia} (victorias en pares semilla x fold, p de Wilcoxon):")
    for nombre in MODELOS:
        if nombre == a.referencia:
            continue
        for clave in ["F1_val", "AUC_val"]:
            g, n, p = comparar_pareado(filas, a.referencia, nombre, clave)
            print(f"  {nombre:15s} {clave:8s}: {g} de {n} | p = {p:.4f}")
    bitacora.cerrar(t0, OUT, f"comparacion_{a.objetivo}", {"Dispositivo": dev, "Modelos": list(MODELOS), "Semillas": a.semillas, "Corridas entrenadas": len(filas)})


if __name__ == "__main__":
    main()
