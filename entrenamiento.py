"""Entrenamiento de la CNN multi-etiqueta: configuración final del proyecto y utilidades.

Lo usan multi_cv.py (validación cruzada) y prueba_final.py (prueba final).
Configuración final: CNN de 5 bloques, aumento de datos, 20 épocas y tasa con coseno.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import experimentos as ex

CONFIG_FINAL = dict(filtros=32, activ="gelu", dropout=0.3, bloques=5)
OPT, LR, WD, BATCH = "adamw", 1.2349482735064392e-4, 0.0, 16
VARIANTES = {
    "base": dict(aug=False, sched=False, epochs=10),
    "mejorada": dict(aug=True, sched=True, epochs=20),
}


def aumentar(x):
    """Volteos horizontales y verticales al azar, y cambios pequeños de brillo y contraste."""
    b, d = x.shape[0], x.device
    x = torch.where(torch.rand(b, 1, 1, 1, device=d) < 0.5, x.flip(3), x)
    x = torch.where(torch.rand(b, 1, 1, 1, device=d) < 0.5, x.flip(2), x)
    x = x * (1 + (torch.rand(b, 1, 1, 1, device=d) * 0.2 - 0.1))
    media = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - media) * (1 + (torch.rand(b, 1, 1, 1, device=d) * 0.2 - 0.1)) + media
    return x.clamp(0, 1)


def preparar(xb, imnet, dev, entrenando):
    """Pasa de uint8 a [0, 1], aplica aumento si se entrena y normaliza si el modelo es preentrenado."""
    x = xb.to(dev).float() / 255.0
    if entrenando:
        x = aumentar(x)
    if imnet:
        x = (x - ex.IMNET_MEAN.to(dev)) / ex.IMNET_STD.to(dev)
    return x


def entrenar_multi(cfg, Xtr, Ytr, epochs, aug, sched, dev, semilla, arq="cnn", imnet=False):
    """Entrena con dos salidas (antracnosis y roña). Cada salida tiene su sigmoide y pos_weight."""
    ex.fijar(semilla)
    m = ex.crear_modelo(arq, cfg).to(dev)
    pos_w = ((1 - Ytr).sum(0) / Ytr.sum(0).clamp(min=1)).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = ex.optimizador(OPT, m.parameters(), LR, WD)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if sched else None
    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)
    m.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(m(preparar(xb, imnet, dev, aug)), yb.to(dev))
            loss.backward()
            opt.step()
        if s:
            s.step()
    return m
