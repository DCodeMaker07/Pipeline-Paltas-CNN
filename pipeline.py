"""Pipeline de la CNN multietiqueta para roña (R) y antracnosis (A) en palta.

Etiquetas por prefijo del nombre: A.* = antracnosis, R.* = roña, S.* = sano.
Salida por imagen: [antracnosis, roña]. Sano = [0, 0]; co-infección = [1, 1].

Uso (PowerShell, dentro de pipeline-palta):
  .\.venv\Scripts\python.exe pipeline.py --data dataset --modo simple --epochs 2
  .\.venv\Scripts\python.exe pipeline.py --data dataset --modo kfold --opt adam --epochs 10
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

IMG = 224
SEED = 42
PREFIJO = {"A": [1.0, 0.0], "R": [0.0, 1.0], "S": [0.0, 0.0]}
NOMBRES = ["antracnosis", "roña"]


def fijar_semilla(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)


def cargar(carpeta):
    rutas = sorted(p for p in Path(carpeta).glob("*.jpg") if p.name[0] in PREFIJO)
    X = np.empty((len(rutas), 3, IMG, IMG), dtype=np.uint8)
    Y = np.empty((len(rutas), 2), dtype=np.float32)
    clases = []
    for i, r in enumerate(rutas):
        img = Image.open(r).convert("RGB").resize((IMG, IMG))
        X[i] = np.asarray(img).transpose(2, 0, 1)
        Y[i] = PREFIJO[r.name[0]]
        clases.append(r.name[0])
    clases = np.array(clases)
    print(f"Imágenes: {len(rutas)}  (A={np.sum(clases == 'A')}, R={np.sum(clases == 'R')}, S={np.sum(clases == 'S')})")
    return torch.from_numpy(X), torch.from_numpy(Y), clases


class PaltaCNN(nn.Module):
    """Arquitectura de la diapositiva 4: 4 bloques conv, GAP, Dense(64), Dropout(0.3), Dense(2)."""

    def __init__(self):
        super().__init__()

        def bloque(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(bloque(3, 16), bloque(16, 32), bloque(32, 64), bloque(64, 128))
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.clf(self.features(x))


def normalizar(xb):
    return xb.float() / 255.0


def contar_parametros(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def crear_optimizador(nombre, params, lr):
    if nombre == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if nombre == "adam":
        return torch.optim.Adam(params, lr=lr)
    if nombre == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    raise ValueError(nombre)


@torch.no_grad()
def predecir(model, X, dev):
    model.eval()
    salida = []
    for i in range(0, len(X), 64):
        xb = normalizar(X[i:i + 64]).to(dev)
        salida.append(torch.sigmoid(model(xb)).cpu())
    return torch.cat(salida).numpy()


def entrenar(Xtr, Ytr, Xva, Yva, epochs, opt_nombre, lr, dev, batch=32):
    model = PaltaCNN().to(dev)
    # pos_weight = N_neg / N_pos por etiqueta: castiga más los falsos negativos de la clase minoritaria.
    pos_weight = ((1 - Ytr).sum(0) / Ytr.sum(0).clamp(min=1)).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = crear_optimizador(opt_nombre, model.parameters(), lr)
    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch, shuffle=True)
    historial = []
    for ep in range(1, epochs + 1):
        model.train()
        suma = 0.0
        for xb, yb in dl:
            xb, yb = normalizar(xb).to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            suma += loss.item() * len(xb)
        p_va = torch.tensor(predecir(model, Xva, dev))
        val_loss = F.binary_cross_entropy(p_va, Yva).item()
        historial.append({"epoca": ep, "loss_train": suma / len(Xtr), "loss_val": val_loss})
        print(f"  epoca {ep}/{epochs}  loss_train={suma / len(Xtr):.4f}  loss_val={val_loss:.4f}")
    return model, historial


def metricas(Y, P):
    Y = np.asarray(Y)
    pred = (P >= 0.5).astype(int)
    out = {}
    for i, nombre in enumerate(NOMBRES):
        y, p = Y[:, i], P[:, i]
        out[f"F1_{nombre}"] = float(f1_score(y, pred[:, i], zero_division=0))
        out[f"AUC_{nombre}"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    out["subset_accuracy"] = float((pred == Y).all(axis=1).mean())
    out["hamming_loss"] = float((pred != Y).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="carpeta con las imágenes A.*, R.*, S.*")
    ap.add_argument("--modo", choices=["simple", "kfold"], default="kfold")
    ap.add_argument("--opt", choices=["sgd", "adam", "adamw"], default="adam")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="resultados")
    args = ap.parse_args()

    fijar_semilla()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", dev)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    X, Y, clases = cargar(args.data)
    print("AVISO: los nombres no tienen ID de fruto, así que la partición es por imagen y puede haber fuga de datos.")

    t = lambda a: torch.as_tensor(a, dtype=torch.long)
    idx = np.arange(len(X))
    idx_trv, idx_te = train_test_split(idx, test_size=0.2, stratify=clases, random_state=SEED)
    Xte, Yte = X[t(idx_te)], Y[t(idx_te)]

    if args.modo == "simple":
        itr, iva = train_test_split(idx_trv, test_size=0.2, stratify=clases[idx_trv], random_state=SEED)
        splits = [(itr, iva)]
    else:
        skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=SEED)
        splits = [(idx_trv[a], idx_trv[b]) for a, b in skf.split(idx_trv, clases[idx_trv])]

    resultados = []
    t0 = time.time()
    for f, (itr, iva) in enumerate(splits, 1):
        print(f"\nFold {f}/{len(splits)}  train={len(itr)}  val={len(iva)}")
        fijar_semilla(SEED + f)
        model, hist = entrenar(X[t(itr)], Y[t(itr)], X[t(iva)], Y[t(iva)],
                               args.epochs, args.opt, args.lr, dev)
        m_va = metricas(Y[t(iva)].numpy(), predecir(model, X[t(iva)], dev))
        m_te = metricas(Yte.numpy(), predecir(model, Xte, dev))
        fila = {f"val_{k}": v for k, v in m_va.items()}
        fila.update({f"test_{k}": v for k, v in m_te.items()})
        fila["fold"] = f
        resultados.append(fila)
        print("  val:", {k: round(v, 4) for k, v in m_va.items()})
        print("  test:", {k: round(v, 4) for k, v in m_te.items()})
        with open(out / f"historial_{args.opt}_fold{f}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["epoca", "loss_train", "loss_val"])
            w.writeheader()
            w.writerows(hist)

    resumen = {}
    for clave in [k for k in resultados[0] if k != "fold"]:
        vals = np.array([r[clave] for r in resultados], dtype=float)
        resumen[clave] = {
            "media": float(np.nanmean(vals)),
            "desv": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    salida = {
        "optimizador": args.opt,
        "modo": args.modo,
        "epocas": args.epochs,
        "lr": args.lr,
        "parametros_entrenables": contar_parametros(PaltaCNN()),
        "segundos": round(time.time() - t0, 1),
        "folds": resultados,
        "resumen": resumen,
    }
    (out / f"resumen_{args.opt}_{args.modo}.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nMedia ± desv (val y test):")
    for k, v in resumen.items():
        print(f"  {k}: {v['media']:.4f} ± {v['desv']:.4f}")
    print(f"\nListo. Resultados en: {out.resolve()}")


if __name__ == "__main__":
    main()
