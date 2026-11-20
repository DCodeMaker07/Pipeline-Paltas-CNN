import argparse
import csv
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

import bitacora
import pipeline as pl

SEED = pl.SEED
OUT = Path("resultados")
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Pregunta 2: este es el espacio de búsqueda de hiperparámetros: los valores que puede tomar
# cada uno. La tasa de aprendizaje no está aquí porque se muestrea aparte, en escala log.
ESPACIO = {
    "bloques": [3, 4, 5],
    "filtros": [8, 16, 32],
    "activ": ["relu", "leaky", "gelu"],
    "dropout": [0.0, 0.2, 0.3, 0.5],
    "opt": ["sgd", "adam", "adamw"],
    "batch": [16, 32],
    "wd": [0.0, 1e-5, 1e-4, 1e-3],
}


def fijar(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prep(xb, imnet=False):
    x = xb.float() / 255.0
    if imnet:
        x = (x - IMNET_MEAN) / IMNET_STD
    return x


# Pregunta 1: aquí defino mi CNN propia, la que elegí como modelo principal.
# Son 4 bloques de Conv 3x3 -> BatchNorm -> GELU -> MaxPool, con 32, 64, 128 y 256 filtros.
# Después va global average pooling, una densa de 64 neuronas con dropout y una salida de 2.
# La elegí por su tamaño: tiene 405,954 parámetros entrenables.
class CNNConfig(nn.Module):
    """CNN de la diapositiva 4 con hiperparámetros configurables."""

    def __init__(self, bloques=4, filtros=32, activ="gelu", dropout=0.3, bn=True):
        super().__init__()
        act = {"relu": nn.ReLU, "leaky": lambda: nn.LeakyReLU(0.1), "gelu": nn.GELU}[activ]
        capas, cin, c = [], 3, filtros
        for _ in range(bloques):
            norm = [nn.BatchNorm2d(c)] if bn else []
            capas += [nn.Conv2d(cin, c, 3, padding=1), *norm, act(), nn.MaxPool2d(2)]
            cin, c = c, c * 2
        self.features = nn.Sequential(*capas)
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(cin, 64), act(), nn.Dropout(dropout), nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.clf(self.features(x))


# Pregunta 1: aquí armo las tres arquitecturas que comparé: mi CNN propia y dos modelos
# preentrenados en ImageNet (MobileNetV2 y ResNet-18). En los preentrenados solo cambio la
# última capa por una de 2 salidas, para que prediga antracnosis y roña.
def crear_modelo(nombre, cfg=None):
    if nombre == "cnn":
        return CNNConfig(**(cfg or {}))
    if nombre == "mobilenet_v2":
        m = models.mobilenet_v2(weights="DEFAULT")
        m.classifier[1] = nn.Linear(m.last_channel, 2)
        return m
    if nombre == "resnet18":
        m = models.resnet18(weights="DEFAULT")
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    raise ValueError(nombre)


def optimizador(nombre, params, lr, wd):
    if nombre == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if nombre == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def entrenar(nombre, cfg, Xtr, Ytr, epochs, opt, lr, wd, batch, imnet, dev):
    model = crear_modelo(nombre, cfg).to(dev)
    pos_w = ((1 - Ytr).sum(0) / Ytr.sum(0).clamp(min=1)).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    o = optimizador(opt, model.parameters(), lr, wd)
    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            o.zero_grad()
            loss = crit(model(prep(xb, imnet).to(dev)), yb.to(dev))
            loss.backward()
            o.step()
        model.train()
    return model


@torch.no_grad()
def predecir(model, X, dev, imnet=False):
    model.eval()
    out = []
    for i in range(0, len(X), 64):
        out.append(torch.sigmoid(model(prep(X[i:i + 64], imnet).to(dev))).cpu())
    return torch.cat(out).numpy()


def cargar_todo(data):
    X, Y, clases = pl.cargar(data)
    idx = np.arange(len(clases))
    trv, te = train_test_split(idx, test_size=0.2, stratify=clases, random_state=SEED)
    tr, va = train_test_split(trv, test_size=0.2, stratify=clases[trv], random_state=SEED)
    t = lambda a: torch.as_tensor(a, dtype=torch.long)
    return {"X": X, "Y": Y, "tr": t(tr), "va": t(va), "te": t(te)}


def guardar_csv(path, filas):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def evaluar_cfg(D, c, epochs, dev):
    cfg = dict(bloques=c["bloques"], filtros=c["filtros"], activ=c["activ"], dropout=c["dropout"])
    fijar()
    m = entrenar("cnn", cfg, D["X"][D["tr"]], D["Y"][D["tr"]], epochs,
                 c["opt"], c["lr"], c["wd"], c["batch"], False, dev)
    met = pl.metricas(D["Y"][D["va"]].numpy(), predecir(m, D["X"][D["va"]], dev))
    met["score"] = (met["F1_antracnosis"] + met["F1_roña"]) / 2
    return met


# Pregunta 2 y 3: genera una configuración al azar dentro de ESPACIO.
# Con la misma semilla (42) siempre salen las mismas configuraciones, así los resultados se reproducen.
def muestrear(rng):
    c = {k: rng.choice(v) for k, v in ESPACIO.items()}
    c["lr"] = float(10 ** rng.uniform(-4, -2))
    return c


# ---------- Pregunta 3: Random Search ----------
# Pregunta 3: aquí implemento Random Search. Pruebo N configuraciones al azar (N = 10 por defecto),
# entreno cada una y calculo su score en validación: (F1 antracnosis + F1 roña) / 2.
# Guardo todas en busqueda.csv y la de mayor score en mejor_config.json.
def cmd_buscar(a, D, dev):
    rng = random.Random(SEED)
    filas = []
    for i in range(1, a.n + 1):
        c = muestrear(rng)
        t0 = time.time()
        met = evaluar_cfg(D, c, a.epochs, dev)
        filas.append({**c, **met, "segundos": round(time.time() - t0, 1), "config": i})
        print(f"[{i}/{a.n}] score={met['score']:.4f}  {c}")
    guardar_csv(OUT / "busqueda.csv", filas)
    mejor = max(filas, key=lambda r: r["score"])
    best = {k: mejor[k] for k in list(ESPACIO) + ["lr"]}
    (OUT / "mejor_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("Mejor configuración:", best, "score:", round(mejor["score"], 4))


# ---------- Pregunta 6: sensibilidad ----------
def cmd_sensibilidad(a, D, dev):
    base = json.loads((OUT / "mejor_config.json").read_text(encoding="utf-8"))
    variaciones = {
        "bloques": [3, 4, 5], "filtros": [8, 16, 32], "dropout": [0.0, 0.2, 0.3, 0.5],
        "lr": [1e-4, 1e-3, 1e-2], "wd": [0.0, 1e-5, 1e-4, 1e-3], "batch": [16, 32],
    }
    filas = []
    for par, valores in variaciones.items():
        for v in valores:
            c = dict(base)
            c[par] = v
            met = evaluar_cfg(D, c, a.epochs, dev)
            filas.append({"parametro": par, "valor": v, "score": met["score"],
                          "hamming_loss": met["hamming_loss"],
                          "F1_antracnosis": met["F1_antracnosis"], "F1_roña": met["F1_roña"]})
            print(f"{par}={v}  score={met['score']:.4f}")
    guardar_csv(OUT / "sensibilidad.csv", filas)
    rango = {}
    for par in variaciones:
        s = [f["score"] for f in filas if f["parametro"] == par]
        rango[par] = max(s) - min(s)
    print("Rango del score por hiperparámetro (mayor = más influyente):")
    for par, r in sorted(rango.items(), key=lambda x: -x[1]):
        print(f"  {par}: {r:.4f}")


# ---------- Pregunta 7: comparación de arquitecturas ----------
def cmd_arquitecturas(a, D, dev):
    # Mismo optimizador (Adam), misma tasa (la ganadora de la búsqueda), mismo lote, mismas épocas y mismo test.
    # Así la única diferencia entre modelos es la arquitectura.
    LR = 1.2349482735064392e-4
    LOTE = 16
    cfgs = {
        "cnn": ("cnn", None, False, LR),
        "mobilenet_v2": ("mobilenet_v2", None, True, LR),
        "resnet18": ("resnet18", None, True, LR),
    }
    Yte = D["Y"][D["te"]].numpy().astype(int)
    preds, filas = {}, []
    for nombre, (arch, cfg, imnet, lr) in cfgs.items():
        t0 = time.time()
        fijar()
        m = entrenar(arch, cfg, D["X"][D["tr"]], D["Y"][D["tr"]], a.epochs,
                     "adam", lr, 0.0, LOTE, imnet, dev)
        P = predecir(m, D["X"][D["te"]], dev, imnet)
        preds[nombre] = (P >= 0.5).astype(int)
        met = pl.metricas(Yte, P)
        filas.append({"arquitectura": nombre, "lr": lr, **met,
                      "segundos": round(time.time() - t0, 1)})
        print(nombre, {k: round(v, 4) for k, v in met.items()})
    guardar_csv(OUT / "arquitecturas.csv", filas)

    comparaciones = []
    nombres = list(cfgs)
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            A, B = nombres[i], nombres[j]
            ok_a = (preds[A] == Yte).all(axis=1)
            ok_b = (preds[B] == Yte).all(axis=1)
            n10 = int(np.sum(ok_a & ~ok_b))
            n01 = int(np.sum(~ok_a & ok_b))
            n = n10 + n01
            p = float(binomtest(min(n10, n01), n, 0.5).pvalue) if n > 0 else 1.0
            comparaciones.append({"A": A, "B": B, "solo_A_acierta": n10,
                                  "solo_B_acierta": n01, "p_mcnemar_exacta": p})
    guardar_csv(OUT / "mcnemar.csv", comparaciones)
    for c in comparaciones:
        print(f"McNemar {c['A']} vs {c['B']}: p={c['p_mcnemar_exacta']:.4f}")


# ---------- Pregunta 8 y 12: ablación de regularización y tamaño de lote ----------
def cmd_ablacion(a, D, dev):
    base = json.loads((OUT / "mejor_config.json").read_text(encoding="utf-8"))
    cfg_reg = dict(bloques=base["bloques"], filtros=base["filtros"], activ=base["activ"], dropout=base["dropout"])
    cfg_sin = dict(cfg_reg, dropout=0.0, bn=False)
    variantes = {
        "con_regularizacion_lote16": (cfg_reg, 16),
        "sin_bn_ni_dropout_lote16": (cfg_sin, 16),
        "con_regularizacion_lote32": (cfg_reg, 32),
    }
    Yva = D["Y"][D["va"]].numpy().astype(int)
    filas = []
    for nombre, (cfg, lote) in variantes.items():
        t0 = time.time()
        fijar()
        m = entrenar("cnn", cfg, D["X"][D["tr"]], D["Y"][D["tr"]], a.epochs,
                     base["opt"], base["lr"], base["wd"], lote, False, dev)
        seg = round(time.time() - t0, 1)
        for conjunto, idx, Y in [("validacion", "va", Yva)]:  # solo validación: la prueba se usa una vez, en pregunta 10
            met = pl.metricas(Y, predecir(m, D["X"][D[idx]], dev))
            filas.append({"variante": nombre, "conjunto": conjunto, "lote": lote, **met,
                          "segundos_entrenamiento": seg})
            print(nombre, conjunto, {k: round(v, 4) for k, v in met.items()})
    guardar_csv(OUT / "ablacion.csv", filas)


# ---------- Pregunta 10: modelo final ----------
def cmd_final(a, D, dev):
    base = json.loads((OUT / "mejor_config.json").read_text(encoding="utf-8"))
    cfg = dict(bloques=base["bloques"], filtros=base["filtros"], activ=base["activ"], dropout=base["dropout"])
    trv = torch.cat([D["tr"], D["va"]])
    fijar()
    m = entrenar("cnn", cfg, D["X"][trv], D["Y"][trv], a.epochs,
                 base["opt"], base["lr"], base["wd"], base["batch"], False, dev)
    torch.save({"state_dict": m.state_dict(), "config": base, "epocas": a.epochs},
               OUT / "modelo_final.pt")
    P = predecir(m, D["X"][D["te"]], dev)
    Yte = D["Y"][D["te"]].numpy().astype(int)
    pred = (P >= 0.5).astype(int)
    met = pl.metricas(Yte, P)
    combos = {"sano (0,0)": (0, 0), "solo antracnosis (1,0)": (1, 0),
              "solo roña (0,1)": (0, 1), "co-infección (1,1)": (1, 1)}
    matriz = []
    for nombre, (ca, cr) in combos.items():
        real = (Yte[:, 0] == ca) & (Yte[:, 1] == cr)
        acierto = (pred[:, 0] == ca) & (pred[:, 1] == cr) & real
        matriz.append({"clase_real": nombre, "n_en_test": int(real.sum()),
                       "aciertos_exactos": int(acierto.sum())})
    salida = {"config": base, "epocas": a.epochs, "metricas_test": met, "matriz_por_combinacion": matriz}
    (OUT / "metricas_finales.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Métricas en test:", {k: round(v, 4) for k, v in met.items()})
    for fila in matriz:
        print(fila)


# ---------- Preguntas 7 y 11: costo computacional ----------
def flops_del_modelo(model, dev):
    total = [0]

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            total[0] += 2 * int(np.prod(out.shape[1:])) * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, nn.Linear):
            total[0] += 2 * mod.in_features * mod.out_features

    hs = [mod.register_forward_hook(hook) for mod in model.modules()
          if isinstance(mod, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, pl.IMG, pl.IMG, device=dev))
    for h in hs:
        h.remove()
    return total[0]


def tiempo_inferencia_ms(model, dev, reps=30):
    model.eval()
    x = torch.zeros(1, 3, pl.IMG, pl.IMG, device=dev)
    ts = []
    with torch.no_grad():
        for _ in range(5):
            model(x)
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


def cmd_complejidad(a, dev):
    filas = []
    for nombre in ["cnn", "mobilenet_v2", "resnet18"]:
        m = crear_modelo(nombre).to(dev)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        flops = flops_del_modelo(m, dev)
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        filas.append({
            "arquitectura": nombre,
            "parametros_entrenables": params,
            "MFLOPs_por_imagen": round(flops / 1e6, 1),
            "memoria_pesos_MB": round(params * 4 / 1e6, 2),
            "tamano_archivo_MB": round(buf.getbuffer().nbytes / 1e6, 2),
            "inferencia_ms": round(tiempo_inferencia_ms(m, dev), 2),
            "dispositivo": str(dev),
        })
        print(filas[-1])
    guardar_csv(OUT / "complejidad.csv", filas)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre in ["buscar", "sensibilidad", "arquitecturas", "final", "complejidad", "ablacion"]:
        s = sub.add_parser(nombre)
        s.add_argument("--data", default="dataset")
        s.add_argument("--epochs", type=int, default=10)
        s.add_argument("--pregunta", type=int, default=None)
        if nombre == "buscar":
            s.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", dev)
    t0 = time.time()
    bitacora.encabezado(f"Comando {a.cmd} de experimentos.py", a.pregunta)
    if a.cmd == "complejidad":
        cmd_complejidad(a, dev)
        bitacora.cerrar(t0, OUT, a.cmd, {"Dispositivo": dev})
        return
    fijar()
    D = cargar_todo(a.data)
    {"buscar": cmd_buscar, "sensibilidad": cmd_sensibilidad, "arquitecturas": cmd_arquitecturas,
     "final": cmd_final, "ablacion": cmd_ablacion}[a.cmd](a, D, dev)
    bitacora.cerrar(t0, OUT, a.cmd, {"Dispositivo": dev, "Épocas": a.epochs})


if __name__ == "__main__":
    main()
