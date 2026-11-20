"""Pregunta 12: recursos de hardware, tamaño de lote y precisión numérica.

Para cada combinación de precisión (fp32, o amp = mezcla de precisión en GPU), tamaño de lote y
semilla, mide:
- tiempo de entrenamiento por época,
- pico de memoria de GPU durante el entrenamiento (solo GPU),
- tiempo de inferencia por imagen con lote de 1 (con calentamiento y muchas repeticiones),
- F1 en VALIDACIÓN (nunca en prueba) para ver si la precisión degrada el modelo.

Luego resume cada combinación con media y desviación estándar sobre las semillas.

Notas:
- Entrena solo con la partición de entrenamiento (428 imágenes) y evalúa en validación (107).
- Para medir en CPU, oculten la GPU antes de ejecutar:  $env:CUDA_VISIBLE_DEVICES=""
  y usen --salida resultados_recursos_cpu para no sobrescribir la corrida de GPU.
- El tiempo de inferencia no depende del lote de entrenamiento: compárenlo por precisión.

Salidas en la carpeta de --salida: hardware.json, recursos_corridas.csv, recursos_resumen.csv, recursos.png.
"""
import argparse
import csv
import ctypes
import json
import os
import platform
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import bitacora
import entrenamiento as en
import experimentos as ex
import pipeline as pl


class _MemoriaWindows(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def ram_gb():
    if os.name == "nt":
        st = _MemoriaWindows()
        st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return round(st.ullTotalPhys / 2**30, 1)
    return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)


def sincronizar(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def hardware(dev):
    info = {"sistema": platform.platform(), "cpu": platform.processor(), "ram_gb": ram_gb(),
            "dispositivo": str(dev), "torch": torch.__version__}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1), cuda=torch.version.cuda)
    else:
        info["gpu"] = "no disponible (se usa CPU)"
    return info


def entrenar_medido(Xtr, Ytr, lote, precision, dev, epocas, semilla):
    ex.fijar(semilla)
    m = ex.crear_modelo("cnn", en.CONFIG_FINAL).to(dev)
    pos_w = ((1 - Ytr).sum(0) / Ytr.sum(0).clamp(min=1)).to(dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = ex.optimizador(en.OPT, m.parameters(), en.LR, en.WD)
    amp = precision == "amp"
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=lote, shuffle=True)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    sincronizar(dev)
    t0 = time.time()
    m.train()
    for _ in range(epocas):
        for xb, yb in dl:
            x = en.preparar(xb, False, dev, False)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                loss = crit(m(x), yb.to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    sincronizar(dev)
    seg_por_epoca = (time.time() - t0) / epocas
    pico_mb = torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None
    return m, seg_por_epoca, pico_mb


def latencia_ms(m, X, dev, amp, reps):
    m.eval()
    x = en.preparar(X[:1], False, dev, False)
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
        for _ in range(30):
            m(x)
        sincronizar(dev)
        t = time.time()
        for _ in range(reps):
            m(x)
        sincronizar(dev)
    return (time.time() - t) / reps * 1000


def resumir(corridas):
    grupos = {}
    for c in corridas:
        grupos.setdefault((c["precision"], c["lote"]), []).append(c)
    resumen = []
    claves = ["seg_por_epoca", "pico_vram_mb", "ms_por_imagen", "F1_val_antracnosis", "F1_val_roña"]
    for (prec, lote), cs in grupos.items():
        fila = {"dispositivo": cs[0]["dispositivo"], "precision": prec, "lote": lote, "semillas": len(cs)}
        for k in claves:
            vals = [c[k] for c in cs if isinstance(c[k], (int, float))]
            if vals:
                fila[k + "_media"] = round(statistics.mean(vals), 4)
                fila[k + "_sd"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
            else:
                fila[k + "_media"] = "no aplica"
                fila[k + "_sd"] = "no aplica"
        resumen.append(fila)
    return resumen


def graficar(resumen, carpeta, dispositivo):
    fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 4.5))
    for prec in sorted({f["precision"] for f in resumen}):
        sub = [f for f in resumen if f["precision"] == prec]
        lotes = [f["lote"] for f in sub]
        izq.errorbar(lotes, [f["seg_por_epoca_media"] for f in sub],
                     yerr=[f["seg_por_epoca_sd"] for f in sub], marker="o", capsize=3, label=prec)
        der.errorbar(lotes, [f["ms_por_imagen_media"] for f in sub],
                     yerr=[f["ms_por_imagen_sd"] for f in sub], marker="o", capsize=3, label=prec)
    izq.set(title="Entrenamiento por época", xlabel="Tamaño de lote", ylabel="segundos")
    der.set(title="Inferencia, lote de 1", xlabel="Tamaño de lote de entrenamiento", ylabel="ms por imagen")
    for eje in (izq, der):
        eje.legend()
        eje.grid(alpha=0.3)
    fig.suptitle(f"Recursos medidos en {dispositivo} (media ± sd sobre semillas)")
    fig.tight_layout()
    fig.savefig(carpeta / "recursos.png", dpi=150)
    plt.close(fig)


def escribir_csv(ruta, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--lotes", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--semillas", type=int, nargs="+", default=[42, 7, 123])
    ap.add_argument("--reps", type=int, default=300, help="repeticiones para medir la inferencia")
    ap.add_argument("--salida", default="resultados_recursos")
    ap.add_argument("--pregunta", type=int, default=12)
    a = ap.parse_args()

    t0 = time.time()
    bitacora.encabezado("Recursos de hardware, tamaño de lote y precisión numérica", a.pregunta)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", dev)
    out = Path(a.salida)
    out.mkdir(exist_ok=True)
    info = hardware(dev)
    (out / "hardware.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Hardware:", info)

    D = ex.cargar_todo(a.data)
    Xtr, Ytr = D["X"][D["tr"]], D["Y"][D["tr"]]
    Xva, Yva = D["X"][D["va"]], D["Y"][D["va"]].numpy().astype(int)
    precisiones = ["fp32", "amp"] if dev.type == "cuda" else ["fp32"]

    corridas = []
    for prec in precisiones:
        for lote in a.lotes:
            for semilla in a.semillas:
                m, seg, pico = entrenar_medido(Xtr, Ytr, lote, prec, dev, a.epocas, semilla)
                met = pl.metricas(Yva, ex.predecir(m, Xva, dev))
                ms = latencia_ms(m, Xva, dev, prec == "amp", a.reps)
                fila = {"dispositivo": str(dev), "precision": prec, "lote": lote, "semilla": semilla,
                        "seg_por_epoca": round(seg, 3),
                        "pico_vram_mb": round(pico, 1) if pico is not None else "no aplica",
                        "ms_por_imagen": round(ms, 4),
                        "F1_val_antracnosis": round(met["F1_antracnosis"], 4),
                        "F1_val_roña": round(met["F1_roña"], 4)}
                corridas.append(fila)
                print(fila)
                del m
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

    resumen = resumir(corridas)
    escribir_csv(out / "recursos_corridas.csv", corridas)
    escribir_csv(out / "recursos_resumen.csv", resumen)
    graficar(resumen, out, str(dev))
    bitacora.cerrar(t0, out, "recursos", {"Dispositivo": dev, "Épocas": a.epocas, "Lotes": a.lotes,
                                         "Semillas": a.semillas, "Repeticiones de inferencia": a.reps,
                                         "Precisiones": precisiones,
                                         "Imágenes de entrenamiento": len(Xtr),
                                         "Imágenes de validación": len(Xva)})


if __name__ == "__main__":
    main()
