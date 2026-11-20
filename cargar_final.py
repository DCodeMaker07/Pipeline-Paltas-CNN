"""Carga el modelo final guardado por prueba_final.py y predice sobre el conjunto de prueba.

No reentrena ni cambia resultados: solo lee resultados_prueba_final_mejorada\\modelo_final.pt,
recalcula las métricas en prueba y las compara con metricas_finales.json.
Pensado para la Pregunta 10 (carga del modelo seleccionado y predicciones).
"""
import argparse
import json
import time
from pathlib import Path

import torch

import bitacora
import experimentos as ex
import pipeline as pl

COMBOS = {"sano (0,0)": (0, 0), "solo antracnosis (1,0)": (1, 0),
          "solo roña (0,1)": (0, 1), "co-infección (1,1)": (1, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", choices=["base", "mejorada"], default="mejorada")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--pregunta", type=int, default=None)
    a = ap.parse_args()

    carpeta = Path("resultados_prueba_final" if a.variante == "base" else "resultados_prueba_final_mejorada")
    ruta_modelo = carpeta / "modelo_final.pt"
    ruta_metricas = carpeta / "metricas_finales.json"
    if not ruta_modelo.exists():
        raise SystemExit(f"No existe {ruta_modelo}. Primero corran prueba_final.py.")

    t0 = time.time()
    bitacora.encabezado(f"Carga del modelo final (variante {a.variante}) y predicción en prueba", a.pregunta)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", dev)

    guardado = torch.load(ruta_modelo, map_location=dev, weights_only=False)
    cfg = {k: v for k, v in guardado["config"].items() if k != "variante"}
    modelo = ex.crear_modelo("cnn", cfg).to(dev)
    modelo.load_state_dict(guardado["state_dict"])
    modelo.eval()
    print("Modelo cargado. Configuración:", cfg, "| épocas de entrenamiento:", guardado["epocas"])

    D = ex.cargar_todo(a.data)
    Xte = D["X"][D["te"]]
    Yte = D["Y"][D["te"]].numpy().astype(int)
    P = ex.predecir(modelo, Xte, dev)
    pred = (P >= 0.5).astype(int)
    met = pl.metricas(Yte, P)

    print("Métricas en prueba (recalculadas):", {k: round(x, 4) for k, x in met.items()})
    if ruta_metricas.exists():
        guardadas = json.loads(ruta_metricas.read_text(encoding="utf-8"))["metricas_test"]
        coincide = all(abs(met[k] - guardadas[k]) < 1e-3 for k in met)
        print("Coinciden con metricas_finales.json:", coincide)

    print("Aciertos exactos por combinación:")
    for nombre, (ca, cr_) in COMBOS.items():
        real = (Yte[:, 0] == ca) & (Yte[:, 1] == cr_)
        acierto = (pred[:, 0] == ca) & (pred[:, 1] == cr_) & real
        print(f"  {nombre}: {int(acierto.sum())}/{int(real.sum())}")

    bitacora.cerrar(t0, carpeta, "cargar_final", {"Dispositivo": dev, "Variante": a.variante,
                                                 "Imágenes de prueba": len(Yte)})


if __name__ == "__main__":
    main()
