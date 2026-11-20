"""Prueba final de la CNN multi-etiqueta (5 bloques): entrena con train+validación (535 imágenes)
y evalúa UNA vez en prueba (134 imágenes).

Variantes (elegidas con validación cruzada, no con la prueba):
  base:     sin aumento, 10 épocas, tasa fija (ya evaluada en prueba antes).
  mejorada: aumento de datos, 20 épocas, tasa con coseno (variante elegida).
Resultados en resultados_prueba_final\\ (base) y resultados_prueba_final_mejorada\\ (mejorada).
No vuelve a evaluar si ya existe el resultado de esa variante, salvo con --forzar.
"""
import argparse
import json
import time
from pathlib import Path

import torch

import bitacora
import entrenamiento as en
import experimentos as ex
import pipeline as pl

COMBOS = {"sano (0,0)": (0, 0), "solo antracnosis (1,0)": (1, 0),
          "solo roña (0,1)": (0, 1), "co-infección (1,1)": (1, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", choices=list(en.VARIANTES), default="mejorada")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--pregunta", type=int, default=None)
    ap.add_argument("--forzar", action="store_true", help="permite volver a evaluar la prueba")
    a = ap.parse_args()
    v = en.VARIANTES[a.variante]
    out = Path("resultados_prueba_final" if a.variante == "base" else "resultados_prueba_final_mejorada")
    archivo = out / "metricas_finales.json"

    t0 = time.time()
    bitacora.encabezado(f"Prueba final de la CNN multi-etiqueta, 5 bloques, variante {a.variante}", a.pregunta)
    if archivo.exists() and not a.forzar:
        raise SystemExit(f"Ya existe {archivo}. La prueba se evalúa una sola vez; usa --forzar para repetirla.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", dev)
    D = ex.cargar_todo(a.data)
    trv = torch.cat([D["tr"], D["va"]])
    modelo = en.entrenar_multi(en.CONFIG_FINAL, D["X"][trv], D["Y"][trv], v["epochs"],
                               v["aug"], v["sched"], dev, 42)

    P = ex.predecir(modelo, D["X"][D["te"]], dev)
    Yte = D["Y"][D["te"]].numpy().astype(int)
    pred = (P >= 0.5).astype(int)
    met = pl.metricas(Yte, P)
    matriz = []
    for nombre, (ca, cr_) in COMBOS.items():
        real = (Yte[:, 0] == ca) & (Yte[:, 1] == cr_)
        acierto = (pred[:, 0] == ca) & (pred[:, 1] == cr_) & real
        matriz.append({"clase_real": nombre, "n_en_test": int(real.sum()),
                       "aciertos_exactos": int(acierto.sum())})

    out.mkdir(exist_ok=True)
    torch.save({"state_dict": modelo.state_dict(), "config": dict(en.CONFIG_FINAL, variante=a.variante),
                "epocas": v["epochs"]}, out / "modelo_final.pt")
    salida = {"config": dict(en.CONFIG_FINAL, variante=a.variante, aumento=v["aug"], coseno=v["sched"]),
              "epocas": v["epochs"], "metricas_test": met, "matriz_por_combinacion": matriz}
    archivo.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Métricas en prueba:", {k: round(x, 4) for k, x in met.items()})
    for fila in matriz:
        print(fila)
    bitacora.cerrar(t0, out, "prueba_final", {"Dispositivo": dev, "Variante": a.variante,
                                             "Épocas": v["epochs"], "Imágenes de entrenamiento": len(trv),
                                             "Imágenes de prueba": len(Yte)})


if __name__ == "__main__":
    main()
