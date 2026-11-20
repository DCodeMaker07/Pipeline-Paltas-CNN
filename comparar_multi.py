"""Compara los modelos multi-etiqueta con la variante mejorada (mismos folds y semillas).

Lee resultados_profundidad\\multi_cv_mejorada.csv (CNN 5 bloques),
multi_cv_mobilenet_v2_mejorada.csv y multi_cv_resnet18_mejorada.csv.
"""
import csv
import statistics
from pathlib import Path

import bitacora
import comparar_rona as cr

OUT = Path("resultados_profundidad")
ARCHIVOS = {
    "cnn_5_bloques": OUT / "multi_cv_mejorada.csv",
    "mobilenet_v2": OUT / "multi_cv_mobilenet_v2_mejorada.csv",
    "resnet18": OUT / "multi_cv_resnet18_mejorada.csv",
}
METRICAS = ["F1_antracnosis", "AUC_antracnosis", "F1_roña", "AUC_roña", "subset_accuracy", "hamming_loss"]


def cargar():
    filas = []
    for nombre, ruta in ARCHIVOS.items():
        with open(ruta, encoding="utf-8") as fh:
            filas += [dict(r, modelo=nombre) for r in csv.DictReader(fh)]
    return filas


def main():
    bitacora.encabezado("Comparación de modelos multi-etiqueta con la variante mejorada")
    filas = cargar()
    print("Resumen (media ± desviación; semillas x folds):")
    for clave in METRICAS:
        partes = []
        for nombre in ARCHIVOS:
            vals = [float(r[clave]) for r in filas if r["modelo"] == nombre]
            partes.append(f"{nombre} {statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}")
        print(f"  {clave:18s}: " + " | ".join(partes))

    print("\nCNN 5 bloques frente a cada preentrenado (pares semilla x fold):")
    for otro in ["mobilenet_v2", "resnet18"]:
        for clave in METRICAS:
            if clave == "hamming_loss":
                g, n, p = cr.comparar_pareado(filas, otro, "cnn_5_bloques", clave)
                etiqueta = f"{otro} con menor error"
            else:
                g, n, p = cr.comparar_pareado(filas, "cnn_5_bloques", otro, clave)
                etiqueta = f"{otro} mejor"
            print(f"  {clave:18s} ({etiqueta}): {g} de {n} | p = {p:.4f}")



if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    bitacora.cerrar(t0, OUT, "comparar_multi", {"Modelos": list(ARCHIVOS)})
