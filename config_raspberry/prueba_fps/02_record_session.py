"""
Paso 2: grabar tus propios datos (características + etiqueta) a CSV.

Uso:
    python 02_record_session.py --subject yo --model face_landmarker.task
Teclas (ventana activa):
    0 = normal   1 = bostezo   2 = cabeceo   3 = microsueño (cierre prolongado simulado)
    q = terminar
Recomendación: graba tomas separadas y cortas por clase; cambia la etiqueta ANTES
de empezar el gesto y vuelve a 0 al terminar.
El CSV guarda marcas de tiempo reales (t_ms): luego se remuestrea a tasa fija.
"""
import argparse
import csv
import os
import time

import cv2

from camera import ThreadedCamera
from face_features import FEATURE_NAMES, FaceFeatureExtractor, draw_overlay

LABEL_NAMES = {0: "normal", 1: "bostezo", 2: "cabeceo", 3: "microsueno"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="identificador de la persona")
    ap.add_argument("--model", default="face_landmarker.task")
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--label", type=int, default=0)
    ap.add_argument("--save-video", action="store_true",
                    help="guarda un .mp4 SOLO para revisar/etiquetar visualmente")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.out_dir, f"{args.subject}_{stamp}")

    cam = ThreadedCamera(args.camera, args.width, args.height, 30)
    extractor = FaceFeatureExtractor(args.model)
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(base + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                 20, (cam.width, cam.height))

    label = args.label
    t_start = time.monotonic()
    last_count, rows = -1, 0

    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "frame_idx", "label"] + FEATURE_NAMES)
        try:
            while True:
                frame, ts, last_count = cam.read(last_count)
                if frame is None:
                    break
                t_ms = (ts - t_start) * 1000.0
                feats, lm = extractor.process(frame, t_ms)
                w.writerow([f"{t_ms:.1f}", last_count, label] + [f"{v:.4f}" for v in feats])
                rows += 1
                if rows % 100 == 0:
                    f.flush()

                if writer is not None:
                    writer.write(frame)
                draw_overlay(frame, lm, feats,
                             [f"ETIQUETA: {label} ({LABEL_NAMES.get(label, '?')})  |  filas: {rows}"])
                cv2.imshow("record", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in (ord("0"), ord("1"), ord("2"), ord("3")):
                    label = int(chr(key))
        finally:
            cam.release()
            extractor.close()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    print(f"Guardado: {base}.csv ({rows} filas)")


if __name__ == "__main__":
    main()
