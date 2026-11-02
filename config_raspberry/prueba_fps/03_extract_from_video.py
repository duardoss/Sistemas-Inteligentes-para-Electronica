"""
Paso 3 (cuando llegue el dataset): videos -> CSV de características, mismo formato
que 02_record_session.py (label = -1 hasta que se asigne con las anotaciones).

Uso:
    python 03_extract_from_video.py RUTA_DATASET --out-dir data/features --model face_landmarker.task
RUTA_DATASET puede ser un video o una carpeta (se busca de forma recursiva).
Es reanudable: si el CSV de un video ya existe, se salta.
Funciona sin ventana (apto para Colab).
"""
import argparse
import csv
import os

import cv2

from face_features import FEATURE_NAMES, FaceFeatureExtractor

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg")


def find_videos(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
        else:
            for root, _, files in os.walk(p):
                for name in sorted(files):
                    if name.lower().endswith(VIDEO_EXT):
                        yield os.path.join(root, name)


def process_video(path, out_csv, model, max_width):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  No se pudo abrir {path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0
    extractor = FaceFeatureExtractor(model)
    idx = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "frame_idx", "label"] + FEATURE_NAMES)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_width and frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            t_ms = idx * 1000.0 / fps
            feats, _ = extractor.process(frame, t_ms)
            w.writerow([f"{t_ms:.1f}", idx, -1] + [f"{v:.4f}" for v in feats])
            idx += 1
            if idx % 500 == 0:
                print(f"  {idx} cuadros...")
    extractor.close()
    cap.release()
    print(f"  Listo: {idx} cuadros a {fps:.1f} FPS -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--model", default="face_landmarker.task")
    ap.add_argument("--out-dir", default="data/features")
    ap.add_argument("--max-width", type=int, default=640,
                    help="reduce cuadros más anchos (acelera); 0 = sin reducir")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for video in find_videos(args.inputs):
        # nombre único a partir de la ruta (evita colisiones entre subcarpetas)
        rel = os.path.splitext(os.path.normpath(video))[0]
        name = rel.replace(os.sep, "__").replace(":", "").strip("._")
        out_csv = os.path.join(args.out_dir, name + ".csv")
        if os.path.exists(out_csv):
            print(f"Saltando (ya existe): {out_csv}")
            continue
        print(f"Procesando {video}")
        process_video(video, out_csv, args.model, args.max_width)


if __name__ == "__main__":
    main()
