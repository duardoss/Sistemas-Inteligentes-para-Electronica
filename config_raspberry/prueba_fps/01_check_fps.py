"""
Paso 1: ¿MediaPipe corre fluido en tu equipo? ¿Los valores tienen sentido?

Uso:
    python 01_check_fps.py --model face_landmarker.task
    python 01_check_fps.py --model face_landmarker.task --seconds 30 --no-display   # por SSH
Teclas (con ventana): q = salir
"""
import argparse
import os
import time

import cv2
import numpy as np

from camera import ThreadedCamera
from face_features import FaceFeatureExtractor, draw_overlay


def cpu_temp_c():
    path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(path):
        with open(path) as f:
            return int(f.read().strip()) / 1000.0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="face_landmarker.task")
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seconds", type=float, default=0, help="0 = hasta pulsar q")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    cam = ThreadedCamera(args.camera, args.width, args.height, 30)
    print(f"Cámara: {cam.width}x{cam.height}, FPS reportados: {cam.reported_fps}")
    extractor = FaceFeatureExtractor(args.model)

    t_start = time.monotonic()
    last_count = -1
    proc_ms, found, n_proc = [], 0, 0
    fps_text, fps_t0, fps_n = "", time.monotonic(), 0
    first_count = None

    try:
        while True:
            frame, ts, last_count = cam.read(last_count)
            if frame is None:
                print("Sin cuadros de la cámara (timeout).")
                break
            if first_count is None:
                first_count = last_count

            t0 = time.perf_counter()
            feats, lm = extractor.process(frame, (ts - t_start) * 1000.0)
            proc_ms.append((time.perf_counter() - t0) * 1000.0)
            n_proc += 1
            found += int(feats[0] > 0)

            fps_n += 1
            now = time.monotonic()
            if now - fps_t0 >= 1.0:
                temp = cpu_temp_c()
                fps_text = f"FPS proc: {fps_n / (now - fps_t0):.1f}"
                if temp is not None:
                    fps_text += f" | CPU {temp:.0f}C"
                fps_t0, fps_n = now, 0

            if not args.no_display:
                draw_overlay(frame, lm, feats, [fps_text])
                cv2.imshow("check_fps", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif fps_text and n_proc % 30 == 0:
                print(fps_text, "| EAR", np.round(feats[1:3], 2), "MAR", round(float(feats[3]), 2),
                      "pitch", round(float(feats[4]), 1))

            if args.seconds and (time.monotonic() - t_start) >= args.seconds:
                break
    finally:
        elapsed = time.monotonic() - t_start
        captured = cam.frames_captured
        cam.release()
        extractor.close()
        cv2.destroyAllWindows()

    if n_proc:
        p = np.array(proc_ms)
        print("\n===== RESUMEN =====")
        print(f"Duración: {elapsed:.1f} s")
        print(f"Cuadros capturados por la cámara: {captured} ({captured / elapsed:.1f} FPS)")
        print(f"Cuadros procesados: {n_proc} ({n_proc / elapsed:.1f} FPS)")
        print(f"Latencia MediaPipe: media {p.mean():.1f} ms | p95 {np.percentile(p, 95):.1f} ms")
        print(f"Cara detectada en {100.0 * found / n_proc:.0f}% de los cuadros")
        temp = cpu_temp_c()
        if temp is not None:
            print(f"Temperatura CPU al final: {temp:.0f} C")


if __name__ == "__main__":
    main()
