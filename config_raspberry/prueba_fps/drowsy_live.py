"""
drowsy_live_buzzer.py  -- se corre en la maquina virtual / Raspberry Pi (con pantalla)

Cámara -> MediaPipe (face_features.py) -> tuberia en vivo (live_pipeline.py) ->
modelo ONNX -> ventana de OpenCV con las probabilidades y una ALARMA VISUAL.
El buzzer en GPIO17 sigue el mismo intervalo de la ALARMA VISUAL.

Archivos que deben estar en la MISMA carpeta que este script:
    face_features.py, camera.py   (los de tu proyecto)
    preprocess.py, live_pipeline.py
y aparte: face_landmarker.task y la carpeta de la corrida (model.onnx + contract.json).

Uso:
    python drowsy_live_buzzer.py --model-dir /home/admin/drowsy_models/NOMBRE_DE_LA_CORRIDA
Opciones: --landmarker RUTA (por defecto face_landmarker.task), --camera 0,
          --width 640 --height 480 --fps 30, --log-csv salida.csv, --no-window
Criterio de alarma (si no se indica, se usa el del contract.json de la corrida):
          --threshold 0.9      probabilidad minima de la clase de alarma
          --hold-seconds 3     segundos seguidos por encima del umbral
          --refractory 5       segundos de silencio despues de una alarma
Teclas: q o ESC para salir.

Como se usa: al empezar, MIRA AL FRENTE con la cabeza en tu postura normal durante
30 s (calibracion). Despues empieza a evaluar. Para probar la alarma: cierra los ojos
unos 2 s seguidos.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from gpiozero import Buzzer

import preprocess as pp
from camera import ThreadedCamera
from face_features import FaceFeatureExtractor, draw_overlay
from live_pipeline import AlarmLogic, LivePipeline

ALARM_DISPLAY_S = 3.0        # cuanto tiempo se queda el cartel de alarma en pantalla
DEFAULT_ALARM = {"threshold": 0.9, "hold_windows": 2, "refractory_s": 5.0}


def draw_bars(img, names, probs, target, thr):
    h = img.shape[0]
    y0 = h - 20 - 22 * len(names)
    for i, (nm, p) in enumerate(zip(names, probs)):
        y = y0 + 22 * i
        hot = i == target and p >= thr
        color = (0, 0, 255) if hot else (0, 200, 0)
        cv2.rectangle(img, (110, y), (110 + int(160 * p), y + 14), color, -1)
        cv2.rectangle(img, (110, y), (270, y + 14), (255, 255, 255), 1)
        cv2.putText(img, f"{nm[:10]:<10} {p:.2f}", (8, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)


def banner(img, text, color):
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 40), color, -1)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--landmarker", default="face_landmarker.task")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--log-csv", default=None)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--hold-seconds", type=float, default=None)
    ap.add_argument("--refractory", type=float, default=None)
    args = ap.parse_args()

    mdir = Path(args.model_dir).expanduser()
    contract = json.loads((mdir / "contract.json").read_text(encoding="utf-8"))
    names = contract["output"]["classes"]
    if contract["input"]["shape"] != [1, pp.WINDOW_STEPS, len(pp.CHANNELS)]:
        raise SystemExit(f"El contrato pide la entrada {contract['input']['shape']}, no la que prepara este programa.")
    if contract.get("smoke_test_only"):
        print("AVISO: modelo de PRUEBA DE HUMO: no sirve para decidir nada real.")
    target = next(i for i, n in enumerate(names) if "microsueno" in n)
    alarm_cfg = contract.get("alarm_thresholds") or {}
    if not alarm_cfg:
        print("AVISO: el contrato no trae umbral de alarma; se usan valores por defecto "
              f"{DEFAULT_ALARM}.")
    thr = alarm_cfg.get("threshold", DEFAULT_ALARM["threshold"])
    hold = alarm_cfg.get("hold_windows", DEFAULT_ALARM["hold_windows"])
    refractory = alarm_cfg.get("refractory_s", DEFAULT_ALARM["refractory_s"])
    step_s = pp.STRIDE_STEPS * pp.STEP_MS / 1000.0     # una evaluacion cada 0.5 s
    source = "contract.json" if alarm_cfg else "valores por defecto"
    if args.threshold is not None or args.hold_seconds is not None or args.refractory is not None:
        source = "linea de comandos (sobrescribe el contract.json)"
    if args.threshold is not None:
        thr = args.threshold
    if args.hold_seconds is not None:
        hold = max(1, int(np.ceil(args.hold_seconds / step_s - 1e-9)))
    if args.refractory is not None:
        refractory = args.refractory
    print(f"Modelo: {contract['run_id']} | alarma: '{names[target]}' >= {thr} durante {hold} evaluacion(es) seguida(s) "
          f"(~{hold * step_s:.1f} s), silencio {refractory} s | criterio de: {source}")

    sess = ort.InferenceSession(str(mdir / "model.onnx"), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    extractor = FaceFeatureExtractor(args.landmarker)
    cam = ThreadedCamera(args.camera, args.width, args.height, args.fps)
    pipe, alarm = LivePipeline(), AlarmLogic(thr, hold, refractory)

    # El buzzer no evalua probabilidades ni tiene un umbral independiente.
    # Solo replica la vigencia del aviso visual (alarm_until).
    buzzer = Buzzer(17)
    buzzer.off()
    buzzer_active = False

    log_f = log_w = None
    if args.log_csv:
        log_f = open(args.log_csv, "w", newline="", encoding="utf-8")
        log_w = csv.writer(log_f)
        log_w.writerow(["t_s"] + names + ["alarm"])

    probs, alarm_until, t0, last_count = None, 0.0, None, -1
    n_frames, t_fps, fps, last_print = 0, time.monotonic(), 0.0, 0.0
    print("Calibrando: mira al frente con tu postura normal durante 30 s...")
    try:
        while True:
            # Mantener la salida sincronizada aun si la camara tarda o falla.
            visual_alarm_active = pipe.state != "calibrando" and time.monotonic() < alarm_until
            if visual_alarm_active != buzzer_active:
                if visual_alarm_active:
                    buzzer.on()
                else:
                    buzzer.off()
                buzzer_active = visual_alarm_active
            frame, ts, count = cam.read(last_count, timeout=2.0)
            if frame is None:
                print("Sin cuadros de la camara (2 s). ¿Esta conectada a la maquina virtual?")
                continue
            last_count = count
            if t0 is None:
                t0 = ts
            t_ms = (ts - t0) * 1000.0
            feats, lm = extractor.process(frame, t_ms)

            if pipe.push_frame(t_ms, feats):
                probs = sess.run(None, {in_name: pipe.window()})[0][0]
                fired = alarm.update(t_ms / 1000.0, float(probs[target]))
                if fired:
                    alarm_until = time.monotonic() + ALARM_DISPLAY_S
                    print(f"[{t_ms / 1000:7.1f} s] *** ALARMA: {names[target]} p={probs[target]:.2f} ***")
                if log_w:
                    log_w.writerow([f"{t_ms / 1000:.2f}"] + [f"{p:.4f}" for p in probs] + [int(fired)])

            # Usar exactamente la condicion del cartel, sin otro threshold.
            visual_alarm_active = pipe.state != "calibrando" and time.monotonic() < alarm_until
            if visual_alarm_active != buzzer_active:
                if visual_alarm_active:
                    buzzer.on()
                else:
                    buzzer.off()
                buzzer_active = visual_alarm_active

            n_frames += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                fps, n_frames, t_fps = n_frames / (now - t_fps), 0, now
            if now - last_print >= 5.0:
                last_print = now
                msg = f"calibrando {100 * pipe.calib_progress:.0f}%" if pipe.state == "calibrando" else \
                    "probs " + " ".join(f"{n[:4]}={p:.2f}" for n, p in zip(names, probs)) if probs is not None else "en vivo"
                print(f"[{t_ms / 1000:7.1f} s] {fps:4.1f} FPS | {msg}")

            if not args.no_window:
                view = frame.copy()
                extra = [f"{fps:.1f} FPS | {pipe.state}"]
                if pipe.state == "en_vivo":
                    extra.append(f"sobre umbral: {alarm.run}/{alarm.hold} evaluaciones")
                if pipe.fail_msg:
                    extra.append(pipe.fail_msg[:48])
                draw_overlay(view, lm, feats, extra)
                if pipe.state == "calibrando":
                    left = max(0.0, (1 - pipe.calib_progress) * pp.CAL_SECONDS)
                    banner(view, f"CALIBRANDO: mira al frente ({left:.0f} s)", (160, 100, 0))
                elif visual_alarm_active:
                    banner(view, f"ALARMA: {names[target].upper()}", (0, 0, 220))
                if probs is not None and pipe.state == "en_vivo":
                    draw_bars(view, names, probs, target, thr)
                cv2.imshow("Deteccion de somnolencia (q para salir)", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        # Apagado garantizado al salir, incluso por Ctrl+C o una excepcion.
        buzzer.off()
        buzzer.close()
        cam.release()
        extractor.close()
        if log_f:
            log_f.close()
        if not args.no_window:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print("Terminado.")


if __name__ == "__main__":
    main()
