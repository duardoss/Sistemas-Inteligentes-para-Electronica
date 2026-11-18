"""
drowsy_live.py  -- se corre en la Raspberry Pi (con pantalla) o en la maquina virtual

Camara -> MediaPipe (face_features.py) -> tuberia en vivo (live_pipeline.py) ->
modelo ONNX -> ventana de OpenCV con las probabilidades y una ALARMA VISUAL.
(El buzzer por GPIO se agrega despues; no se usa aqui.)

AL TERMINAR (q, ESC o Ctrl+C) IMPRIME UN RECUENTO y lo guarda en un archivo
resumen_sesion_AAAAMMDD_HHMMSS.txt (en la carpeta donde ejecutaste el programa):
  A. equipo (si es una Raspberry Pi fisica o no), B. resultados de la corrida de Colab
  (macro-F1, matriz de confusion), C. regla de alarma elegida y usada y sus falsas alarmas
  por hora, D. FPS y latencias MEDIDOS en esta sesion, E. alarmas y gestos marcados,
  F. pruebas de aceptacion (CUMPLE / NO CUMPLE / N/A).
Los resultados de Colab se leen de metrics.json, alarm_config.json y contract.json de la
carpeta de la corrida; las mediciones de FPS y latencia son de ESTA sesion.

Archivos que deben estar en la MISMA carpeta que este script:
    face_features.py, camera.py   (los de tu proyecto)
    preprocess.py, live_pipeline.py
y aparte: face_landmarker.task y la carpeta de la corrida (model.onnx + contract.json y,
para el recuento completo, metrics.json y alarm_config.json).

Uso:
    python drowsy_live.py --model-dir /home/admin/drowsy_models/NOMBRE_DE_LA_CORRIDA
Opciones: --landmarker RUTA (por defecto face_landmarker.task), --camera 0,
          --width 640 --height 480 --fps 30, --log-csv salida.csv, --no-window
Criterio de alarma (si no se indica, se usa el del contract.json de la corrida):
          --threshold 0.9      probabilidad minima de la clase de alarma
          --hold-seconds 3     segundos seguidos por encima del umbral
          --refractory 5       segundos de silencio despues de una alarma
Teclas: q o ESC para salir; g para MARCAR el inicio de un gesto simulado.

Como se usa: al empezar, MIRA AL FRENTE con la cabeza en tu postura normal durante
30 s (calibracion). Despues empieza a evaluar.

Para las PRUEBAS DE ACEPTACION (deteccion y latencia):
  - Cada vez que EMPIECES un gesto simulado (ojos cerrados ~2 s, cabeceo), pulsa g
    (con --no-window, pulsa Enter en la terminal). El programa mide cuantas de tus marcas
    activaron la alarma y en cuantos segundos. Las alarmas SIN marca cercana se cuentan
    como falsas alarmas.
  - Para medir falsas alarmas por hora: uso normal, sin marcas, al menos 30 min.
"""
import argparse
import csv
import json
import platform
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import preprocess as pp
from camera import ThreadedCamera
from face_features import FaceFeatureExtractor, draw_overlay
from live_pipeline import AlarmLogic, LivePipeline

ALARM_DISPLAY_S = 3.0        # cuanto tiempo se queda el cartel de alarma en pantalla
DEFAULT_ALARM = {"threshold": 0.9, "hold_windows": 2, "refractory_s": 5.0}

# --- Criterios de aceptacion. OJO: son PROPUESTAS del asistente (no vienen de una norma);
#     cambialos aqui si tu proyecto exige otros.
ACCEPT = {
    "fps_min": 10.0,                 # el modelo trabaja a 10 Hz: al menos ~1 cuadro por paso
    "frame_latency_p95_ms": 200.0,   # captura -> caracteristicas -> (inferencia) por cuadro
    "onnx_p95_ms": 20.0,             # una inferencia del modelo
    "face_pct_min": 80.0,            # % de cuadros con cara
    "detect_rate_min": 0.90,         # gestos marcados que activan la alarma
    "alarm_latency_median_s": 3.0,   # segundos desde la marca hasta la alarma (mediana)
    "min_marks": 10,                 # marcas minimas para que la tasa de deteccion tenga sentido
    "min_live_s_for_fa": 1800.0,     # sesion minima (30 min) para juzgar falsas alarmas/hora
    "max_fa_per_hour": 1.0,          # meta de falsas alarmas por hora (se toma del contrato si existe)
}
MATCH_S = 8.0     # una alarma se asigna a una marca si ocurre hasta 8 s despues de ella
EARLY_S = 0.5     # ... o hasta 0.5 s antes


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Equipo y estado de la Pi
# ---------------------------------------------------------------------------
def run_cmd(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def hw_info():
    model = None
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00\n ")
    except Exception:  # noqa: BLE001
        pass
    return {"model": model, "is_pi": bool(model and "Raspberry Pi" in model),
            "machine": platform.machine(), "system": platform.system(), "python": platform.python_version()}


def pi_status():
    return {"throttled": run_cmd(["vcgencmd", "get_throttled"]), "temp": run_cmd(["vcgencmd", "measure_temp"]),
            "clock": run_cmd(["vcgencmd", "measure_clock", "arm"])}


# ---------------------------------------------------------------------------
# Estadisticas
# ---------------------------------------------------------------------------
def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def mean(a):
    return float(np.mean(a)) if len(a) else float("nan")


def match_marks(marks, alarms):
    """Asigna alarmas a marcas de gesto. Devuelve (detectadas, latencias, alarmas_sin_marca)."""
    detected, lat, used = 0, [], set()
    for m in marks:
        hits = [a for a in alarms if m - EARLY_S <= a <= m + MATCH_S]
        if hits:
            detected += 1
            lat.append(hits[0] - m)
            used.update(hits)
    return detected, lat, [a for a in alarms if a not in used]


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def fmt_cm(cm, names, indent="  "):
    cm = np.array(cm)
    k = len(names)
    lines = [indent + "filas = clase REAL, columnas = clase PREDICHA (ventanas)",
             indent + "columnas: " + "  ".join(str(i) + "=" + n for i, n in enumerate(names))]
    lines.append(indent + " " * 22 + "".join(f"{'p' + str(i):>8}" for i in range(k)) + f"{'recall':>9}")
    for i in range(k):
        row_sum = cm[i].sum()
        rec = cm[i, i] / row_sum if row_sum else float("nan")
        label = str(i) + ":" + names[i][:19]
        lines.append(indent + f"{label:<22}" + "".join(f"{int(v):>8d}" for v in cm[i]) + f"{rec:>9.2f}")
    prec = [cm[j, j] / cm[:, j].sum() if cm[:, j].sum() else float("nan") for j in range(k)]
    lines.append(indent + f"{'precision':<22}" + "".join(f"{p:>8.2f}" for p in prec))
    return lines


def fmt_alarm_eval(title, m):
    if not m:
        return [f"  {title}: no disponible"]
    lat = "-"
    if m.get("latency_median_s") is not None:
        lat = f"{m['latency_median_s']:.1f} s (p90 {m['latency_p90_s']:.1f} s)"
    return [f"  {title}: eventos detectados {m['detected']}/{m['events']} (recall {m['recall']:.2f}) | "
            f"falsas alarmas {m['false_alarms']} en {m['hours']:.2f} h = {m['fa_per_hour']:.2f}/h "
            f"(cota 95 % aprox. {m['fa_per_hour_upper95_approx']:.1f}/h) | latencia mediana {lat}"]


def colab_sections(mdir, names, contract, used_rule):
    """Secciones B y C: resultados de la corrida de Colab (leidos de la carpeta del modelo)."""
    L = ["B. RESULTADOS DE LA CORRIDA (Colab; sujetos que el modelo NO vio durante el entrenamiento)"]
    metrics = load_json(mdir / "metrics.json")
    alarm_cfg_file = load_json(mdir / "alarm_config.json")
    if not metrics:
        L.append("  No se encontro metrics.json en la carpeta de la corrida (copiala completa desde Drive).")
    elif contract.get("smoke_test_only") or not metrics.get("reported") or "cv" not in metrics:
        L.append("  Corrida de PRUEBA DE HUMO: no tiene metricas de validacion con sentido.")
    else:
        cv = metrics["cv"]
        L.append(f"  Validacion cruzada: macro-F1 por sujeto = {cv['macro_f1_mean']:.3f} +- {cv['macro_f1_std']:.3f}")
        L.append("  por sujeto: " + ", ".join(f"{s}={v:.3f}" for s, v in cv["per_subject"].items()))
        L.append("  Matriz de confusion (validacion cruzada):")
        L += fmt_cm(cv["confusion_matrix"], names, indent="    ")
        ft = metrics.get("frozen_test")
        if ft:
            L.append(f"  Prueba congelada: macro-F1 por sujeto = {ft['macro_f1_mean']:.3f} +- {ft['macro_f1_std']:.3f}"
                     "  (" + ", ".join(f"{s}={v:.3f}" for s, v in ft["per_subject"].items()) + ")")
            L.append("  Matriz de confusion (prueba congelada):")
            L += fmt_cm(ft["confusion_matrix"], names, indent="    ")
    L += ["", "C. REGLA DE ALARMA Y FALSAS ALARMAS POR HORA"]
    chosen = contract.get("alarm_thresholds") or {}
    if chosen:
        L.append(f"  Elegida en Colab (contract.json): '{chosen.get('class')}' con probabilidad >= {chosen.get('threshold')} "
                 f"durante {chosen.get('hold_windows')} evaluacion(es) seguida(s) (1 evaluacion = 0.5 s); "
                 f"silencio {chosen.get('refractory_s')} s; meta {chosen.get('max_false_alarms_per_hour')} falsas alarmas/hora")
    else:
        L.append("  El contract.json no trae regla de alarma (corrida sin validacion por sujeto).")
    if alarm_cfg_file and "meets_max_in_cv" in alarm_cfg_file and not alarm_cfg_file["meets_max_in_cv"]:
        L.append("  AVISO: en Colab NINGUNA regla cumplio el maximo de falsas alarmas; se eligio la de menos falsas alarmas.")
    L.append(f"  Usada en ESTA sesion: probabilidad >= {used_rule['thr']} durante {used_rule['hold']} evaluacion(es) "
             f"(~{used_rule['hold'] * used_rule['step_s']:.1f} s); silencio {used_rule['refractory']} s | origen: {used_rule['source']}")
    if metrics and metrics.get("alarm"):
        L += fmt_alarm_eval("Evaluacion en Colab, validacion cruzada", metrics["alarm"].get("cv"))
        L += fmt_alarm_eval("Evaluacion en Colab, prueba congelada  ", metrics["alarm"].get("frozen_test"))
        if chosen and used_rule["source"].startswith("linea"):
            L.append("  OJO: esas cifras de Colab son de la regla ELEGIDA en Colab, no de la usada en esta sesion.")
    else:
        L.append("  No hay evaluacion de alarma por evento guardada en metrics.json.")
    return L


def acceptance_rows(S, hw, st1, max_fa):
    rows = []

    def add(name, crit, value, ok):
        rows.append((name, crit, value, "N/A" if ok is None else ("CUMPLE" if ok else "NO CUMPLE")))

    add("Equipo = Raspberry Pi fisica", "es una Raspberry Pi", hw["model"] or f"{hw['system']} {hw['machine']}", hw["is_pi"])
    note = "" if hw["is_pi"] else " (NO es Pi)"
    add("FPS medio", f">= {ACCEPT['fps_min']:.0f}", f"{S['fps_mean']:.1f}{note}",
        None if S["frames"] < 60 else S["fps_mean"] >= ACCEPT["fps_min"])
    add("Latencia por cuadro (p95)", f"<= {ACCEPT['frame_latency_p95_ms']:.0f} ms", f"{pct(S['frame_lat'], 95):.0f} ms{note}",
        None if not S["frame_lat"] else pct(S["frame_lat"], 95) <= ACCEPT["frame_latency_p95_ms"])
    add("Inferencia del modelo (p95)", f"<= {ACCEPT['onnx_p95_ms']:.0f} ms", f"{pct(S['onnx'], 95):.1f} ms{note}",
        None if not S["onnx"] else pct(S["onnx"], 95) <= ACCEPT["onnx_p95_ms"])
    add("Cuadros con cara", f">= {ACCEPT['face_pct_min']:.0f} %", f"{S['face_pct']:.1f} %",
        None if S["frames"] == 0 else S["face_pct"] >= ACCEPT["face_pct_min"])
    add("Calibracion sin reinicios", "0 reinicios", f"{S['calib_restarts']} reinicio(s)",
        None if (S["live_s"] <= 0 and S["calib_restarts"] == 0) else S["calib_restarts"] == 0)
    thr_ok = st1["throttled"].strip().endswith("0x0") if st1.get("throttled") else None
    add("Sin limitacion por voltaje/temperatura", "get_throttled = 0x0", st1.get("throttled") or "sin vcgencmd", thr_ok)
    n_marks = len(S["marks"])
    if n_marks:
        add("Deteccion de gestos marcados", f">= {100 * ACCEPT['detect_rate_min']:.0f} % (min. {ACCEPT['min_marks']} marcas)",
            f"{S['detected']}/{n_marks} = {100 * S['detected'] / n_marks:.0f} %",
            None if n_marks < ACCEPT["min_marks"] else S["detected"] / n_marks >= ACCEPT["detect_rate_min"])
        add("Latencia de alarma (mediana)", f"<= {ACCEPT['alarm_latency_median_s']:.0f} s",
            f"{np.median(S['lat']):.1f} s" if S["lat"] else "ninguna alarma",
            None if not S["lat"] else float(np.median(S["lat"])) <= ACCEPT["alarm_latency_median_s"])
    else:
        add("Deteccion de gestos marcados", "pulsa g al empezar cada gesto", "sin marcas", None)
        add("Latencia de alarma (mediana)", "requiere marcas", "sin marcas", None)
    fa_val = f"{len(S['false'])} en {S['exposure_h'] * 60:.1f} min = {S['fa_per_h']:.2f}/h (cota {S['fa_upper']:.1f}/h)"
    enough = S["live_s"] >= ACCEPT["min_live_s_for_fa"]
    add("Falsas alarmas por hora", f"<= {max_fa:g}/h, sesion >= {ACCEPT['min_live_s_for_fa'] / 60:.0f} min", fa_val,
        None if not enough else S["fa_per_h"] <= max_fa)
    return rows


def build_report(mdir, contract, names, target, used_rule, S, hw, st0, st1, mp_ver):
    L = ["=" * 100, "RECUENTO DE LA SESION", "=" * 100,
         f"Fecha: {datetime.now():%Y-%m-%d %H:%M:%S} | corrida: {contract.get('run_id')} | regimen: {contract.get('regime')}",
         f"Clases: {names} | clase de alarma: '{names[target]}'", ""]
    L.append("A. EQUIPO")
    L.append(f"  {hw['model'] or 'Modelo de equipo desconocido'} | {hw['system']} {hw['machine']} | Python {hw['python']} | "
             f"onnxruntime {getattr(ort, '__version__', '?')} | mediapipe {mp_ver}")
    if not hw["is_pi"]:
        L.append("  *** ESTE EQUIPO NO ES UNA RASPBERRY PI (o no se pudo detectar). Las cifras de FPS y latencia "
                 "NO valen como medidas de la Pi fisica. ***")
    for k, lbl in (("throttled", "get_throttled"), ("temp", "temperatura"), ("clock", "reloj CPU")):
        if st0.get(k) or st1.get(k):
            L.append(f"  {lbl}: inicio = {st0.get(k)} | final = {st1.get(k)}")
    if contract.get("smoke_test_only"):
        L.append("  *** El modelo es de PRUEBA DE HUMO: no sirve para decidir nada real. ***")
    L.append("")
    L += colab_sections(mdir, names, contract, used_rule)
    L += ["", "D. MEDICIONES EN ESTA SESION" + ("" if hw["is_pi"] else " (equipo que NO es una Raspberry Pi)")]
    L.append(f"  Duracion: {S['total_s']:.0f} s en total | {S['live_s']:.0f} s evaluando (despues de la calibracion) | "
             f"cuadros: {S['frames']} | con cara: {S['face_pct']:.1f} %")
    fps_min = min(S["fps_samples"]) if S["fps_samples"] else float("nan")
    if S["fps_samples"]:
        L.append(f"  FPS: medio {S['fps_mean']:.1f} | mediana {pct(S['fps_samples'], 50):.1f} | minimo {fps_min:.1f}")
    else:
        L.append(f"  FPS: medio {S['fps_mean']:.1f} (sesion demasiado corta para medianas por segundo)")
    lat_max = max(S["frame_lat"]) if S["frame_lat"] else float("nan")
    L.append(f"  Latencia por cuadro (captura -> caracteristicas -> inferencia): media {mean(S['frame_lat']):.0f} ms | "
             f"mediana {pct(S['frame_lat'], 50):.0f} | p95 {pct(S['frame_lat'], 95):.0f} | max {lat_max:.0f} ms")
    L.append(f"  MediaPipe por cuadro: media {mean(S['mp']):.0f} ms | p95 {pct(S['mp'], 95):.0f} ms")
    L.append(f"  Inferencia del modelo ({len(S['onnx'])} evaluaciones): media {mean(S['onnx']):.2f} ms | p95 {pct(S['onnx'], 95):.2f} ms")
    L.append(f"  Reinicios de calibracion: {S['calib_restarts']}")
    L += ["", "E. ALARMAS Y GESTOS MARCADOS"]
    alarm_txt = ""
    if S["alarms"]:
        alarm_txt = " | instantes (s): " + ", ".join(f"{a:.1f}" for a in S["alarms"][:12]) + (" ..." if len(S["alarms"]) > 12 else "")
    L.append(f"  Alarmas: {len(S['alarms'])}{alarm_txt}")
    n_marks = len(S["marks"])
    if n_marks:
        lat_txt = ""
        if S["lat"]:
            lat_txt = (f" | latencia de alarma: mediana {np.median(S['lat']):.1f} s, p90 {pct(S['lat'], 90):.1f} s, "
                       f"max {max(S['lat']):.1f} s")
        L.append(f"  Gestos marcados: {n_marks} | detectados: {S['detected']} ({100 * S['detected'] / n_marks:.0f} %){lat_txt}")
        missed = [m for m in S["marks"] if not any(m - EARLY_S <= a <= m + MATCH_S for a in S["alarms"])]
        if missed:
            L.append("  Marcas sin alarma (s): " + ", ".join(f"{m:.1f}" for m in missed[:12]))
    else:
        L.append("  Sin gestos marcados (pulsa g, o Enter con --no-window, al empezar cada gesto simulado).")
    L.append(f"  Falsas alarmas (alarmas sin marca cercana): {len(S['false'])} | tiempo evaluado sin gestos: "
             f"{S['exposure_h'] * 60:.1f} min | {S['fa_per_h']:.2f} por hora (cota 95 % aprox. {S['fa_upper']:.1f}/h)")
    max_fa = (contract.get("alarm_thresholds") or {}).get("max_false_alarms_per_hour", ACCEPT["max_fa_per_hour"])
    L += ["", "F. PRUEBAS DE ACEPTACION (criterios PROPUESTOS por el asistente; se editan en ACCEPT al inicio del script)"]
    L.append(f"  {'prueba':<40}{'criterio':<40}{'valor':<56}resultado")
    for name, crit, val, status in acceptance_rows(S, hw, st1, max_fa):
        L.append(f"  {name:<40}{crit:<40}{val:<56}{status}")
    L += ["", "NOTAS: la latencia de alarma se mide desde tu marca (g) y por eso incluye tu tiempo de reaccion al pulsar; "
              "las cifras de Colab son de sujetos no vistos; los gestos son simulados, no somnolencia real."]
    return L


# ---------------------------------------------------------------------------
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
    used_rule = {"thr": thr, "hold": hold, "refractory": refractory, "step_s": step_s, "source": source}

    hw = hw_info()
    st0 = pi_status()
    if not hw["is_pi"]:
        print("AVISO: este equipo no se identifica como Raspberry Pi; los FPS y latencias NO valen para la Pi fisica.")
    try:
        import mediapipe as _mp
        mp_ver = _mp.__version__
    except Exception:  # noqa: BLE001
        mp_ver = "?"

    sess = ort.InferenceSession(str(mdir / "model.onnx"), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    extractor = FaceFeatureExtractor(args.landmarker)
    cam = ThreadedCamera(args.camera, args.width, args.height, args.fps)
    pipe, alarm = LivePipeline(), AlarmLogic(thr, hold, refractory)

    log_f = log_w = None
    if args.log_csv:
        log_f = open(args.log_csv, "w", newline="", encoding="utf-8")
        log_w = csv.writer(log_f)
        log_w.writerow(["t_s"] + names + ["alarm"])

    # Marcas de gestos: tecla g (ventana) o Enter en la terminal (--no-window)
    mark_q = queue.SimpleQueue()
    if args.no_window and sys.stdin is not None and sys.stdin.isatty():
        def _read_stdin():
            try:
                for _ in sys.stdin:
                    mark_q.put(1)
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=_read_stdin, daemon=True).start()

    probs, alarm_until, t0, last_count = None, 0.0, None, -1
    n_frames, t_fps, fps, last_print = 0, time.monotonic(), 0.0, 0.0
    frames, frames_face, ts_first, ts_last, t_last_ms = 0, 0, None, None, 0.0
    fps_samples, frame_lat, mp_ms, onnx_ms = [], [], [], []
    alarms, marks = [], []
    live_t0, calib_restarts, prev_prog = None, 0, 0.0

    def add_mark():
        if pipe.state != "en_vivo":
            print("  (marca ignorada: todavia calibrando)")
            return
        marks.append(t_last_ms / 1000.0)
        print(f"[{t_last_ms / 1000:7.1f} s] marca de gesto #{len(marks)}")

    print("Calibrando: mira al frente con tu postura normal durante 30 s...")
    print("Para pruebas de aceptacion: pulsa g (o Enter con --no-window) al EMPEZAR cada gesto simulado.")
    try:
        while True:
            frame, ts, count = cam.read(last_count, timeout=2.0)
            if frame is None:
                print("Sin cuadros de la camara (2 s). ¿Esta conectada? (en maquina virtual: pasarla a la VM)")
                continue
            last_count = count
            if t0 is None:
                t0 = ts
            if ts_first is None:
                ts_first = ts
            ts_last = ts
            t_ms = (ts - t0) * 1000.0
            t_last_ms = t_ms

            t_a = time.perf_counter()
            feats, lm = extractor.process(frame, t_ms)
            mp_ms.append((time.perf_counter() - t_a) * 1000.0)
            frames += 1
            frames_face += int(feats[0] == 1)

            if pipe.push_frame(t_ms, feats):
                t_b = time.perf_counter()
                probs = sess.run(None, {in_name: pipe.window()})[0][0]
                onnx_ms.append((time.perf_counter() - t_b) * 1000.0)
                fired = alarm.update(t_ms / 1000.0, float(probs[target]))
                if fired:
                    alarms.append(t_ms / 1000.0)
                    alarm_until = time.monotonic() + ALARM_DISPLAY_S
                    print(f"[{t_ms / 1000:7.1f} s] *** ALARMA: {names[target]} p={probs[target]:.2f} ***")
                if log_w:
                    log_w.writerow([f"{t_ms / 1000:.2f}"] + [f"{p:.4f}" for p in probs] + [int(fired)])
            frame_lat.append((time.monotonic() - ts) * 1000.0)

            if live_t0 is None and pipe.state == "en_vivo":
                live_t0 = t_ms / 1000.0
            if pipe.state == "calibrando":
                prog = pipe.calib_progress
                if prog < prev_prog - 0.5:
                    calib_restarts += 1
                prev_prog = prog

            n_frames += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                fps, n_frames, t_fps = n_frames / (now - t_fps), 0, now
                fps_samples.append(fps)
            if now - last_print >= 5.0:
                last_print = now
                msg = f"calibrando {100 * pipe.calib_progress:.0f}%" if pipe.state == "calibrando" else \
                    "probs " + " ".join(f"{n[:4]}={p:.2f}" for n, p in zip(names, probs)) if probs is not None else "en vivo"
                print(f"[{t_ms / 1000:7.1f} s] {fps:4.1f} FPS | {msg}")

            while not mark_q.empty():
                mark_q.get()
                add_mark()

            if not args.no_window:
                view = frame.copy()
                extra = [f"{fps:.1f} FPS | {pipe.state} | marcas: {len(marks)}"]
                if pipe.state == "en_vivo":
                    extra.append(f"sobre umbral: {alarm.run}/{alarm.hold} evaluaciones")
                if pipe.fail_msg:
                    extra.append(pipe.fail_msg[:48])
                draw_overlay(view, lm, feats, extra)
                if pipe.state == "calibrando":
                    left = max(0.0, (1 - pipe.calib_progress) * pp.CAL_SECONDS)
                    banner(view, f"CALIBRANDO: mira al frente ({left:.0f} s)", (160, 100, 0))
                elif time.monotonic() < alarm_until:
                    banner(view, f"ALARMA: {names[target].upper()}", (0, 0, 220))
                if probs is not None and pipe.state == "en_vivo":
                    draw_bars(view, names, probs, target, thr)
                cv2.imshow("Deteccion de somnolencia (q salir, g marcar gesto)", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("g"):
                    add_mark()
    except KeyboardInterrupt:
        pass
    finally:
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

    # ------------------------------ recuento final ------------------------------
    try:
        st1 = pi_status()
        total_s = (ts_last - ts_first) if ts_first is not None else 0.0
        live_s = (t_last_ms / 1000.0 - live_t0) if live_t0 is not None else 0.0
        detected, lat, false = match_marks(marks, alarms)
        exposure_h = max(live_s - len(marks) * (MATCH_S + EARLY_S), 0.0) / 3600.0
        fa = len(false)
        fa_per_h = fa / exposure_h if exposure_h > 0 else float("nan")
        upper = ((3.0 if fa == 0 else fa + 1.96 * np.sqrt(fa) + 2.0) / exposure_h) if exposure_h > 0 else float("nan")
        S = {"frames": frames, "face_pct": 100.0 * frames_face / frames if frames else 0.0,
             "fps_mean": frames / total_s if total_s > 0 else float("nan"), "fps_samples": fps_samples,
             "frame_lat": frame_lat, "mp": mp_ms, "onnx": onnx_ms, "calib_restarts": calib_restarts,
             "total_s": total_s, "live_s": live_s, "alarms": alarms, "marks": marks, "detected": detected,
             "lat": lat, "false": false, "exposure_h": exposure_h, "fa_per_h": fa_per_h, "fa_upper": upper}
        text = "\n".join(build_report(mdir, contract, names, target, used_rule, S, hw, st0, st1, mp_ver))
        print("\n" + text)
        out = Path.cwd() / f"resumen_sesion_{datetime.now():%Y%m%d_%H%M%S}.txt"
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nRecuento guardado en: {out}")
    except Exception as e:  # noqa: BLE001
        print(f"No se pudo generar el recuento final: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
