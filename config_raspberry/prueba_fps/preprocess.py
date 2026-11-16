"""
preprocess.py  (pasos S3-S5)

CSV de características (03_extract_from_video.py) + etiquetas por cuadro (.txt de
04_record_video.py)  ->  ventanas listas para entrenar.

Qué hace, por cada PASADA (= mismo sujeto, escenario y NN; las 4 tomas):
  1. Remuestrea cada toma a 10 Hz (un paso = 100 ms). Las características se
     promedian dentro del paso; la etiqueta del paso es la mayoritaria (en
     empate gana la clase de evento).
  2. Calibra con los primeros 30 s de la toma "normal" de ESA pasada y normaliza
     todas las tomas de la pasada con esos valores:
        EAR_norm = EAR / mediana_EAR     (cada ojo con su propia mediana)
        MAR, pitch, yaw, roll = valor - mediana
  3. Cara perdida: el canal "valid" avisa a la red. Los huecos de hasta 2 s se
     rellenan con el ultimo valor; pasado eso, con el valor neutro (postura de
     calibracion).
  4. Corta ventanas de 40 pasos (4 s) cada 5 pasos (0.5 s). Una ventana es de
     evento si el gesto ocupa >= min(30 % de la ventana, 1.0 s); si no, es 0.
     Se descartan las ventanas con >50 % de cuadros invalidos Y etiqueta 0.

Si existe data/manifest.csv (build_manifest.py), se respeta: se ignoran los videos con
QC = FAIL y, si a una pasada le falta la toma "normal", se calibra con la "normal" mas
cercana del mismo sujeto/escenario (columna calib_pass del manifiesto).

La estandarizacion (media/desviacion por canal) NO se hace aqui: se calcula en el
notebook de Colab solo con los sujetos de entrenamiento, para no filtrar
informacion del sujeto de prueba.

Solo necesita numpy. Se puede correr en PC, Colab o donde sea.

Uso (terminal de PyCharm, desde cualquier carpeta):
    python preprocess.py
Salidas (en data/processed):  windows.npz, windows_meta.csv, calibration.csv
Opciones: --features-dir, --dataset-dir, --manifest, --out-dir (por defecto, tus rutas de drowsy)
"""
import argparse
import csv
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURACION (lo unico que se toca)
# ----------------------------------------------------------------------------
DEFAULT_ROOT = Path(r"C:\Users\esteb\Esteban\2026-2\Sistemas inteligentes\drowsy")

STEP_MS = 100            # 10 Hz
CAL_SECONDS = 30         # calibracion: primeros 30 s de la toma normal
MIN_CAL_STEPS = 100      # minimo de pasos validos para aceptar la calibracion
WINDOW_STEPS = 40        # 4 s
STRIDE_STEPS = 5         # 0.5 s
EVENT_MIN_FRAC = 0.30    # el gesto ocupa >= 30 % de la ventana ...
EVENT_MIN_STEPS = 10     # ... o >= 1.0 s (se usa la condicion mas facil de cumplir)
MAX_INVALID_FRAC = 0.50  # descartar ventanas de etiqueta 0 con mas invalidos que esto
MAX_FILL_STEPS = 20      # rellenar huecos de cara perdida hasta 2 s
EAR_CLIP = (0.0, 0.5)    # recorte de valores basura (EAR normal abierto ~0.3)
MAR_CLIP = (0.0, 1.5)
EAR_NORM_CLIP = (0.0, 2.0)

TAKES = ("normal", "bostezo", "cabeceo", "microsueno")
CLASS_NAMES = ["normal", "bostezo", "cabeceo", "microsueno"]
CHANNELS = ["ear_l_norm", "ear_r_norm", "mar", "pitch", "yaw", "roll",
            "d_ear", "d_mar", "d_pitch", "valid"]
# Orden de las 6 caracteristicas continuas tras el remuestreo
BASE = ["ear_l", "ear_r", "mar", "pitch", "yaw", "roll"]
NEUTRAL = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
def load_csv(path):
    """CSV de 03_extract: t_ms, frame_idx, label, face_found, ear_l, ear_r, mar, pitch, yaw, roll."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t_ms = data[:, 0]
    feats = data[:, 3:10].copy()  # face_found + 6 caracteristicas
    return t_ms, feats


def load_labels(path):
    text = Path(path).read_text(encoding="utf-8").split()
    return np.array([int(x) for x in text], dtype=np.int64)


# ----------------------------------------------------------------------------
# S3: remuestreo a 10 Hz
# ----------------------------------------------------------------------------
def resample(t_ms, feats, labels):
    """Devuelve vals (n,6) con NaN en pasos invalidos, step_valid (n,), step_lab (n,)."""
    feats = feats.copy()
    feats[:, 1:3] = np.clip(feats[:, 1:3], *EAR_CLIP)
    feats[:, 3] = np.clip(feats[:, 3], *MAR_CLIP)

    bins = (t_ms // STEP_MS).astype(np.int64)
    n = int(bins.max()) + 1
    tot = np.bincount(bins, minlength=n)
    ok = (feats[:, 0] == 1) & np.isfinite(feats[:, 1:]).all(axis=1)
    vcnt = np.bincount(bins, weights=ok.astype(float), minlength=n)
    step_valid = (tot > 0) & (vcnt > 0) & (vcnt >= np.ceil(0.5 * tot))

    vals = np.full((n, 6), np.nan)
    for j in range(6):
        s = np.bincount(bins, weights=np.where(ok, feats[:, 1 + j], 0.0), minlength=n)
        vals[:, j] = np.where(step_valid, s / np.maximum(vcnt, 1), np.nan)

    lab_counts = np.zeros((n, 4))
    np.add.at(lab_counts, (bins, labels), 1)
    step_lab = 3 - np.argmax(lab_counts[:, ::-1], axis=1)  # empate -> clase mayor
    step_lab = np.where(tot > 0, step_lab, 0)
    return vals, step_valid, step_lab


# ----------------------------------------------------------------------------
# S4: calibracion y normalizacion
# ----------------------------------------------------------------------------
def calibrate(vals, step_valid):
    n_cal = CAL_SECONDS * 1000 // STEP_MS
    cal = vals[:n_cal][step_valid[:n_cal]]
    if len(cal) < MIN_CAL_STEPS:
        raise ValueError(f"calibracion con solo {len(cal)} pasos validos (minimo {MIN_CAL_STEPS})")
    med = np.median(cal, axis=0)
    if med[0] < 0.05 or med[1] < 0.05:
        raise ValueError(f"EAR de calibracion demasiado bajo ({med[0]:.3f}, {med[1]:.3f}): "
                         "¿ojos cerrados o cara mal detectada?")
    return med


def normalize(vals, med):
    out = vals.copy()
    out[:, 0] = np.clip(out[:, 0] / med[0], *EAR_NORM_CLIP)
    out[:, 1] = np.clip(out[:, 1] / med[1], *EAR_NORM_CLIP)
    out[:, 2:] = out[:, 2:] - med[2:]
    return out


# ----------------------------------------------------------------------------
# S5: relleno de huecos, canales y ventanas
# ----------------------------------------------------------------------------
def fill_gaps(x, valid, neutral):
    y = x.copy()
    last, gap = neutral, 0
    for i in range(len(x)):
        if valid[i]:
            last, gap = x[i], 0
        else:
            gap += 1
            y[i] = last if gap <= MAX_FILL_STEPS else neutral
    return y


def build_channels(norm, step_valid):
    y = np.stack([fill_gaps(norm[:, j], step_valid, NEUTRAL[j]) for j in range(6)], axis=1)
    ear_mean = 0.5 * (y[:, 0] + y[:, 1])
    d_ear = np.diff(ear_mean, prepend=ear_mean[0])
    d_mar = np.diff(y[:, 2], prepend=y[0, 2])
    d_pitch = np.diff(y[:, 3], prepend=y[0, 3])
    x = np.column_stack([y, d_ear, d_mar, d_pitch, step_valid.astype(float)])
    return x.astype(np.float32)


def cut_windows(x, step_lab, step_valid):
    """Devuelve lista de (inicio_paso, etiqueta, fraccion_invalida, pasos_de_evento)."""
    L, S = WINDOW_STEPS, STRIDE_STEPS
    thr = min(EVENT_MIN_FRAC * L, EVENT_MIN_STEPS)
    out = []
    for s in range(0, len(x) - L + 1, S):
        counts = np.bincount(step_lab[s:s + L], minlength=4)[1:]  # clases 1..3
        label = int(3 - np.argmax(counts[::-1])) if counts.max() >= thr else 0
        inv = 1.0 - float(step_valid[s:s + L].mean())
        if inv > MAX_INVALID_FRAC and label == 0:
            continue
        out.append((s, label, inv, int(counts.sum())))
    return out


# ----------------------------------------------------------------------------
# Orquestacion
# ----------------------------------------------------------------------------
def parse_csv_name(path):
    """...__s02__glasses__bostezo_01.csv -> ('s02','glasses','bostezo','01')."""
    parts = Path(path).stem.split("__")
    if len(parts) < 3:
        return None
    subject, scenario, take_nn = parts[-3:]
    if "_" not in take_nn:
        return None
    take, nn = take_nn.rsplit("_", 1)
    if take not in TAKES:
        return None
    return subject, scenario, take, nn


def load_manifest(path):
    """{(sujeto, escenario, toma, NN): (estado_QC, calib_pass)}  o None si no existe el manifiesto."""
    path = Path(path)
    if not path.exists():
        return None
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["subject"], r["scenario"], r["take"], r["pass"])] = (r["qc_state"], r.get("calib_pass", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=str(DEFAULT_ROOT / "data" / "features"))
    ap.add_argument("--dataset-dir", default=str(DEFAULT_ROOT / "dataset_own"))
    ap.add_argument("--manifest", default=str(DEFAULT_ROOT / "data" / "manifest.csv"))
    ap.add_argument("--out-dir", default=str(DEFAULT_ROOT / "data" / "processed"))
    args = ap.parse_args()

    features_dir, dataset_dir, out_dir = Path(args.features_dir), Path(args.dataset_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    if manifest is None:
        print("AVISO: no hay manifiesto; no se aplica el QC ni la 'normal mas cercana'. "
              "Corre build_manifest.py primero.")

    # Indice de CSV utilizables: (sujeto, escenario, toma, NN) -> ruta
    index = {}
    for csv_path in sorted(features_dir.glob("*.csv")):
        info = parse_csv_name(csv_path)
        if info is None:
            print(f"Ignorado (nombre no reconocido): {csv_path.name}")
            continue
        if manifest is not None:
            state = manifest.get(info, (None, ""))[0]
            if state is None:
                print(f"Ignorado (no esta en el manifiesto; corre build_manifest.py): {csv_path.name}")
                continue
            if state == "FAIL":
                print(f"Ignorado (QC = FAIL): {csv_path.name}")
                continue
        index[info] = csv_path
    if not index:
        print(f"No encontre CSV utilizables en {features_dir}")
        return

    passes = {}
    for (subject, scenario, take, nn) in index:
        passes.setdefault((subject, scenario, nn), set()).add(take)

    X_all, rows, cal_rows = [], [], []
    for (subject, scenario, nn), takes in sorted(passes.items()):
        tag = f"{subject}/{scenario}/{nn}"
        # Toma "normal" que calibra esta pasada: la propia o la mas cercana (manifiesto)
        calib_nn = nn
        if manifest is not None:
            for t in TAKES:
                c = manifest.get((subject, scenario, t, nn), (None, ""))[1]
                if c:
                    calib_nn = c
                    break
        note = "" if calib_nn == nn else f"  [calibra con normal_{calib_nn}]"
        print(f"\n=== Pasada {tag}  (tomas: {', '.join(sorted(takes))}){note}")
        calib_csv = index.get((subject, scenario, "normal", calib_nn))
        if calib_csv is None:
            print("  SALTADA: no hay una toma 'normal' utilizable para calibrar.")
            continue
        try:
            t, f = load_csv(calib_csv)
            vals, valid, _ = resample(t, f, np.zeros(len(t), dtype=np.int64))
            med = calibrate(vals, valid)
        except Exception as e:  # noqa: BLE001
            print(f"  SALTADA: no se pudo calibrar ({e})")
            continue
        cal_rows.append([subject, scenario, nn] + [f"{v:.4f}" for v in med] + [calib_nn])
        print("  calibracion (" + ", ".join(f"{n}={v:.3f}" for n, v in zip(BASE, med)) + ")")

        for take in TAKES:
            key = (subject, scenario, take, nn)
            if key not in index:
                print(f"  {take:<10} no disponible (falta el CSV o el QC lo marco FAIL)")
                continue
            t, f = load_csv(index[key])
            lab_path = dataset_dir / subject / scenario / f"{take}_{nn}.txt"
            if not lab_path.exists():
                print(f"  {take:<10} SALTADA: no existe {lab_path}")
                continue
            lab = load_labels(lab_path)
            if len(lab) != len(t):
                print(f"  {take:<10} SALTADA: {len(t)} cuadros en el CSV vs {len(lab)} etiquetas")
                continue
            if lab.min() < 0 or lab.max() > 3:
                print(f"  {take:<10} SALTADA: etiquetas fuera de 0-3")
                continue

            vals, valid, step_lab = resample(t, f, lab)
            x = build_channels(normalize(vals, med), valid)
            wins = cut_windows(x, step_lab, valid)
            for s, label, inv, n_ev in wins:
                X_all.append(x[s:s + WINDOW_STEPS])
                rows.append([subject, scenario, nn, take, label, f"{s * STEP_MS / 1000:.1f}",
                             f"{inv:.3f}", n_ev])
            counts = np.bincount([w[1] for w in wins], minlength=4) if wins else np.zeros(4, int)
            print(f"  {take:<10} {len(x)} pasos, {100 * (1 - valid.mean()):.1f} % invalidos | ventanas por clase "
                  + " ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in range(4)))

    if not X_all:
        print("\nNo se genero ninguna ventana.")
        return

    X = np.stack(X_all).astype(np.float32)
    meta = np.array(rows, dtype=object)
    y = meta[:, 4].astype(np.int64)
    np.savez_compressed(
        out_dir / "windows.npz",
        X=X, y=y,
        subject=meta[:, 0].astype(str), scenario=meta[:, 1].astype(str),
        pass_nn=meta[:, 2].astype(str), take=meta[:, 3].astype(str),
        t_start_s=meta[:, 5].astype(np.float32), invalid_frac=meta[:, 6].astype(np.float32),
        event_steps=meta[:, 7].astype(np.int64),
        channels=np.array(CHANNELS), class_names=np.array(CLASS_NAMES),
    )
    with open(out_dir / "windows_meta.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "scenario", "pass", "take", "label", "t_start_s", "invalid_frac", "event_steps"])
        w.writerows(rows)
    with open(out_dir / "calibration.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "scenario", "pass"] + BASE + ["calib_pass"])
        w.writerows(cal_rows)

    print(f"\nTOTAL: {len(X)} ventanas, forma X = {X.shape}")
    for c in range(4):
        print(f"  {CLASS_NAMES[c]:<11} {int((y == c).sum())}")
    print(f"Guardado en {out_dir}")


if __name__ == "__main__":
    main()
