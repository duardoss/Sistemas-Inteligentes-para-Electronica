"""
live_pipeline.py  -- solo numpy; se usa en la Pi (y en la maquina virtual)

Version "cuadro a cuadro" de preprocess.py: las MISMAS operaciones que se hicieron al
entrenar, para que el modelo vea en vivo exactamente lo que vio al entrenarse:
  1. recorte de EAR/MAR basura, promedio por paso de 100 ms con marcas de tiempo reales
  2. calibracion con los primeros 30 s (el conductor mira al frente)
  3. normalizacion, relleno de huecos por cara perdida (hasta 2 s) y 10 canales
  4. ventana de 40 pasos (4 s); una nueva ventana cada 5 pasos (0.5 s)
y AlarmLogic: la alarma suena si la probabilidad de microsueno >= umbral durante
`hold` ventanas seguidas, con un periodo de silencio (misma regla que evaluate_events.py).

Todas las constantes vienen de preprocess.py (fuente unica); este archivo debe estar en
la misma carpeta que preprocess.py.
"""
from collections import deque

import numpy as np

import preprocess as pp

NEUTRAL = list(pp.NEUTRAL)
N_CAL_STEPS = pp.CAL_SECONDS * 1000 // pp.STEP_MS


class LivePipeline:
    def __init__(self, record=False):
        self.record = record          # solo para pruebas: guarda todos los canales
        self.all_rows = []
        self.reset()

    def reset(self):
        self.cur_bin = None
        self.acc = []                 # cuadros del paso de 100 ms en curso
        self.raw_vals, self.raw_valid = [], []
        self.med = None
        self.buf = deque(maxlen=pp.WINDOW_STEPS)
        self._last, self._gap, self._prev = list(NEUTRAL), 0, None
        self.live_steps = 0
        self.fail_msg = ""
        self.last_step_valid = False
        self._due = False

    # ------------------------------------------------------------------ estado
    @property
    def state(self):
        return "en_vivo" if self.med is not None else "calibrando"

    @property
    def calib_progress(self):
        return min(len(self.raw_vals) / N_CAL_STEPS, 1.0)

    # ------------------------------------------------------------------ entrada
    def push_frame(self, t_ms, feats):
        """
        t_ms: milisegundos desde el primer cuadro. feats: vector de 7 de FaceFeatureExtractor
        (face_found, ear_l, ear_r, mar, pitch, yaw, roll). Devuelve True cuando toca evaluar
        una ventana nueva con el modelo.
        """
        self._due = False
        f = np.asarray(feats, dtype=np.float64).copy()
        f[1:3] = np.clip(f[1:3], *pp.EAR_CLIP)
        f[3] = np.clip(f[3], *pp.MAR_CLIP)
        b = int(t_ms // pp.STEP_MS)
        if self.cur_bin is None:
            self.cur_bin = b
        while b > self.cur_bin:       # cierra el paso en curso y los pasos sin ningun cuadro
            self._finalize_bin()
            self.cur_bin += 1
        self.acc.append(f)
        return self._due

    def _finalize_bin(self):
        if self.acc:
            a = np.array(self.acc)
            ok = (a[:, 0] == 1) & np.isfinite(a[:, 1:]).all(axis=1)
            tot, vcnt = len(a), int(ok.sum())
            valid = vcnt > 0 and vcnt >= np.ceil(0.5 * tot)
            vals = a[ok, 1:7].mean(axis=0) if valid else np.full(6, np.nan)
        else:
            valid, vals = False, np.full(6, np.nan)
        self.acc = []
        self.last_step_valid = bool(valid)
        self._on_step(vals, valid)

    # ------------------------------------------------------------------ pasos
    def _on_step(self, vals, valid):
        if self.med is None:
            self.raw_vals.append(vals)
            self.raw_valid.append(valid)
            if len(self.raw_vals) >= N_CAL_STEPS:
                try:
                    med = pp.calibrate(np.array(self.raw_vals), np.array(self.raw_valid))
                except ValueError as e:
                    self.fail_msg = f"calibracion fallida ({e}); se reinicia"
                    self.raw_vals, self.raw_valid = [], []
                    return
                self.fail_msg = ""
                self.med = med
                for v, ok in zip(self.raw_vals, self.raw_valid):   # rellena el buffer con la calibracion
                    self._emit(v, ok)
                self.raw_vals, self.raw_valid = [], []
            return
        self._emit(vals, valid)
        self.live_steps += 1
        if len(self.buf) == pp.WINDOW_STEPS and self.live_steps % pp.STRIDE_STEPS == 0:
            self._due = True

    def _emit(self, vals, valid):
        if valid:
            x = pp.normalize(vals[None, :], self.med)[0]
            self._last, self._gap, y = list(x), 0, x
        else:
            self._gap += 1
            y = np.array(self._last if self._gap <= pp.MAX_FILL_STEPS else NEUTRAL)
        ear_mean = 0.5 * (y[0] + y[1])
        cur = (ear_mean, y[2], y[3])
        d = (0.0, 0.0, 0.0) if self._prev is None else tuple(c - p for c, p in zip(cur, self._prev))
        self._prev = cur
        row = np.array(list(y) + list(d) + [1.0 if valid else 0.0], dtype=np.float32)
        self.buf.append(row)
        if self.record:
            self.all_rows.append(row)

    # ------------------------------------------------------------------ salida
    def window(self):
        """Ventana (1, 40, 10) float32 lista para el modelo."""
        return np.asarray(self.buf, dtype=np.float32)[None]


class AlarmLogic:
    def __init__(self, threshold, hold_windows, refractory_s):
        self.thr, self.hold, self.refractory = threshold, hold_windows, refractory_s
        self.run, self.last = 0, -1e9

    def update(self, t_s, p):
        """t_s: instante en que termina la ventana (s). Devuelve True si suena la alarma."""
        self.run = self.run + 1 if p >= self.thr else 0
        if self.run >= self.hold and t_s - self.last >= self.refractory:
            self.last = t_s
            return True
        return False
