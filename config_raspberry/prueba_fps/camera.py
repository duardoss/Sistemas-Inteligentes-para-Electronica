"""Captura de cámara en un hilo propio: siempre se entrega el cuadro más reciente.

Prueba varios modos de apertura hasta encontrar uno que SÍ entregue cuadros
(en Windows, DirectShow suele funcionar mejor que el modo por defecto).
"""
import sys
import threading
import time

import cv2


def _backends():
    if sys.platform.startswith("linux"):
        return [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]
    if sys.platform.startswith("win"):
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
    return [("ANY", cv2.CAP_ANY)]


def _try_open(index, backend, width, height, fps, use_mjpg):
    """Abre la cámara y confirma que entrega un cuadro (espera hasta ~5 s)."""
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None, None
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, frame
        time.sleep(0.05)
    cap.release()
    return None, None


class ThreadedCamera:
    def __init__(self, index=0, width=640, height=480, fps=30):
        self.cap, first = None, None
        # Primero con MJPG, luego sin forzar formato
        for use_mjpg in (True, False):
            for name, backend in _backends():
                cap, frame = _try_open(index, backend, width, height, fps, use_mjpg)
                if cap is not None:
                    print(f"Cámara {index} abierta con {name} (MJPG={use_mjpg})")
                    self.cap, first = cap, frame
                    break
            if self.cap is not None:
                break
        if self.cap is None:
            raise RuntimeError(
                f"No se pudo obtener video de la cámara {index} con ningún modo. "
                "¿Otra aplicación la está usando? Prueba otro índice (--camera)."
            )

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or first.shape[1]
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or first.shape[0]
        self.reported_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self._lock = threading.Lock()
        self._frame = first
        self._ts = time.monotonic()
        self._count = 1
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._ts = time.monotonic()
                self._count += 1

    def read(self, last_count=-1, timeout=5.0):
        """Espera un cuadro NUEVO (count > last_count). Devuelve (frame, ts, count)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None and self._count > last_count:
                    return self._frame, self._ts, self._count
            time.sleep(0.001)
        return None, 0.0, last_count

    @property
    def frames_captured(self):
        with self._lock:
            return self._count

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()