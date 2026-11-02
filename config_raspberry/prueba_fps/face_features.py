"""
Extracción de características faciales con MediaPipe Face Landmarker.

IMPORTANTE: este mismo módulo se usa en la Raspberry Pi (en vivo), en tu PC
y en Colab (videos del dataset). Así el modelo se entrena y se usa con
exactamente las mismas fórmulas.

Vector de salida por cuadro (FEATURE_NAMES):
    face_found, ear_l, ear_r, mar, pitch, yaw, roll
"""
import math

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

FEATURE_NAMES = ["face_found", "ear_l", "ear_r", "mar", "pitch", "yaw", "roll"]

# Índices de FaceMesh (los primeros 468 puntos). "Derecho/izquierdo" = del sujeto.
# Orden de cada ojo: p1 esquina externa, p2 y p3 párpado superior,
#                    p4 esquina interna, p5 y p6 párpado inferior.
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Boca (labios internos): esquinas y 3 pares verticales
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
MOUTH_PAIRS = [(82, 87), (13, 14), (312, 317)]


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def eye_aspect_ratio(p):
    """EAR (Soukupová y Čech, 2016). p: 6 puntos (x, y) en píxeles."""
    horizontal = _dist(p[0], p[3])
    if horizontal < 1e-6:
        return float("nan")
    vertical = _dist(p[1], p[5]) + _dist(p[2], p[4])
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(lm):
    """MAR: apertura vertical media de la boca / ancho de la boca."""
    horizontal = _dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])
    if horizontal < 1e-6:
        return float("nan")
    vertical = np.mean([_dist(lm[a], lm[b]) for a, b in MOUTH_PAIRS])
    return float(vertical / horizontal)


def euler_from_matrix(m):
    """
    Matriz de transformación facial 4x4 de MediaPipe -> (pitch, yaw, roll) en grados.
    Convención R = Rz(roll) @ Ry(yaw) @ Rx(pitch).
    OJO: el signo de cada ángulo se verifica de forma empírica (ver instrucciones);
    para el modelo lo importante es que sea consistente, ya que luego se resta
    la postura neutra del conductor.
    """
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    scale = np.linalg.norm(r[:, 0])
    if scale < 1e-9:
        return float("nan"), float("nan"), float("nan")
    r = r / scale
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    # Evita saltos de +-180 si el eje z viene invertido
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return pitch, yaw, roll


class FaceFeatureExtractor:
    """Envuelve MediaPipe Face Landmarker en modo VIDEO (con seguimiento)."""

    def __init__(self, model_path, min_confidence=0.5):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr, timestamp_ms):
        """
        Devuelve (features, landmarks_px).
        features: np.array float32 con FEATURE_NAMES (NaN si no hay cara).
        landmarks_px: array (N, 2) en píxeles, o None si no hay cara.
        """
        ts = int(timestamp_ms)
        if ts <= self._last_ts:  # MediaPipe exige timestamps estrictamente crecientes
            ts = self._last_ts + 1
        self._last_ts = ts

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, ts)

        feats = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
        feats[0] = 0.0
        if not result.face_landmarks:
            return feats, None

        lm = np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]],
                      dtype=np.float32)  # en píxeles: evita distorsión por aspecto
        feats[0] = 1.0
        feats[1] = eye_aspect_ratio(lm[LEFT_EYE])
        feats[2] = eye_aspect_ratio(lm[RIGHT_EYE])
        feats[3] = mouth_aspect_ratio(lm)
        if result.facial_transformation_matrixes:
            feats[4:7] = euler_from_matrix(result.facial_transformation_matrixes[0])
        return feats, lm

    def close(self):
        self._landmarker.close()


def draw_overlay(frame, lm, feats, extra_lines=()):
    """Dibuja los puntos usados y los valores sobre el cuadro."""
    if lm is not None:
        for idx in RIGHT_EYE + LEFT_EYE + [MOUTH_LEFT, MOUTH_RIGHT] + \
                [i for pair in MOUTH_PAIRS for i in pair]:
            x, y = lm[idx]
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        lines = [
            f"EAR L {feats[1]:.2f}  R {feats[2]:.2f}",
            f"MAR {feats[3]:.2f}",
            f"pitch {feats[4]:+.0f}  yaw {feats[5]:+.0f}  roll {feats[6]:+.0f}",
        ]
    else:
        lines = ["SIN CARA"]
    y = 22
    for text in list(lines) + list(extra_lines):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
