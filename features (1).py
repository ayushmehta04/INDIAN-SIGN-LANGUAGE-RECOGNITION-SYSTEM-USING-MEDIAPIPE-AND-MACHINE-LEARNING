"""Hand landmark extraction (MediaPipe) shared by train.py and app.py.

Each image -> 127 numbers:
  2 hands x 21 landmarks x (x, y, z) = 126, plus 1 value = number of hands seen.
Hands are ordered left-to-right by wrist position, and all points are centred and
scaled, so the features do not depend on where the hand is in the frame or how
close it is to the camera.
"""
import os
import threading
import urllib.request

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

N_LANDMARKS = 21
N_FEATURES = 2 * N_LANDMARKS * 3 + 1
FEATURE_NAMES = [f"f{i}" for i in range(N_FEATURES)]


def ensure_model(path=MODEL_PATH):
    """Download the MediaPipe hand model once (about 8 MB)."""
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Downloading MediaPipe hand model (one time)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as e:
        raise RuntimeError(
            "Could not download hand_landmarker.task. Download it manually from\n"
            f"  {MODEL_URL}\nand put it in the 'models' folder. ({e})"
        )
    return path


def landmarks_to_features(hands):
    """hands: list of (21, 3) arrays. Returns a (127,) float32 vector or None."""
    if not hands:
        return None
    hands = hands[:2]
    pts = np.concatenate(hands, axis=0).astype(np.float32)
    pts = pts - pts.mean(axis=0)
    scale = float(np.abs(pts[:, :2]).max())
    if scale < 1e-6:
        return None
    pts /= scale
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    for i in range(len(hands)):
        block = pts[i * N_LANDMARKS:(i + 1) * N_LANDMARKS]
        vec[i * 63:(i + 1) * 63] = block.reshape(-1)
    vec[-1] = len(hands)
    return vec


class HandExtractor:
    """Wraps MediaPipe HandLandmarker. Safe to call from several Flask threads."""

    def __init__(self, num_hands=2, min_confidence=0.4):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        path = ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()

    def detect(self, bgr):
        """Returns list of (21, 3) arrays (normalised x, y, z), sorted left to right."""
        h, w = bgr.shape[:2]
        if max(h, w) > 640:
            s = 640 / max(h, w)
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._detector.detect(mp_image)
        hands = [
            np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            for lm in result.hand_landmarks
        ]
        hands.sort(key=lambda a: a[0, 0])
        return hands

    def features(self, bgr):
        hands = self.detect(bgr)
        return landmarks_to_features(hands), hands

    def close(self):
        self._detector.close()
