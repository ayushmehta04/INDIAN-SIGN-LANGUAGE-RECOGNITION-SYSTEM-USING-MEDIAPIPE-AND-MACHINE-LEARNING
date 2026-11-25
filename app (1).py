"""Flask backend for the ISL recognizer.

Run:  python app.py      then open http://127.0.0.1:5000
"""
import base64
import os

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE, "models", "isl_model.joblib")

app = Flask(__name__)
_state = {"bundle": None, "extractor": None}


def get_bundle():
    if _state["bundle"] is None:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError("Model not found. Run train.py first.")
        _state["bundle"] = joblib.load(MODEL_FILE)
    return _state["bundle"]


def get_extractor():
    if _state["extractor"] is None:
        from features import HandExtractor

        _state["extractor"] = HandExtractor()
    return _state["extractor"]


def decode_image(data_url):
    payload = data_url.split(",", 1)[-1]
    arr = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    ok = os.path.exists(MODEL_FILE)
    info = {"model_ready": ok}
    if ok:
        b = get_bundle()
        info.update(model=b["name"], accuracy=b["accuracy"], classes=len(b["classes"]))
    return jsonify(info)


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    if not body.get("image"):
        return jsonify(ok=False, error="No image received"), 400
    try:
        bundle = get_bundle()
        extractor = get_extractor()
    except Exception as e:  # missing model file, mediapipe model download failed, ...
        return jsonify(ok=False, error=str(e)), 503

    img = decode_image(body["image"])
    if img is None:
        return jsonify(ok=False, error="Could not read image"), 400
    flip = bool(body.get("flip"))
    if flip:
        img = cv2.flip(img, 1)

    feats, hands = extractor.features(img)
    if feats is None:
        return jsonify(ok=True, hands=0, label=None, confidence=0.0, top=[], landmarks=[])

    model = bundle["model"]
    proba = model.predict_proba(feats.reshape(1, -1))[0]
    order = np.argsort(proba)[::-1][:3]
    classes = bundle["classes"]
    top = [{"label": str(classes[i]), "confidence": round(float(proba[i]), 4)} for i in order]

    lms = []
    for h in hands:
        pts = [[float(1 - x) if flip else float(x), float(y)] for x, y, _ in h]
        lms.append(pts)

    return jsonify(ok=True, hands=len(hands), label=top[0]["label"],
                   confidence=top[0]["confidence"], top=top, landmarks=lms)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
