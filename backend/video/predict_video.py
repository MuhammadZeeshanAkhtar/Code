"""Video facial emotion prediction utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from backend.utils.io import MODELS_DIR, load_json
from backend.utils.labels import EMOTION_LABELS, depression_likelihood_from_probs
from backend.video.preprocess_video import extract_face_frames


def predict_video(file_path: str | Path, model_path: str | Path | None = None) -> dict:
    """Predict facial emotion probabilities for one uploaded video."""
    model_path = Path(model_path) if model_path else MODELS_DIR / "video_model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Video model not found: {model_path}. Train it first.")
    class_names = load_json(MODELS_DIR / "class_names.json", {str(i): label for i, label in enumerate(EMOTION_LABELS)})
    model = load_model(model_path)
    frames = extract_face_frames(file_path)[np.newaxis, ...]
    probs = model.predict(frames, verbose=0)[0]
    probabilities = {class_names[str(i)]: float(probs[i]) for i in range(len(probs))}
    best_idx = int(np.argmax(probs))
    return {
        "label": class_names[str(best_idx)],
        "confidence": round(float(probs[best_idx]), 4),
        "probabilities": probabilities,
        "depression_likelihood": depression_likelihood_from_probs(probabilities),
    }
