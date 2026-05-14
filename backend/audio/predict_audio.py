"""Audio emotion prediction utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from backend.audio.preprocess_audio import extract_feature_stack
from backend.utils.io import MODELS_DIR, load_json
from backend.utils.labels import EMOTION_LABELS, depression_likelihood_from_probs


def predict_audio(file_path: str | Path, model_path: str | Path | None = None) -> dict:
    """Predict emotion probabilities for one uploaded WAV/MP3 file."""
    model_path = Path(model_path) if model_path else MODELS_DIR / "audio_model.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Audio model not found: {model_path}. Train it first.")
    class_names = load_json(MODELS_DIR / "class_names.json", {str(i): label for i, label in enumerate(EMOTION_LABELS)})
    model = load_model(model_path)
    features = extract_feature_stack(file_path)[np.newaxis, ...]
    probs = model.predict(features, verbose=0)[0]
    probabilities = {class_names[str(i)]: float(probs[i]) for i in range(len(probs))}
    best_idx = int(np.argmax(probs))
    return {
        "label": class_names[str(best_idx)],
        "confidence": round(float(probs[best_idx]), 4),
        "probabilities": probabilities,
        "depression_likelihood": depression_likelihood_from_probs(probabilities),
    }
