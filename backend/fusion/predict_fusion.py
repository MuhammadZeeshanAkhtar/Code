"""Prediction entry points for multimodal fusion."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from backend.fusion.fusion_model import probs_to_vector, weighted_late_fusion
from backend.utils.io import MODELS_DIR, load_json
from backend.utils.labels import EMOTION_LABELS, depression_likelihood_from_probs


def predict_fusion(audio_result: dict, video_result: dict, model_path: str | Path | None = None, audio_weight: float | None = None) -> dict:
    """Return final multimodal emotion/depression prediction.

    If fusion_model.h5 exists, use the trained MLP. Otherwise use weighted late
    fusion, which is reliable for FYP demos and does not require extra data.
    """
    audio_probs = audio_result.get("probabilities", {})
    video_probs = video_result.get("probabilities", {})
    model_path = Path(model_path) if model_path else MODELS_DIR / "fusion_model.h5"

    if model_path.exists():
        model = load_model(model_path)
        features = np.concatenate([probs_to_vector(audio_probs), probs_to_vector(video_probs)])[np.newaxis, ...]
        probs = model.predict(features, verbose=0)[0]
        probabilities = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
        best_idx = int(np.argmax(probs))
        return {
            "label": EMOTION_LABELS[best_idx],
            "confidence": round(float(probs[best_idx]), 4),
            "probabilities": probabilities,
            "depression_likelihood": depression_likelihood_from_probs(probabilities),
            "fusion_method": "trained_mlp_late_fusion",
            "audio_result": audio_result,
            "video_result": video_result,
        }

    config = load_json(MODELS_DIR / "fusion_config.json", {}) or {}
    weight = audio_weight if audio_weight is not None else float(config.get("audio_weight", 0.5))
    fused = weighted_late_fusion(audio_probs, video_probs, audio_weight=weight)
    fused["audio_result"] = audio_result
    fused["video_result"] = video_result
    return fused
