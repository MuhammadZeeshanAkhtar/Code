"""Shared emotion labels and depression-risk helpers."""
from __future__ import annotations

EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
]

RAVDESS_EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

NEGATIVE_EMOTIONS = {"sad", "fear", "angry", "disgust"}


def normalize_label(label: str) -> str:
    """Map common dataset label spellings to the project label set."""
    value = label.strip().lower().replace("fearful", "fear").replace("surprised", "surprise")
    aliases = {
        "happiness": "happy",
        "sadness": "sad",
        "anger": "angry",
        "fearful": "fear",
        "surprised": "surprise",
    }
    return aliases.get(value, value)


def depression_likelihood_from_probs(probabilities: dict[str, float]) -> float:
    """Estimate depression-related likelihood from negative affect probabilities.

    This is a screening signal for an FYP demo, not a medical diagnosis.
    """
    score = 0.0
    weights = {"sad": 0.45, "fear": 0.2, "angry": 0.15, "disgust": 0.1, "neutral": 0.1}
    for emotion, weight in weights.items():
        score += float(probabilities.get(emotion, 0.0)) * weight
    calm_happy = float(probabilities.get("happy", 0.0)) + float(probabilities.get("calm", 0.0))
    score = max(0.0, min(1.0, score + max(0.0, 0.2 - calm_happy) * 0.25))
    return round(score, 4)
