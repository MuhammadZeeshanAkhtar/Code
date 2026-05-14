"""Trainable and rule-based late fusion for audio/video emotion predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from backend.utils.io import MODELS_DIR, ensure_dirs, save_json
from backend.utils.labels import EMOTION_LABELS, depression_likelihood_from_probs

DEFAULT_AUDIO_WEIGHT = 0.5
DEFAULT_VIDEO_WEIGHT = 0.5


def build_fusion_model(input_dim: int = 16, num_classes: int = 8) -> models.Model:
    """Build a small MLP for late fusion over audio/video probabilities."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="late_fusion_mlp")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def probs_to_vector(probabilities: dict[str, float]) -> np.ndarray:
    """Convert a probability dictionary to label-order vector."""
    return np.array([float(probabilities.get(label, 0.0)) for label in EMOTION_LABELS], dtype="float32")


def weighted_late_fusion(audio_probs: dict[str, float], video_probs: dict[str, float], audio_weight: float = DEFAULT_AUDIO_WEIGHT) -> dict:
    """Combine audio and video probabilities with weighted late fusion."""
    audio_weight = float(audio_weight)
    video_weight = 1.0 - audio_weight
    audio_vec = probs_to_vector(audio_probs)
    video_vec = probs_to_vector(video_probs)
    final_vec = (audio_weight * audio_vec) + (video_weight * video_vec)
    if final_vec.sum() > 0:
        final_vec = final_vec / final_vec.sum()
    probabilities = {label: float(final_vec[i]) for i, label in enumerate(EMOTION_LABELS)}
    best_idx = int(np.argmax(final_vec))
    return {
        "label": EMOTION_LABELS[best_idx],
        "confidence": round(float(final_vec[best_idx]), 4),
        "probabilities": probabilities,
        "depression_likelihood": depression_likelihood_from_probs(probabilities),
        "fusion_method": "weighted_late_fusion",
        "audio_weight": round(audio_weight, 4),
        "video_weight": round(video_weight, 4),
    }


def train_from_csv(csv_path: str | Path, epochs: int = 100, batch_size: int = 16) -> Path:
    """Train fusion_model.h5 from a CSV of saved unimodal probabilities.

    Expected columns: audio_<label>, video_<label>, label.
    """
    ensure_dirs()
    df = pd.read_csv(csv_path)
    feature_cols = [f"audio_{label}" for label in EMOTION_LABELS] + [f"video_{label}" for label in EMOTION_LABELS]
    missing = [col for col in feature_cols + ["label"] if col not in df.columns]
    if missing:
        raise ValueError(f"Fusion CSV missing columns: {missing}")
    x = df[feature_cols].astype("float32").values
    y_idx = df["label"].map({label: idx for idx, label in enumerate(EMOTION_LABELS)}).astype(int).values
    y = to_categorical(y_idx, num_classes=len(EMOTION_LABELS))
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y_idx)

    model_path = MODELS_DIR / "fusion_model.h5"
    model = build_fusion_model(input_dim=x.shape[1], num_classes=len(EMOTION_LABELS))
    callbacks = [
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    model.save(model_path)
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    save_json(MODELS_DIR / "fusion_config.json", {"method": "mlp", "input_labels": EMOTION_LABELS, "val_accuracy": float(val_acc), "val_loss": float(val_loss), "history": history.history})
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with audio/video probability columns and label")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(f"Saved fusion model to {train_from_csv(args.csv, args.epochs, args.batch_size)}")
