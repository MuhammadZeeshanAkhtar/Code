"""Train the CNN+LSTM audio emotion model."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from backend.audio.preprocess_audio import batch_extract
from backend.utils.dataset import AUDIO_EXTENSIONS, discover_labeled_files
from backend.utils.io import MODELS_DIR, REPORTS_DIR, ensure_dirs, save_json
from backend.utils.labels import EMOTION_LABELS


def build_audio_model(input_shape: tuple[int, int, int], num_classes: int) -> models.Model:
    """Build a compact CNN+LSTM model for audio feature sequences."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Permute((2, 1, 3))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="audio_cnn_lstm_emotion")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _split_indices(labels: list[str], groups: list[str | None], test_size: float = 0.3):
    indices = np.arange(len(labels))
    if all(groups) and len(set(groups)) >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, temp_idx = next(splitter.split(indices, labels, groups))
        temp_groups = [groups[i] for i in temp_idx]
        temp_labels = [labels[i] for i in temp_idx]
        val_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        val_rel, test_rel = next(val_test.split(temp_idx, temp_labels, temp_groups))
        return train_idx, temp_idx[val_rel], temp_idx[test_rel]
    train_idx, temp_idx = train_test_split(indices, test_size=test_size, random_state=42, stratify=labels)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=temp_labels)
    return train_idx, val_idx, test_idx


def train(dataset_dir: str, epochs: int = 30, batch_size: int = 16) -> Path:
    """Train and save audio_model.h5 plus metrics."""
    ensure_dirs()
    samples = discover_labeled_files(dataset_dir, AUDIO_EXTENSIONS)
    paths, labels, groups = zip(*samples)
    valid = [i for i, label in enumerate(labels) if label in EMOTION_LABELS]
    paths = [paths[i] for i in valid]
    labels = [labels[i] for i in valid]
    groups = [groups[i] for i in valid]

    train_idx, val_idx, test_idx = _split_indices(labels, groups)
    x_train = batch_extract([paths[i] for i in train_idx])
    x_val = batch_extract([paths[i] for i in val_idx])
    x_test = batch_extract([paths[i] for i in test_idx])
    y_all = np.array([EMOTION_LABELS.index(label) for label in labels])
    y_train = to_categorical(y_all[train_idx], num_classes=len(EMOTION_LABELS))
    y_val = to_categorical(y_all[val_idx], num_classes=len(EMOTION_LABELS))
    y_test = to_categorical(y_all[test_idx], num_classes=len(EMOTION_LABELS))

    model_path = MODELS_DIR / "audio_model.h5"
    model = build_audio_model(x_train.shape[1:], len(EMOTION_LABELS))
    callbacks = [
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    model.save(model_path)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    preds = model.predict(x_test).argmax(axis=1)
    true = y_test.argmax(axis=1)
    report = classification_report(true, preds, target_names=EMOTION_LABELS, output_dict=True, zero_division=0)
    save_json(REPORTS_DIR / "audio_metrics.json", {"test_loss": float(test_loss), "test_accuracy": float(test_acc), "classification_report": report, "history": history.history})
    save_json(MODELS_DIR / "class_names.json", {str(i): label for i, label in enumerate(EMOTION_LABELS)})

    cm = confusion_matrix(true, preds, labels=list(range(len(EMOTION_LABELS))))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "audio_confusion_matrix.png")
    plt.close()
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to RAVDESS audio dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(f"Saved audio model to {train(args.dataset, args.epochs, args.batch_size)}")
