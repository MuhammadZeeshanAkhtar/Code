"""One-command training runner for the FYP backend.

Use this after placing RAVDESS under backend/datasets/ravdess:

    python -m backend.train_all --dataset backend/datasets/ravdess

The script trains:
1. backend/models/audio_model.h5
2. backend/models/video_model.h5

Fusion works immediately with weighted late fusion. If you later generate a CSV
of audio/video probabilities, train fusion with backend.fusion.fusion_model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from backend.audio.train_audio_model import train as train_audio
from backend.utils.dataset import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, discover_labeled_files
from backend.utils.io import MODELS_DIR, REPORTS_DIR, ensure_dirs
from backend.video.train_video_model import train as train_video


def verify_dataset(dataset_dir: str | Path) -> dict[str, int]:
    """Check that the dataset contains labeled audio and video files."""
    audio_samples = discover_labeled_files(dataset_dir, AUDIO_EXTENSIONS)
    video_samples = discover_labeled_files(dataset_dir, VIDEO_EXTENSIONS)
    return {"audio_files": len(audio_samples), "video_files": len(video_samples)}


def train_all(dataset_dir: str | Path, audio_epochs: int, video_epochs: int, audio_batch_size: int, video_batch_size: int) -> dict[str, str]:
    """Train audio and video models and return saved paths."""
    ensure_dirs()
    counts = verify_dataset(dataset_dir)
    print(f"Dataset verified: {counts['audio_files']} audio files, {counts['video_files']} video files")
    print("Training audio model...")
    audio_model_path = train_audio(str(dataset_dir), epochs=audio_epochs, batch_size=audio_batch_size)
    print("Training video model...")
    video_model_path = train_video(str(dataset_dir), epochs=video_epochs, batch_size=video_batch_size)
    print("Training complete.")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Reports saved in: {REPORTS_DIR}")
    return {"audio_model": str(audio_model_path), "video_model": str(video_model_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio and video emotion models from one dataset folder.")
    parser.add_argument("--dataset", default="backend/datasets/ravdess", help="Path where RAVDESS is extracted")
    parser.add_argument("--audio-epochs", type=int, default=30)
    parser.add_argument("--video-epochs", type=int, default=20)
    parser.add_argument("--audio-batch-size", type=int, default=16)
    parser.add_argument("--video-batch-size", type=int, default=4)
    args = parser.parse_args()
    train_all(args.dataset, args.audio_epochs, args.video_epochs, args.audio_batch_size, args.video_batch_size)
