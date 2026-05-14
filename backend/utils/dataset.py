"""Dataset discovery helpers for RAVDESS-style audio/video files."""
from __future__ import annotations

from pathlib import Path

from backend.utils.labels import RAVDESS_EMOTION_CODES, normalize_label

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def label_from_ravdess_filename(path: str | Path) -> str | None:
    """Read the emotion label from a RAVDESS filename.

    RAVDESS filenames are like 03-01-05-01-02-01-12.mp4 where the third
    token is the emotion code.
    """
    stem = Path(path).stem
    parts = stem.split("-")
    if len(parts) >= 3 and parts[2] in RAVDESS_EMOTION_CODES:
        return RAVDESS_EMOTION_CODES[parts[2]]
    return None


def actor_from_ravdess_filename(path: str | Path) -> str | None:
    parts = Path(path).stem.split("-")
    return parts[-1] if len(parts) >= 7 else None


def discover_labeled_files(dataset_dir: str | Path, extensions: set[str]) -> list[tuple[Path, str, str | None]]:
    """Discover labeled media files from RAVDESS or label-named folders."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    samples: list[tuple[Path, str, str | None]] = []
    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in extensions:
            continue
        label = label_from_ravdess_filename(file_path)
        if label is None:
            parent_label = normalize_label(file_path.parent.name)
            label = parent_label if parent_label else None
        if label:
            samples.append((file_path, label, actor_from_ravdess_filename(file_path)))
    if not samples:
        raise ValueError(f"No labeled files found in {root}. Expected RAVDESS names or label folders.")
    return samples
