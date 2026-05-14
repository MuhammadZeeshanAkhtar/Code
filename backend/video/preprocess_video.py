"""Video preprocessing for facial emotion detection."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_SIZE = 224
MAX_FRAMES = 16


def _detect_face_haar(frame: np.ndarray) -> np.ndarray:
    """Detect and crop the largest face; fall back to center crop."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        pad = int(0.15 * max(w, h))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        return frame[y1:y2, x1:x2]
    height, width = frame.shape[:2]
    side = min(height, width)
    y1 = (height - side) // 2
    x1 = (width - side) // 2
    return frame[y1:y1 + side, x1:x1 + side]


def extract_face_frames(video_path: str | Path, max_frames: int = MAX_FRAMES, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Extract evenly spaced face crops from an MP4/video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_indices = np.linspace(0, max(total_frames - 1, 0), max_frames, dtype=int)
    frames: list[np.ndarray] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        face = _detect_face_haar(frame)
        face = cv2.resize(face, (image_size, image_size))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        frames.append(face.astype("float32") / 255.0)
    cap.release()
    if not frames:
        raise ValueError(f"No readable frames found in video: {video_path}")
    while len(frames) < max_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames[:max_frames], axis=0)


def batch_extract_videos(video_paths: list[str | Path]) -> np.ndarray:
    """Extract fixed frame tensors for a batch of videos."""
    return np.stack([extract_face_frames(path) for path in video_paths], axis=0)
