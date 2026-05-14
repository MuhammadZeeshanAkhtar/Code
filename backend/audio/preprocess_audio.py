"""Audio preprocessing and feature extraction for emotion detection."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION_SECONDS = 10
N_MFCC = 40
N_MELS = 64
MAX_TIME_STEPS = 216


def load_audio(file_path: str | Path, sample_rate: int = SAMPLE_RATE, duration: int = DURATION_SECONDS) -> np.ndarray:
    """Load WAV/MP3 audio as mono waveform with fixed duration."""
    waveform, _ = librosa.load(str(file_path), sr=sample_rate, mono=True, duration=duration)
    target_len = sample_rate * duration
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    return waveform[:target_len]


def _fix_time(feature: np.ndarray, max_steps: int = MAX_TIME_STEPS) -> np.ndarray:
    """Pad or crop a 2D feature matrix to a fixed time dimension."""
    if feature.shape[1] < max_steps:
        feature = np.pad(feature, ((0, 0), (0, max_steps - feature.shape[1])), mode="constant")
    return feature[:, :max_steps]


def extract_feature_stack(file_path: str | Path) -> np.ndarray:
    """Extract MFCC, mel spectrogram, chroma, and spectral contrast features.

    Returns a fixed-size tensor shaped (128, MAX_TIME_STEPS, 1), ready for a
    CNN/LSTM model. The feature count is 40 + 64 + 12 + 7 = 123, then padded
    to 128 rows for cleaner model dimensions.
    """
    y = load_audio(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE)

    features = [_fix_time(x) for x in (mfcc, mel_db, chroma, contrast)]
    stacked = np.vstack(features).astype("float32")
    if stacked.shape[0] < 128:
        stacked = np.pad(stacked, ((0, 128 - stacked.shape[0]), (0, 0)), mode="constant")

    mean = stacked.mean(axis=1, keepdims=True)
    std = stacked.std(axis=1, keepdims=True) + 1e-6
    normalized = (stacked - mean) / std
    return normalized[..., np.newaxis].astype("float32")


def batch_extract(file_paths: list[str | Path]) -> np.ndarray:
    """Extract features for multiple audio files."""
    return np.stack([extract_feature_stack(path) for path in file_paths], axis=0)
