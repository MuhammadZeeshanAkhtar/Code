"""Optional Kaggle dataset downloader.

Requires Kaggle credentials configured via ~/.kaggle/kaggle.json or environment
variables. The module intentionally fails with a clear message when credentials
or the kaggle package are unavailable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


DATASETS = {
    "ravdess": "alenken/multimodal-emotion-recognition-ravdess",
}


def download_kaggle_dataset(name: str, output_dir: str | Path) -> Path:
    """Download and unzip a supported Kaggle dataset."""
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset '{name}'. Supported: {sorted(DATASETS)}")
    output_path = Path(output_dir) / name
    output_path.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASETS[name], path=str(output_path), unzip=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ravdess", choices=sorted(DATASETS))
    parser.add_argument("--output", default="backend/datasets")
    args = parser.parse_args()
    print(download_kaggle_dataset(args.name, args.output))
