"""I/O helpers shared by training, prediction, and API modules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_BACKEND = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_BACKEND / "models"
REPORTS_DIR = PROJECT_BACKEND / "reports"
UPLOADS_DIR = PROJECT_BACKEND / "uploads"


def ensure_dirs() -> None:
    for directory in (MODELS_DIR, REPORTS_DIR, UPLOADS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
