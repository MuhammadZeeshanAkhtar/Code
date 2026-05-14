# Multimodal Depression & Emotion Detection Backend

This repository contains backend AI modules for an FYP-style multimodal system:

- audio emotion/depression-pattern detection,
- video facial emotion/depression-pattern detection,
- late-fusion final prediction,
- Flask REST API endpoints for frontend integration.

> The depression score is a non-clinical screening signal based on emotion probabilities. It is not a medical diagnosis.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

TensorFlow automatically uses GPU when a compatible CUDA environment is installed. The helper `backend/utils/gpu.py` can enable GPU memory growth.

## Optional dataset download

Configure Kaggle credentials, then run:

```bash
python -m backend.utils.download_datasets --name ravdess --output backend/datasets
```

You can also manually place RAVDESS, CREMA-D, FER2013, MELD, or DAIC-WOZ-derived files under `backend/datasets/`. RAVDESS filenames are detected automatically.

## Train models

```bash
python -m backend.audio.train_audio_model --dataset backend/datasets/ravdess --epochs 30
python -m backend.video.train_video_model --dataset backend/datasets/ravdess --epochs 20
```

Outputs:

- `backend/models/audio_model.h5`
- `backend/models/video_model.h5`
- `backend/reports/audio_metrics.json`
- `backend/reports/video_metrics.json`
- confusion matrix images under `backend/reports/`

Fusion can run without training using weighted late fusion. To train an MLP fusion model, provide a CSV with columns `audio_<label>`, `video_<label>`, and `label`:

```bash
python -m backend.fusion.fusion_model --csv backend/datasets/fusion_predictions.csv
```

## Run API

```bash
python -m backend.api.app
```

Endpoints:

- `POST /predict-audio` with form-data file field `file` and optional `session_id`
- `POST /predict-video` with form-data file field `file` and optional `session_id`
- `POST /predict-final` with JSON `{ "session_id": "..." }`

The API returns JSON predictions with label, confidence, class probabilities, and depression likelihood score.
