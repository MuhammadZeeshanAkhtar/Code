# Multimodal Depression & Emotion Detection Backend

This repository contains backend AI modules for an FYP-style multimodal system:

- audio emotion/depression-pattern detection,
- video facial emotion/depression-pattern detection,
- late-fusion final prediction,
- Flask REST API endpoints for frontend integration.

> The depression score is a non-clinical screening signal based on emotion probabilities. It is not a medical diagnosis.


## Quick answer: dataset kahan import/download/train hota hai?

Aapko 26 files samajhne ki zaroorat nahi. Important files ye hain:

| Kaam | File | Command / Notes |
|---|---|---|
| Dataset download | `backend/utils/download_datasets.py` | `python -m backend.utils.download_datasets --name ravdess --output backend/datasets` |
| Dataset folder guide | `backend/datasets/README.md` | RAVDESS unzip karke `backend/datasets/ravdess/` mein rakho |
| Dataset label import/detect | `backend/utils/dataset.py` | RAVDESS filename se emotion code read hota hai |
| Audio training | `backend/audio/train_audio_model.py` | `python -m backend.audio.train_audio_model --dataset backend/datasets/ravdess --epochs 30` |
| Video training | `backend/video/train_video_model.py` | `python -m backend.video.train_video_model --dataset backend/datasets/ravdess --epochs 20` |
| Dono models ek command se train | `backend/train_all.py` | `python -m backend.train_all --dataset backend/datasets/ravdess` |
| Audio prediction | `backend/audio/predict_audio.py` | API ke through `/predict-audio` use hota hai |
| Video prediction | `backend/video/predict_video.py` | API ke through `/predict-video` use hota hai |
| Final fusion | `backend/fusion/predict_fusion.py` | API ke through `/predict-final` use hota hai |

Manual download ka easiest tareeqa:

1. Kaggle dataset open karo: <https://www.kaggle.com/datasets/alenken/multimodal-emotion-recognition-ravdess>
2. Download button se ZIP download karo.
3. ZIP extract karke folder yahan rakho: `backend/datasets/ravdess/`
4. Phir run karo:

```bash
python -m backend.train_all --dataset backend/datasets/ravdess --audio-epochs 30 --video-epochs 20
```

Training ke baad models yahan save honge:

```text
backend/models/audio_model.h5
backend/models/video_model.h5
```

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
