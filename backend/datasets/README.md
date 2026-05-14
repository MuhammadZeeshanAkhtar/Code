# Dataset Placement Guide

This folder is where you put or download datasets for training.

## Option 1: Manual download from Kaggle

1. Open this dataset page:
   <https://www.kaggle.com/datasets/alenken/multimodal-emotion-recognition-ravdess>
2. Click **Download**.
3. Unzip it into:

```text
backend/datasets/ravdess/
```

After extraction, this folder can contain nested folders. The training code searches recursively and detects RAVDESS labels from filenames like:

```text
03-01-05-01-02-01-12.mp4
```

The third value is the emotion code:

```text
01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fear, 07 disgust, 08 surprise
```

## Option 2: Automatic Kaggle download

Install dependencies first:

```bash
pip install -r backend/requirements.txt
```

Configure Kaggle API credentials, then run:

```bash
python -m backend.utils.download_datasets --name ravdess --output backend/datasets
```

Kaggle API credentials are required. If automatic download fails, use manual download.

## Train after dataset is ready

Train both audio and video models with one command:

```bash
python -m backend.train_all --dataset backend/datasets/ravdess --audio-epochs 30 --video-epochs 20
```

Or train separately:

```bash
python -m backend.audio.train_audio_model --dataset backend/datasets/ravdess --epochs 30
python -m backend.video.train_video_model --dataset backend/datasets/ravdess --epochs 20
```

Saved outputs:

```text
backend/models/audio_model.h5
backend/models/video_model.h5
backend/reports/audio_metrics.json
backend/reports/video_metrics.json
backend/reports/audio_confusion_matrix.png
backend/reports/video_confusion_matrix.png
```
