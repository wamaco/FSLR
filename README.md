# FSLR / AnsabI mo?

AnsabI mo? is an early-stage **Filipino Sign Language Recognition (FSLR)** project for an AI course.
The goal is to recognize FSL gestures from webcam/video input and convert them into text, with optional text-to-speech in later milestones.

## Features (Milestone 1 Scaffold)
- Dynamic FSL pipeline scaffold (video -> landmarks -> sequence tensor -> classifier)
- Baseline GRU sequence classifier for dynamic recognition
- Dataset and training script skeletons
- Evaluation utilities (accuracy, precision, recall, F1, confusion matrix, latency)
- CLI prediction entrypoint
- Minimal Gradio app with mode selector and feedback logging

## Tech Stack
- Python 3.10-3.11
- OpenCV + MediaPipe (landmark extraction)
- PyTorch (modeling/training)
- scikit-learn (metrics)
- Gradio (UI scaffold)

## Project Structure
```text
FSLR/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── feedback/
├── models/
├── reports/
│   └── figures/
├── notebooks/
│   └── mediapipe.ipynb
├── pyproject.toml
└── README.md
```

## Setup (using `uv`)
```bash
uv sync
```

If you need PyTorch CPU extras:
```bash
uv sync --extra cpu
```

## Preprocess raw videos
Expected raw layout (example):
```text
data/raw/
  hello/
    sample1.mp4
  thank_you/
    sample2.mp4
```

Then run in Python:
```python
from src.preprocessing import save_processed_dataset
save_processed_dataset("data/raw", "data/processed")
```

> TODO: align preprocessing with the exact notebook frame-selection and normalization logic.

## How to Train
```bash
uv run python -m src.train \
  --processed-dir data/processed \
  --labels-csv data/processed/labels.csv \
  --epochs 10 \
  --batch-size 8
```

Model checkpoint is saved to:
- `models/best_dynamic_model.pth`

## How to Evaluate
Use `src.evaluate` utilities in scripts/notebooks:
- `classification_metrics(y_true, y_pred)`
- `measure_inference_latency(model, sample_batch, runs=50)`

> TODO: add a dedicated CLI evaluator once train/val/test manifests are finalized.

## How to Predict
```bash
uv run python -m src.predict --video path/to/video.mp4 --model models/best_dynamic_model.pth
```

Outputs:
- predicted label
- confidence score

## How to Run the App
```bash
uv run python -m src.app
```

Current app includes:
- mode selection (`Dynamic Words/Phrases`, `Static Alphabet`)
- video upload
- placeholder prediction result + confidence
- feedback logging to `data/feedback/feedback_log.csv`

## Current Status / Limitations
- This milestone is a **scaffold**, not a finished recognizer.
- No packaged FSL-105 dataset is included in this repo.
- Preprocessing currently uses uniform frame sampling; notebook motion-score sampling still needs to be ported.
- Static alphabet model is not implemented yet.
- App prediction is currently placeholder logic until trained models and label maps are finalized.

## Next Steps
1. Finalize FSL-105 dataset ingestion and labels manifest (`sample_id,label`).
2. Port exact notebook motion-scoring and feature normalization to `src/preprocessing.py`.
3. Train baseline dynamic model and log curves/metrics.
4. Add static alphabet classifier pipeline.
5. Replace placeholder app inference with deployed model calls.
6. Add optional text-to-speech output.
