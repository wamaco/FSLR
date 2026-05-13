from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def measure_inference_latency(model, sample_batch, device: str = "cpu", runs: int = 50) -> dict[str, float]:
    """Measure inference latency in milliseconds for a prepared sample batch."""
    import torch

    model = model.to(device)
    sample_batch = sample_batch.to(device)
    model.eval()

    timings = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(sample_batch)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)

    arr = np.array(timings, dtype=np.float64)
    return {"latency_ms_mean": float(arr.mean()), "latency_ms_std": float(arr.std())}
