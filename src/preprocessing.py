from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("mediapipe is required for preprocessing") from exc

FEATURE_SIZE = 258  # 33*4 (pose xyz+vis) + 21*3*2 (hands xyz)


def extract_landmarks_from_frame(frame: np.ndarray) -> np.ndarray:
    """Extract a single 258-d landmark feature vector from one frame.

    TODO: align exact normalization/motion-aware feature engineering with notebook logic.
    """
    if frame is None:
        return np.zeros(FEATURE_SIZE, dtype=np.float32)

    with mp.solutions.holistic.Holistic(static_image_mode=True) as holistic:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

    features = []

    # Pose: 33 landmarks * (x, y, z, visibility)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        features.extend([0.0] * (33 * 4))

    # Left + right hand: 21 landmarks each * (x, y, z)
    for hand_landmarks in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand_landmarks:
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * (21 * 3))

    arr = np.asarray(features, dtype=np.float32)
    if arr.shape[0] != FEATURE_SIZE:
        padded = np.zeros(FEATURE_SIZE, dtype=np.float32)
        padded[: min(FEATURE_SIZE, arr.shape[0])] = arr[:FEATURE_SIZE]
        return padded
    return arr


def _uniform_indices(num_items: int, target: int) -> np.ndarray:
    if num_items <= 0:
        return np.zeros(target, dtype=int)
    return np.linspace(0, num_items - 1, num=target, dtype=int)


def video_to_tensor(video_path: str | Path, num_frames: int = 32) -> np.ndarray:
    """Convert a video into a fixed-length [num_frames, 258] tensor-like array.

    Uses simple uniform frame sampling for this milestone scaffold.
    TODO: port notebook motion-score sampling for dynamic gesture emphasis.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return np.zeros((num_frames, FEATURE_SIZE), dtype=np.float32)

    idx = _uniform_indices(len(frames), num_frames)
    sampled = [extract_landmarks_from_frame(frames[i]) for i in idx]
    return np.asarray(sampled, dtype=np.float32)


def save_processed_dataset(raw_dir: str | Path, output_dir: str | Path, extensions: Iterable[str] = (".mp4", ".avi", ".mov")) -> None:
    """Process raw videos into .npy tensors.

    Expects raw_dir/<label>/*.mp4 style layout. Writes output_dir/<label>/<video_stem>.npy.
    TODO: integrate FSL-105 metadata and deterministic train/val/test split manifest generation.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
        out_label = output_dir / label_dir.name
        out_label.mkdir(parents=True, exist_ok=True)

        for video_path in label_dir.iterdir():
            if video_path.suffix.lower() not in extensions:
                continue
            arr = video_to_tensor(video_path)
            np.save(out_label / f"{video_path.stem}.npy", arr)
