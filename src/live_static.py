from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from .static_preprocessing import draw_hand_landmarks, extract_hand_landmarks_from_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live static FSL webcam preview/prediction.")
    parser.add_argument("--model", default="models/static_fsl_model.joblib")
    parser.add_argument("--label-map", default="models/static_label_map.json")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--preview-only", action="store_true")
    return parser.parse_args()


def load_model_and_labels(model_path: Path, label_map_path: Path):
    clf = joblib.load(model_path)
    payload = json.loads(label_map_path.read_text(encoding="utf-8"))
    classes = payload.get("classes", [])
    return clf, classes


def main() -> None:
    args = parse_args()

    clf = None
    classes: list[str] = []
    if not args.preview_only:
        model_path = Path(args.model)
        label_map_path = Path(args.label_map)
        if not model_path.exists() or not label_map_path.exists():
            raise FileNotFoundError("Model or label map missing. Train first or run with --preview-only.")
        clf, classes = load_model_and_labels(model_path, label_map_path)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_index}")

    win = "FSL Static Live"
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        overlay = draw_hand_landmarks(frame)
        text = "Preview only"

        if not args.preview_only:
            feats = extract_hand_landmarks_from_image(frame)
            if feats is None:
                text = "No hand detected"
            else:
                probs = clf.predict_proba(np.array([feats], dtype=np.float32))[0]
                idx = int(np.argmax(probs))
                pred = classes[idx] if idx < len(classes) else str(clf.classes_[idx])
                conf = float(probs[idx])
                text = f"Pred: {pred} ({conf:.2f})"

        cv2.putText(overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(win, overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
