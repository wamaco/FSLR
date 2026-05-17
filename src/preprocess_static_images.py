from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from .static_preprocessing import extract_hand_landmarks_from_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess static FSL alphabet images into landmark CSV.")
    parser.add_argument("--input-dir", default="data/raw/static")
    parser.add_argument("--output-csv", default="data/processed/static/static_landmarks.csv")
    parser.add_argument("--failed-csv", default="data/processed/static/failed_images.csv")
    return parser.parse_args()


def list_images(label_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    failed_csv = Path(args.failed_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    failed_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failed_rows: list[dict] = []

    label_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    print("Labels found:", [p.name for p in label_dirs])

    total_images = 0
    success = 0
    failed = 0

    for label_dir in label_dirs:
        label = label_dir.name
        images = list_images(label_dir)
        print(f"{label}: {len(images)} images")
        total_images += len(images)

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                failed += 1
                failed_rows.append({"label": label, "source_path": str(img_path), "reason": "imread_failed"})
                continue

            feats = extract_hand_landmarks_from_image(image)
            if feats is None:
                failed += 1
                failed_rows.append({"label": label, "source_path": str(img_path), "reason": "no_hand_detected"})
                continue

            row = {"label": label, "source_path": str(img_path)}
            for i, v in enumerate(feats.tolist()):
                row[f"f{i}"] = v
            rows.append(row)
            success += 1

    feature_cols = [f"f{i}" for i in range(63)]
    out_df = pd.DataFrame(rows, columns=["label", "source_path", *feature_cols])
    fail_df = pd.DataFrame(failed_rows, columns=["label", "source_path", "reason"])

    out_df.to_csv(output_csv, index=False)
    fail_df.to_csv(failed_csv, index=False)

    print(f"Total images: {total_images}")
    print(f"Successful detections: {success}")
    print(f"Failed detections: {failed}")
    print(f"Saved: {output_csv}")
    print(f"Saved failures: {failed_csv}")


if __name__ == "__main__":
    main()
