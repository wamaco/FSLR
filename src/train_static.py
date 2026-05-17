from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train static FSL alphabet baseline model.")
    parser.add_argument("--data", default="data/processed/static/static_landmarks.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_path}")

    df = pd.read_csv(data_path)
    feature_cols = [f"f{i}" for i in range(63)]
    missing = [c for c in ["label", *feature_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "static_fsl_model.joblib"
    label_map_path = models_dir / "static_label_map.json"

    joblib.dump(clf, model_path)
    label_map_path.write_text(json.dumps({"classes": clf.classes_.tolist()}, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved label map: {label_map_path}")


if __name__ == "__main__":
    main()
