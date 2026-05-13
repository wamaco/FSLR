from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ProcessedFSLDataset(Dataset):
    """Dataset for processed tensor sequences and labels.

    labels CSV format:
      sample_id,label
    where sample_id matches <processed_dir>/<sample_id>.npy (or nested path if provided).
    """

    def __init__(self, processed_dir: str | Path, labels_csv: str | Path) -> None:
        self.processed_dir = Path(processed_dir)
        self.labels_csv = Path(labels_csv)

        if not self.processed_dir.exists():
            raise FileNotFoundError(f"Processed directory not found: {self.processed_dir}")
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Labels CSV not found: {self.labels_csv}")

        self.df = pd.read_csv(self.labels_csv)
        required_cols = {"sample_id", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Labels CSV missing columns: {sorted(missing)}")

        labels = sorted(self.df["label"].unique().tolist())
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_path = self.processed_dir / f"{row['sample_id']}.npy"
        if not sample_path.exists():
            raise FileNotFoundError(f"Processed sample missing: {sample_path}")

        x = np.load(sample_path)
        y = self.label_to_idx[row["label"]]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
