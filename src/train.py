from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import ProcessedFSLDataset
from .model import DynamicFSLGRU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline dynamic FSL recognizer.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory containing .npy tensors.")
    parser.add_argument("--labels-csv", default="data/processed/labels.csv", help="CSV with columns: sample_id,label")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: point --processed-dir and --labels-csv to actual prepared FSL-105 split artifacts.
    dataset = ProcessedFSLDataset(args.processed_dir, args.labels_csv)

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DynamicFSLGRU(num_classes=len(dataset.label_to_idx)).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "best_dynamic_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "label_to_idx": dataset.label_to_idx}, save_path)
            print(f"Saved new best model to: {save_path}")


if __name__ == "__main__":
    main()
