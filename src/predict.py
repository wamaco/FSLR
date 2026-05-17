from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import DynamicFSLGRU
from .preprocessing import video_to_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for dynamic FSL recognition on a video clip.")
    parser.add_argument("--video", required=True, help="Path to input video clip")
    parser.add_argument("--model", default="models/best_dynamic_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {ckpt_path}. Train first with `uv run python -m src.train ...`"
        )

    ckpt = torch.load(ckpt_path, map_location=args.device)
    label_to_idx = ckpt.get("label_to_idx", {"UNKNOWN": 0})
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    model = DynamicFSLGRU(num_classes=len(label_to_idx)).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = video_to_tensor(args.video)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(args.device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    label = idx_to_label.get(int(pred_idx.item()), "UNKNOWN")
    print(f"predicted_label={label}")
    print(f"confidence={float(conf.item()):.4f}")


if __name__ == "__main__":
    main()
