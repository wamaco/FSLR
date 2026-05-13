from __future__ import annotations

import torch
from torch import nn


class DynamicFSLGRU(nn.Module):
    """Simple GRU baseline for dynamic FSL recognition."""

    def __init__(self, input_size: int = 258, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, frames, features] where features defaults to 258
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.classifier(last)
