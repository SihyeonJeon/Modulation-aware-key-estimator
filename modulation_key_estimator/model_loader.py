from __future__ import annotations

import os
from pathlib import Path

import torch

from .model import MultiStreamKeyPredictor


def default_checkpoint_path() -> Path:
    override = os.environ.get("MODEL_CHECKPOINT_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "assets" / "key_model.pt"


def load_model(checkpoint_path: str | Path | None = None) -> MultiStreamKeyPredictor:
    checkpoint = Path(checkpoint_path) if checkpoint_path else default_checkpoint_path()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint}. "
            "Set MODEL_CHECKPOINT_PATH or keep assets/key_model.pt in the package."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiStreamKeyPredictor(d_model=128, nhead=4, num_layers=4)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval().to(device)
    return model
