from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

import torch

from .model import MultiStreamKeyPredictor

CHECKPOINT_URL = (
    "https://github.com/SihyeonJeon/Modulation-aware-key-estimator/"
    "releases/download/v0.1.0/key_model.pt"
)
CHECKPOINT_SHA256 = "7d04741637b013030138d70239ccaab95cc9a7e0cf78d27e05b98bead8c90c4f"


def cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    if root:
        return Path(root).expanduser() / "modulation-aware-key-estimator"
    return Path.home() / ".cache" / "modulation-aware-key-estimator"


def cached_checkpoint_path() -> Path:
    return cache_dir() / "key_model.pt"


def configured_checkpoint_path() -> Path:
    override = os.environ.get("MODEL_CHECKPOINT_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return cached_checkpoint_path()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_checkpoint(path: Path | None = None) -> Path:
    checkpoint = path or cached_checkpoint_path()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and _sha256(checkpoint) == CHECKPOINT_SHA256:
        return checkpoint

    tmp = checkpoint.with_suffix(".pt.tmp")
    with urlopen(CHECKPOINT_URL, timeout=60) as response, tmp.open("wb") as fp:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fp.write(chunk)

    actual = _sha256(tmp)
    if actual != CHECKPOINT_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded checkpoint hash mismatch: expected {CHECKPOINT_SHA256}, got {actual}"
        )
    tmp.replace(checkpoint)
    return checkpoint


def default_checkpoint_path() -> Path:
    override = os.environ.get("MODEL_CHECKPOINT_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return ensure_checkpoint()


def load_model(checkpoint_path: str | Path | None = None) -> MultiStreamKeyPredictor:
    checkpoint = Path(checkpoint_path) if checkpoint_path else default_checkpoint_path()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint}. "
            "Set MODEL_CHECKPOINT_PATH or let the default loader download the release checkpoint."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiStreamKeyPredictor(d_model=128, nhead=4, num_layers=4)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval().to(device)
    return model
