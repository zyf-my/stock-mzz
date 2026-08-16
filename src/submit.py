from __future__ import annotations

from pathlib import Path

import numpy as np


def save_submission(pred: np.ndarray, path: str | Path) -> Path:
    """Save test predictions so `np.load(path)` returns the array directly."""
    arr = np.asarray(pred, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"submission must be 2-D (Tt, S), got {arr.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    loaded = np.load(path)
    if loaded.shape != arr.shape or loaded.dtype != np.float32:
        raise RuntimeError("saved file is not reloadable as float32 (Tt, S)")
    return path
