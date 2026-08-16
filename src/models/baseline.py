"""Cross-section LightGBM baseline (stage 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class LightGBMBaseline:
    """Fit on train rows, predict a (T, S) score panel for a split."""

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.model = None

    def fit(self, x, y) -> None:
        raise NotImplementedError("阶段 2：在此调用 lightgbm.LGBMRegressor.fit")

    def predict_panel(self, split_data: dict[str, Any], fill_invalid: float = 0.0):
        raise NotImplementedError("阶段 2：输出形状 (T_split, S) 的 float32 分数")

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        raise NotImplementedError
