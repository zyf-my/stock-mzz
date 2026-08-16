"""Time-axis models (stage 4): historical stats, GRU/TCN, small Transformer."""

from __future__ import annotations

from typing import Any


class TemporalModel:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def fit(self, panel: dict[str, Any]) -> None:
        raise NotImplementedError("阶段 4：窗口只能使用当前日之前的特征")

    def predict_panel(self, panel: dict[str, Any]):
        raise NotImplementedError
