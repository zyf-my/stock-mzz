"""Blend time-axis and cross-section scores (stage 5)."""

from __future__ import annotations

import numpy as np


def linear_blend(temporal: np.ndarray, cross_section: np.ndarray, weight: float) -> np.ndarray:
    """weight 是时序支的权重，0 表示纯截面，1 表示纯时序。"""
    if temporal.shape != cross_section.shape:
        raise ValueError(f"shape mismatch: {temporal.shape} vs {cross_section.shape}")
    w = float(weight)
    return (w * temporal + (1.0 - w) * cross_section).astype(np.float32)


class FusionModel:
    def fit(self, temporal_valid, cs_valid, label, mask) -> float:
        raise NotImplementedError("阶段 5：只在 valid 上估融合权重")

    def predict(self, temporal, cs, weight: float) -> np.ndarray:
        return linear_blend(temporal, cs, weight)
