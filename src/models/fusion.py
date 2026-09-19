"""Blend time-axis and cross-section scores (stage 5).

Weights and gates are fit on valid only, then locked for test.
All transforms use the same day's scores / categories / coverage — no future days.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rankdata


def coverage_gate_blend(
    temporal: np.ndarray,
    cross_section: np.ndarray,
    mask_x: np.ndarray,
    tau: float,
    weight_low: float,
    weight_high: float,
    space: str = "raw",
) -> np.ndarray:
    """Same-day coverage gate. High coverage uses weight_high on the temporal branch."""
    if space == "rank":
        temporal = panel_cs_rank(temporal, mask_x)
        cross_section = panel_cs_rank(cross_section, mask_x)
    elif space != "raw":
        raise ValueError(f"unknown coverage space {space!r}")
    n_x = np.asarray(mask_x).sum(axis=1)
    high = n_x >= float(tau)
    pred = np.empty_like(temporal, dtype=np.float32)
    pred[~high] = linear_blend(temporal[~high], cross_section[~high], weight_low)
    pred[high] = linear_blend(temporal[high], cross_section[high], weight_high)
    return pred


def coverage_soft_gate_blend(
    temporal: np.ndarray,
    cross_section: np.ndarray,
    mask_x: np.ndarray,
    tau_lo: float,
    tau_hi: float,
    weight_low: float,
    weight_high: float,
    space: str = "raw",
    extrapolate: bool = True,
) -> np.ndarray:
    """Linear coverage ramp between tau_lo→w_lo and tau_hi→w_hi; optional linear extrapolation above tau_hi."""
    if space == "rank":
        temporal = panel_cs_rank(temporal, mask_x)
        cross_section = panel_cs_rank(cross_section, mask_x)
    elif space != "raw":
        raise ValueError(f"unknown coverage space {space!r}")
    n_x = np.asarray(mask_x).sum(axis=1).astype(np.float64)
    lo, hi = float(tau_lo), float(tau_hi)
    span = max(hi - lo, 1.0)
    t = (n_x - lo) / span
    if not extrapolate:
        t = np.clip(t, 0.0, 1.0)
    w = float(weight_low) + t * (float(weight_high) - float(weight_low))
    pred = np.empty_like(temporal, dtype=np.float32)
    for day in range(temporal.shape[0]):
        wd = float(w[day])
        pred[day] = linear_blend(temporal[day], cross_section[day], wd)
    return pred


def dual_regime_blend(
    base: np.ndarray,
    expert: np.ndarray,
    mask_x: np.ndarray,
    cov_lo: float,
    cov_hi: float,
    *,
    extrapolate: bool = True,
    w_cap: float = 1.5,
) -> np.ndarray:
    """Per-day linear ramp: cov_lo→base, cov_hi→expert; optional extrapolation above cov_hi."""
    n_x = np.asarray(mask_x).sum(axis=1).astype(np.float64)
    span = max(float(cov_hi) - float(cov_lo), 1.0)
    w = (n_x - float(cov_lo)) / span
    if not extrapolate:
        w = np.clip(w, 0.0, 1.0)
    else:
        w = np.clip(w, 0.0, float(w_cap))
    out = np.empty_like(base, dtype=np.float32)
    for day in range(base.shape[0]):
        wd = float(w[day])
        out[day] = ((1.0 - wd) * base[day] + wd * expert[day]).astype(np.float32)
    return out


def linear_blend(temporal: np.ndarray, cross_section: np.ndarray, weight: float) -> np.ndarray:
    """weight 是时序支的权重，0 表示纯截面，1 表示纯时序。"""
    if temporal.shape != cross_section.shape:
        raise ValueError(f"shape mismatch: {temporal.shape} vs {cross_section.shape}")
    w = float(weight)
    return (w * temporal + (1.0 - w) * cross_section).astype(np.float32)


def panel_cs_rank(pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Each day: rank among mask=True stocks. Invalid stays 0."""
    out = np.zeros_like(pred, dtype=np.float32)
    for t in range(pred.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if int(m.sum()) < 2:
            continue
        out[t, m] = rankdata(pred[t, m], method="average").astype(np.float32)
    return out


def shrink_to_day_mean(pred: np.ndarray, mask: np.ndarray, strength: np.ndarray | float) -> np.ndarray:
    """score = (1-s) * pred + s * same-day mean. s=0 keeps pred; s=1 is constant (RankIC 0)."""
    out = np.array(pred, dtype=np.float32, copy=True)
    if np.isscalar(strength):
        s_day = np.full(pred.shape[0], float(strength), dtype=np.float64)
    else:
        s_day = np.asarray(strength, dtype=np.float64)
        if s_day.shape[0] != pred.shape[0]:
            raise ValueError("strength length must match days")
    s_day = np.clip(s_day, 0.0, 1.0)
    for t in range(pred.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if int(m.sum()) < 2:
            continue
        mu = float(out[t, m].mean())
        st = float(s_day[t])
        out[t, m] = ((1.0 - st) * out[t, m] + st * mu).astype(np.float32)
    return out


def neutralize_industry(pred: np.ndarray, industry: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract same-day industry mean of scores. Stock-axis residual."""
    out = np.array(pred, dtype=np.float32, copy=True)
    for t in range(pred.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if int(m.sum()) < 2:
            continue
        scores = out[t, m].copy()
        groups = np.asarray(industry[t, m])
        for gid in np.unique(groups):
            idx = groups == gid
            if int(idx.sum()) < 2:
                continue
            scores[idx] -= float(scores[idx].mean())
        out[t, m] = scores
    return out


class FusionModel:
    """Search a small valid-only menu, then lock."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = dict(cfg or {})
        self.locked: dict[str, Any] = {}

    def fit(
        self,
        temporal_valid: np.ndarray,
        cs_valid: np.ndarray,
        label: np.ndarray,
        mask_y: np.ndarray,
        mask_x: np.ndarray,
        industry: np.ndarray | None = None,
        scorer=None,
    ) -> dict[str, Any]:
        from src.metrics import mean_rank_ic

        score_fn = scorer or mean_rank_ic
        grid = [float(x) for x in (self.cfg.get("weight_grid") or [0.0, 0.25, 0.5, 0.75, 1.0])]
        results: list[dict[str, Any]] = []

        def consider(name: str, pred: np.ndarray, **meta: Any) -> None:
            ic = float(score_fn(pred, label, mask_y))
            row = {"name": name, "ic": ic, **meta}
            results.append(row)

        consider("cs_only", cs_valid, weight=0.0)
        consider("temporal_only", temporal_valid, weight=1.0)

        for w in grid:
            consider("raw_blend", linear_blend(temporal_valid, cs_valid, w), weight=w)

        t_rank = panel_cs_rank(temporal_valid, mask_x)
        c_rank = panel_cs_rank(cs_valid, mask_x)
        for w in grid:
            consider("rank_blend", linear_blend(t_rank, c_rank, w), weight=w, space="rank")

        if industry is not None:
            t_neu = neutralize_industry(t_rank, industry, mask_x)
            c_neu = neutralize_industry(c_rank, industry, mask_x)
            for w in grid:
                consider(
                    "rank_blend_industry",
                    linear_blend(t_neu, c_neu, w),
                    weight=w,
                    space="rank_industry",
                )

        n_x = np.asarray(mask_x).sum(axis=1)
        tau = float(np.median(n_x))
        high = n_x >= tau
        for w_low in (0.5, 0.75, 1.0):
            for w_high in (0.5, 0.75, 1.0):
                pred = np.empty_like(t_rank, dtype=np.float32)
                pred[~high] = linear_blend(t_rank[~high], c_rank[~high], w_low)
                pred[high] = linear_blend(t_rank[high], c_rank[high], w_high)
                consider(
                    "coverage_gate",
                    pred,
                    weight_low=w_low,
                    weight_high=w_high,
                    tau=tau,
                    space="rank",
                )

        results.sort(key=lambda r: r["ic"], reverse=True)
        self.locked = dict(results[0])
        self.locked["valid_leaderboard"] = results[:8]
        return self.locked

    def predict(
        self,
        temporal: np.ndarray,
        cs: np.ndarray,
        mask_x: np.ndarray,
        industry: np.ndarray | None = None,
        spec: dict[str, Any] | None = None,
    ) -> np.ndarray:
        spec = spec or self.locked
        if not spec:
            raise RuntimeError("fusion is not fitted")
        name = spec["name"]
        if name == "cs_only":
            return np.asarray(cs, dtype=np.float32)
        if name == "temporal_only":
            return np.asarray(temporal, dtype=np.float32)
        if name == "raw_blend":
            return linear_blend(temporal, cs, float(spec["weight"]))

        t_rank = panel_cs_rank(temporal, mask_x)
        c_rank = panel_cs_rank(cs, mask_x)
        if spec.get("space") == "rank_industry":
            if industry is None:
                raise ValueError("industry neutralization needs cat_x[..., 6]")
            t_rank = neutralize_industry(t_rank, industry, mask_x)
            c_rank = neutralize_industry(c_rank, industry, mask_x)
        if name in {"rank_blend", "rank_blend_industry"}:
            return linear_blend(t_rank, c_rank, float(spec["weight"]))
        if name == "coverage_gate":
            n_x = np.asarray(mask_x).sum(axis=1)
            tau = float(spec["tau"])
            high = n_x >= tau
            pred = np.empty_like(t_rank, dtype=np.float32)
            pred[~high] = linear_blend(t_rank[~high], c_rank[~high], float(spec["weight_low"]))
            pred[high] = linear_blend(t_rank[high], c_rank[high], float(spec["weight_high"]))
            return pred
        raise ValueError(f"unknown fusion {name!r}")
