"""Same-day CS-z linear models: batch ridge and causal online ridge (train-only updates)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _design(
    cs_t: np.ndarray,
    idx: np.ndarray,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(cs_t[idx], dtype=np.float64)
    pieces = [np.ones((idx.size, 1), dtype=np.float64), x]
    if extra is not None:
        pieces.append(np.broadcast_to(np.asarray(extra, dtype=np.float64).reshape(1, -1), (idx.size, extra.size)))
    return np.concatenate(pieces, axis=1)


def _sample_idx(mask_y: np.ndarray, max_stocks: int | None, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(mask_y, dtype=bool))
    if max_stocks and idx.size > int(max_stocks):
        idx = np.sort(rng.choice(idx, size=int(max_stocks), replace=False))
    return idx


def fit_ridge(
    cs_sel: np.ndarray,
    data: dict[str, Any],
    start: int,
    end: int,
    *,
    lam: float = 10.0,
    max_stocks: int | None = 2000,
    seed: int = 42,
    extras_fn=None,
) -> np.ndarray:
    """OLS with L2 on concatenated train days. extras_fn(t)->(k,) broadcast to all stocks."""
    rng = np.random.default_rng(seed)
    xtx = None
    xty = None
    y_key = "y1" if "y1" in data else "y2"
    for t in range(int(start), int(end)):
        idx = _sample_idx(data["mask_y"][t], max_stocks, rng)
        if idx.size < 8:
            continue
        extra = extras_fn(t) if extras_fn is not None else None
        x = _design(cs_sel[t], idx, extra)
        y = np.asarray(data[y_key][t, idx], dtype=np.float64)
        if xtx is None:
            xtx = x.T @ x
            xty = x.T @ y
        else:
            xtx += x.T @ x
            xty += x.T @ y
    if xtx is None:
        raise RuntimeError("no train rows for ridge")
    d = xtx.shape[0]
    xtx = xtx + float(lam) * np.eye(d)
    return np.linalg.solve(xtx, xty)


def fit_online_ridge(
    cs_sel: np.ndarray,
    data: dict[str, Any],
    start: int,
    end: int,
    *,
    lam: float = 10.0,
    forget: float = 0.997,
    max_stocks: int | None = 2000,
    seed: int = 42,
    extras_fn=None,
) -> np.ndarray:
    """Causal: each day updates after that day's labels. Returns final weights (train freeze)."""
    rng = np.random.default_rng(seed)
    xtx = None
    xty = None
    w = None
    y_key = "y1" if "y1" in data else "y2"
    for t in range(int(start), int(end)):
        idx = _sample_idx(data["mask_y"][t], max_stocks, rng)
        if idx.size < 8:
            continue
        extra = extras_fn(t) if extras_fn is not None else None
        x = _design(cs_sel[t], idx, extra)
        y = np.asarray(data[y_key][t, idx], dtype=np.float64)
        if xtx is None:
            d = x.shape[1]
            xtx = float(lam) * np.eye(d)
            xty = np.zeros(d, dtype=np.float64)
        xtx *= float(forget)
        xty *= float(forget)
        xtx += x.T @ x
        xty += x.T @ y
        w = np.linalg.solve(xtx, xty)
    if w is None:
        raise RuntimeError("no train rows for online ridge")
    return w


def predict_panel(
    cs_sel: np.ndarray,
    split: dict[str, Any],
    weights: np.ndarray,
    *,
    extras_fn=None,
    fill_invalid: float = 0.0,
) -> np.ndarray:
    n_days, n_stocks = split["mask_x"].shape
    start = int(split.get("start", 0))
    out = np.full((n_days, n_stocks), float(fill_invalid), dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64)
    for local_t in range(n_days):
        g = start + local_t
        m = np.asarray(split["mask_x"][local_t], dtype=bool)
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        extra = extras_fn(g) if extras_fn is not None else None
        x = _design(cs_sel[g], idx, extra)
        out[local_t, idx] = (x @ w).astype(np.float32)
    return out
