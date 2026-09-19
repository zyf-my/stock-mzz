"""LambdaRankIC objective for LightGBM (Lin et al., 2026).

Directly targets Spearman Rank IC via pairwise lambda gradients weighted by
|Delta RankIC| from rank swaps. See Algorithm 1 in the paper.
"""

from __future__ import annotations

import numpy as np


def _group_grad_hess(
    s: np.ndarray,
    y_raw: np.ndarray,
    *,
    max_pair_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(s.shape[0])
    g_loc = np.zeros(n, dtype=np.float64)
    h_loc = np.zeros(n, dtype=np.float64)
    if n < 3:
        return g_loc, h_loc

    order_y = np.argsort(-y_raw, kind="mergesort")
    y_rank = np.empty(n, dtype=np.float64)
    y_rank[order_y] = np.arange(1, n + 1, dtype=np.float64)

    order_s = np.argsort(-s, kind="mergesort")
    pred_rank = np.empty(n, dtype=np.float64)
    pred_rank[order_s] = np.arange(1, n + 1, dtype=np.float64)

    denom = float(n * (n * n - 1))
    if denom <= 0:
        return g_loc, h_loc

    n_pairs = n * (n - 1) // 2
    k = min(int(max_pair_samples), n_pairs)
    if n_pairs <= k:
        i_idx, j_idx = np.triu_indices(n, k=1)
    else:
        i_idx = rng.integers(0, n, size=k, dtype=np.int64)
        j_idx = rng.integers(0, n, size=k, dtype=np.int64)
        ok = i_idx != j_idx
        i_idx, j_idx = i_idx[ok], j_idx[ok]
        if i_idx.size == 0:
            return g_loc, h_loc

    yi = y_rank[i_idx]
    yj = y_rank[j_idx]
    swap = yi < yj
    i2 = np.where(swap, j_idx, i_idx)
    j2 = np.where(swap, i_idx, j_idx)
    yi = y_rank[i2]
    yj = y_rank[j2]
    keep = yi > yj
    i2, j2 = i2[keep], j2[keep]
    yi, yj = yi[keep], yj[keep]
    if i2.size == 0:
        return g_loc, h_loc

    ri = pred_rank[i2]
    rj = pred_rank[j2]
    delta = 12.0 * np.abs(rj - ri) * np.abs(yi - yj) / denom
    si = s[i2]
    sj = s[j2]
    p = 1.0 / (1.0 + np.exp(-(si - sj)))
    lam = (p - 1.0) * delta
    h_ij = 2.0 * p * (1.0 - p) * delta

    np.add.at(g_loc, i2, lam)
    np.add.at(g_loc, j2, -lam)
    np.add.at(h_loc, i2, h_ij)
    np.add.at(h_loc, j2, h_ij)
    return g_loc, h_loc


def lambdarank_ic_grad_hess(
    preds: np.ndarray,
    labels: np.ndarray,
    group_sizes: np.ndarray,
    *,
    max_pair_samples: int = 512,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gradient and hessian w.r.t. predictions for one boosting step."""
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    grad = np.zeros_like(preds)
    hess = np.zeros_like(preds)
    rng = np.random.default_rng(seed)
    offset = 0
    for gsize in np.asarray(group_sizes, dtype=np.int64):
        n = int(gsize)
        sl = slice(offset, offset + n)
        g_loc, h_loc = _group_grad_hess(
            preds[sl],
            labels[sl],
            max_pair_samples=max_pair_samples,
            rng=rng,
        )
        grad[sl] = g_loc
        hess[sl] = np.maximum(h_loc, 1e-8)
        offset += n
    return grad, hess


def make_lambdarank_ic_objective(max_pair_samples: int = 512, seed: int = 42):
    """LightGBM custom objective closure."""

    def _fobj(preds: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        labels = train_data.get_label()
        group = train_data.get_group()
        if group is None:
            raise ValueError("LambdaRankIC requires group sizes (one query per day)")
        return lambdarank_ic_grad_hess(
            preds,
            labels,
            group,
            max_pair_samples=max_pair_samples,
            seed=seed,
        )

    return _fobj
