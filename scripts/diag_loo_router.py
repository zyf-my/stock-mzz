"""Leave-one-day-out router on valid. Analysis only — does not write a submission.

If coverage + tree/GRU disagreement can pick the daily winner, LOO RankIC
shows how much of the 0.14 gap is reachable without a new model class.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import linear_blend  # noqa: E402


def day_disagree(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape[0], dtype=np.float64)
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(a[t]) & np.isfinite(b[t])
        if int(m.sum()) < 8:
            continue
        r = spearmanr(a[t, m], b[t, m]).statistic
        out[t] = 1.0 - float(r) if np.isfinite(r) else 1.0
    return out


def main() -> None:
    splits, src = load_eval_splits(splits=("valid",), dump_if_missing=False)
    valid = splits["valid"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    n_x = np.asarray(mx).sum(axis=1).astype(np.float64)
    tree = np.load(ROOT / "outputs/fusion_valid.npy")
    gru = np.load(ROOT / "outputs/gru_no_today_recent_n2000_valid.npy")
    mlp = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    names = ["tree", "gru", "mlp"]
    preds = [tree, gru, mlp]
    series = np.vstack([rank_ic_series(p, y, my) for p in preds])
    winner = np.nanargmax(series, axis=0)
    disagree = day_disagree(gru, tree, mx)
    x = np.column_stack(
        [
            (n_x - n_x.mean()) / (n_x.std() + 1e-8),
            (disagree - disagree.mean()) / (disagree.std() + 1e-8),
            n_x * disagree / 1e6,
        ]
    )
    n = x.shape[0]
    picked = np.empty(n, dtype=np.int32)
    for t in range(n):
        tr = np.ones(n, dtype=bool)
        tr[t] = False
        # nearest-centroid by class
        cents = []
        for k in range(3):
            sel = tr & (winner == k)
            if int(sel.sum()) == 0:
                cents.append(np.zeros(x.shape[1]))
            else:
                cents.append(x[sel].mean(axis=0))
        cents = np.stack(cents)
        picked[t] = int(np.argmin(np.linalg.norm(cents - x[t], axis=1)))
    loo_pred = np.empty_like(tree)
    for t in range(n):
        loo_pred[t] = preds[int(picked[t])][t]
    acc = float((picked == winner).mean())
    print(f"loaded {src}")
    print(f"class prior { {names[k]: float((winner == k).mean()) for k in range(3)} }")
    print(f"LOO pick acc={acc:.3f}  counts={ {names[k]: int((picked == k).sum()) for k in range(3)} }")
    print(f"LOO hard-pick RankIC={mean_rank_ic(loo_pred, y, my):.6f}")
    # soft blend from distances
    soft = np.empty_like(tree)
    for t in range(n):
        tr = np.ones(n, dtype=bool)
        tr[t] = False
        cents = []
        for k in range(3):
            sel = tr & (winner == k)
            cents.append(x[sel].mean(axis=0) if int(sel.sum()) else np.zeros(x.shape[1]))
        dist = np.linalg.norm(np.stack(cents) - x[t], axis=1) + 1e-6
        w = 1.0 / dist
        w = w / w.sum()
        soft[t] = w[0] * tree[t] + w[1] * gru[t] + w[2] * mlp[t]
    print(f"LOO soft-blend RankIC={mean_rank_ic(soft, y, my):.6f}")
    print(f"oracle RankIC={float(np.nanmean(np.nanmax(series, axis=0))):.6f}")
    # coverage-only 3-bin pick (no valid-fit of class): Q1 tree, Q4 gru, else tree
    edges = np.nanpercentile(n_x, [0, 25, 75, 100])
    rule = np.empty_like(tree)
    for t in range(n):
        if n_x[t] >= edges[2]:
            rule[t] = gru[t]
        else:
            rule[t] = tree[t]
    print(f"rule Q4=GRU else tree RankIC={mean_rank_ic(rule, y, my):.6f}")
    print(f"equal blend 0.5 tree/gru RankIC={mean_rank_ic(linear_blend(gru, tree, 0.5), y, my):.6f}")


if __name__ == "__main__":
    main()
