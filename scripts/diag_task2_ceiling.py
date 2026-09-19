"""y2 branch complementarity and coverage. No training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import coverage_gate_blend, linear_blend  # noqa: E402

OUT = ROOT / "outputs" / "task2"
TAU = 4546.0


def _daily(a, b, mask):
    vals = []
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(a[t]) & np.isfinite(b[t])
        if int(m.sum()) < 8:
            continue
        r = spearmanr(a[t, m], b[t, m]).statistic
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan")


def _ic_on(pred, y, my, day_mask):
    s = rank_ic_series(pred, y, my)
    s = s.copy()
    s[~np.asarray(day_mask, dtype=bool)] = np.nan
    return float(np.nanmean(s))


def main() -> None:
    splits, src = load_eval_splits(splits=("valid",), dump_if_missing=False)
    valid = splits["valid"]
    y = split_label_array(valid, "y2")
    my, mx = valid["mask_y"], valid["mask_x"]
    nx = np.asarray(mx).sum(axis=1)
    qs = np.quantile(nx, [0.25, 0.5, 0.75])
    print(f"cache={src} cov {nx.min():.0f}-{nx.max():.0f} p25/50/75={qs}")

    names = [
        "baseline_valid.npy",
        "hist_lgbm_valid.npy",
        "fusion_valid.npy",
        "gru_x6_with_today_valid.npy",
        "gru_only6_with_today_valid.npy",
        "gru_next6_with_today_valid.npy",
        "gru_y2cols_valid.npy",
        "cs_mlp_valid.npy",
        "cs_mlp_only6_valid.npy",
        "fusion_x6_today_valid.npy",
    ]
    loaded = {n: np.load(OUT / n) for n in names if (OUT / n).is_file()}
    print("single-model valid")
    for n, arr in loaded.items():
        print(f"  {n:36s} {mean_rank_ic(arr, y, my):.6f}")

    keys = list(loaded)
    print("daily Spearman")
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            r = _daily(loaded[a], loaded[b], mx)
            if r < 0.75:
                print(f"  {a} vs {b}: {r:.3f}")

    stacks = np.stack([loaded[k] for k in keys], axis=0)
    series = [rank_ic_series(loaded[k], y, my) for k in keys]
    series = np.stack(series, axis=0)
    best = np.nanmax(series, axis=0)
    print(f"day-oracle of existing branches {float(np.nanmean(best)):.6f}")

    labels = ["Q1", "Q2", "Q3", "Q4"]
    edges = [0, qs[0], qs[1], qs[2], 1e9]
    print("coverage quartiles")
    header = "bucket n " + " ".join(f"{k.replace('_valid.npy','')[:12]:>12}" for k in keys)
    print(header)
    for i, lab in enumerate(labels):
        dm = (nx >= edges[i]) & (nx < edges[i + 1])
        ics = [f"{_ic_on(loaded[k], y, my, dm):12.4f}" for k in keys]
        print(f"{lab} {int(dm.sum()):3d} " + " ".join(ics))

    tree = loaded["fusion_valid.npy"]
    gru = loaded["gru_x6_with_today_valid.npy"]
    print("gate grid raw (GRU vs tree)")
    best_ic, best = -1.0, None
    for tau in (np.quantile(nx, q) for q in (0.4, 0.5, 0.6, 0.7)):
        for w_lo in (0.15, 0.25, 0.4):
            for w_hi in (0.25, 0.4, 0.6, 0.85):
                pred = coverage_gate_blend(gru, tree, mx, float(tau), w_lo, w_hi, "raw")
                ic = mean_rank_ic(pred, y, my)
                if ic > best_ic:
                    best_ic, best = ic, (float(tau), w_lo, w_hi)
    print(f"  best {best} ic={best_ic:.6f}  locked y1-gate {mean_rank_ic(coverage_gate_blend(gru, tree, mx, TAU, 0.25, 0.6, 'raw'), y, my):.6f}")
    print(f"  gru only {mean_rank_ic(gru, y, my):.6f} tree only {mean_rank_ic(tree, y, my):.6f}")


if __name__ == "__main__":
    main()
