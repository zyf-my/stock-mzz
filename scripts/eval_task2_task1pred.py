"""Can task1 scores help y2? Diagnostic only. Does not write a main."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402

OUT2 = ROOT / "outputs" / "task2"
OUT1 = ROOT / "outputs"


def _q4(pred, y, my, nx):
    s = rank_ic_series(pred, y, my).copy()
    s[nx < float(np.quantile(nx, 0.75))] = np.nan
    return float(np.nanmean(s))


def _daily(a, b, mask):
    from scipy.stats import spearmanr

    vals = []
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(a[t]) & np.isfinite(b[t])
        if int(m.sum()) < 8:
            continue
        r = spearmanr(a[t, m], b[t, m]).statistic
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    splits, _ = load_eval_splits(splits=("valid",), dump_if_missing=False)
    valid = splits["valid"]
    y2 = split_label_array(valid, "y2")
    y1 = split_label_array(valid, "y1")
    my, mx = valid["mask_y"], valid["mask_x"]
    nx = np.asarray(mx).sum(axis=1)

    cands = [
        OUT1 / "fusion_x6_today_valid.npy",
        ROOT / "submissions" / "task1_fusion_x6_today.npy",
        OUT2 / "fusion_rank_gate_valid.npy",
    ]
    t1 = None
    for p in cands[:2]:
        if p.is_file():
            arr = np.load(p)
            if arr.shape == y2.shape:
                t1 = arr
                print(f"task1 pred {p}")
                break
            print(f"skip {p} shape {arr.shape} vs valid {y2.shape}")
    if t1 is None:
        print("no task1 valid pred with matching shape")
        return

    t2 = np.load(OUT2 / "fusion_rank_gate_valid.npy")
    print(f"task1 vs y1 {mean_rank_ic(t1, y1, my):.6f}")
    print(f"task1 vs y2 {mean_rank_ic(t1, y2, my):.6f}  Q4 {_q4(t1, y2, my, nx):.6f}")
    print(f"task2 vs y2 {mean_rank_ic(t2, y2, my):.6f}")
    print(f"task1 vs task2 daily Spearman {_daily(t1, t2, mx):.3f}")

    blender = FusionModel({"weight_grid": [0.15]})
    print("rank blend task1 pred into task2 (weight = task1)")
    for w in (0.15, 0.25, 0.4, 0.5):
        blender.locked = {"name": "rank_blend", "weight": w, "space": "rank", "ic": float("nan")}
        out = blender.predict(t1, t2, mx)
        print(f"  w_t1={w:.2f}  y2={mean_rank_ic(out, y2, my):.6f}  Q4={_q4(out, y2, my, nx):.6f}")


if __name__ == "__main__":
    main()
