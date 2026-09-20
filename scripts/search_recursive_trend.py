"""Search a causal recursive signal with a two-day trend term."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split  # noqa: E402
from src.metrics import rank_ic_series  # noqa: E402
from src.models.fusion import panel_cs_rank  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs"


def rank_day(x, mask):
    x = np.asarray(x, dtype=np.float32)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    return panel_cs_rank(np.nan_to_num(x, nan=0.0)[None, :], m[None, :])[0]


def recursive_trend(current, mask, prev1, prev2, weight, decay, scale, beta):
    out = np.zeros_like(current, dtype=np.float32)
    p1 = np.asarray(prev1, dtype=np.float32).copy()
    p2 = np.asarray(prev2, dtype=np.float32).copy()
    v1 = np.isfinite(p1)
    v2 = np.isfinite(p2)
    for t in range(current.shape[0]):
        cur_r = rank_day(current[t], mask[t])
        r1 = rank_day(p1, mask[t] & v1)
        r2 = rank_day(p2, mask[t] & v2)
        use = mask[t] & v1 & np.isfinite(current[t])
        trend = r1 + float(beta) * (r1 - r2)
        wt = float(weight) * (float(decay) ** (t / float(scale)))
        out[t] = cur_r
        out[t, use] = ((1.0 - wt) * cur_r[use] + wt * trend[use]).astype(np.float32)
        p2, v2 = p1, v1
        p1, v1 = out[t], np.asarray(mask[t], dtype=bool) & np.isfinite(current[t])
    return out


def main():
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    cur_v = np.load(OUT / "fusion_alpha_platform_mix_valid.npy")
    cur_t = np.load(OUT / "fusion_alpha_platform_mix_test.npy")
    y = np.asarray(data["y1"], dtype=np.float32)
    p1v, p2v = y[valid["start"] - 1], y[valid["start"] - 2]
    p1t, p2t = y[test["start"] - 1], y[test["start"] - 2]
    rows, candidates = [], []
    for beta in [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]:
        for weight in [.65, .75, .85, .9]:
            for decay in [.005, .01, .02]:
                pv = recursive_trend(cur_v, valid["mask_x"], p1v, p2v, weight, decay, 45, beta)
                s = rank_ic_series(pv, valid["y1"], valid["mask_y"])
                row = (float(np.nanmean(s)), float(np.nanmean(s[-60:])), beta, weight, decay)
                rows.append(row); candidates.append((row, pv))
    rows.sort(reverse=True, key=lambda x: x[0])
    print("BEST", rows[:15])
    (OUT / "recursive_trend_grid.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row, pv in sorted(candidates, key=lambda x: x[0][0], reverse=True)[:5]:
        _, _, beta, weight, decay = row
        tag = f"recursive_trend_b{int(beta*100):+03d}_w{int(weight*100):02d}_d{int(decay*1000):03d}"
        pt = recursive_trend(cur_t, test["mask_x"], p1t, p2t, weight, decay, 45, beta)
        np.save(OUT / f"fusion_alpha_{tag}_valid.npy", pv)
        np.save(OUT / f"fusion_alpha_{tag}_test.npy", pt)
        save_submission(pt, ROOT / f"submissions/task1_fusion_alpha_{tag}.npy")
        print("WROTE", f"submissions/task1_fusion_alpha_{tag}.npy")


if __name__ == "__main__":
    main()
