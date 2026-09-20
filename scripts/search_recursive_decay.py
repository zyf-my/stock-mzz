"""Search a causal recursive blend whose lag weight decays with forecast age."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import panel_cs_rank  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs"


def rank_day(x, mask):
    x = np.asarray(x, dtype=np.float32)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    return panel_cs_rank(np.nan_to_num(x, nan=0.0)[None, :], m[None, :])[0]


def recursive_decay(current, mask, initial_prev, weight, decay):
    out = np.zeros_like(current, dtype=np.float32)
    prev = np.asarray(initial_prev, dtype=np.float32).copy()
    prev_valid = np.isfinite(prev)
    for t in range(current.shape[0]):
        cur_r = rank_day(current[t], mask[t])
        prev_r = rank_day(prev, mask[t] & prev_valid)
        use_prev = mask[t] & prev_valid & np.isfinite(current[t])
        wt = float(weight) * (float(decay) ** (t / 60.0))
        out[t] = cur_r
        out[t, use_prev] = ((1.0 - wt) * cur_r[use_prev] + wt * prev_r[use_prev]).astype(np.float32)
        prev = out[t]
        prev_valid = np.asarray(mask[t], dtype=bool) & np.isfinite(current[t])
    return out


def main():
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    cur_v = np.load(OUT / "fusion_alpha_platform_mix_valid.npy")
    cur_t = np.load(OUT / "fusion_alpha_platform_mix_test.npy")
    prev_v = np.asarray(data["y1"][valid["start"] - 1], dtype=np.float32)
    prev_t = np.asarray(data["y1"][test["start"] - 1], dtype=np.float32)
    rows = []
    candidates = []
    for w in [0.45, .5, .55, .6, .65, .7, .75, .8, .85, .9]:
        for decay in [.50, .65, .75, .85, .90, .95, .98, 1.0]:
            pv = recursive_decay(cur_v, valid["mask_x"], prev_v, w, decay)
            s = rank_ic_series(pv, valid["y1"], valid["mask_y"])
            row = (float(np.nanmean(s)), float(np.nanmean(s[-60:])), float(w), float(decay))
            rows.append(row)
            candidates.append((row, pv))
    rows.sort(reverse=True, key=lambda x: x[0])
    print("BEST", rows[:20])
    (OUT / "recursive_decay_grid.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    best = candidates[max(range(len(candidates)), key=lambda i: candidates[i][0][0])]
    if best[0][0] > 0.129307581:
        _, _, w, decay = best[0]
        pt = recursive_decay(cur_t, test["mask_x"], prev_t, w, decay)
        tag = f"recursive_decay_w{int(w*100):02d}_d{int(decay*100):02d}"
        np.save(OUT / f"fusion_alpha_{tag}_valid.npy", best[1])
        np.save(OUT / f"fusion_alpha_{tag}_test.npy", pt)
        save_submission(pt, ROOT / f"submissions/task1_fusion_alpha_{tag}.npy")
        (OUT / f"fusion_alpha_{tag}_lock.json").write_text(
            json.dumps({"valid_ic": best[0][0], "last60_ic": best[0][1], "weight": w, "decay": decay, "causal": True}, indent=2),
            encoding="utf-8",
        )
        print("WROTE", f"submissions/task1_fusion_alpha_{tag}.npy")


if __name__ == "__main__":
    main()
