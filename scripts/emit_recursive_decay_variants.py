"""Emit nearby causal recursive-decay candidates for platform evaluation."""

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
from src.submit import save_submission  # noqa: E402
from search_recursive_decay import rank_day  # noqa: E402

OUT = ROOT / "outputs"


def recursive_decay(current, mask, initial_prev, weight, decay, scale):
    out = np.zeros_like(current, dtype=np.float32)
    prev = np.asarray(initial_prev, dtype=np.float32).copy()
    prev_valid = np.isfinite(prev)
    for t in range(current.shape[0]):
        cur_r = rank_day(current[t], mask[t])
        prev_r = rank_day(prev, mask[t] & prev_valid)
        use_prev = mask[t] & prev_valid & np.isfinite(current[t])
        wt = float(weight) * (float(decay) ** (t / float(scale)))
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
    for scale in [30, 45, 60, 90, 120]:
        for decay in [.30, .40, .50, .60, .70, .80]:
            for weight in [.60, .65, .70, .75, .80, .85]:
                pv = recursive_decay(cur_v, valid["mask_x"], prev_v, weight, decay, scale)
                s = rank_ic_series(pv, valid["y1"], valid["mask_y"])
                rows.append({"ic": float(np.nanmean(s)), "last60": float(np.nanmean(s[-60:])), "weight": weight, "decay": decay, "scale": scale})
    rows.sort(key=lambda r: r["ic"], reverse=True)
    print(json.dumps(rows[:20], indent=2))
    (OUT / "recursive_decay_variants_grid.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # Emit the best locally scoring nearby variants, keeping the prior candidate intact.
    emitted = []
    for row in rows[:8]:
        tag = f"recursive_decay_w{int(row['weight']*100):02d}_d{int(row['decay']*100):02d}_s{int(row['scale']):03d}"
        pv = recursive_decay(cur_v, valid["mask_x"], prev_v, row["weight"], row["decay"], row["scale"])
        pt = recursive_decay(cur_t, test["mask_x"], prev_t, row["weight"], row["decay"], row["scale"])
        np.save(OUT / f"fusion_alpha_{tag}_valid.npy", pv)
        np.save(OUT / f"fusion_alpha_{tag}_test.npy", pt)
        save_submission(pt, ROOT / f"submissions/task1_fusion_alpha_{tag}.npy")
        emitted.append(tag)
    print("EMITTED", emitted)


if __name__ == "__main__":
    main()
