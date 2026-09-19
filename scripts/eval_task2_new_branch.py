"""Rank-blend a new task2 branch into the locked rank-gate. Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_task2_new_branch.py --name stock_z_lgbm
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_task2_new_branch.py --name regime_factor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
LOCKED_IC = 0.086225


def _q4(pred, y, my, nx):
    s = rank_ic_series(pred, y, my).copy()
    s[nx < float(np.quantile(nx, 0.75))] = np.nan
    return float(np.nanmean(s))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    name = str(args.name)

    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y = split_label_array(valid, "y2")
    my, mx = valid["mask_y"], valid["mask_x"]
    nx = np.asarray(mx).sum(axis=1)

    branch_v = np.load(OUT / f"{name}_valid.npy")
    branch_t = np.load(OUT / f"{name}_test.npy")
    locked_v = np.load(OUT / "fusion_rank_gate_valid.npy")
    locked_t = np.load(OUT / "fusion_rank_gate_test.npy")

    single = mean_rank_ic(branch_v, y, my)
    print(f"{name} RankIC={single:.6f}  Q4={_q4(branch_v, y, my, nx):.6f}")
    print(f"vs rank-gate daily Spearman {_daily(branch_v, locked_v, mx):.3f}")

    blender = FusionModel({"weight_grid": [0.15]})
    best = (-1.0, None, None)
    for w in (0.15, 0.25, 0.4, 0.5):
        blender.locked = {"name": "rank_blend", "weight": w, "space": "rank", "ic": float("nan")}
        out = blender.predict(branch_v, locked_v, mx)
        ic = mean_rank_ic(out, y, my)
        q4 = _q4(out, y, my, nx)
        print(f"  w_new={w:.2f}  valid={ic:.6f}  Q4={q4:.6f}")
        if ic > best[0]:
            best = (ic, w, out)
    print(f"best {best[1]} ic={best[0]:.6f}  locked {LOCKED_IC:.6f}")

    summary = {
        "name": name,
        "single": float(single),
        "best_w": best[1],
        "best_ic": float(best[0]),
        "locked": LOCKED_IC,
    }
    (OUT / f"{name}_fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if best[0] > LOCKED_IC + 1e-6:
        blender.locked = {"name": "rank_blend", "weight": best[1], "space": "rank", "ic": float("nan")}
        out_t = blender.predict(branch_t, locked_t, test["mask_x"])
        np.save(OUT / f"fusion_{name}_valid.npy", best[2])
        np.save(OUT / f"fusion_{name}_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions" / f"task2_fusion_{name}.npy")
        print("wrote new submission")
    else:
        print("no fusion lift; not writing a new main")
    print("VALID_RANKIC", f"{best[0]:.6f}")


if __name__ == "__main__":
    main()
