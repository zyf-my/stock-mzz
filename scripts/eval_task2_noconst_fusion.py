"""Plug noconst baseline into locked task2 recipe. Does not overwrite main npy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402

OUT = ROOT / "outputs" / "task2"
TREE_TAU = 4491.0
TREE_W_LOW = 0.5
TREE_W_HIGH = 0.75
GATE_TAU = 4546.0
W_LOW = 0.25
W_HIGH = 0.6


def _q4(pred, y, my, nx):
    q3 = float(np.quantile(nx, 0.75))
    s = rank_ic_series(pred, y, my)
    s = s.copy()
    s[nx < q3] = np.nan
    return float(np.nanmean(s))


def main() -> None:
    splits, src = load_eval_splits(splits=("valid",), dump_if_missing=False)
    valid = splits["valid"]
    y = split_label_array(valid, "y2")
    my, mx = valid["mask_y"], valid["mask_x"]
    nx = np.asarray(mx).sum(axis=1)
    print(f"cache={src}")

    base = np.load(OUT / "baseline_noconst_valid.npy")
    hist = np.load(OUT / "hist_lgbm_valid.npy")
    old_tree = np.load(OUT / "fusion_valid.npy")
    x6 = np.load(OUT / "gru_x6_with_today_valid.npy")
    o6 = np.load(OUT / "gru_only6_with_today_valid.npy")
    n6 = np.load(OUT / "gru_next6_with_today_valid.npy")
    mlp = np.load(OUT / "cs_mlp_valid.npy")
    mlp6 = np.load(OUT / "cs_mlp_only6_valid.npy")
    locked = np.load(OUT / "fusion_x6_today_valid.npy")

    print(f"noconst baseline {mean_rank_ic(base, y, my):.6f}  Q4 {_q4(base, y, my, nx):.6f}")
    print(f"old baseline     {mean_rank_ic(np.load(OUT / 'baseline_valid.npy'), y, my):.6f}")

    tree = coverage_gate_blend(hist, base, mx, TREE_TAU, TREE_W_LOW, TREE_W_HIGH, "rank")
    print(
        f"tree (locked rule) {mean_rank_ic(tree, y, my):.6f}  "
        f"Q4 {_q4(tree, y, my, nx):.6f}  old_tree {mean_rank_ic(old_tree, y, my):.6f}"
    )

    ens = linear_blend(n6, linear_blend(o6, x6, 0.4), 0.15)
    gated = coverage_gate_blend(ens, tree, mx, GATE_TAU, W_LOW, W_HIGH, "raw")
    mlp_ens = linear_blend(mlp6, mlp, 0.4)
    blender = FusionModel({"weight_grid": [0.15]})
    blender.locked = {"name": "rank_blend", "weight": 0.15, "space": "rank", "ic": float("nan")}
    out = blender.predict(mlp_ens, gated, mx)
    ic = mean_rank_ic(out, y, my)
    print(
        f"full recipe {ic:.6f}  Q4 {_q4(out, y, my, nx):.6f}  "
        f"locked {mean_rank_ic(locked, y, my):.6f}"
    )

    print("test-proxy constant w_gru on new tree")
    for w in (0.25, 0.4, 0.6, 0.75, 0.85):
        pred = blender.predict(mlp_ens, linear_blend(ens, tree, w), mx)
        print(f"  w_gru={w:.2f}  valid={mean_rank_ic(pred, y, my):.6f}  Q4={_q4(pred, y, my, nx):.6f}")

    summary = {
        "noconst_baseline": float(mean_rank_ic(base, y, my)),
        "noconst_tree": float(mean_rank_ic(tree, y, my)),
        "noconst_full": float(ic),
        "locked_full": float(mean_rank_ic(locked, y, my)),
    }
    (OUT / "noconst_fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
