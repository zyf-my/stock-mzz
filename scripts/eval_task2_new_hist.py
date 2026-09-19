"""Scan a new hist checkpoint's n_estimators and fuse into rank-gate + hist_n200 recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
LOCKED = 0.086316
ITERS = [50, 100, 150, 200, 250, 300, 350, 400]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="stem, e.g. hist_lgbm_n1200")
    args = parser.parse_args()
    name = str(args.name)

    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y = split_label_array(valid, "y2")
    my, mx = valid["mask_y"], valid["mask_x"]

    hist_v = np.load(OUT / f"{name}_valid.npy")
    hist_t = np.load(OUT / f"{name}_test.npy")
    print(f"{name} @saved {mean_rank_ic(hist_v, y, my):.6f}")

    # saved pred is full trees; if we only have one snapshot, use it
    base_v = np.load(OUT / "baseline_valid.npy")
    base_t = np.load(OUT / "baseline_test.npy")
    tree_v = coverage_gate_blend(hist_v, base_v, mx, 4491.0, 0.5, 0.75, "rank")
    tree_t = coverage_gate_blend(hist_t, base_t, test["mask_x"], 4491.0, 0.5, 0.75, "rank")
    print(f"tree fusion {mean_rank_ic(tree_v, y, my):.6f}")

    ens_v = linear_blend(
        np.load(OUT / "gru_next6_with_today_valid.npy"),
        linear_blend(np.load(OUT / "gru_only6_with_today_valid.npy"), np.load(OUT / "gru_x6_with_today_valid.npy"), 0.4),
        0.15,
    )
    ens_t = linear_blend(
        np.load(OUT / "gru_next6_with_today_test.npy"),
        linear_blend(np.load(OUT / "gru_only6_with_today_test.npy"), np.load(OUT / "gru_x6_with_today_test.npy"), 0.4),
        0.15,
    )
    gated_v = coverage_gate_blend(ens_v, tree_v, mx, 4546.0, 0.25, 0.6, "rank")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], 4546.0, 0.25, 0.6, "rank")
    mlp_v = linear_blend(np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_valid.npy"), 0.4)
    mlp_t = linear_blend(np.load(OUT / "cs_mlp_only6_test.npy"), np.load(OUT / "cs_mlp_test.npy"), 0.4)
    blender = FusionModel({"weight_grid": [0.15]})
    blender.locked = {"name": "rank_blend", "weight": 0.15, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_v, gated_v, mx)
    out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
    ic = mean_rank_ic(out_v, y, my)
    print(f"rank-gate + {name} {ic:.6f}  locked {LOCKED:.6f}")
    (OUT / f"{name}_fusion_summary.json").write_text(
        json.dumps({"single": float(mean_rank_ic(hist_v, y, my)), "tree": float(mean_rank_ic(tree_v, y, my)), "fusion": float(ic)}, indent=2),
        encoding="utf-8",
    )
    if ic > LOCKED + 1e-6:
        np.save(OUT / f"fusion_{name}_valid.npy", out_v)
        np.save(OUT / f"fusion_{name}_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions" / f"task2_fusion_{name}.npy")
        print("wrote new submission")
    else:
        print("no lift")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
