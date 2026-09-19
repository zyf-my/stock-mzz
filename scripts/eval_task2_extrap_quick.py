"""Quick manual extrap bag combos."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_task2_tree_bag_v2 import _fuse, _metrics, _rank_bag  # noqa: E402
from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import panel_cs_rank  # noqa: E402

OUT = ROOT / "outputs" / "task2"
RECIPE = OUT / "recipes" / "TREE_BAG_091467.json"
CORE = ["hist_lgbm_n240", "hist_lgbm_rankic", "hist_lgbm_mkt_rel"]
CANDIDATES = [
    "hist_lgbm_rankic_mkt",
    "hist_lgbm_extrap_finetune",
    "hist_lgbm_rankic_s43",
    "hist_lgbm_rankic_s44",
    "hist_lgbm_extrap",
]

def main() -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    splits, _ = load_eval_splits(splits=("valid",), dump_if_missing=False)
    valid = splits["valid"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    hi = mx.sum(1) >= 4670
    base_m = _metrics(_fuse(_rank_bag(CORE, "valid", mx), "valid", recipe), valid)
    print("base", base_m)
    for s in CANDIDATES:
        p = OUT / f"{s}_valid.npy"
        if not p.is_file():
            continue
        r = panel_cs_rank(np.load(p), mx)
        print(
            f"{s:30s} rank valid={mean_rank_ic(r, y, my):.6f} "
            f"hi31={mean_rank_ic(r[hi], y[hi], my[hi]):.6f}"
        )

    combos = [
        ["hist_lgbm_n240", "hist_lgbm_rankic_mkt", "hist_lgbm_mkt_rel"],
        ["hist_lgbm_n240", "hist_lgbm_rankic", "hist_lgbm_rankic_mkt", "hist_lgbm_mkt_rel"],
        ["hist_lgbm_n240", "hist_lgbm_rankic", "hist_lgbm_rankic_s43", "hist_lgbm_mkt_rel"],
        ["hist_lgbm_n240", "hist_lgbm_rankic", "hist_lgbm_rankic_s44", "hist_lgbm_mkt_rel"],
        CORE + ["hist_lgbm_rankic_s43"],
        CORE + ["hist_lgbm_rankic_mkt"],
    ]
    for combo in combos:
        combo = [c for c in combo if (OUT / f"{c}_valid.npy").is_file()]
        if len(combo) < 2:
            continue
        m = _metrics(_fuse(_rank_bag(combo, "valid", mx), "valid", recipe), valid)
        tag = "+".join(c.replace("hist_lgbm_", "") for c in combo)
        ok = m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6 and m["valid_ic"] >= base_m["valid_ic"] - 0.0005
        print(f"{tag:55s} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f} {'PASS' if ok else 'fail'}")


if __name__ == "__main__":
    main()
