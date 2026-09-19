"""Focused hi31-first bag search on promising stems + write candidate."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_task2_tree_bag_v2 import (  # noqa: E402
    HIGH_TAU,
    OUT,
    RECIPE_PATH,
    _fuse,
    _load_recipe,
    _metrics,
    _rank_bag,
)
from src.dataset import load_eval_splits  # noqa: E402
from src.submit import save_submission  # noqa: E402

FOCUS = [
    "hist_lgbm_n200",
    "hist_lgbm_n210",
    "hist_lgbm_n225",
    "hist_lgbm_n240",
    "hist_lgbm_n250",
    "hist_lgbm_rankic",
    "hist_lgbm",
    "hist_lgbm_mkt_rel",
]


def main() -> None:
    recipe = _load_recipe()
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    mx, mxt = valid["mask_x"], test["mask_x"]

    base_combo = recipe["hist_bag"]
    base_m = _metrics(_fuse(_rank_bag(base_combo, "valid", mx), "valid", recipe), valid)
    print(f"TREE_BAG base valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f}")

    present = [s for s in FOCUS if (OUT / f"{s}_valid.npy").is_file()]
    print(f"focus stems: {present}\n")

    passing: list[tuple[str, dict, list[str]]] = []
    for k in range(2, min(len(present) + 1, 5)):
        for combo in combinations(present, k):
            tag = "+".join(s.replace("hist_lgbm_", "") for s in combo)
            m = _metrics(_fuse(_rank_bag(list(combo), "valid", mx), "valid", recipe), valid)
            ok = m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6 and m["valid_ic"] >= base_m["valid_ic"] - 0.0005
            if ok and (m["hi31_ic"] > base_m["hi31_ic"] + 1e-6 or m["valid_ic"] > base_m["valid_ic"] + 1e-6):
                passing.append((tag, m, list(combo)))
                print(f"  PASS+ {tag:45s} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f} dhi={m['hi31_ic']-base_m['hi31_ic']:+.6f}")

    if not passing:
        print("no lift over TREE_BAG")
        return

    best_tag, best_m, best_combo = max(passing, key=lambda x: (x[1]["hi31_ic"], x[1]["valid_ic"]))
    print(f"\nbest: {best_tag} valid={best_m['valid_ic']:.6f} hi31={best_m['hi31_ic']:.6f}")

    hist_t = _rank_bag(best_combo, "test", mxt)
    out_t = _fuse(hist_t, "test", recipe)
    save_submission(out_t, ROOT / "submissions" / "task2_fusion_tree_bag_hi31.npy")
    np.save(OUT / "fusion_tree_bag_hi31_valid.npy", _fuse(_rank_bag(best_combo, "valid", mx), "valid", recipe))

    meta = {
        "name": f"TREE_BAG_HI31_{best_tag}",
        **recipe,
        "hist_bag": best_combo,
        **best_m,
        "baseline_valid": base_m["valid_ic"],
        "baseline_hi31": base_m["hi31_ic"],
        "platform_test_prev": 0.091445,
    }
    (OUT / "fusion_tree_bag_hi31_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "recipes" / f"TREE_BAG_HI31_{best_tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote submissions/task2_fusion_tree_bag_hi31.npy")


if __name__ == "__main__":
    main()
