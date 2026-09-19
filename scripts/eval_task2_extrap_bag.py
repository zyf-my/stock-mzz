"""Search extrap + multi-seed rankic stems in TREE_BAG_091467 shell (hi31-first)."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_task2_tree_bag_v2 import (  # noqa: E402
    _fuse,
    _metrics,
    _rank_bag,
)
from src.dataset import load_eval_splits  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
RECIPE_PATH = OUT / "recipes" / "TREE_BAG_091467.json"

CORE = ["hist_lgbm_n240", "hist_lgbm_rankic", "hist_lgbm_mkt_rel"]
NEW_STEMS = [
    "hist_lgbm_extrap",
    "hist_lgbm_rankic_mkt",
    "hist_lgbm_extrap_finetune",
    "hist_lgbm_rankic_s43",
    "hist_lgbm_rankic_s44",
]


def main() -> None:
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    mx, mxt = valid["mask_x"], test["mask_x"]

    base_m = _metrics(_fuse(_rank_bag(CORE, "valid", mx), "valid", recipe), valid)
    print(f"TREE_BAG_091467 base valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f}")

    present_new = [s for s in NEW_STEMS if (OUT / f"{s}_valid.npy").is_file()]
    if not present_new:
        print("no new stems found; train extrap/rankic configs first")
        return
    print(f"new stems: {present_new}\n")

    passing: list[tuple[str, dict, list[str]]] = []

    # replace one core stem or add to bag
    for new in present_new:
        for combo in [
            [new] + [c for c in CORE if c != "hist_lgbm_mkt_rel"],
            [c for c in CORE if c != "hist_lgbm_rankic"] + [new],
            CORE + [new],
        ]:
            combo = list(dict.fromkeys(combo))
            tag = "+".join(s.replace("hist_lgbm_", "") for s in combo)
            m = _metrics(_fuse(_rank_bag(combo, "valid", mx), "valid", recipe), valid)
            ok = m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6 and m["valid_ic"] >= base_m["valid_ic"] - 0.0005
            lift = m["hi31_ic"] - base_m["hi31_ic"]
            if ok and lift > 1e-6:
                passing.append((tag, m, combo))
                print(f"  PASS+ {tag:50s} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f} dhi={lift:+.6f}")

    for k in range(2, min(len(present_new) + 1, 4)):
        for extra in combinations(present_new, k):
            combo = CORE + list(extra)
            tag = "+".join(s.replace("hist_lgbm_", "") for s in combo)
            m = _metrics(_fuse(_rank_bag(combo, "valid", mx), "valid", recipe), valid)
            ok = m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6 and m["valid_ic"] >= base_m["valid_ic"] - 0.0005
            if ok and m["hi31_ic"] > base_m["hi31_ic"] + 1e-6:
                passing.append((tag, m, combo))
                print(f"  PASS+ {tag:50s} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f}")

    if not passing:
        print("\nno extrap bag beat base hi31")
        return

    best_tag, best_m, best_combo = max(passing, key=lambda x: (x[1]["hi31_ic"], x[1]["valid_ic"]))
    print(f"\nbest: {best_tag} valid={best_m['valid_ic']:.6f} hi31={best_m['hi31_ic']:.6f}")

    out_t = _fuse(_rank_bag(best_combo, "test", mxt), "test", recipe)
    save_submission(out_t, ROOT / "submissions" / "task2_fusion_extrap_bag.npy")
    np.save(OUT / "fusion_extrap_bag_valid.npy", _fuse(_rank_bag(best_combo, "valid", mx), "valid", recipe))

    meta = {
        "name": f"EXTRAP_BAG_{best_tag}",
        **recipe,
        "hist_bag": best_combo,
        **best_m,
        "baseline_valid": base_m["valid_ic"],
        "baseline_hi31": base_m["hi31_ic"],
        "platform_test_prev": 0.091467,
    }
    (OUT / "fusion_extrap_bag_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "recipes" / f"EXTRAP_BAG_{best_tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote submissions/task2_fusion_extrap_bag.npy")
    print("VALID_RANKIC", f"{best_m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
