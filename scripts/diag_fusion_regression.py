"""Compare fusion recipes: find why test dropped 0.091354 -> 0.090172."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"

RECIPES = {
    "OLD_091354": {"ow": 0.7, "nw": 0.34, "glo": 0.28, "ghi": 0.92, "mm": 0.0, "mw": 0.1},
    "OLD_091004": {"ow": 0.7, "nw": 0.34, "glo": 0.28, "ghi": 0.92, "mm": 0.4, "mw": 0.06},
    "NEW_gate1": {"ow": 0.76, "nw": 0.38, "glo": 0.26, "ghi": 1.0, "mm": 0.0, "mw": 0.1},
    "NEW_summary": {"ow": 0.72, "nw": 0.36, "glo": 0.26, "ghi": 1.0, "mm": 0.0, "mw": 0.1},
}


def build(recipe: dict, split: str) -> np.ndarray:
    splits, _ = load_eval_splits(splits=(split,), dump_if_missing=False)
    data = splits[split]
    mx = data["mask_x"]
    lt = lambda n: np.load(OUT / f"{n}_{split}.npy")
    tree = coverage_gate_blend(lt("hist_lgbm_n200"), lt("baseline"), mx, 4614.0, 0.45, 1.0, "rank")
    x6, o6, n6 = lt("gru_x6_with_today"), lt("gru_only6_with_today"), lt("gru_next6_with_today")
    ens = linear_blend(n6, linear_blend(o6, x6, recipe["ow"]), recipe["nw"])
    gated = coverage_gate_blend(ens, tree, mx, 4670.0, recipe["glo"], recipe["ghi"], "rank")
    mlp = linear_blend(lt("cs_mlp_only6"), lt("cs_mlp"), recipe["mm"])
    mw = recipe["mw"]
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp, gated, mx)


def main() -> None:
    splits, _ = load_eval_splits(splits=("valid",), dump_if_missing=False)
    valid = splits["valid"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    hi = np.asarray(mx).sum(axis=1) >= 4670.0

    preds = {}
    for name, r in RECIPES.items():
        pv = build(r, "valid")
        pt = build(r, "test")
        preds[name] = (pv, pt)
        s = rank_ic_series(pv, y, my)
        s[~hi] = np.nan
        print(
            f"{name:12s} valid={mean_rank_ic(pv, y, my):.6f} "
            f"hi31={float(np.nanmean(s)):.6f} test_mean={float(np.nanmean(pt)):.2f}"
        )

    sub_path = ROOT / "submissions" / "task2_fusion_best.npy"
    if sub_path.is_file():
        cur = np.load(sub_path)
        for name, (_, pt) in preds.items():
            m = np.isfinite(cur) & np.isfinite(pt)
            r = spearmanr(cur[m], pt[m]).statistic
            print(f"sub vs {name:12s} spearman={r:.6f} max_diff={float(np.max(np.abs(cur - pt))):.6f}")

    # restore OLD_091354
    import json

    r = RECIPES["OLD_091354"]
    out_t = preds["OLD_091354"][1]
    save_submission(out_t, sub_path)
    summary = {
        "ic": float(mean_rank_ic(preds["OLD_091354"][0], y, my)),
        "platform_test": 0.091354,
        "tree": {"tau": 4614.0, "w_lo": 0.45, "w_hi": 1.0},
        "gate": {"tau": 4670.0, "w_lo": r["glo"], "w_hi": r["ghi"]},
        "ens": {"only6_w": r["ow"], "next6_w": r["nw"]},
        "mlp_w": r["mw"],
        "mlp6_mix": r["mm"],
        "x6": "single_s42",
        "notes": "restored after gate w_hi=1.0 hurt test 0.091354->0.090172",
    }
    (OUT / "fusion_best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(OUT / "fusion_best_valid.npy", preds["OLD_091354"][0])
    print("restored task2_fusion_best.npy to OLD_091354 recipe")
    print("VALID_RANKIC", f"{summary['ic']:.6f}")


if __name__ == "__main__":
    main()
