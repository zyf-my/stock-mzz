"""Expand hist bag search: more stems + hi31-first weights under TREE_BAG shell."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
HIGH_TAU = 4670.0
RECIPE_PATH = OUT / "recipes" / "TREE_BAG_091445.json"

CANDIDATE_STEMS = [
    "hist_lgbm_n200",
    "hist_lgbm_n225",
    "hist_lgbm_rankic",
    "hist_lgbm_mkt_rel",
    "hist_lgbm_reg",
    "hist_lgbm_n1200",
    "hist_lgbm",
    "hist_lgbm_y2cols",
    "hist_lgbm_covw",
]


def _load_recipe() -> dict:
    return json.loads(RECIPE_PATH.read_text(encoding="utf-8"))


def _fuse(hist: np.ndarray, split: str, recipe: dict) -> np.ndarray:
    lt = lambda s: np.load(OUT / f"{s}_{split}.npy")
    splits, _ = load_eval_splits(splits=(split,), dump_if_missing=False)
    mx = splits[split]["mask_x"]
    tp, gp = recipe["tree"], recipe["gate"]
    ow, nw = recipe["ens"]["only6_w"], recipe["ens"]["next6_w"]
    mw = float(recipe["mlp_w"])
    tree = coverage_gate_blend(hist, lt("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens = linear_blend(
        lt(recipe["next6"]),
        linear_blend(lt(recipe["only6"]), lt(recipe["x6"]), ow),
        nw,
    )
    gated = coverage_gate_blend(ens, tree, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp = lt(recipe["mlp"])
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp, gated, mx)


def _metrics(pred: np.ndarray, valid: dict) -> dict:
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    ic = mean_rank_ic(pred, y, my)
    hi = np.asarray(mx).sum(axis=1) >= HIGH_TAU
    ic_hi = mean_rank_ic(pred[hi], y[hi], my[hi])
    return {"valid_ic": float(ic), "hi31_ic": float(ic_hi)}


def _rank_bag(stems: list[str], split: str, mx: np.ndarray, weights: list[float] | None = None) -> np.ndarray:
    ranks = [panel_cs_rank(np.load(OUT / f"{s}_{split}.npy"), mx) for s in stems]
    if weights is None:
        return np.mean(ranks, axis=0).astype(np.float32)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = np.zeros_like(ranks[0], dtype=np.float32)
    for wi, r in zip(w, ranks):
        out += float(wi) * r
    return out


def main() -> None:
    recipe = _load_recipe()
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    mx = valid["mask_x"]
    mxt = test["mask_x"]

    present = [s for s in CANDIDATE_STEMS if (OUT / f"{s}_valid.npy").is_file()]
    base_combo = recipe["hist_bag"]
    base_hist = _rank_bag(base_combo, "valid", mx)
    base_m = _metrics(_fuse(base_hist, "valid", recipe), valid)
    print(f"TREE_BAG base: valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f}")
    print(f"candidate stems: {present}\n")

    rows: list[tuple[str, dict, list[str], list[float] | None]] = []

    # equal-weight combos
    for k in range(2, min(len(present) + 1, 5)):
        for combo in combinations(present, k):
            tag = "+".join(s.replace("hist_lgbm_", "") for s in combo)
            hist_v = _rank_bag(list(combo), "valid", mx)
            m = _metrics(_fuse(hist_v, "valid", recipe), valid)
            rows.append((tag, m, list(combo), None))
            ok = m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6 and m["valid_ic"] >= base_m["valid_ic"] - 0.0005
            print(f"  {tag:45s} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f} gate={'PASS' if ok else 'FAIL'}")

    # weighted search on core trio (hi31-first, then valid)
    core = [s for s in ["hist_lgbm_n200", "hist_lgbm_n225", "hist_lgbm_rankic"] if s in present]
    if len(core) == 3:
        print("\nweighted core trio:")
        best_w: tuple[str, dict, list[float]] | None = None
        for w0 in np.arange(0.15, 0.66, 0.05):
            for w1 in np.arange(0.15, 0.66, 0.05):
                w2 = 1.0 - w0 - w1
                if w2 < 0.10:
                    continue
                hist_v = _rank_bag(core, "valid", mx, [w0, w1, w2])
                m = _metrics(_fuse(hist_v, "valid", recipe), valid)
                tag = f"wt_{w0:.2f}_{w1:.2f}_{w2:.2f}"
                ok = m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6 and m["valid_ic"] >= base_m["valid_ic"] - 0.0005
                if ok and (best_w is None or (m["hi31_ic"], m["valid_ic"]) > (best_w[1]["hi31_ic"], best_w[1]["valid_ic"])):
                    best_w = (tag, m, [w0, w1, w2])
        if best_w:
            tag, m, ws = best_w
            print(f"  best weighted PASS: {tag} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f}")
            rows.append((tag, m, core, ws))
        else:
            print("  no weighted combo passed gate")

    # pick best passing row (hi31 first, then valid)
    passing = [
        (tag, m, combo, ws)
        for tag, m, combo, ws in rows
        if m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6 and m["valid_ic"] >= base_m["valid_ic"] - 0.0005
    ]
    if not passing:
        print("\nno combo passed test-sim gate")
        print("VALID_RANKIC", f"{base_m['valid_ic']:.6f}")
        return

    best_tag, best_m, best_combo, best_ws = max(passing, key=lambda x: (x[1]["hi31_ic"], x[1]["valid_ic"]))
    print(f"\nbest PASS: {best_tag} valid={best_m['valid_ic']:.6f} hi31={best_m['hi31_ic']:.6f}")

    if best_m["hi31_ic"] <= base_m["hi31_ic"] + 1e-6 and best_m["valid_ic"] <= base_m["valid_ic"] + 1e-6:
        print("no meaningful lift over TREE_BAG")
        print("VALID_RANKIC", f"{base_m['valid_ic']:.6f}")
        return

    hist_t = _rank_bag(best_combo, "test", mxt, best_ws)
    out_t = _fuse(hist_t, "test", recipe)
    save_submission(out_t, ROOT / "submissions" / "task2_fusion_tree_bag_v2.npy")
    np.save(OUT / "fusion_tree_bag_v2_valid.npy", _fuse(_rank_bag(best_combo, "valid", mx, best_ws), "valid", recipe))

    meta = {
        "name": f"TREE_BAG_V2_{best_tag}",
        **recipe,
        "hist_bag": best_combo,
        **({"hist_bag_weights": best_ws} if best_ws else {}),
        **best_m,
        "baseline_valid": base_m["valid_ic"],
        "baseline_hi31": base_m["hi31_ic"],
        "platform_test_prev": 0.091445,
    }
    (OUT / "fusion_tree_bag_v2_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (OUT / "recipes" / f"TREE_BAG_V2_{best_tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote submissions/task2_fusion_tree_bag_v2.npy")
    print("VALID_RANKIC", f"{best_m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
