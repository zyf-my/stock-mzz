"""Bag hist LightGBM checkpoints (rank-average) under OLD_091354 fusion shell."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402

OUT = ROOT / "outputs" / "task2"
HIGH_TAU = 4670.0
OLD = OUT / "recipes" / "OLD_091354.json"

HIST_STEMS = [
    "hist_lgbm_n200",
    "hist_lgbm_n225",
    "hist_lgbm_rankic",
    "hist_lgbm_n1200",
]


def _load_old() -> dict:
    return json.loads(OLD.read_text(encoding="utf-8"))


def _rank_bag(stems: list[str], split: str, mask_x: np.ndarray) -> np.ndarray:
    arrs = [np.load(OUT / f"{s}_{split}.npy") for s in stems]
    ranks = [panel_cs_rank(a, mask_x) for a in arrs]
    return np.mean(ranks, axis=0).astype(np.float32)


def _fusion(hist: np.ndarray, split: str, recipe: dict) -> np.ndarray:
    lt = lambda s: np.load(OUT / f"{s}_{split}.npy")
    splits, _ = load_eval_splits(splits=(split,), dump_if_missing=False)
    mx = splits[split]["mask_x"]
    tp, gp = recipe["tree"], recipe["gate"]
    ow, nw = recipe["ens"]["only6_w"], recipe["ens"]["next6_w"]
    mw = float(recipe["mlp_w"])
    mm = float(recipe.get("mlp6_mix", 0.0))
    tree = coverage_gate_blend(hist, lt("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens = linear_blend(
        lt(recipe.get("next6", "gru_next6_with_today")),
        linear_blend(
            lt(recipe.get("only6", "gru_only6_with_today")),
            lt(recipe.get("x6", "gru_x6_with_today")),
            ow,
        ),
        nw,
    )
    gated = coverage_gate_blend(ens, tree, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp = lt(recipe.get("mlp", "cs_mlp")) if mm <= 0 else linear_blend(lt("cs_mlp_only6"), lt("cs_mlp"), mm)
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp, gated, mx)


def _metrics(pred: np.ndarray, valid: dict) -> dict:
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    ic = mean_rank_ic(pred, y, my)
    s = rank_ic_series(pred, y, my)
    hi = np.asarray(mx).sum(axis=1) >= HIGH_TAU
    s_hi = s.copy()
    s_hi[~hi] = np.nan
    return {"valid_ic": float(ic), "hi31_ic": float(np.nanmean(s_hi)), "neg_days": int(np.nansum(s < 0))}


def main() -> None:
    recipe = _load_old()
    splits, _ = load_eval_splits(splits=("valid",), dump_if_missing=False)
    valid = splits["valid"]
    base_m = _metrics(_fusion(np.load(OUT / "hist_lgbm_n200_valid.npy"), "valid", recipe), valid)
    print(f"OLD hist@200 only: valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f}")

    mx = valid["mask_x"]
    present = [s for s in HIST_STEMS if (OUT / f"{s}_valid.npy").is_file()]
    print(f"hist stems: {present}")

    rows: list[tuple[str, dict]] = []
    for k in (2, 3, len(present)):
        if k > len(present):
            continue
        for combo in combinations(present, k):
            tag = "+".join(s.replace("hist_lgbm_", "") for s in combo)
            hist_v = _rank_bag(list(combo), "valid", mx)
            m = _metrics(_fusion(hist_v, "valid", recipe), valid)
            rows.append((tag, m))
            print(f"  {tag:30s} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f} neg={m['neg_days']}")

    rows.sort(key=lambda x: (x[1]["valid_ic"], x[1]["hi31_ic"]), reverse=True)
    passing = [
        (tag, m)
        for tag, m in rows
        if m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6 and m["valid_ic"] >= base_m["valid_ic"] - 0.0005
    ]
    if passing:
        best_tag, best_m = max(passing, key=lambda x: (x[1]["valid_ic"], x[1]["hi31_ic"]))
    else:
        best_tag, best_m = rows[0]
    ok = bool(passing)
    print(f"\nbest bag: {best_tag} valid={best_m['valid_ic']:.6f} hi31={best_m['hi31_ic']:.6f}")
    print(f"test-sim gate vs OLD: {'PASS' if ok else 'FAIL'}")

    if ok and best_tag != "n200":
        combo = []
        for part in best_tag.split("+"):
            stem = f"hist_lgbm_{part}" if not part.startswith("hist_") else part
            if not stem.startswith("hist_lgbm_"):
                stem = f"hist_lgbm_{part}"
            combo.append(stem)
        hist_t = _rank_bag(combo, "test", load_eval_splits(splits=("test",), dump_if_missing=False)[0]["test"]["mask_x"])
        out_t = _fusion(hist_t, "test", recipe)
        from src.submit import save_submission

        save_submission(out_t, ROOT / "submissions" / "task2_fusion_tree_bag.npy")
        np.save(OUT / "fusion_tree_bag_valid.npy", _fusion(_rank_bag(combo, "valid", mx), "valid", recipe))
        meta = {
            "name": f"tree_bag_{best_tag}",
            **recipe,
            "hist_bag": combo,
            **best_m,
            "baseline_valid": base_m["valid_ic"],
            "baseline_hi31": base_m["hi31_ic"],
        }
        (OUT / "fusion_tree_bag_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        bag_recipe = OUT / "recipes" / f"tree_bag_{best_tag}.json"
        bag_recipe.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"wrote submissions/task2_fusion_tree_bag.npy + {bag_recipe.name}")

    print("VALID_RANKIC", f"{best_m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
