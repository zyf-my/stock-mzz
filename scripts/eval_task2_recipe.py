"""Build task2 fusion from recipe JSON; report valid / hi31 / test-sim.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_task2_recipe.py --recipe outputs/task2/recipes/OLD_091354.json
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_task2_recipe.py --recipe outputs/task2/recipes/hicov.json --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
HIGH_TAU = 4670.0


def _load_recipe(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mlp_panel(recipe: dict, split: str) -> np.ndarray:
    lt = lambda s: np.load(OUT / f"{s}_{split}.npy")
    mm = float(recipe.get("mlp6_mix", 0.0))
    if mm <= 0.0:
        return lt(recipe.get("mlp", "cs_mlp"))
    return linear_blend(lt("cs_mlp_only6"), lt(recipe.get("mlp", "cs_mlp")), mm)


def build(recipe: dict, split: str) -> np.ndarray:
    lt = lambda s: np.load(OUT / f"{s}_{split}.npy")
    tp, gp = recipe["tree"], recipe["gate"]
    ow, nw = recipe["ens"]["only6_w"], recipe["ens"]["next6_w"]
    x6 = lt(recipe.get("x6", "gru_x6_with_today"))
    o6 = lt(recipe.get("only6", "gru_only6_with_today"))
    n6 = lt(recipe.get("next6", "gru_next6_with_today"))
    splits, _ = load_eval_splits(splits=(split,), dump_if_missing=False)
    mx = splits[split]["mask_x"]
    if recipe.get("hist_bag"):
        from src.models.fusion import panel_cs_rank

        ranks = [panel_cs_rank(lt(s), mx) for s in recipe["hist_bag"]]
        hist = np.mean(ranks, axis=0).astype(np.float32)
    else:
        hist = lt(recipe.get("hist", "hist_lgbm_n200"))
    tree = coverage_gate_blend(
        hist, lt("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank"
    )
    ens = linear_blend(n6, linear_blend(o6, x6, ow), nw)
    gated = coverage_gate_blend(ens, tree, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mw = float(recipe["mlp_w"])
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    return blender.predict(_mlp_panel(recipe, split), gated, mx)


def metrics(recipe: dict) -> dict:
    splits, _ = load_eval_splits(splits=("valid",), dump_if_missing=False)
    valid = splits["valid"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    pred = build(recipe, "valid")
    ic = mean_rank_ic(pred, y, my)
    s = rank_ic_series(pred, y, my)
    hi = np.asarray(mx).sum(axis=1) >= HIGH_TAU
    s_hi = s.copy()
    s_hi[~hi] = np.nan
    return {
        "valid_ic": float(ic),
        "hi31_ic": float(np.nanmean(s_hi)),
        "neg_days": int(np.nansum(s < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--write", action="store_true", help="write task2_fusion_candidate.npy if test-sim ok")
    parser.add_argument("--baseline", default=str(OUT / "recipes" / "OLD_091354.json"))
    args = parser.parse_args()

    recipe = _load_recipe(ROOT / args.recipe if not Path(args.recipe).is_absolute() else Path(args.recipe))
    m = metrics(recipe)
    base = _load_recipe(ROOT / args.baseline)
    base_m = metrics(base) if (ROOT / args.baseline).is_file() else {"hi31_ic": 0.0, "valid_ic": 0.0}

    print(f"recipe={recipe.get('name', args.recipe)}")
    print(f"  valid={m['valid_ic']:.6f}  hi31={m['hi31_ic']:.6f}  neg={m['neg_days']}")
    print(f"  baseline valid={base_m['valid_ic']:.6f}  hi31={base_m['hi31_ic']:.6f}")

    ok = m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6 and m["valid_ic"] >= base_m["valid_ic"] - 0.0005
    print(f"  test-sim gate: {'PASS' if ok else 'FAIL'} (hi31 not down, valid not -0.0005)")

    if args.write and ok:
        out_t = build(recipe, "test")
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_candidate.npy")
        np.save(OUT / "fusion_candidate_valid.npy", build(recipe, "valid"))
        meta = {**recipe, **m, "baseline_hi31": base_m["hi31_ic"]}
        (OUT / "fusion_candidate_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("wrote submissions/task2_fusion_candidate.npy (not fusion_best)")
    elif args.write:
        print("skip write: failed test-sim gate")

    print("VALID_RANKIC", f"{m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
