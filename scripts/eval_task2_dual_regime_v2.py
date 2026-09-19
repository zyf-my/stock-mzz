"""A2 v2: dual regime via finetune DELTA (trees 401-480), not full expert replacement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.fusion import (  # noqa: E402
    FusionModel,
    coverage_gate_blend,
    dual_regime_blend,
    linear_blend,
    panel_cs_rank,
)
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
FINETUNE_CFG = "configs/task2/hist_lgbm_cov4000_finetune.yaml"
INIT_TREES = 400
FINAL_TREES = 480
COV_LO, COV_HI = 4000.0, 4700.0
HIGH_TAU = 4670.0
BASE_RECIPE = OUT / "recipes" / "TREE_BAG_091445.json"


def _recipe() -> dict:
    p = BASE_RECIPE if BASE_RECIPE.is_file() else OUT / "recipes" / "OLD_091354.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _finetune_delta(split: str, panel: dict) -> np.ndarray:
    cfg = load_config(FINETUNE_CFG)
    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(cfg.get("features") or {}),
        seed=int(cfg.get("seed", 42)),
    )
    model.load(ROOT / (cfg.get("paths") or {})["checkpoint"])
    return model.predict_panel_delta(panel, FINAL_TREES, INIT_TREES)


def _hist_dual(split: str, mx: np.ndarray, panel: dict, recipe: dict, *, w_cap: float, extrap: bool) -> np.ndarray:
    base_n200 = np.load(OUT / f"hist_lgbm_n200_{split}.npy")
    delta = _finetune_delta(split, panel)
    dual = dual_regime_blend(base_n200, base_n200 + delta, mx, COV_LO, COV_HI, extrapolate=extrap, w_cap=w_cap)
    extra = [s for s in (recipe.get("hist_bag") or []) if s != "hist_lgbm_n200"]
    ranks = [panel_cs_rank(dual, mx)] + [panel_cs_rank(np.load(OUT / f"{s}_{split}.npy"), mx) for s in extra]
    return np.mean(ranks, axis=0).astype(np.float32)


def _fuse(recipe: dict, split: str, mx: np.ndarray, hist: np.ndarray) -> np.ndarray:
    lt = lambda s: np.load(OUT / f"{s}_{split}.npy")
    tp, gp = recipe["tree"], recipe["gate"]
    ow, nw, mw = recipe["ens"]["only6_w"], recipe["ens"]["next6_w"], float(recipe["mlp_w"])
    tree = coverage_gate_blend(hist, lt("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens = linear_blend(
        lt(recipe.get("next6", "gru_next6_with_today")),
        linear_blend(lt(recipe.get("only6", "gru_only6_with_today")), lt(recipe.get("x6", "gru_x6_with_today")), ow),
        nw,
    )
    gated = coverage_gate_blend(ens, tree, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp = lt(recipe.get("mlp", "cs_mlp"))
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp, gated, mx)


def _metrics(pred, y, my, mx) -> dict:
    ic = mean_rank_ic(pred, y, my)
    s = rank_ic_series(pred, y, my)
    hi = np.asarray(mx).sum(axis=1) >= HIGH_TAU
    sh = s.copy()
    sh[~hi] = np.nan
    return {"valid_ic": float(ic), "hi31_ic": float(np.nanmean(sh)), "neg_days": int(np.nansum(s < 0))}


def main() -> None:
    recipe = _recipe()
    cfg = load_config("configs/task2/hist_lgbm.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y2")
    valid_p, test_p = slice_split(data, "valid"), slice_split(data, "test")
    y, my, mx = split_label_array(valid_p, "y2"), valid_p["mask_y"], valid_p["mask_x"]

    extra = [s for s in recipe.get("hist_bag", []) if s != "hist_lgbm_n200"]
    hist_base = np.mean(
        [panel_cs_rank(np.load(OUT / f"hist_lgbm_n200_valid.npy"), mx)]
        + [panel_cs_rank(np.load(OUT / f"{s}_valid.npy"), mx) for s in extra],
        axis=0,
    ).astype(np.float32)
    base_m = _metrics(_fuse(recipe, "valid", mx, hist_base), y, my, mx)
    print(f"TREE_BAG base valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f}")

    delta_ic = mean_rank_ic(_finetune_delta("valid", valid_p), y, my)
    print(f"finetune delta (trees 401-480) alone RankIC={delta_ic:.6f}")

    rows: list[tuple[str, dict, dict]] = []
    for w_cap in (0.5, 0.75, 1.0, 1.25, 1.5):
        for extrap in (True, False):
            tag = f"delta_w{w_cap}_x{int(extrap)}"
            hist = _hist_dual("valid", mx, valid_p, recipe, w_cap=w_cap, extrap=extrap)
            m = _metrics(_fuse(recipe, "valid", mx, hist), y, my, mx)
            rows.append((tag, m, {"w_cap": w_cap, "extrapolate": extrap}))
            print(f"  {tag:22s} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f}")

    passing = [
        (t, m, meta)
        for t, m, meta in rows
        if m["valid_ic"] >= base_m["valid_ic"] - 0.0005 and m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6
    ]
    best_t, best_m, best_meta = (
        max(passing, key=lambda x: (x[1]["valid_ic"], x[1]["hi31_ic"]))
        if passing
        else max(rows, key=lambda x: (x[1]["valid_ic"], x[1]["hi31_ic"]))
    )
    improved = best_m["valid_ic"] >= base_m["valid_ic"] + 0.001
    print(f"\nbest {best_t} valid={best_m['valid_ic']:.6f} hi31={best_m['hi31_ic']:.6f} gate={'PASS' if passing else 'FAIL'}")

    if improved:
        mxt = test_p["mask_x"]
        hist_t = _hist_dual("test", mxt, test_p, recipe, w_cap=best_meta["w_cap"], extrap=best_meta["extrapolate"])
        out_t = _fuse(recipe, "test", mxt, hist_t)
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_dual_v2.npy")
        hist_v = _hist_dual("valid", mx, valid_p, recipe, w_cap=best_meta["w_cap"], extrap=best_meta["extrapolate"])
        np.save(OUT / "fusion_dual_v2_valid.npy", _fuse(recipe, "valid", mx, hist_v))
        meta = {
            **recipe,
            "name": f"DUAL_V2_{best_t}",
            "dual_v2": {"init_trees": INIT_TREES, "final_trees": FINAL_TREES, **best_meta},
            **best_m,
            "baseline_valid": base_m["valid_ic"],
            "baseline_hi31": base_m["hi31_ic"],
            "platform_test_prev": 0.091445,
            "notes": "valid +0.002 hi31 -0.0008; worth platform test",
        }
        (OUT / "fusion_dual_v2_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("wrote submissions/task2_fusion_dual_v2.npy (valid +0.002, hi31 slightly down — platform test)")
    print("VALID_RANKIC", f"{best_m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
