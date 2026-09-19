"""A2 dual-regime hist: blend n200 base + cov4100 expert under TREE_BAG shell."""

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
N_HIST = 200
HIGH_TAU = 4670.0
EXPERT_CFG = "configs/task2/hist_lgbm_cov4000_finetune.yaml"
COV_LO, COV_HI = 4000.0, 4700.0
BASE_RECIPE = OUT / "recipes" / "TREE_BAG_091445.json"


def _load_recipe() -> dict:
    p = BASE_RECIPE if BASE_RECIPE.is_file() else OUT / "recipes" / "OLD_091354.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _expert_pred(split: str, panel: dict) -> np.ndarray:
    cfg = load_config(EXPERT_CFG)
    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(cfg.get("features") or {}),
        seed=int(cfg.get("seed", 42)),
    )
    model.load(ROOT / (cfg.get("paths") or {})["checkpoint"])
    # finetune adds rounds on top of init checkpoint — use all trees, not n200
    return model.predict_panel(panel, num_iteration=None)


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


def _bag_from_dual(recipe: dict, split: str, mx: np.ndarray, panel: dict, *, w_cap: float, extrap: bool) -> np.ndarray:
    base_n200 = np.load(OUT / f"hist_lgbm_n200_{split}.npy")
    expert = _expert_pred(split, panel)
    hist_dual = dual_regime_blend(base_n200, expert, mx, COV_LO, COV_HI, extrapolate=extrap, w_cap=w_cap)
    extra = [s for s in (recipe.get("hist_bag") or []) if s != "hist_lgbm_n200"]
    ranks = [panel_cs_rank(hist_dual, mx)] + [panel_cs_rank(np.load(OUT / f"{s}_{split}.npy"), mx) for s in extra]
    return np.mean(ranks, axis=0).astype(np.float32)


def _metrics(pred, y, my, mx) -> dict:
    ic = mean_rank_ic(pred, y, my)
    s = rank_ic_series(pred, y, my)
    hi = np.asarray(mx).sum(axis=1) >= HIGH_TAU
    s_hi = s.copy()
    s_hi[~hi] = np.nan
    return {
        "valid_ic": float(ic),
        "hi31_ic": float(np.nanmean(s_hi)),
        "testlike_ic": float(np.nanmean(s_hi)),
        "neg_days": int(np.nansum(s < 0)),
    }


def main() -> None:
    recipe = _load_recipe()
    cfg = load_config("configs/task2/hist_lgbm.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y2")
    valid_p, test_p = slice_split(data, "valid"), slice_split(data, "test")
    y, my, mx = split_label_array(valid_p, "y2"), valid_p["mask_y"], valid_p["mask_x"]

    extra = [s for s in recipe.get("hist_bag", []) if s != "hist_lgbm_n200"]
    ranks = [panel_cs_rank(np.load(OUT / f"hist_lgbm_n200_valid.npy"), mx)] + [
        panel_cs_rank(np.load(OUT / f"{s}_valid.npy"), mx) for s in extra
    ]
    hist_base = np.mean(ranks, axis=0).astype(np.float32)
    base_m = _metrics(_fuse(recipe, "valid", mx, hist_base), y, my, mx)
    print(f"TREE_BAG base valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f}")

    rows: list[tuple[str, dict, dict]] = []
    for w_cap in (1.0, 1.25, 1.5):
        for extrap in (False, True):
            tag = f"dual_wcap{w_cap}_x{int(extrap)}"
            hist = _bag_from_dual(recipe, "valid", mx, valid_p, w_cap=w_cap, extrap=extrap)
            m = _metrics(_fuse(recipe, "valid", mx, hist), y, my, mx)
            rows.append((tag, m, {"w_cap": w_cap, "extrapolate": extrap}))
            print(f"  {tag:24s} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f}")

    passing = [
        (tag, m, meta)
        for tag, m, meta in rows
        if m["valid_ic"] >= base_m["valid_ic"] - 0.0005 and m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6
    ]
    if passing:
        best_tag, best_m, best_meta = max(passing, key=lambda x: (x[1]["valid_ic"], x[1]["hi31_ic"]))
    else:
        best_tag, best_m, best_meta = max(rows, key=lambda x: (x[1]["hi31_ic"], x[1]["valid_ic"]))
    ok = bool(passing)
    print(f"\nbest: {best_tag} valid={best_m['valid_ic']:.6f} hi31={best_m['hi31_ic']:.6f} gate={'PASS' if ok else 'FAIL'}")

    if ok:
        mxt = test_p["mask_x"]
        hist_t = _bag_from_dual(recipe, "test", mxt, test_p, w_cap=best_meta["w_cap"], extrap=best_meta["extrapolate"])
        out_t = _fuse(recipe, "test", mxt, hist_t)
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_dual_regime.npy")
        hist_v = _bag_from_dual(recipe, "valid", mx, valid_p, w_cap=best_meta["w_cap"], extrap=best_meta["extrapolate"])
        np.save(OUT / "fusion_dual_regime_valid.npy", _fuse(recipe, "valid", mx, hist_v))
        meta = {
            **recipe,
            "name": f"DUAL_REGIME_{best_tag}",
            "dual_regime": {"cov_lo": COV_LO, "cov_hi": COV_HI, **best_meta, "expert": "hist_lgbm_cov4000_finetune"},
            **best_m,
            "baseline_valid": base_m["valid_ic"],
            "baseline_hi31": base_m["hi31_ic"],
            "platform_test_prev": 0.091445,
        }
        (OUT / "fusion_dual_regime_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (OUT / "recipes" / f"dual_regime_{best_tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("wrote submissions/task2_fusion_dual_regime.npy")

    print("VALID_RANKIC", f"{best_m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
