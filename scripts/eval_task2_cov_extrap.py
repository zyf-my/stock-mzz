"""Coverage extrapolation: mkt_rel / covw hist + soft gate under TREE_BAG shell."""

from __future__ import annotations

import json
import sys
from itertools import combinations
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
    coverage_soft_gate_blend,
    linear_blend,
    panel_cs_rank,
)
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
N_HIST = 200
BASE_RECIPE = OUT / "recipes" / "TREE_BAG_091445.json"
NEW_HISTS = ["hist_lgbm_mkt_rel", "hist_lgbm_covw"]
HIGH_TAU = 4670.0


def _load_recipe() -> dict:
    p = BASE_RECIPE if BASE_RECIPE.is_file() else OUT / "recipes" / "OLD_091354.json"
    return json.loads(p.read_text(encoding="utf-8"))


_PANEL: dict[str, dict] = {}


def _split_panel(name: str) -> dict:
    if name not in _PANEL:
        cfg = load_config("configs/task2/hist_lgbm.yaml")
        data = load_panel(str(resolve_data_path(cfg, None)))
        drop_other_label(data, "y2")
        _PANEL["valid"] = slice_split(data, "valid")
        _PANEL["test"] = slice_split(data, "test")
    return _PANEL[name]


def _load_hist(stem: str, split: str) -> np.ndarray:
    if stem not in NEW_HISTS:
        return np.load(OUT / f"{stem}_{split}.npy")
    cfg = load_config(f"configs/task2/{stem}.yaml")
    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(cfg.get("features") or {}),
        seed=int(cfg.get("seed", 42)),
    )
    model.load(ROOT / (cfg.get("paths") or {})["checkpoint"])
    return model.predict_panel(_split_panel(split), num_iteration=N_HIST)


def _rank_bag(stems: list[str], split: str, mx: np.ndarray) -> np.ndarray:
    arrs = [_load_hist(s, split) for s in stems]
    ranks = [panel_cs_rank(a, mx) for a in arrs]
    return np.mean(ranks, axis=0).astype(np.float32)


def _fuse(
    recipe: dict,
    split_data: dict,
    mx: np.ndarray,
    hist: np.ndarray,
    *,
    tree_soft: tuple[float, float, float, float] | None = None,
    gate_soft: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    split = "valid" if mx.shape[0] == 243 else "test"
    lt = lambda s: np.load(OUT / f"{s}_{split}.npy")
    tp, gp = recipe["tree"], recipe["gate"]
    ow, nw, mw = recipe["ens"]["only6_w"], recipe["ens"]["next6_w"], float(recipe["mlp_w"])
    mm = float(recipe.get("mlp6_mix", 0.0))
    base = lt("baseline")
    if tree_soft:
        tree = coverage_soft_gate_blend(hist, base, mx, *tree_soft, "rank", extrapolate=True)
    else:
        tree = coverage_gate_blend(hist, base, mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens = linear_blend(lt(recipe.get("next6", "gru_next6_with_today")), linear_blend(lt(recipe.get("only6", "gru_only6_with_today")), lt(recipe.get("x6", "gru_x6_with_today")), ow), nw)
    if gate_soft:
        gated = coverage_soft_gate_blend(ens, tree, mx, *gate_soft, "rank", extrapolate=True)
    else:
        gated = coverage_gate_blend(ens, tree, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp = lt(recipe.get("mlp", "cs_mlp")) if mm <= 0 else linear_blend(lt("cs_mlp_only6"), lt("cs_mlp"), mm)
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp, gated, mx)


def _metrics(pred: np.ndarray, y, my, mx) -> dict:
    ic = mean_rank_ic(pred, y, my)
    s = rank_ic_series(pred, y, my)
    hi = np.asarray(mx).sum(axis=1) >= HIGH_TAU
    s_hi = s.copy()
    s_hi[~hi] = np.nan
    cov = np.asarray(mx).sum(axis=1)
    testlike = cov >= 4670
    s_tl = s.copy()
    s_tl[~testlike] = np.nan
    return {
        "valid_ic": float(ic),
        "hi31_ic": float(np.nanmean(s_hi)),
        "testlike_ic": float(np.nanmean(s_tl)),
        "neg_days": int(np.nansum(s < 0)),
    }


def main() -> None:
    recipe = _load_recipe()
    splits, _ = __import__("src.dataset", fromlist=["load_eval_splits"]).load_eval_splits(
        splits=("valid", "test"), dump_if_missing=False
    )
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]

    base_bag = list(recipe.get("hist_bag") or ["hist_lgbm_n200"])
    base_hist = _rank_bag(base_bag, "valid", mx)
    base_m = _metrics(_fuse(recipe, valid, mx, base_hist), y, my, mx)
    print(f"TREE_BAG base valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f} testlike={base_m['testlike_ic']:.6f}")

    candidates: list[str] = base_bag + NEW_HISTS
    rows: list[tuple[str, dict]] = []

    for k in (3, 4):
        if k > len(candidates):
            continue
        for combo in combinations(candidates, k):
            if not any(h in combo for h in NEW_HISTS):
                continue
            tag = "+".join(s.replace("hist_lgbm_", "") for s in combo)
            hist = _rank_bag(list(combo), "valid", mx)
            m = _metrics(_fuse(recipe, valid, mx, hist), y, my, mx)
            rows.append((f"bag:{tag}", m))
            print(f"  {tag:40s} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f} tl={m['testlike_ic']:.6f}")

    tp, gp = recipe["tree"], recipe["gate"]
    hist = base_hist
    for label, soft in [
        ("tree_soft", (4200.0, 4724.0, tp["w_lo"], tp["w_hi"])),
        ("gate_soft", None),
    ]:
        if label == "tree_soft":
            m = _metrics(_fuse(recipe, valid, mx, hist, tree_soft=soft), y, my, mx)
        else:
            m = _metrics(
                _fuse(recipe, valid, mx, hist, gate_soft=(4200.0, 4724.0, gp["w_lo"], gp["w_hi"])),
                y,
                my,
                mx,
            )
        rows.append((label, m))
        print(f"  {label:40s} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f} tl={m['testlike_ic']:.6f}")

    passing = [
        (tag, m)
        for tag, m in rows
        if m["valid_ic"] >= base_m["valid_ic"] - 0.0005
        and m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6
        and m["testlike_ic"] >= base_m["testlike_ic"] - 0.001
    ]
    if passing:
        best_tag, best_m = max(passing, key=lambda x: (x[1]["testlike_ic"], x[1]["valid_ic"], x[1]["hi31_ic"]))
    else:
        best_tag, best_m = max(rows, key=lambda x: (x[1]["testlike_ic"], x[1]["valid_ic"]))
    ok = bool(passing)
    print(f"\nbest: {best_tag} valid={best_m['valid_ic']:.6f} hi31={best_m['hi31_ic']:.6f} testlike={best_m['testlike_ic']:.6f}")
    print(f"gate: {'PASS' if ok else 'FAIL'}")

    if ok:
        mxt = test["mask_x"]
        if best_tag.startswith("bag:"):
            parts = best_tag[4:].split("+")
            combo = [f"hist_lgbm_{p}" for p in parts]
            hist_t = _rank_bag(combo, "test", mxt)
            out_t = _fuse(recipe, test, mxt, hist_t)
            meta = {**recipe, "name": f"cov_extrap_{best_tag[4:]}", "hist_bag": combo, **best_m, "platform_test_prev": 0.091445}
        elif best_tag == "tree_soft":
            hist_t = _rank_bag(base_bag, "test", mxt)
            out_t = _fuse(recipe, test, mxt, hist_t, tree_soft=(4200.0, 4724.0, tp["w_lo"], tp["w_hi"]))
            meta = {**recipe, "name": "cov_extrap_tree_soft", "tree_soft": [4200, 4724, tp["w_lo"], tp["w_hi"]], **best_m}
        else:
            hist_t = _rank_bag(base_bag, "test", mxt)
            out_t = _fuse(recipe, test, mxt, hist_t, gate_soft=(4200.0, 4724.0, gp["w_lo"], gp["w_hi"]))
            meta = {**recipe, "name": "cov_extrap_gate_soft", "gate_soft": [4200, 4724, gp["w_lo"], gp["w_hi"]], **best_m}
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_cov_extrap.npy")
        np.save(OUT / "fusion_cov_extrap_valid.npy", _fuse(recipe, valid, mx, hist if best_tag.startswith("bag:") else base_hist))
        (OUT / "fusion_cov_extrap_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("wrote submissions/task2_fusion_cov_extrap.npy")

    print("VALID_RANKIC", f"{best_m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
