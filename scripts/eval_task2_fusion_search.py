"""Grid search on hist@200 + existing branches. Valid-only; lock best then write test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_eval_splits, load_panel, slice_split, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
LOCKED = 0.086316
HIST_N = 200


def _load(name: str) -> np.ndarray:
    return np.load(OUT / name)


def _ensure_hist_n200() -> None:
    if (OUT / "hist_lgbm_n200_valid.npy").is_file():
        return
    cfg = load_config("configs/task2/hist_lgbm.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y2")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(cfg.get("features") or {}),
        seed=int(cfg.get("seed", 42)),
    )
    model.load(ROOT / (cfg.get("paths") or {})["checkpoint"])
    np.save(OUT / "hist_lgbm_n200_valid.npy", model.predict_panel(valid, num_iteration=HIST_N))
    np.save(OUT / "hist_lgbm_n200_test.npy", model.predict_panel(test, num_iteration=HIST_N))


def main() -> None:
    _ensure_hist_n200()
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y = split_label_array(valid, "y2")
    my, mx = valid["mask_y"], valid["mask_x"]

    hist_v = _load("hist_lgbm_n200_valid.npy")
    base_v = _load("baseline_valid.npy")
    x6_v = _load("gru_x6_with_today_valid.npy")
    o6_v = _load("gru_only6_with_today_valid.npy")
    n6_v = _load("gru_next6_with_today_valid.npy")
    mlp_v = _load("cs_mlp_valid.npy")
    mlp6_v = _load("cs_mlp_only6_valid.npy")
    d600 = OUT / "gru_x6_d600_valid.npy"
    if d600.is_file():
        x6_alt = np.load(d600)
        print(f"x6_d600 single {mean_rank_ic(x6_alt, y, my):.6f}")

    print("1) tree fusion grid (hist@200 + baseline@400)")
    best_tree = (-1.0, None, None)
    for tau in (4347.0, 4491.0, 4546.0, 4614.0):
        for w_lo in (0.25, 0.5, 0.7):
            for w_hi in (0.5, 0.75, 0.85, 1.0):
                tree = coverage_gate_blend(hist_v, base_v, mx, tau, w_lo, w_hi, "rank")
                ic = mean_rank_ic(tree, y, my)
                if ic > best_tree[0]:
                    best_tree = (ic, (tau, w_lo, w_hi), tree)
    print(f"  best tree {best_tree[1]} ic={best_tree[0]:.6f}")

    print("2) GRU ensemble grid")
    best_ens = (-1.0, None)
    for ow in (0.25, 0.4, 0.5, 0.6):
        for nw in (0.0, 0.1, 0.15, 0.25):
            ens = linear_blend(n6_v, linear_blend(o6_v, x6_v, ow), nw)
            ic = mean_rank_ic(ens, y, my)
            if ic > best_ens[0]:
                best_ens = (ic, (ow, nw))
    print(f"  best ens {best_ens[1]} ic={best_ens[0]:.6f}  locked (0.4,0.15)={mean_rank_ic(linear_blend(n6_v, linear_blend(o6_v, x6_v, 0.4), 0.15), y, my):.6f}")

    ow, nw = best_ens[1]
    ens_v = linear_blend(n6_v, linear_blend(o6_v, x6_v, ow), nw)
    tree_v = best_tree[2]

    print("3) rank gate on best tree + best ens")
    best_gate = (-1.0, None, None)
    for tau in (4491.0, 4546.0, 4614.0):
        for w_lo in (0.15, 0.25, 0.4):
            for w_hi in (0.4, 0.6, 0.75, 0.85):
                gated = coverage_gate_blend(ens_v, tree_v, mx, tau, w_lo, w_hi, "rank")
                ic = mean_rank_ic(gated, y, my)
                if ic > best_gate[0]:
                    best_gate = (ic, (tau, w_lo, w_hi), gated)
    print(f"  best gate {best_gate[1]} ic={best_gate[0]:.6f}")

    print("4) mlp stack weight")
    mlp_ens = linear_blend(mlp6_v, mlp_v, 0.4)
    blender = FusionModel({"weight_grid": [0.15]})
    best_full = (-1.0, None, None)
    for mw in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
        out = blender.predict(mlp_ens, best_gate[2], mx)
        ic = mean_rank_ic(out, y, my)
        if ic > best_full[0]:
            best_full = (ic, mw, out)
    print(f"  best mlp_w={best_full[1]} ic={best_full[0]:.6f}  locked 0.15={mean_rank_ic(blender.predict(mlp_ens, best_gate[2], mx), y, my):.6f}")

    # optional: swap x6 with d600 in best ens weights
    if d600.is_file():
        ens_d = linear_blend(n6_v, linear_blend(o6_v, x6_alt, ow), nw)
        gated_d = coverage_gate_blend(ens_d, tree_v, mx, *best_gate[1], "rank")
        blender.locked = {"name": "rank_blend", "weight": best_full[1], "space": "rank", "ic": float("nan")}
        ic_d = mean_rank_ic(blender.predict(mlp_ens, gated_d, mx), y, my)
        print(f"  with x6_d600 in ens {ic_d:.6f}")

    summary = {
        "locked_ic": LOCKED,
        "best_tree": {"ic": best_tree[0], "params": {"tau": best_tree[1][0], "w_lo": best_tree[1][1], "w_hi": best_tree[1][2]}},
        "best_ens": {"ic": best_ens[0], "only6_w": ow, "next6_w": nw},
        "best_gate": {"ic": best_gate[0], "params": {"tau": best_gate[1][0], "w_lo": best_gate[1][1], "w_hi": best_gate[1][2]}},
        "best_full": {"ic": best_full[0], "mlp_w": best_full[1]},
    }
    (OUT / "fusion_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if best_full[0] > LOCKED + 1e-6:
        tau, w_lo, w_hi = best_gate[1]
        hist_t = _load("hist_lgbm_n200_test.npy")
        base_t = _load("baseline_test.npy")
        tree_t = coverage_gate_blend(hist_t, base_t, test["mask_x"], tau, w_lo, w_hi, "rank")
        x6_t = _load("gru_x6_with_today_test.npy")
        o6_t = _load("gru_only6_with_today_test.npy")
        n6_t = _load("gru_next6_with_today_test.npy")
        ens_t = linear_blend(n6_t, linear_blend(o6_t, x6_t, ow), nw)
        gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], tau, w_lo, w_hi, "rank")
        mlp_t = linear_blend(_load("cs_mlp_only6_test.npy"), _load("cs_mlp_test.npy"), 0.4)
        blender.locked = {"name": "rank_blend", "weight": best_full[1], "space": "rank", "ic": float("nan")}
        out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
        np.save(OUT / "fusion_search_valid.npy", best_full[2])
        np.save(OUT / "fusion_search_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_search.npy")
        print("wrote submissions/task2_fusion_search.npy")
    else:
        print("no lift over hist_n200 main")
    print("VALID_RANKIC", f"{best_full[0]:.6f}")


if __name__ == "__main__":
    main()
