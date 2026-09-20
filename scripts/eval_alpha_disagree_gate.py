"""Use DoubleEnsemble component disagreement as a causal daily confidence signal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import build_sample_features, drop_other_label, load_panel, slice_split  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs"
TREE_W = 0.7
GATE_TAU, GATE_LO, GATE_HI = 4546.0, 0.25, 0.6
MLP6_W, MLP_W = 0.4, 0.15


def component_panel(model: LightGBMBaseline, split: dict, which: str) -> np.ndarray:
    out = np.zeros(split["num_x"].shape[:2], dtype=np.float32)
    for t in range(out.shape[0]):
        mask = np.asarray(split["mask_x"][t], dtype=bool)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        x = build_sample_features(
            split["num_x"][t], split["cat_x"][t], mask, idx, model.cat_indices,
            model.feature_cfg, global_t=int(split["start"]) + t,
            panel_num_x=split.get("panel_num_x"), panel_mask_x=split.get("panel_mask_x"),
        )
        if which == "base":
            p = model.booster.predict(x)
        else:
            p = model.booster2.predict(x[:, model.feat_idx2])
        out[t, idx] = np.asarray(p, dtype=np.float32)
    return out


def daily_disagreement(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d = np.zeros(a.shape[0], dtype=np.float64)
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if m.sum() > 2:
            ar = panel_cs_rank(a[t:t + 1], mask[t:t + 1])[0]
            br = panel_cs_rank(b[t:t + 1], mask[t:t + 1])[0]
            scale = max(float(np.std(ar[m])), 1.0)
            d[t] = float(np.mean(np.abs(ar[m] - br[m])) / scale)
    return d


def build_final(tree, x6, o6, n6, mlp, mlp6, mx):
    ens = linear_blend(n6, linear_blend(o6, x6, 0.4), 0.15)
    gated = coverage_gate_blend(ens, tree, mx, GATE_TAU, GATE_LO, GATE_HI, "raw")
    mlp_ens = linear_blend(mlp6, mlp, MLP6_W)
    fm = FusionModel({"weight_grid": [MLP_W]})
    fm.locked = {"name": "rank_blend", "weight": MLP_W, "space": "rank", "ic": float("nan")}
    return fm.predict(mlp_ens, gated, mx)


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    path = resolve_data_path(cfg, None)
    data = load_panel(str(path))
    drop_other_label(data, "y1")
    valid = slice_split(data, "valid")
    test = slice_split(data, "test")
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    mxt = test["mask_x"]

    model = LightGBMBaseline({}, {}, 42)
    model.load(ROOT / "checkpoints/hist_lgbm_alpha_x.txt")
    model.feature_cfg = json.loads((ROOT / "checkpoints/hist_lgbm_alpha_x.txt.meta.json").read_text(encoding="utf-8"))["feature_cfg"]
    model.cat_indices = json.loads((ROOT / "checkpoints/hist_lgbm_alpha_x.txt.meta.json").read_text(encoding="utf-8"))["cat_indices"]
    a_v = component_panel(model, valid, "base")
    b_v = component_panel(model, valid, "hard")
    a_t = component_panel(model, test, "base")
    b_t = component_panel(model, test, "hard")
    np.save(OUT / "hist_lgbm_alpha_x_base_valid.npy", a_v)
    np.save(OUT / "hist_lgbm_alpha_x_hard_valid.npy", b_v)
    np.save(OUT / "hist_lgbm_alpha_x_base_test.npy", a_t)
    np.save(OUT / "hist_lgbm_alpha_x_hard_test.npy", b_t)

    x6_v, x6_t = np.load(OUT / "gru_x6_with_today_valid.npy"), np.load(OUT / "gru_x6_with_today_test.npy")
    o6_v, o6_t = np.load(OUT / "gru_only6_with_today_valid.npy"), np.load(OUT / "gru_only6_with_today_test.npy")
    n6_v, n6_t = np.load(OUT / "gru_next6_with_today_valid.npy"), np.load(OUT / "gru_next6_with_today_test.npy")
    mlp_v, mlp_t = np.load(OUT / "cs_mlp_valid.npy"), np.load(OUT / "cs_mlp_test.npy")
    mlp6_v, mlp6_t = np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_only6_test.npy")
    base_v, base_t = np.load(OUT / "baseline_valid.npy"), np.load(OUT / "baseline_test.npy")
    dis_v, dis_t = daily_disagreement(a_v, b_v, mx), daily_disagreement(a_t, b_t, mxt)
    print("component disagreement quantiles", np.quantile(dis_v, [0,.25,.5,.75,1]).round(4).tolist())

    rows = []
    for q in [.25, .4, .5, .6, .75]:
        cut = float(np.quantile(dis_v, q))
        for wh in [.3, .4, .5, .6, .7, .8, 1.0]:
            w = np.where(dis_v >= cut, wh, 0.5)
            tree_v = np.empty_like(a_v)
            tree_t = np.empty_like(a_t)
            for t in range(tree_v.shape[0]):
                tree_v[t] = w[t] * a_v[t] + (1-w[t]) * b_v[t]
            # Test must use the frozen valid threshold and weights.
            wt = np.where(dis_t >= cut, wh, 0.5)
            for t in range(tree_t.shape[0]):
                tree_t[t] = wt[t] * a_t[t] + (1-wt[t]) * b_t[t]
            tree_v = linear_blend(tree_v, base_v, TREE_W)
            tree_t = linear_blend(tree_t, base_t, TREE_W)
            out_v = build_final(tree_v, x6_v, o6_v, n6_v, mlp_v, mlp6_v, mx)
            ic = mean_rank_ic(out_v, y, my)
            rows.append((ic, q, wh, cut))
    rows.sort(reverse=True)
    print("best", rows[:10])
    best = rows[0]
    # Only promote if the lift is at least 0.0003 on valid; otherwise preserve main.
    if best[0] > 0.122823 + 0.0003:
        _, q, wh, cut = best
        wv = np.where(dis_v >= cut, wh, 0.5); wt = np.where(dis_t >= cut, wh, 0.5)
        tv = np.where(wv[:, None] == 0.5, 0.5*a_v + 0.5*b_v, wv[:, None]*a_v + (1-wv[:, None])*b_v)
        tt = np.where(wt[:, None] == 0.5, 0.5*a_t + 0.5*b_t, wt[:, None]*a_t + (1-wt[:, None])*b_t)
        tv, tt = linear_blend(tv, base_v, TREE_W), linear_blend(tt, base_t, TREE_W)
        out_v = build_final(tv, x6_v, o6_v, n6_v, mlp_v, mlp6_v, mx)
        out_t = build_final(tt, x6_t, o6_t, n6_t, mlp_t, mlp6_t, mxt)
        np.save(OUT / "fusion_alpha_disagree_valid.npy", out_v)
        np.save(OUT / "fusion_alpha_disagree_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions/task1_fusion_alpha_disagree.npy")
        (OUT / "fusion_alpha_disagree_lock.json").write_text(
            json.dumps(
                {
                    "valid_ic": float(best[0]),
                    "disagreement_quantile": float(q),
                    "disagreement_cut": float(cut),
                    "ordinary_tree_weight_high_disagreement": float(wh),
                    "ordinary_tree_weight_other_days": 0.5,
                    "tree_baseline_weight": TREE_W,
                    "gate": {"tau": GATE_TAU, "w_low": GATE_LO, "w_high": GATE_HI},
                    "note": "candidate only; platform test not run",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("wrote task1_fusion_alpha_disagree.npy")
    print("VALID_RANKIC", f"{best[0]:.6f}")


if __name__ == "__main__":
    main()
