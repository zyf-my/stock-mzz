"""Search smooth day-level component gates and small prediction blends."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402

OUT = ROOT / "outputs"
TREE_W = 0.7
GATE_TAU, GATE_LO, GATE_HI = 4546.0, 0.25, 0.6
MLP6_W, MLP_W = 0.4, 0.15


def daily_disagreement(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d = np.zeros(a.shape[0], dtype=np.float64)
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if m.sum() > 2:
            ar = panel_cs_rank(a[t:t + 1], mask[t:t + 1])[0]
            br = panel_cs_rank(b[t:t + 1], mask[t:t + 1])[0]
            d[t] = float(np.mean(np.abs(ar[m] - br[m])))
    return d


def final(tree, x6, o6, n6, mlp, mlp6, mx):
    ens = linear_blend(n6, linear_blend(o6, x6, 0.4), 0.15)
    gated = coverage_gate_blend(ens, tree, mx, GATE_TAU, GATE_LO, GATE_HI, "raw")
    mlp_ens = linear_blend(mlp6, mlp, MLP6_W)
    fm = FusionModel({"weight_grid": [MLP_W]})
    fm.locked = {"name": "rank_blend", "weight": MLP_W, "space": "rank", "ic": float("nan")}
    return fm.predict(mlp_ens, gated, mx)


def final_fast(tree, x6, o6, n6, mlp, mlp6, mx, mlp_rank=None):
    """Same locked fusion, avoiding repeated ranking of its fixed MLP branch."""
    ens = linear_blend(n6, linear_blend(o6, x6, 0.4), 0.15)
    gated = coverage_gate_blend(ens, tree, mx, GATE_TAU, GATE_LO, GATE_HI, "raw")
    if mlp_rank is None:
        mlp_ens = linear_blend(mlp6, mlp, MLP6_W)
        mlp_rank = panel_cs_rank(mlp_ens, mx)
    return (MLP_W * mlp_rank + (1.0 - MLP_W) * panel_cs_rank(gated, mx)).astype(np.float32)


def rank_blend(a, b, w, mask):
    ar = panel_cs_rank(a, mask)
    br = panel_cs_rank(b, mask)
    return w * ar + (1.0 - w) * br


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    y, my, mx, mxt = valid["y1"], valid["mask_y"], valid["mask_x"], test["mask_x"]
    av, bv = np.load(OUT / "hist_lgbm_alpha_x_base_valid.npy"), np.load(OUT / "hist_lgbm_alpha_x_hard_valid.npy")
    at, bt = np.load(OUT / "hist_lgbm_alpha_x_base_test.npy"), np.load(OUT / "hist_lgbm_alpha_x_hard_test.npy")
    x6v, x6t = np.load(OUT / "gru_x6_with_today_valid.npy"), np.load(OUT / "gru_x6_with_today_test.npy")
    o6v, o6t = np.load(OUT / "gru_only6_with_today_valid.npy"), np.load(OUT / "gru_only6_with_today_test.npy")
    n6v, n6t = np.load(OUT / "gru_next6_with_today_valid.npy"), np.load(OUT / "gru_next6_with_today_test.npy")
    mlpv, mlpt = np.load(OUT / "cs_mlp_valid.npy"), np.load(OUT / "cs_mlp_test.npy")
    mlp6v, mlp6t = np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_only6_test.npy")
    basev, baset = np.load(OUT / "baseline_valid.npy"), np.load(OUT / "baseline_test.npy")
    disv, dist = daily_disagreement(av, bv, mx), daily_disagreement(at, bt, mxt)
    mlp_rank = panel_cs_rank(linear_blend(mlp6v, mlpv, MLP6_W), mx)

    rows = []
    # Smooth gates: hard component gets more weight as disagreement rises.
    z = (disv - np.median(disv)) / max(float(np.quantile(disv, .9) - np.quantile(disv, .1)), 1e-9)
    zt = (dist - np.median(disv)) / max(float(np.quantile(disv, .9) - np.quantile(disv, .1)), 1e-9)
    for beta in np.linspace(-1.0, 1.0, 21):
        for center in [-0.2, 0.0, 0.2]:
            wv = np.clip(0.5 + beta * (z - center), 0.15, 0.85)
            wt = np.clip(0.5 + beta * (zt - center), 0.15, 0.85)
            tv = wv[:, None] * av + (1 - wv[:, None]) * bv
            tt = wt[:, None] * at + (1 - wt[:, None]) * bt
            ov = final_fast(linear_blend(tv, basev, TREE_W), x6v, o6v, n6v, mlpv, mlp6v, mx, mlp_rank)
            ic = mean_rank_ic(ov, y, my)
            rows.append((float(ic), "smooth", float(beta), float(center)))
    # A smaller piecewise search around the strongest previously found region.
    for q in [.15, .2, .25, .3, .4, .5, .6, .7, .8]:
        cut = float(np.quantile(disv, q))
        for wh in [.2, .3, .4, .5, .6, .7, .8]:
            wv = np.where(disv >= cut, wh, 0.5)
            wt = np.where(dist >= cut, wh, 0.5)
            tv = wv[:, None] * av + (1 - wv[:, None]) * bv
            tt = wt[:, None] * at + (1 - wt[:, None]) * bt
            ov = final_fast(linear_blend(tv, basev, TREE_W), x6v, o6v, n6v, mlpv, mlp6v, mx, mlp_rank)
            ic = mean_rank_ic(ov, y, my)
            rows.append((float(ic), "piecewise", float(q), float(wh)))
    rows.sort(reverse=True)
    print("gate_best", rows[:20])

    # Small rank blends against existing branches, using the locked candidate as A.
    candidate = np.load(OUT / "fusion_alpha_disagree_valid.npy")
    candidate_t = np.load(OUT / "fusion_alpha_disagree_test.npy")
    for name in [
        "fusion_alpha_x_valid.npy", "fusion_next6_wt_mlp6_valid.npy", "fusion_x6_cov_mlp_valid.npy",
        "fusion_wt_mlp6_valid.npy", "fusion_x6_only6_mlp6_valid.npy", "fusion_nfull_cov_mlp_valid.npy",
    ]:
        p = OUT / name
        if not p.exists():
            continue
        other = np.load(p)
        for w in np.linspace(.7, .99, 30):
            out = rank_blend(candidate, other, w, my)
            ic = mean_rank_ic(out, y, my)
            print("branch", name, float(w), float(ic)) if ic > 0.12314 else None

    (OUT / "smooth_gate_grid.json").write_text(json.dumps(rows[:50], indent=2), encoding="utf-8")
    print("BEST_VALID", f"{rows[0][0]:.9f}")


if __name__ == "__main__":
    main()
