"""Search final MLP and alpha-disagreement branch weights without retraining."""

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
from src.models.fusion import coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402

OUT = ROOT / "outputs"


def disagreement(a, b, mask):
    ar, br = panel_cs_rank(a, mask), panel_cs_rank(b, mask)
    d = np.zeros(a.shape[0], dtype=float)
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if m.sum() > 2:
            d[t] = np.mean(np.abs(ar[t, m] - br[t, m]))
    return d


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    mxt = test["mask_x"]
    x6v, x6t = np.load(OUT / "gru_x6_with_today_valid.npy"), np.load(OUT / "gru_x6_with_today_test.npy")
    o6v, o6t = np.load(OUT / "gru_only6_with_today_valid.npy"), np.load(OUT / "gru_only6_with_today_test.npy")
    n6v, n6t = np.load(OUT / "gru_next6_with_today_valid.npy"), np.load(OUT / "gru_next6_with_today_test.npy")
    mlpv, mlpt = np.load(OUT / "cs_mlp_valid.npy"), np.load(OUT / "cs_mlp_test.npy")
    mlp6v, mlp6t = np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_only6_test.npy")
    basev, baset = np.load(OUT / "baseline_valid.npy"), np.load(OUT / "baseline_test.npy")
    av, at = np.load(OUT / "hist_lgbm_alpha_x_base_valid.npy"), np.load(OUT / "hist_lgbm_alpha_x_base_test.npy")
    hv, ht = np.load(OUT / "hist_lgbm_alpha_x_hard_valid.npy"), np.load(OUT / "hist_lgbm_alpha_x_hard_test.npy")
    d_v, d_t = disagreement(av, hv, mx), disagreement(at, ht, mxt)
    cut = float(np.quantile(d_v, .75))
    wv = np.where(d_v >= cut, .3, .5)
    wt = np.where(d_t >= cut, .3, .5)
    tv = linear_blend(wv[:, None] * av + (1 - wv[:, None]) * hv, basev, .7)
    tt = linear_blend(wt[:, None] * at + (1 - wt[:, None]) * ht, baset, .7)
    ens_v = linear_blend(n6v, linear_blend(o6v, x6v, .4), .15)
    ens_t = linear_blend(n6t, linear_blend(o6t, x6t, .4), .15)
    gated_v = coverage_gate_blend(ens_v, tv, mx, 4546.0, .25, .6, "raw")
    gated_t = coverage_gate_blend(ens_t, tt, mxt, 4546.0, .25, .6, "raw")
    mlp_rank_v = panel_cs_rank(linear_blend(mlp6v, mlpv, .4), mx)
    mlp_rank_t = panel_cs_rank(linear_blend(mlp6t, mlpt, .4), mxt)
    gate_rank_v = panel_cs_rank(gated_v, mx)
    gate_rank_t = panel_cs_rank(gated_t, mxt)
    branch_v = panel_cs_rank(np.load(OUT / "fusion_x6_only6_mlp6_valid.npy"), mx)
    branch_t = panel_cs_rank(np.load(OUT / "fusion_x6_only6_mlp6_test.npy"), mxt)
    rows = []
    for mw in np.arange(0.0, .101, .01):
        alpha_v = (mw * mlp_rank_v + (1 - mw) * gate_rank_v).astype(np.float32)
        alpha_t = (mw * mlp_rank_t + (1 - mw) * gate_rank_t).astype(np.float32)
        for aw in np.arange(.75, .951, .01):
            out = (aw * panel_cs_rank(alpha_v, mx) + (1 - aw) * branch_v).astype(np.float32)
            ic = mean_rank_ic(out, y, my)
            rows.append((float(ic), float(mw), float(aw)))
    rows.sort(reverse=True)
    print("BEST", rows[:30])
    (OUT / "alpha_final_weights_grid.json").write_text(json.dumps(rows[:100], indent=2), encoding="utf-8")
    best_mw, best_aw = 0.0, 0.86
    alpha_v = (best_mw * mlp_rank_v + (1 - best_mw) * gate_rank_v).astype(np.float32)
    alpha_t = (best_mw * mlp_rank_t + (1 - best_mw) * gate_rank_t).astype(np.float32)
    out_v = (best_aw * panel_cs_rank(alpha_v, mx) + (1 - best_aw) * branch_v).astype(np.float32)
    out_t = (best_aw * panel_cs_rank(alpha_t, mxt) + (1 - best_aw) * branch_t).astype(np.float32)
    np.save(OUT / "fusion_alpha_no_mlp_valid.npy", out_v)
    np.save(OUT / "fusion_alpha_no_mlp_test.npy", out_t)
    from src.submit import save_submission  # noqa: E402
    save_submission(out_t, ROOT / "submissions/task1_fusion_alpha_no_mlp.npy")
    (OUT / "fusion_alpha_no_mlp_lock.json").write_text(
        json.dumps(
            {
                "valid_ic": float(mean_rank_ic(out_v, y, my)),
                "mlp_weight": best_mw,
                "alpha_disagree_final_weight": best_aw,
                "alpha_disagree_gate": {"quantile": 0.75, "ordinary_high": 0.3, "ordinary_other": 0.5},
                "note": "candidate only; platform test not run",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("WROTE", "submissions/task1_fusion_alpha_no_mlp.npy")


if __name__ == "__main__":
    main()
