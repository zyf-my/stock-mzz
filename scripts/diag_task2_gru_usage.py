"""Diagnose GRU vs tree contribution in fusion_best recipe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402

OUT = ROOT / "outputs" / "task2"


def _daily_spearman(a, b, mask):
    vals = []
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(a[t]) & np.isfinite(b[t])
        if int(m.sum()) < 8:
            continue
        r = spearmanr(a[t, m], b[t, m]).statistic
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan")


def _q_ic(pred, y, my, nx, qlo=0.0, qhi=1.0):
    s = rank_ic_series(pred, y, my).copy()
    if qlo > 0 or qhi < 1:
        lo, hi = np.quantile(nx, [qlo, qhi])
        day_m = (nx >= lo) & (nx <= hi if qhi < 1 else nx <= nx.max())
        s[~day_m] = np.nan
    return float(np.nanmean(s))


def main() -> None:
    s = json.loads((OUT / "fusion_best_summary.json").read_text(encoding="utf-8"))
    tp, gp = s["tree"], s["gate"]
    ow, nw, mw = s["ens"]["only6_w"], s["ens"]["next6_w"], s["mlp_w"]

    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    nx = np.asarray(mx).sum(axis=1)

    hist = np.load(OUT / "hist_lgbm_n200_valid.npy")
    base = np.load(OUT / "baseline_valid.npy")
    x6 = np.load(OUT / "gru_x6_with_today_valid.npy")
    o6 = np.load(OUT / "gru_only6_with_today_valid.npy")
    n6 = np.load(OUT / "gru_next6_with_today_valid.npy")
    mlp = np.load(OUT / "cs_mlp_valid.npy")
    mlp6 = np.load(OUT / "cs_mlp_only6_valid.npy")
    best = np.load(ROOT / "submissions" / "task2_fusion_best.npy")
    # best.npy is test - reload valid from components

    tree = coverage_gate_blend(hist, base, mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens = linear_blend(n6, linear_blend(o6, x6, ow), nw)
    gated = coverage_gate_blend(ens, tree, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp_ens = linear_blend(mlp6, mlp, 0.4)
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    fused = blender.predict(mlp_ens, gated, mx)

    print("=== single-branch valid RankIC ===")
    for name, arr in [
        ("hist@200", hist),
        ("baseline", base),
        ("tree_fusion", tree),
        ("gru_x6", x6),
        ("gru_only6", o6),
        ("gru_next6", n6),
        ("gru_ens", ens),
        ("gated(tree+gru)", gated),
        ("mlp_stack", mlp_ens),
        ("final_fused", fused),
    ]:
        print(f"  {name:18s} {mean_rank_ic(arr, y, my):.6f}")

    print("\n=== coverage quartiles (final fused) ===")
    for q, label in [(0, "Q1 low"), (0.25, "Q2"), (0.5, "Q3"), (0.75, "Q4 high")]:
        print(f"  {label:8s} {_q_ic(fused, y, my, nx, q, q + 0.25 if q < 0.75 else 1.0):.6f}")

    high = nx >= gp["tau"]
    low = ~high
    print(f"\n=== gate split tau={gp['tau']} (valid cov {nx.min():.0f}-{nx.max():.0f}) ===")
    print(f"  days low={int(low.sum())} high={int(high.sum())}")
    for name, arr in [("gru_ens", ens), ("tree", tree), ("gated", gated), ("final", fused)]:
        ic_lo = mean_rank_ic(arr[low], y[low], my[low])
        ic_hi = mean_rank_ic(arr[high], y[high], my[high])
        print(f"  {name:10s} low={ic_lo:.6f}  high={ic_hi:.6f}")

    print("\n=== daily Spearman vs final ===")
    for name, arr in [("gru_ens", ens), ("tree", tree), ("gated", gated), ("mlp", mlp_ens)]:
        print(f"  {name:10s} {_daily_spearman(arr, fused, mx):.3f}")

    print("\n=== if GRU removed (pure tree + mlp) ===")
    pure_tree = blender.predict(mlp_ens, tree, mx)
    print(f"  tree+mlp only     {mean_rank_ic(pure_tree, y, my):.6f}")
    pure_gru = blender.predict(mlp_ens, ens, mx)
    print(f"  gru+mlp only      {mean_rank_ic(pure_gru, y, my):.6f}")
    print(f"  current gated     {mean_rank_ic(fused, y, my):.6f}")

    print("\n=== rank-space effective weight check (high-cov days) ===")
    gr = panel_cs_rank(ens, mx)
    tr = panel_cs_rank(tree, mx)
    # implied blend on high days
    implied = gp["w_hi"] * gr + (1 - gp["w_hi"]) * tr
    print(f"  implied gate high {mean_rank_ic(implied, y, my):.6f}  (w_hi={gp['w_hi']})")

    # test split if preds exist
    if (OUT / "fusion_best_valid.npy").is_file():
        pass
    yt, myt, mxt = split_label_array(test, "y2"), test["mask_y"], test["mask_x"]
    nxt = np.asarray(mxt).sum(axis=1)
    hist_t = np.load(OUT / "hist_lgbm_n200_test.npy")
    base_t = np.load(OUT / "baseline_test.npy")
    x6t = np.load(OUT / "gru_x6_with_today_test.npy")
    o6t = np.load(OUT / "gru_only6_with_today_test.npy")
    n6t = np.load(OUT / "gru_next6_with_today_test.npy")
    mlp_t = linear_blend(np.load(OUT / "cs_mlp_only6_test.npy"), np.load(OUT / "cs_mlp_test.npy"), 0.4)
    tree_t = coverage_gate_blend(hist_t, base_t, mxt, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens_t = linear_blend(n6t, linear_blend(o6t, x6t, ow), nw)
    gated_t = coverage_gate_blend(ens_t, tree_t, mxt, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    fused_t = blender.predict(mlp_t, gated_t, mxt)
    print("\n=== test split (rebuilt) ===")
    print(f"  gru_ens  {mean_rank_ic(ens_t, yt, myt):.6f}")
    print(f"  tree     {mean_rank_ic(tree_t, yt, myt):.6f}")
    print(f"  gated    {mean_rank_ic(gated_t, yt, myt):.6f}")
    print(f"  final    {mean_rank_ic(fused_t, yt, myt):.6f}")
    print(f"  submission file vs rebuilt diff {np.nanmax(np.abs(best - fused_t)):.6f}")
    high_t = nxt >= gp["tau"]
    print(f"  test high-cov days {int(high_t.sum())}/{len(high_t)} cov range {nxt.min():.0f}-{nxt.max():.0f}")


if __name__ == "__main__":
    main()
