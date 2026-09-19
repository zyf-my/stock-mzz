"""Diagnose why recent-tree fusion valid rose but platform test fell.

Does not train. Does not overwrite submissions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402

TREE_W = 0.7
ONLY6_W = 0.4
NEXT6_W = 0.15
MLP6_W = 0.4
MLP_STACK_W = 0.15
TAU = 4546.0
W_LOW = 0.25
W_HIGH = 0.6


def _stats(name: str, a: np.ndarray, mask: np.ndarray) -> None:
    m = np.asarray(mask, dtype=bool) & np.isfinite(a)
    day_std = []
    for t in range(a.shape[0]):
        mt = m[t]
        if int(mt.sum()) >= 8:
            day_std.append(float(np.std(a[t, mt])))
    print(
        f"  {name:28s} shape={a.shape} mean={float(a[m].mean()):.5f} "
        f"std={float(a[m].std()):.5f} day_std={float(np.mean(day_std)):.5f} "
        f"nan={int((~np.isfinite(a)).sum())} zero={(a == 0).mean():.3f}"
    )


def _ic_on(pred, y, my, day_mask) -> float:
    series = rank_ic_series(pred, y, my)
    out = series.copy()
    out[~np.asarray(day_mask, dtype=bool)] = np.nan
    return float(np.nanmean(out))


def main() -> None:
    splits, src = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    print(f"cache={src}")
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    mxt = test["mask_x"]
    nx_v = np.asarray(mx).sum(axis=1)
    nx_t = np.asarray(mxt).sum(axis=1)
    qs = np.quantile(nx_v, [0.25, 0.5, 0.75])
    print(
        f"valid cov min/p25/p50/p75/max={nx_v.min():.0f}/{qs[0]:.0f}/{qs[1]:.0f}/{qs[2]:.0f}/{nx_v.max():.0f} "
        f"frac>=4546={(nx_v >= TAU).mean():.3f} frac>=4650={(nx_v >= 4650).mean():.3f}"
    )
    print(
        f"test  cov min/p25/p50/p75/max={nx_t.min():.0f}/{np.quantile(nx_t, 0.25):.0f}/"
        f"{np.quantile(nx_t, 0.5):.0f}/{np.quantile(nx_t, 0.75):.0f}/{nx_t.max():.0f} "
        f"frac>=4546={(nx_t >= TAU).mean():.3f} frac>=4650={(nx_t >= 4650).mean():.3f}"
    )

    hist_r_v = np.load(ROOT / "outputs/hist_lgbm_recent_valid.npy")
    hist_r_t = np.load(ROOT / "outputs/hist_lgbm_recent_test.npy")
    hist_o_v = np.load(ROOT / "outputs/hist_lgbm_valid.npy")
    hist_o_t = np.load(ROOT / "outputs/hist_lgbm_test.npy")
    base_v = np.load(ROOT / "outputs/baseline_valid.npy")
    base_t = np.load(ROOT / "outputs/baseline_test.npy")
    old_tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    old_tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    recon_v = linear_blend(hist_o_v, base_v, TREE_W)
    recon_t = linear_blend(hist_o_t, base_t, TREE_W)
    print(f"fusion_valid vs 0.7*hist002+0.3*base maxabs={np.max(np.abs(old_tree_v - recon_v)):.6e}")
    print(f"fusion_test  vs 0.7*hist002+0.3*base maxabs={np.max(np.abs(old_tree_t - recon_t)):.6e}")

    tree_v = linear_blend(hist_r_v, base_v, TREE_W)
    tree_t = linear_blend(hist_r_t, base_t, TREE_W)

    x6_v = np.load(ROOT / "outputs/gru_x6_with_today_valid.npy")
    x6_t = np.load(ROOT / "outputs/gru_x6_with_today_test.npy")
    o6_v = np.load(ROOT / "outputs/gru_only6_with_today_valid.npy")
    o6_t = np.load(ROOT / "outputs/gru_only6_with_today_test.npy")
    n6_v = np.load(ROOT / "outputs/gru_next6_with_today_valid.npy")
    n6_t = np.load(ROOT / "outputs/gru_next6_with_today_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
    mlp6_v = np.load(ROOT / "outputs/cs_mlp_only6_valid.npy")
    mlp6_t = np.load(ROOT / "outputs/cs_mlp_only6_test.npy")
    ens_v = linear_blend(n6_v, linear_blend(o6_v, x6_v, ONLY6_W), NEXT6_W)
    ens_t = linear_blend(n6_t, linear_blend(o6_t, x6_t, ONLY6_W), NEXT6_W)

    print("VALID scale")
    for name, arr in [
        ("hist_002", hist_o_v),
        ("hist_recent", hist_r_v),
        ("baseline", base_v),
        ("tree_002", old_tree_v),
        ("tree_recent", tree_v),
        ("gru_ens", ens_v),
    ]:
        _stats(name, arr, mx)
    print("TEST scale")
    for name, arr in [
        ("hist_002", hist_o_t),
        ("hist_recent", hist_r_t),
        ("baseline", base_t),
        ("tree_002", old_tree_t),
        ("tree_recent", tree_t),
        ("gru_ens", ens_t),
    ]:
        _stats(name, arr, mxt)

    print("VALID RankIC by coverage")
    edges = [0, qs[0], qs[1], qs[2], 1e9]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    buckets = [
        ("all", np.ones(len(nx_v), dtype=bool)),
        ("cov<4546", nx_v < TAU),
        ("cov>=4546", nx_v >= TAU),
        ("cov>=4650", nx_v >= 4650),
    ]
    for i, lab in enumerate(labels):
        buckets.append((lab, (nx_v >= edges[i]) & (nx_v < edges[i + 1])))

    gated_r = coverage_gate_blend(ens_v, tree_v, mx, TAU, W_LOW, W_HIGH, "raw")
    gated_o = coverage_gate_blend(ens_v, old_tree_v, mx, TAU, W_LOW, W_HIGH, "raw")
    blender = FusionModel({"weight_grid": [MLP_STACK_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_STACK_W, "space": "rank", "ic": float("nan")}
    mlp_ens = linear_blend(mlp6_v, mlp_v, MLP6_W)
    fus_r = blender.predict(mlp_ens, gated_r, mx)
    fus_o = blender.predict(mlp_ens, gated_o, mx)
    saved_v = np.load(ROOT / "outputs/fusion_recent_tree_valid.npy")
    saved_t = np.load(ROOT / "outputs/fusion_recent_tree_test.npy")
    sub = np.load(ROOT / "submissions/task1_fusion_recent_tree.npy")
    main_t = np.load(ROOT / "submissions/task1_fusion_x6_today.npy")
    print(f"saved valid vs recomputed maxabs={np.max(np.abs(saved_v - fus_r)):.6e}")
    print(f"submission vs saved test maxabs={np.max(np.abs(sub - saved_t)):.6e}")
    print(f"submission vs main test maxabs={np.max(np.abs(sub - main_t)):.6e}")
    print(f"submission shape={sub.shape} dtype={sub.dtype}")

    rows = [
        ("hist_002", hist_o_v),
        ("hist_recent", hist_r_v),
        ("tree_002", old_tree_v),
        ("tree_recent", tree_v),
        ("gate_002", gated_o),
        ("gate_recent", gated_r),
        ("fusion_002", fus_o),
        ("fusion_recent", fus_r),
    ]
    header = f"{'bucket':<12} n " + " ".join(f"{n:>13}" for n, _ in rows)
    print(header)
    for lab, dm in buckets:
        n = int(dm.sum())
        ics = [f"{_ic_on(arr, y, my, dm):13.6f}" for _, arr in rows]
        print(f"{lab:<12} {n:3d} " + " ".join(ics))

    print("VALID_RANKIC_HIGHCOV4546", f"{_ic_on(fus_r, y, my, nx_v >= TAU):.6f}")
    print("VALID_RANKIC_HIGHCOV4546_002", f"{_ic_on(fus_o, y, my, nx_v >= TAU):.6f}")
    print("VALID_RANKIC_HIGHCOV4650", f"{_ic_on(fus_r, y, my, nx_v >= 4650):.6f}")
    print("VALID_RANKIC_HIGHCOV4650_002", f"{_ic_on(fus_o, y, my, nx_v >= 4650):.6f}")


if __name__ == "__main__":
    main()
