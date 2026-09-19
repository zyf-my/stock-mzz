"""Task1 ceiling + scale + rank-gate probe. No overwrite of main submission."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402

OUT = ROOT / "outputs"
HIGH_TAU = 4670.0
LOCKED = dict(tau=4546.0, w_lo=0.25, w_hi=0.6, only6_w=0.4, next6_w=0.15, mlp6_w=0.4, mlp_w=0.15)


def _load(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_valid.npy")


def _ic(pred, y, my, hi=None) -> float:
    if hi is None:
        return float(mean_rank_ic(pred, y, my))
    return float(mean_rank_ic(pred[hi], y[hi], my[hi]))


def _fuse(ens, tree, mlp, mx, space: str, tau: float, w_lo: float, w_hi: float, mlp_w: float):
    gated = coverage_gate_blend(ens, tree, mx, tau, w_lo, w_hi, space)
    blender = FusionModel({"weight_grid": [mlp_w]})
    blender.locked = {"name": "rank_blend", "weight": mlp_w, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp, gated, mx)


def main() -> None:
    splits, src = load_eval_splits(splits=("valid",), dump_if_missing=False)
    valid = splits["valid"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    cov = np.asarray(mx).sum(axis=1)
    hi = cov >= HIGH_TAU
    print(f"cache={src} days={len(cov)} cov={cov.min():.0f}-{cov.max():.0f} hi31={int(hi.sum())}")

    names = [
        "baseline",
        "hist_lgbm",
        "fusion",
        "gru_x6_with_today",
        "gru_only6_with_today",
        "gru_next6_with_today",
        "cs_mlp",
        "cs_mlp_only6",
        "fusion_x6_today",
    ]
    print("\n== branch IC / scale ==")
    series = []
    for n in names:
        p = OUT / f"{n}_valid.npy"
        if not p.is_file():
            print(f"  missing {n}")
            continue
        a = np.load(p)
        m = np.isfinite(a) & np.asarray(mx, dtype=bool)
        print(
            f"  {n:28s} valid={_ic(a, y, my):.6f} hi31={_ic(a, y, my, hi):.6f} "
            f"mean={float(np.nanmean(a[m])):.3f} std={float(np.nanstd(a[m])):.3f}"
        )
        series.append(rank_ic_series(a, y, my))

    if series:
        oracle = np.nanmax(np.stack(series, axis=0), axis=0)
        print(f"\nday-oracle valid={float(np.nanmean(oracle)):.6f} hi31={float(np.nanmean(oracle[hi])):.6f}")

    x6, o6, n6 = _load("gru_x6_with_today"), _load("gru_only6_with_today"), _load("gru_next6_with_today")
    tree, mlp, mlp6 = _load("fusion"), _load("cs_mlp"), _load("cs_mlp_only6")
    ens_raw = linear_blend(n6, linear_blend(o6, x6, LOCKED["only6_w"]), LOCKED["next6_w"])
    ens_rank = np.mean(
        [panel_cs_rank(n6, mx), panel_cs_rank(linear_blend(o6, x6, LOCKED["only6_w"]), mx)],
        axis=0,
    )
    # proper rank bag of three GRU stems
    ens_rank3 = np.mean([panel_cs_rank(x6, mx), panel_cs_rank(o6, mx), panel_cs_rank(n6, mx)], axis=0).astype(np.float32)
    mlp_ens = linear_blend(mlp6, mlp, LOCKED["mlp6_w"])
    locked = _fuse(ens_raw, tree, mlp_ens, mx, "raw", LOCKED["tau"], LOCKED["w_lo"], LOCKED["w_hi"], LOCKED["mlp_w"])
    print(f"\nlocked raw-gate valid={_ic(locked, y, my):.6f} hi31={_ic(locked, y, my, hi):.6f}")

    print("\n== rank-space gate (weights frozen) ==")
    for space, ens in [("raw", ens_raw), ("rank", ens_raw), ("rank", ens_rank3)]:
        tag = f"{space} ens={'rank3' if ens is ens_rank3 else 'rawmix'}"
        pred = _fuse(ens, tree, mlp_ens, mx, space, LOCKED["tau"], LOCKED["w_lo"], LOCKED["w_hi"], LOCKED["mlp_w"])
        print(f"  {tag:24s} valid={_ic(pred, y, my):.6f} hi31={_ic(pred, y, my, hi):.6f}")

    print("\n== rank-gate w_hi grid (tau/w_lo/mlp frozen) ==")
    best = None
    for w_hi in (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90):
        for ens_name, ens in [("rawmix", ens_raw), ("rank3", ens_rank3)]:
            pred = _fuse(ens, tree, mlp_ens, mx, "rank", LOCKED["tau"], LOCKED["w_lo"], w_hi, LOCKED["mlp_w"])
            v, h = _ic(pred, y, my), _ic(pred, y, my, hi)
            print(f"  rank {ens_name:6s} w_hi={w_hi:.2f}  valid={v:.6f} hi31={h:.6f}")
            if best is None or (h, v) > (best[0], best[1]):
                best = (h, v, ens_name, w_hi)

    print("\n== raw-gate w_hi grid (control) ==")
    for w_hi in (0.50, 0.60, 0.70, 0.80):
        pred = _fuse(ens_raw, tree, mlp_ens, mx, "raw", LOCKED["tau"], LOCKED["w_lo"], w_hi, LOCKED["mlp_w"])
        print(f"  raw  w_hi={w_hi:.2f}  valid={_ic(pred, y, my):.6f} hi31={_ic(pred, y, my, hi):.6f}")

    print("\n== constant GRU weight in rank space (test-like: all high cov) ==")
    blender = FusionModel({"weight_grid": [LOCKED["mlp_w"]]})
    blender.locked = {"name": "rank_blend", "weight": LOCKED["mlp_w"], "space": "rank", "ic": float("nan")}
    for w in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
        mixed = linear_blend(panel_cs_rank(ens_raw, mx), panel_cs_rank(tree, mx), w)
        pred = blender.predict(mlp_ens, mixed, mx)
        print(
            f"  const rank w_gru={w:.2f}  valid={_ic(pred, y, my):.6f} hi31={_ic(pred, y, my, hi):.6f}"
        )

    print("best hi31 among rank-gate grid:", best)
    print("VALID_RANKIC", f"{_ic(locked, y, my):.6f}")


if __name__ == "__main__":
    main()
