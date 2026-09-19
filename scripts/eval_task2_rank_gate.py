"""y2 GRU vs tree in rank space. Raw blend is a no-op if scales differ."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402

OUT = ROOT / "outputs" / "task2"
WS = [0.0, 0.15, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 1.0]


def _q4(pred, y, my, nx):
    s = rank_ic_series(pred, y, my).copy()
    s[nx < float(np.quantile(nx, 0.75))] = np.nan
    return float(np.nanmean(s))


def _scale(name, a, mx):
    m = np.asarray(mx, dtype=bool)
    v = np.asarray(a)[m]
    v = v[np.isfinite(v)]
    print(f"  {name:20s} mean={v.mean():+.4f} std={v.std():.4f} p50={np.median(v):+.4f}")


def main() -> None:
    splits, _ = load_eval_splits(splits=("valid",), dump_if_missing=False)
    valid = splits["valid"]
    y = split_label_array(valid, "y2")
    my, mx = valid["mask_y"], valid["mask_x"]
    nx = np.asarray(mx).sum(axis=1)

    x6 = np.load(OUT / "gru_x6_with_today_valid.npy")
    o6 = np.load(OUT / "gru_only6_with_today_valid.npy")
    n6 = np.load(OUT / "gru_next6_with_today_valid.npy")
    tree = np.load(OUT / "fusion_valid.npy")
    mlp = np.load(OUT / "cs_mlp_valid.npy")
    mlp6 = np.load(OUT / "cs_mlp_only6_valid.npy")
    ens = linear_blend(n6, linear_blend(o6, x6, 0.4), 0.15)
    mlp_ens = linear_blend(mlp6, mlp, 0.4)

    print("raw scales on mask_x")
    for name, arr in (("x6", x6), ("only6", o6), ("next6", n6), ("ens", ens), ("tree", tree), ("mlp", mlp)):
        _scale(name, arr, mx)

    print("constant RANK blend ens vs tree (test proxy)")
    blender = FusionModel({"weight_grid": [0.5]})
    best = (-1.0, None)
    for w in WS:
        blender.locked = {"name": "rank_blend", "weight": w, "space": "rank", "ic": float("nan")}
        pred = blender.predict(ens, tree, mx)
        ic = mean_rank_ic(pred, y, my)
        q4 = _q4(pred, y, my, nx)
        if ic > best[0]:
            best = (ic, w)
        print(f"  w_gru={w:.2f}  valid={ic:.6f}  Q4={q4:.6f}")
    print(f"best rank constant {best}")

    print("coverage RANK gate (report Q4; do not pick tau)")
    for tau in (4491.0, 4546.0, 4614.0):
        for w_lo in (0.25, 0.4, 0.6):
            for w_hi in (0.25, 0.4, 0.6, 0.75):
                pred = coverage_gate_blend(ens, tree, mx, tau, w_lo, w_hi, "rank")
                ic = mean_rank_ic(pred, y, my)
                if ic >= 0.082:
                    print(f"  tau={tau:.0f} lo={w_lo} hi={w_hi}  {ic:.6f}  Q4={_q4(pred, y, my, nx):.6f}")

    print("best rank constant + mlp stack")
    blender.locked = {"name": "rank_blend", "weight": best[1], "space": "rank", "ic": float("nan")}
    gated = blender.predict(ens, tree, mx)
    for mw in (0.0, 0.15, 0.25, 0.4, 0.5):
        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
        out = blender.predict(mlp_ens, gated, mx)
        print(
            f"  mlp_w={mw:.2f}  valid={mean_rank_ic(out, y, my):.6f}  "
            f"Q4={_q4(out, y, my, nx):.6f}"
        )

    summary = {"best_rank_w_gru": best[1], "best_rank_ic": best[0]}
    (OUT / "rank_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
