"""3-way rank blend: tree + GRU ens + MLP ens. Global weights only (no day router)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import linear_blend, panel_cs_rank  # noqa: E402

OUT = ROOT / "outputs" / "task2"
GRID = [0.0, 0.15, 0.25, 0.4, 0.5, 0.6, 0.75]


def _q4(pred, y, my, nx):
    s = rank_ic_series(pred, y, my).copy()
    s[nx < float(np.quantile(nx, 0.75))] = np.nan
    return float(np.nanmean(s))


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

    rt, rg, rm = panel_cs_rank(tree, mx), panel_cs_rank(ens, mx), panel_cs_rank(mlp_ens, mx)
    print("3-way rank w_tree / w_gru / w_mlp")
    best = (-1.0, None)
    for wt in GRID:
        for wg in GRID:
            wm = 1.0 - wt - wg
            if wm < -1e-9 or wm > 1.0 + 1e-9:
                continue
            pred = (wt * rt + wg * rg + wm * rm).astype(np.float32)
            ic = mean_rank_ic(pred, y, my)
            if ic > best[0]:
                best = (ic, (wt, wg, wm))
                print(f"  {wt:.2f}/{wg:.2f}/{wm:.2f}  {ic:.6f}  Q4={_q4(pred, y, my, nx):.6f} *")
    print(f"best {best}")

    # 4-way: split only6 out of ens (most complementary to trees)
    ro = panel_cs_rank(o6, mx)
    rx = panel_cs_rank(x6, mx)
    print("4-way tree / x6 / only6 / mlp, equal and a few hand weights")
    for name, w in (
        ("equal", (0.25, 0.25, 0.25, 0.25)),
        ("tree-heavy", (0.4, 0.25, 0.2, 0.15)),
        ("locked-like", (0.51, 0.255, 0.085, 0.15)),  # ~ gate 0.4 gru of which 0.6 x6 + 0.4 only6, mlp 0.15
    ):
        pred = (w[0] * rt + w[1] * rx + w[2] * ro + w[3] * rm).astype(np.float32)
        print(f"  {name:12s} {mean_rank_ic(pred, y, my):.6f}  Q4={_q4(pred, y, my, nx):.6f}")


if __name__ == "__main__":
    main()
