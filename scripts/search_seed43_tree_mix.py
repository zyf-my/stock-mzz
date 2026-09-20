"""Search a conservative blend of the main and seed=43 alpha-x tree branches."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402

OUT = ROOT / "outputs"


def final(x6, o6, n6, tree, mlp, mlp6, mx):
    ens = linear_blend(n6, linear_blend(o6, x6, 0.4), 0.15)
    gated = coverage_gate_blend(ens, tree, mx, 4546.0, 0.25, 0.6, "raw")
    mlp_ens = linear_blend(mlp6, mlp, 0.4)
    fm = FusionModel({"weight_grid": [0.15]})
    fm.locked = {"name": "rank_blend", "weight": 0.15, "space": "rank", "ic": float("nan")}
    return fm.predict(mlp_ens, gated, mx)


def main() -> None:
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    mxt = test["mask_x"]
    x6v, x6t = np.load(OUT / "gru_x6_with_today_valid.npy"), np.load(OUT / "gru_x6_with_today_test.npy")
    o6v, o6t = np.load(OUT / "gru_only6_with_today_valid.npy"), np.load(OUT / "gru_only6_with_today_test.npy")
    n6v, n6t = np.load(OUT / "gru_next6_with_today_valid.npy"), np.load(OUT / "gru_next6_with_today_test.npy")
    mlpv, mlpt = np.load(OUT / "cs_mlp_valid.npy"), np.load(OUT / "cs_mlp_test.npy")
    mlp6v, mlp6t = np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_only6_test.npy")
    basev, baset = np.load(OUT / "baseline_valid.npy"), np.load(OUT / "baseline_test.npy")
    mainv, maint = np.load(OUT / "hist_lgbm_alpha_x_valid.npy"), np.load(OUT / "hist_lgbm_alpha_x_test.npy")
    s43v, s43t = np.load(OUT / "hist_lgbm_alpha_x_s43_valid.npy"), np.load(OUT / "hist_lgbm_alpha_x_s43_test.npy")
    main_tree_v = linear_blend(mainv, basev, 0.7)
    main_tree_t = linear_blend(maint, baset, 0.7)
    s43_tree_v = linear_blend(s43v, basev, 0.7)
    s43_tree_t = linear_blend(s43t, baset, 0.7)
    rows = []
    for w in np.arange(0.0, 0.501, 0.025):
        tv = linear_blend(main_tree_v, s43_tree_v, 1.0 - w)
        tt = linear_blend(main_tree_t, s43_tree_t, 1.0 - w)
        pv = final(x6v, o6v, n6v, tv, mlpv, mlp6v, mx)
        pt = final(x6t, o6t, n6t, tt, mlpt, mlp6t, mxt)
        s = rank_ic_series(pv, y, my)
        rows.append((float(mean_rank_ic(pv, y, my)), float(np.nanmean(s[-60:])), float(w), pv, pt))
        print("w_s43", w, "valid", rows[-1][0], "last60", rows[-1][1])
    rows.sort(key=lambda x: x[0], reverse=True)
    best = rows[0]
    (OUT / "seed43_tree_mix_grid.json").write_text(
        json.dumps([(r[0], r[1], r[2]) for r in rows], indent=2), encoding="utf-8"
    )
    print("BEST", best[0], "LAST60", best[1], "W_S43", best[2])


if __name__ == "__main__":
    main()
