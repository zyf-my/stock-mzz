"""Simulate test regime: all days use high-cov blend (rank GRU vs tree). Valid-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
BEST = 0.088799
HIGH_TAU = 4670.0


def _ic_subset(pred, y, my, day_mask) -> float:
    s = rank_ic_series(pred, y, my)
    s[~day_mask] = np.nan
    return float(np.nanmean(s))


def main() -> None:
    summary = json.loads((OUT / "fusion_best_summary.json").read_text(encoding="utf-8"))
    tp = summary["tree"]
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    nx = np.asarray(mx).sum(axis=1)
    high = nx >= HIGH_TAU

    tree = coverage_gate_blend(
        np.load(OUT / "hist_lgbm_n200_valid.npy"),
        np.load(OUT / "baseline_valid.npy"),
        mx,
        tp["tau"],
        tp["w_lo"],
        tp["w_hi"],
        "rank",
    )
    x6 = np.load(OUT / "gru_x6_with_today_valid.npy")
    o6 = np.load(OUT / "gru_only6_with_today_valid.npy")
    n6 = np.load(OUT / "gru_next6_with_today_valid.npy")
    mlp = linear_blend(np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_valid.npy"), 0.4)

    tr = panel_cs_rank(tree, mx)
    best = (-1.0, None)
    for ow in (0.7, 0.75, 0.8):
        for nw in (0.34, 0.38, 0.42):
            ens = linear_blend(n6, linear_blend(o6, x6, ow), nw)
            er = panel_cs_rank(ens, mx)
            for gw in (0.90, 0.92, 0.95, 0.98, 1.0):
                gated = linear_blend(er, tr, gw)
                for mw in (0.0, 0.04, 0.06):
                    blender = FusionModel({"weight_grid": [mw]})
                    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
                    out = blender.predict(mlp, gated, mx)
                    ic_all = mean_rank_ic(out, y, my)
                    ic_hi = _ic_subset(out, y, my, high)
                    score = 0.5 * ic_all + 0.5 * ic_hi
                    if score > best[0]:
                        best = (score, (ow, nw, gw, mw, ic_all, ic_hi))
                        print(f"  * ow={ow} nw={nw} gru_w={gw} mlp={mw} all={ic_all:.6f} hi={ic_hi:.6f}")

    ow, nw, gw, mw, ic_all, ic_hi = best[1]
    print(f"BEST test-sim ow={ow} nw={nw} gru_rank_w={gw} mlp={mw} all={ic_all:.6f} hi={ic_hi:.6f} prev={BEST:.6f}")

    if ic_all > BEST + 1e-6:
        tree_t = coverage_gate_blend(
            np.load(OUT / "hist_lgbm_n200_test.npy"),
            np.load(OUT / "baseline_test.npy"),
            test["mask_x"],
            tp["tau"],
            tp["w_lo"],
            tp["w_hi"],
            "rank",
        )
        ens_t = linear_blend(
            np.load(OUT / "gru_next6_with_today_test.npy"),
            linear_blend(
                np.load(OUT / "gru_only6_with_today_test.npy"),
                np.load(OUT / "gru_x6_with_today_test.npy"),
                ow,
            ),
            nw,
        )
        gated_t = linear_blend(panel_cs_rank(ens_t, test["mask_x"]), panel_cs_rank(tree_t, test["mask_x"]), gw)
        mlp_t = linear_blend(np.load(OUT / "cs_mlp_only6_test.npy"), np.load(OUT / "cs_mlp_test.npy"), 0.4)
        blender = FusionModel({"weight_grid": [mw]})
        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
        out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_testsim.npy")
        print("wrote task2_fusion_testsim.npy")
    print("VALID_RANKIC", f"{ic_all:.6f}")


if __name__ == "__main__":
    main()
