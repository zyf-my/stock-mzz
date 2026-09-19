"""Re-search gate w_hi on high-coverage days only (test-like regime). Valid-only."""

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
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
BEST = 0.088799
COV_TAU = 4670.0


def _v(n: str) -> np.ndarray:
    return np.load(OUT / f"{n}_valid.npy")


def _t(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_test.npy")


def _ic_high(pred, y, my, mx, tau: float) -> float:
    nx = np.asarray(mx).sum(axis=1)
    high = nx >= tau
    s = rank_ic_series(pred, y, my)
    s[~high] = np.nan
    return float(np.nanmean(s))


def main() -> None:
    summary = json.loads((OUT / "fusion_best_summary.json").read_text(encoding="utf-8"))
    tp = summary["tree"]
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    nx = np.asarray(mx).sum(axis=1)
    n_high = int((nx >= COV_TAU).sum())
    print(f"valid high-cov days (>={COV_TAU}): {n_high}/{len(nx)}")

    tree = coverage_gate_blend(_v("hist_lgbm_n200"), _v("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    mlp = linear_blend(_v("cs_mlp_only6"), _v("cs_mlp"), 0.4)
    x6, o6, n6 = _v("gru_x6_with_today"), _v("gru_only6_with_today"), _v("gru_next6_with_today")

    print("=== GRU single on high-cov valid days ===")
    for name, arr in [("x6", x6), ("only6", o6), ("next6", n6), ("tree", tree)]:
        print(f"  {name} high-cov IC {_ic_high(arr, y, my, mx, COV_TAU):.6f}  full {_ic_high(arr, y, my, mx, 0):.6f}")

    best = (-1.0, None, None, None)
    for ow in (0.65, 0.7, 0.75, 0.8):
        for nw in (0.30, 0.34, 0.38, 0.42):
            ens = linear_blend(n6, linear_blend(o6, x6, ow), nw)
            for w_hi in (0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.0):
                for w_lo in (0.20, 0.28, 0.35):
                    gated = coverage_gate_blend(ens, tree, mx, COV_TAU, w_lo, w_hi, "rank")
                    for mw in (0.04, 0.06, 0.08):
                        blender = FusionModel({"weight_grid": [mw]})
                        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
                        out = blender.predict(mlp, gated, mx)
                        ic_full = mean_rank_ic(out, y, my)
                        ic_high = _ic_high(out, y, my, mx, COV_TAU)
                        # score: prioritize full valid but require high-cov not to collapse
                        score = 0.6 * ic_full + 0.4 * ic_high
                        if score > best[0]:
                            best = (score, (ow, nw, w_lo, w_hi, mw, ic_full, ic_high), out, None)
                            print(f"  * ow={ow} nw={nw} w_lo={w_lo} w_hi={w_hi} mlp={mw} full={ic_full:.6f} high={ic_high:.6f}")

    ow, nw, w_lo, w_hi, mw, ic_full, ic_high = best[1]
    print(f"\nBEST ow={ow} nw={nw} w_lo={w_lo} w_hi={w_hi} mlp={mw}")
    print(f"  full valid={ic_full:.6f}  high-cov={ic_high:.6f}  prev={BEST:.6f}")

    if ic_full > BEST + 1e-6:
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
        gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], COV_TAU, w_lo, w_hi, "rank")
        mlp_t = linear_blend(np.load(OUT / "cs_mlp_only6_test.npy"), np.load(OUT / "cs_mlp_test.npy"), 0.4)
        blender = FusionModel({"weight_grid": [mw]})
        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
        out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_best.npy")
        summary.update(
            {
                "ic": ic_full,
                "gate": {"tau": COV_TAU, "w_lo": w_lo, "w_hi": w_hi},
                "ens": {"only6_w": ow, "next6_w": nw},
                "mlp_w": mw,
                "high_cov_ic": ic_high,
            }
        )
        (OUT / "fusion_best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("wrote task2_fusion_best.npy")
    print("VALID_RANKIC", f"{ic_full:.6f}")


if __name__ == "__main__":
    main()
