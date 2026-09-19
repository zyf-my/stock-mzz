"""Optimize fusion for high-coverage valid days (test proxy). Valid-only."""

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
COV_TAU = 4670.0
BEST_FULL = 0.088799
BEST_HI = 0.139527


def _v(n: str) -> np.ndarray:
    return np.load(OUT / f"{n}_valid.npy")


def _ic_high(pred, y, my, mx, tau: float = COV_TAU) -> float:
    nx = np.asarray(mx).sum(axis=1)
    s = rank_ic_series(pred, y, my)
    s[nx < tau] = np.nan
    return float(np.nanmean(s))


def _recipe(summary, x6, o6, n6, mlp, mlp6, mx, ow, nw, w_lo, w_hi, mm, sw):
    tp, gp = summary["tree"], summary["gate"]
    tree = coverage_gate_blend(_v("hist_lgbm_n200"), _v("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens = linear_blend(n6, linear_blend(o6, x6, ow), nw)
    gated = coverage_gate_blend(ens, tree, mx, COV_TAU, w_lo, w_hi, "rank")
    mlp_ens = linear_blend(mlp6, mlp, mm)
    blender = FusionModel({"weight_grid": [sw]})
    blender.locked = {"name": "rank_blend", "weight": sw, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp_ens, gated, mx)


def main() -> None:
    summary = json.loads((OUT / "fusion_best_summary.json").read_text(encoding="utf-8"))
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]

    x6 = np.load(OUT / "gru_x6_with_today_valid.npy")
    o6 = np.load(OUT / "gru_only6_with_today_valid.npy")
    n6 = np.load(OUT / "gru_next6_with_today_valid.npy")
    mlp = np.load(OUT / "cs_mlp_valid.npy")
    mlp6 = np.load(OUT / "cs_mlp_only6_valid.npy")

    best = (-1.0, None, None, None)
    for ow in (0.65, 0.7, 0.75, 0.8):
        for nw in (0.30, 0.34, 0.38):
            for w_hi in (0.92, 0.95, 0.98, 1.0):
                for w_lo in (0.25, 0.28):
                    for mm in (0.3, 0.4):
                        for sw in (0.04, 0.06):
                            out = _recipe(summary, x6, o6, n6, mlp, mlp6, mx, ow, nw, w_lo, w_hi, mm, sw)
                            ic = mean_rank_ic(out, y, my)
                            ich = _ic_high(out, y, my, mx)
                            # prioritize high31 (test proxy), tie-break full valid
                            score = ich + 0.05 * ic
                            if score > best[0]:
                                best = (score, (ow, nw, w_lo, w_hi, mm, sw, ic, ich), out, None)
                                print(f"  * o6={ow} n6={nw} w_hi={w_hi} mm={mm} sw={sw} full={ic:.6f} hi31={ich:.6f}")

    ow, nw, w_lo, w_hi, mm, sw, ic, ich = best[1]
    print(f"\nBEST full={ic:.6f} hi31={ich:.6f} (prev full={BEST_FULL} hi={BEST_HI})")

    improved = ich > BEST_HI + 1e-6 or (abs(ich - BEST_HI) < 1e-6 and ic > BEST_FULL + 1e-6)
    if improved:
        tp = summary["tree"]
        lv, lt = lambda n: np.load(OUT / f"{n}_valid.npy"), lambda n: np.load(OUT / f"{n}_test.npy")
        tree_t = coverage_gate_blend(lt("hist_lgbm_n200"), lt("baseline"), test["mask_x"], tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
        ens_t = linear_blend(lt("gru_next6_with_today"), linear_blend(lt("gru_only6_with_today"), np.load(OUT / "gru_x6_with_today_test.npy"), ow), nw)
        gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], COV_TAU, w_lo, w_hi, "rank")
        mlp_t = linear_blend(lt("cs_mlp_only6"), lt("cs_mlp"), mm)
        blender = FusionModel({"weight_grid": [sw]})
        blender.locked = {"name": "rank_blend", "weight": sw, "space": "rank", "ic": float("nan")}
        out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_hicov.npy")
        summary.update({
            "ic": ic, "high_cov_ic": ich,
            "ens": {"only6_w": ow, "next6_w": nw},
            "gate": {"tau": COV_TAU, "w_lo": w_lo, "w_hi": w_hi},
            "mlp_w": sw, "mlp6_mix": mm,
        })
        (OUT / "fusion_best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("wrote task2_fusion_hicov.npy (high31-optimized candidate)")
    print("VALID_RANKIC", f"{ic:.6f}", f"HIGH31={ich:.6f}")


if __name__ == "__main__":
    main()
