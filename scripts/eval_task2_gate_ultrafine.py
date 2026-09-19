"""Ultra-fine gate + mlp around 0.088516 peak."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
BEST = 0.088516
OW, NW = 0.6, 0.25


def _v(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_valid.npy")


def _t(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_test.npy")


def main() -> None:
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]

    tree_v = coverage_gate_blend(_v("hist_lgbm_n200"), _v("baseline"), mx, 4614.0, 0.5, 1.0, "rank")
    ens_v = linear_blend(_v("gru_next6_with_today"), linear_blend(_v("gru_only6_with_today"), _v("gru_x6_with_today"), OW), NW)
    mlp_v = linear_blend(_v("cs_mlp_only6"), _v("cs_mlp"), 0.4)

    best = (-1.0, None, None, None)
    for tau in (4660.0, 4670.0, 4680.0, 4690.0, 4700.0):
        for w_lo in (0.28, 0.30, 0.32):
            for w_hi in (0.86, 0.87, 0.88, 0.89, 0.90):
                gated = coverage_gate_blend(ens_v, tree_v, mx, tau, w_lo, w_hi, "rank")
                for mw in (0.04, 0.05, 0.06, 0.07):
                    blender = FusionModel({"weight_grid": [mw]})
                    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
                    out = blender.predict(mlp_v, gated, mx)
                    ic = mean_rank_ic(out, y, my)
                    if ic > best[0]:
                        best = (ic, (tau, w_lo, w_hi, mw), out, None)

    tau, w_lo, w_hi, mw = best[1]
    print(f"BEST tau={tau} w_lo={w_lo} w_hi={w_hi} mlp={mw} ic={best[0]:.6f} prev={BEST:.6f}")

    summary = {
        "ens": {"only6_w": OW, "next6_w": NW},
        "tree": {"tau": 4614.0, "w_lo": 0.5, "w_hi": 1.0},
        "gate": {"tau": tau, "w_lo": w_lo, "w_hi": w_hi},
        "mlp_w": mw,
        "ic": best[0],
    }
    (OUT / "fusion_best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if best[0] >= BEST - 1e-9:
        tree_t = coverage_gate_blend(_t("hist_lgbm_n200"), _t("baseline"), test["mask_x"], 4614.0, 0.5, 1.0, "rank")
        ens_t = linear_blend(_t("gru_next6_with_today"), linear_blend(_t("gru_only6_with_today"), _t("gru_x6_with_today"), OW), NW)
        gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], tau, w_lo, w_hi, "rank")
        mlp_t = linear_blend(_t("cs_mlp_only6"), _t("cs_mlp"), 0.4)
        blender = FusionModel({"weight_grid": [mw]})
        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
        out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_best.npy")
        print("wrote task2_fusion_best.npy")
    print("VALID_RANKIC", f"{best[0]:.6f}")


if __name__ == "__main__":
    main()
