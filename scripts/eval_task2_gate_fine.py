"""Fine gate tau/w around resync best; mlp_w=0.06 fixed."""

from __future__ import annotations

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
BEST = 0.087525
OW, NW, MW = 0.6, 0.25, 0.06


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
    blender = FusionModel({"weight_grid": [MW]})
    blender.locked = {"name": "rank_blend", "weight": MW, "space": "rank", "ic": float("nan")}

    best = (-1.0, None, None)
    for tau in (4620.0, 4650.0, 4680.0, 4700.0, 4720.0, 4750.0, 4780.0):
        for w_lo in (0.30, 0.32, 0.35, 0.38):
            for w_hi in (0.80, 0.82, 0.85, 0.88):
                gated = coverage_gate_blend(ens_v, tree_v, mx, tau, w_lo, w_hi, "rank")
                out = blender.predict(mlp_v, gated, mx)
                ic = mean_rank_ic(out, y, my)
                if ic > best[0]:
                    best = (ic, (tau, w_lo, w_hi), out)
                    print(f"  * tau={tau} w_lo={w_lo} w_hi={w_hi} ic={ic:.6f}")

    print(f"BEST gate {best[1]} ic={best[0]:.6f} prev={BEST:.6f}")
    if best[0] > BEST + 1e-6:
        tau, w_lo, w_hi = best[1]
        tree_t = coverage_gate_blend(_t("hist_lgbm_n200"), _t("baseline"), test["mask_x"], 4614.0, 0.5, 1.0, "rank")
        ens_t = linear_blend(_t("gru_next6_with_today"), linear_blend(_t("gru_only6_with_today"), _t("gru_x6_with_today"), OW), NW)
        gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], tau, w_lo, w_hi, "rank")
        mlp_t = linear_blend(_t("cs_mlp_only6"), _t("cs_mlp"), 0.4)
        out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_best.npy")
        print("wrote task2_fusion_best.npy")
    print("VALID_RANKIC", f"{best[0]:.6f}")


if __name__ == "__main__":
    main()
