"""Fine mlp_w around gate_resync best (0.05). Valid-only."""

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
BEST = 0.087522
OW, NW = 0.6, 0.25


def _v(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_valid.npy")


def _t(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_test.npy")


def main() -> None:
    summary = json.loads((OUT / "fusion_gate_resync_summary.json").read_text(encoding="utf-8"))
    tp = summary["best_tree"]["params"]
    gp = summary["best_gate"]["params"]
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]

    tree_v = coverage_gate_blend(_v("hist_lgbm_n200"), _v("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens_v = linear_blend(_v("gru_next6_with_today"), linear_blend(_v("gru_only6_with_today"), _v("gru_x6_with_today"), OW), NW)
    gated_v = coverage_gate_blend(ens_v, tree_v, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp_v = linear_blend(_v("cs_mlp_only6"), _v("cs_mlp"), 0.4)

    blender = FusionModel({"weight_grid": [0.05]})
    best = (-1.0, None, None)
    for mw in (0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1):
        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
        out = blender.predict(mlp_v, gated_v, mx)
        ic = mean_rank_ic(out, y, my)
        print(f"mlp_w={mw:.2f} ic={ic:.6f}")
        if ic > best[0]:
            best = (ic, mw, out)

    print(f"BEST mlp_w={best[1]} ic={best[0]:.6f} prev={BEST:.6f}")
    if best[0] > BEST + 1e-6:
        tree_t = coverage_gate_blend(_t("hist_lgbm_n200"), _t("baseline"), test["mask_x"], tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
        ens_t = linear_blend(_t("gru_next6_with_today"), linear_blend(_t("gru_only6_with_today"), _t("gru_x6_with_today"), OW), NW)
        gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
        mlp_t = linear_blend(_t("cs_mlp_only6"), _t("cs_mlp"), 0.4)
        blender.locked = {"name": "rank_blend", "weight": best[1], "space": "rank", "ic": float("nan")}
        out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_gate_resync_tune.npy")
        print("wrote task2_fusion_gate_resync_tune.npy")
    print("VALID_RANKIC", f"{best[0]:.6f}")


if __name__ == "__main__":
    main()
