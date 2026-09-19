"""Re-search tree/gate/mlp with locked GRU ens (only6=0.6, next6=0.25). Valid-only."""

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
BEST = 0.086952
OW, NW = 0.6, 0.25


def _v(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_valid.npy")


def _t(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_test.npy")


def main() -> None:
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]

    hist_v, base_v = _v("hist_lgbm_n200"), _v("baseline")
    ens_v = linear_blend(_v("gru_next6_with_today"), linear_blend(_v("gru_only6_with_today"), _v("gru_x6_with_today"), OW), NW)
    mlp_v = linear_blend(_v("cs_mlp_only6"), _v("cs_mlp"), 0.4)

    print(f"locked ens ow={OW} nw={NW} ic={mean_rank_ic(ens_v, y, my):.6f}")

    best_tree = (-1.0, None, None)
    for tau in (4347.0, 4491.0, 4546.0, 4614.0, 4700.0):
        for w_lo in (0.25, 0.4, 0.5, 0.6, 0.7):
            for w_hi in (0.5, 0.75, 0.85, 1.0):
                tree = coverage_gate_blend(hist_v, base_v, mx, tau, w_lo, w_hi, "rank")
                ic = mean_rank_ic(tree, y, my)
                if ic > best_tree[0]:
                    best_tree = (ic, (tau, w_lo, w_hi), tree)
    print(f"best tree {best_tree[1]} ic={best_tree[0]:.6f}")

    tree_v = best_tree[2]
    best_gate = (-1.0, None, None)
    for tau in (4491.0, 4546.0, 4614.0, 4700.0):
        for w_lo in (0.1, 0.15, 0.25, 0.35, 0.4):
            for w_hi in (0.4, 0.5, 0.6, 0.75, 0.85):
                gated = coverage_gate_blend(ens_v, tree_v, mx, tau, w_lo, w_hi, "rank")
                ic = mean_rank_ic(gated, y, my)
                if ic > best_gate[0]:
                    best_gate = (ic, (tau, w_lo, w_hi), gated)
    print(f"best gate {best_gate[1]} ic={best_gate[0]:.6f}")

    blender = FusionModel({"weight_grid": [0.08]})
    best_full = (-1.0, None, None)
    for mw in (0.0, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2):
        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
        out = blender.predict(mlp_v, best_gate[2], mx)
        ic = mean_rank_ic(out, y, my)
        if ic > best_full[0]:
            best_full = (ic, mw, out)
    print(f"best mlp_w={best_full[1]} ic={best_full[0]:.6f}  prev={BEST:.6f}")

    summary = {
        "ens": {"only6_w": OW, "next6_w": NW},
        "best_tree": {"ic": best_tree[0], "params": {"tau": best_tree[1][0], "w_lo": best_tree[1][1], "w_hi": best_tree[1][2]}},
        "best_gate": {"ic": best_gate[0], "params": {"tau": best_gate[1][0], "w_lo": best_gate[1][1], "w_hi": best_gate[1][2]}},
        "best_full": {"ic": best_full[0], "mlp_w": best_full[1]},
    }
    (OUT / "fusion_gate_resync_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if best_full[0] > BEST + 1e-6:
        tp, gp = best_tree[1], best_gate[1]
        tree_t = coverage_gate_blend(_t("hist_lgbm_n200"), _t("baseline"), test["mask_x"], *tp, "rank")
        ens_t = linear_blend(
            _t("gru_next6_with_today"),
            linear_blend(_t("gru_only6_with_today"), _t("gru_x6_with_today"), OW),
            NW,
        )
        gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], *gp, "rank")
        mlp_t = linear_blend(_t("cs_mlp_only6"), _t("cs_mlp"), 0.4)
        blender.locked = {"name": "rank_blend", "weight": best_full[1], "space": "rank", "ic": float("nan")}
        out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
        np.save(OUT / "fusion_gate_resync_valid.npy", best_full[2])
        np.save(OUT / "fusion_gate_resync_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_gate_resync.npy")
        print("wrote submissions/task2_fusion_gate_resync.npy")
    print("VALID_RANKIC", f"{best_full[0]:.6f}")


if __name__ == "__main__":
    main()
