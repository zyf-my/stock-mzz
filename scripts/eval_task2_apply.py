"""Apply fine-tuned weights around current best (fast)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"


def main() -> None:
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y2"], valid["mask_y"], valid["mask_x"]
    lt = lambda s: np.load(OUT / f"{s}_valid.npy")
    ltt = lambda s: np.load(OUT / f"{s}_test.npy")

    tree_p = {"tau": 4614.0, "weight_low": 0.45, "weight_high": 1.0}
    ens_p = {"only6_w": 0.7, "next6_w": 0.34}
    best = (-1.0, None, None)

    tree_v = coverage_gate_blend(
        lt("hist_lgbm_n200"), lt("baseline"), mx, tree_p["tau"], tree_p["weight_low"], tree_p["weight_high"], "rank"
    )
    ens_v = linear_blend(
        lt("gru_next6_with_today"),
        linear_blend(lt("gru_only6_with_today"), lt("gru_x6_with_today"), ens_p["only6_w"]),
        ens_p["next6_w"],
    )
    mlp_v = lt("cs_mlp")

    for w_hi in (0.92, 0.94, 0.96, 0.98, 1.0):
        for w_lo in (0.26, 0.28, 0.30):
            gated = coverage_gate_blend(ens_v, tree_v, mx, 4670.0, w_lo, w_hi, "rank")
            for mw in (0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14):
                blender = FusionModel({"weight_grid": [mw]})
                blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
                out = blender.predict(mlp_v, gated, mx)
                ic = mean_rank_ic(out, y, my)
                if ic > best[0]:
                    best = (ic, (w_lo, w_hi, mw), out)

    w_lo, w_hi, mw = best[1]
    print(f"BEST ic={best[0]:.6f} gate=4670/{w_lo}/{w_hi} mlp_w={mw}")

    tree_t = coverage_gate_blend(
        ltt("hist_lgbm_n200"), ltt("baseline"), test["mask_x"], 4614.0, 0.45, 1.0, "rank"
    )
    ens_t = linear_blend(
        ltt("gru_next6_with_today"),
        linear_blend(ltt("gru_only6_with_today"), ltt("gru_x6_with_today"), ens_p["only6_w"]),
        ens_p["next6_w"],
    )
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], 4670.0, w_lo, w_hi, "rank")
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    out_t = blender.predict(ltt("cs_mlp"), gated_t, test["mask_x"])
    save_submission(out_t, ROOT / "submissions" / "task2_fusion_best.npy")

    summary = {
        "ic": float(best[0]),
        "platform_test": 0.091354,
        "tree": {"tau": 4614.0, "w_lo": 0.45, "w_hi": 1.0},
        "gate": {"tau": 4670.0, "w_lo": w_lo, "w_hi": w_hi},
        "ens": ens_p,
        "mlp_w": mw,
        "mlp6_mix": 0.0,
        "x6": "single_s42",
    }
    (OUT / "fusion_best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(OUT / "fusion_best_valid.npy", best[2])
    print("wrote task2_fusion_best.npy")
    print("VALID_RANKIC", f"{best[0]:.6f}")


if __name__ == "__main__":
    main()
