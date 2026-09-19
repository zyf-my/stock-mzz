"""Fine mlp_w under branch-search GRU weights ow=0.6 nw=0.25."""

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
OW, NW = 0.6, 0.25
BEST = 0.086952


def main() -> None:
    summary = json.loads((OUT / "fusion_search_summary.json").read_text(encoding="utf-8"))
    tp = summary["best_tree"]["params"]
    gp = summary["best_gate"]["params"]
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    tree_v = coverage_gate_blend(
        np.load(OUT / "hist_lgbm_n200_valid.npy"),
        np.load(OUT / "baseline_valid.npy"),
        mx,
        tp["tau"],
        tp["w_lo"],
        tp["w_hi"],
        "rank",
    )
    tree_t = coverage_gate_blend(
        np.load(OUT / "hist_lgbm_n200_test.npy"),
        np.load(OUT / "baseline_test.npy"),
        test["mask_x"],
        tp["tau"],
        tp["w_lo"],
        tp["w_hi"],
        "rank",
    )
    ens_v = linear_blend(
        np.load(OUT / "gru_next6_with_today_valid.npy"),
        linear_blend(
            np.load(OUT / "gru_only6_with_today_valid.npy"),
            np.load(OUT / "gru_x6_with_today_valid.npy"),
            OW,
        ),
        NW,
    )
    ens_t = linear_blend(
        np.load(OUT / "gru_next6_with_today_test.npy"),
        linear_blend(
            np.load(OUT / "gru_only6_with_today_test.npy"),
            np.load(OUT / "gru_x6_with_today_test.npy"),
            OW,
        ),
        NW,
    )
    gated_v = coverage_gate_blend(ens_v, tree_v, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp_v = linear_blend(np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_valid.npy"), 0.4)
    mlp_t = linear_blend(np.load(OUT / "cs_mlp_only6_test.npy"), np.load(OUT / "cs_mlp_test.npy"), 0.4)
    blender = FusionModel({"weight_grid": [0.08]})
    best = (-1.0, None, None)
    for mw in (0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12):
        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
        out_v = blender.predict(mlp_v, gated_v, mx)
        ic = mean_rank_ic(out_v, y, my)
        print(f"mlp_w={mw:.2f}  {ic:.6f}")
        if ic > best[0]:
            best = (ic, mw, out_v)
    out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
    blender.locked = {"name": "rank_blend", "weight": best[1], "space": "rank", "ic": float("nan")}
    out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
    print(f"best mlp_w={best[1]} ic={best[0]:.6f} locked={BEST:.6f}")
    if best[0] >= BEST - 1e-9:
        np.save(OUT / "fusion_branch_tune_valid.npy", best[2])
        np.save(OUT / "fusion_branch_tune_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_branch_tune.npy")
        print("wrote task2_fusion_branch_tune.npy")
    print("VALID_RANKIC", f"{best[0]:.6f}")


if __name__ == "__main__":
    main()
