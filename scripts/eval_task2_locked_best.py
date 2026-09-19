"""Rebuild task2_fusion_best.npy from fusion_best_summary.json (rank gate + hist@200)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"


def main() -> None:
    summary = json.loads((OUT / "fusion_best_summary.json").read_text(encoding="utf-8"))
    tp = summary["tree"]
    gp = summary["gate"]
    ow, nw = summary["ens"]["only6_w"], summary["ens"]["next6_w"]
    mm = summary.get("mlp6_mix", 0.4)
    sw = summary["mlp_w"]

    splits, _ = load_eval_splits(splits=("test",), dump_if_missing=False)
    test = splits["test"]
    mx = test["mask_x"]
    lt = lambda n: np.load(OUT / f"{n}_test.npy")

    x6 = lt("gru_x6_with_today")
    if summary.get("x6_bag_seeds"):
        seeds = summary["x6_bag_seeds"]
        arrs = [lt("gru_x6_with_today" if s == 42 else f"gru_x6_with_today_s{s}") for s in seeds]
        x6 = np.mean(arrs, axis=0).astype(np.float32)

    o6 = lt("gru_only6_with_today")
    if summary.get("hist_bag"):
        ranks = [panel_cs_rank(lt(s), mx) for s in summary["hist_bag"]]
        hist_t = np.mean(ranks, axis=0).astype(np.float32)
    else:
        hist_t = lt("hist_lgbm_n200")
    tree_t = coverage_gate_blend(hist_t, lt("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens_t = linear_blend(lt("gru_next6_with_today"), linear_blend(o6, x6, ow), nw)
    gated_t = coverage_gate_blend(ens_t, tree_t, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp_t = lt("cs_mlp") if mm <= 0.0 else linear_blend(lt("cs_mlp_only6"), lt("cs_mlp"), mm)
    blender = FusionModel({"weight_grid": [sw]})
    blender.locked = {"name": "rank_blend", "weight": sw, "space": "rank", "ic": float("nan")}
    out_t = blender.predict(mlp_t, gated_t, mx)
    save_submission(out_t, ROOT / "submissions" / "task2_fusion_best.npy")
    print("wrote submissions/task2_fusion_best.npy from fusion_best_summary.json")


if __name__ == "__main__":
    main()
