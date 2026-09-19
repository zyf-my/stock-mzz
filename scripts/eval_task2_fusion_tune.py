"""Fast tweaks on cached preds: mlp_w grid, x6 variants."""

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
BEST = 0.086676


def _recipe(x6_v, x6_t, summary, valid, test, y, my, mx, mlp_v, mlp_t, mw):
    tp = summary["best_tree"]["params"]
    gp = summary["best_gate"]["params"]
    ow, nw = summary["best_ens"]["only6_w"], summary["best_ens"]["next6_w"]
    hist_v = np.load(OUT / "hist_lgbm_n200_valid.npy")
    hist_t = np.load(OUT / "hist_lgbm_n200_test.npy")
    base_v, base_t = np.load(OUT / "baseline_valid.npy"), np.load(OUT / "baseline_test.npy")
    tree_v = coverage_gate_blend(hist_v, base_v, mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    tree_t = coverage_gate_blend(hist_t, base_t, test["mask_x"], tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    o6_v = np.load(OUT / "gru_only6_with_today_valid.npy")
    o6_t = np.load(OUT / "gru_only6_with_today_test.npy")
    n6_v = np.load(OUT / "gru_next6_with_today_valid.npy")
    n6_t = np.load(OUT / "gru_next6_with_today_test.npy")
    ens_v = linear_blend(n6_v, linear_blend(o6_v, x6_v, ow), nw)
    ens_t = linear_blend(n6_t, linear_blend(o6_t, x6_t, ow), nw)
    gated_v = coverage_gate_blend(ens_v, tree_v, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_v, gated_v, mx)
    out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
    return float(mean_rank_ic(out_v, y, my)), out_v, out_t


def main() -> None:
    summary = json.loads((OUT / "fusion_search_summary.json").read_text(encoding="utf-8"))
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    x6_v = np.load(OUT / "gru_x6_with_today_valid.npy")
    x6_t = np.load(OUT / "gru_x6_with_today_test.npy")
    mlp_v = linear_blend(np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_valid.npy"), 0.4)
    mlp_t = linear_blend(np.load(OUT / "cs_mlp_only6_test.npy"), np.load(OUT / "cs_mlp_test.npy"), 0.4)

    print("mlp_w fine grid (x6 default)")
    best = (-1.0, None, None, None)
    for mw in [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2]:
        ic, ov, ot = _recipe(x6_v, x6_t, summary, valid, test, y, my, mx, mlp_v, mlp_t, mw)
        print(f"  mlp_w={mw:.2f}  {ic:.6f}")
        if ic > best[0]:
            best = (ic, mw, ov, ot)

    d600_v = OUT / "gru_x6_d600_valid.npy"
    if d600_v.is_file():
        print("blend x6 and d600 in ensemble")
        for alpha in (0.25, 0.5, 0.75):
            mix_v = alpha * x6_v + (1 - alpha) * np.load(d600_v)
            mix_t = alpha * x6_t + (1 - alpha) * np.load(OUT / "gru_x6_d600_test.npy")
            ic, ov, ot = _recipe(mix_v, mix_t, summary, valid, test, y, my, mx, mlp_v, mlp_t, best[1])
            print(f"  alpha_x6={alpha:.2f} mlp_w={best[1]:.2f}  {ic:.6f}")
            if ic > best[0]:
                best = (ic, best[1], ov, ot)

    print(f"best {best[0]:.6f} mlp_w={best[1]}  locked {BEST:.6f}")
    if best[0] > BEST - 1e-9:
        np.save(OUT / "fusion_tune_valid.npy", best[2])
        np.save(OUT / "fusion_tune_test.npy", best[3])
        save_submission(best[3], ROOT / "submissions" / "task2_fusion_tune.npy")
        print("wrote task2_fusion_tune.npy")
    print("VALID_RANKIC", f"{best[0]:.6f}")


if __name__ == "__main__":
    main()
