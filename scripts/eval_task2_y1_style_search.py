"""Search y1-style GRU/MLP weights under locked task2 tree+gate. Valid-only."""

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
BEST = 0.088799


def _v(n: str) -> np.ndarray:
    return np.load(OUT / f"{n}_valid.npy")


def _build(summary, x6, o6, n6, mlp, mlp6, mx, only6_w, next6_w, mlp6_mix, mlp_stack_w):
    tp, gp = summary["tree"], summary["gate"]
    tree = coverage_gate_blend(_v("hist_lgbm_n200"), _v("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens = linear_blend(n6, linear_blend(o6, x6, only6_w), next6_w)
    gated = coverage_gate_blend(ens, tree, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp_ens = linear_blend(mlp6, mlp, mlp6_mix)
    blender = FusionModel({"weight_grid": [mlp_stack_w]})
    blender.locked = {"name": "rank_blend", "weight": mlp_stack_w, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp_ens, gated, mx), ens


def main() -> None:
    summary = json.loads((OUT / "fusion_best_summary.json").read_text(encoding="utf-8"))
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]

    x6, o6, n6 = _v("gru_x6_with_today"), _v("gru_only6_with_today"), _v("gru_next6_with_today")
    mlp, mlp6 = _v("cs_mlp"), _v("cs_mlp_only6")

    print("=== y1 locked recipe on y2 branches ===")
    out, ens = _build(summary, x6, o6, n6, mlp, mlp6, mx, 0.4, 0.15, 0.4, 0.15)
    print(f"  y1 exact (o6=0.4 n6=0.15 mlp6=0.4 stack=0.15) ens={mean_rank_ic(ens,y,my):.6f} full={mean_rank_ic(out,y,my):.6f}")

    best = (-1.0, None, None, None)
    grids = [
        ("y1-style", [(0.4, 0.15)]),
        ("current-like", [(0.7, 0.34), (0.65, 0.3), (0.75, 0.3)]),
        ("x6-heavy", [(0.4, 0.25), (0.5, 0.25), (0.4, 0.3)]),
    ]
    for tag, ow_nw in grids:
        for only6_w, next6_w in ow_nw:
            for mlp6_mix in (0.3, 0.4, 0.5):
                for mlp_stack_w in (0.04, 0.06, 0.08, 0.10, 0.15):
                    out, _ = _build(summary, x6, o6, n6, mlp, mlp6, mx, only6_w, next6_w, mlp6_mix, mlp_stack_w)
                    ic = mean_rank_ic(out, y, my)
                    if ic > best[0]:
                        best = (ic, (only6_w, next6_w, mlp6_mix, mlp_stack_w), out, tag)
                        print(f"  * [{tag}] o6={only6_w} n6={next6_w} mlp6={mlp6_mix} stk={mlp_stack_w} ic={ic:.6f}")

    ow, nw, mm, sw = best[1]
    print(f"\nBEST o6={ow} n6={nw} mlp6_mix={mm} stack={sw} ic={best[0]:.6f} prev={BEST:.6f}")

    if best[0] > BEST + 1e-6:
        tp, gp = summary["tree"], summary["gate"]
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
        gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
        mlp_t = linear_blend(np.load(OUT / "cs_mlp_only6_test.npy"), np.load(OUT / "cs_mlp_test.npy"), mm)
        blender = FusionModel({"weight_grid": [sw]})
        blender.locked = {"name": "rank_blend", "weight": sw, "space": "rank", "ic": float("nan")}
        out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_best.npy")
        summary.update({"ic": best[0], "ens": {"only6_w": ow, "next6_w": nw}, "mlp_w": sw, "mlp6_mix": mm})
        (OUT / "fusion_best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("wrote task2_fusion_best.npy")
    print("VALID_RANKIC", f"{best[0]:.6f}")


if __name__ == "__main__":
    main()
