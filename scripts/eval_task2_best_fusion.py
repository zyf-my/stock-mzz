"""Swap a branch into fusion_best recipe and re-search local weights. Valid-only."""

from __future__ import annotations

import argparse
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


def _load_summary() -> dict:
    return json.loads((OUT / "fusion_best_summary.json").read_text(encoding="utf-8"))


def _v(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_valid.npy")


def _t(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_test.npy")


def _recipe(summary, x6n, o6n, n6n, mlpn, mlp6n, mask, ow, nw, mw, *, split: str = "valid"):
    load = _v if split == "valid" else _t
    tp = summary["tree"]
    gp = summary["gate"]
    tree = coverage_gate_blend(load("hist_lgbm_n200"), load("baseline"), mask, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens = linear_blend(load(n6n), linear_blend(load(o6n), load(x6n), ow), nw)
    gated = coverage_gate_blend(ens, tree, mask, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp = linear_blend(load(mlp6n), load(mlpn), 0.4)
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp, gated, mask)


def search_next6(summary, n6n: str, best_ic: float, tag: str) -> float:
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    ow, nw0 = summary["ens"]["only6_w"], summary["ens"]["next6_w"]
    mw = float(summary["mlp_w"])

    print(f"[{tag}] next6={n6n} single {mean_rank_ic(_v(n6n), y, my):.6f}")
    best = (-1.0, None, None, None)
    for ow in (0.65, 0.7, 0.75):
        for nw in (0.25, 0.30, 0.34, 0.38, 0.42):
            for mw in (0.04, 0.06, 0.08, 0.10):
                out = _recipe(summary, "gru_x6_with_today", "gru_only6_with_today", n6n, "cs_mlp", "cs_mlp_only6", mx, ow, nw, mw)
                ic = mean_rank_ic(out, y, my)
                if ic > best[0]:
                    best = (ic, (ow, nw, mw), out, f"ow={ow} nw={nw} mlp={mw}")
                    print(f"  * {best[3]} ic={ic:.6f}")

    print(f"BEST {best[3]} ic={best[0]:.6f} prev={best_ic:.6f}")
    if best[0] > best_ic + 1e-6:
        ow, nw, mw = best[1]
        out_t = _recipe(summary, "gru_x6_with_today", "gru_only6_with_today", n6n, "cs_mlp", "cs_mlp_only6", test["mask_x"], ow, nw, mw, split="test")
        save_submission(out_t, ROOT / "submissions" / f"task2_fusion_{tag}.npy")
        print(f"wrote task2_fusion_{tag}.npy")
    print("VALID_RANKIC", f"{best[0]:.6f}")
    return float(best[0])


def search_mlp_y2cols(summary, best_ic: float) -> float:
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    mlpn, mlp6n = "cs_mlp_y2cols", "cs_mlp_only6_y2cols"
    print(f"[mlp_y2cols] single mlp={mean_rank_ic(_v(mlpn), y, my):.6f} only6={mean_rank_ic(_v(mlp6n), y, my):.6f}")

    ow, nw = summary["ens"]["only6_w"], summary["ens"]["next6_w"]
    best = (-1.0, None, None, None)
    for mlp_mix in (0.3, 0.4, 0.5):
        for mw in (0.06, 0.10, 0.15, 0.20, 0.25):
            tp = summary["tree"]
            gp = summary["gate"]
            tree = coverage_gate_blend(_v("hist_lgbm_n200"), _v("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
            ens = linear_blend(_v("gru_next6_with_today"), linear_blend(_v("gru_only6_with_today"), _v("gru_x6_with_today"), ow), nw)
            gated = coverage_gate_blend(ens, tree, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
            mlp = linear_blend(_v(mlp6n), _v(mlpn), mlp_mix)
            blender = FusionModel({"weight_grid": [mw]})
            blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
            out = blender.predict(mlp, gated, mx)
            ic = mean_rank_ic(out, y, my)
            if ic > best[0]:
                best = (ic, (mlp_mix, mw), out, f"mix={mlp_mix} mlp_w={mw}")
                print(f"  * {best[3]} ic={ic:.6f}")

    print(f"BEST {best[3]} ic={best[0]:.6f} prev={best_ic:.6f}")
    if best[0] > best_ic + 1e-6:
        mlp_mix, mw = best[1]
        tp, gp = summary["tree"], summary["gate"]
        tree_t = coverage_gate_blend(_t("hist_lgbm_n200"), _t("baseline"), test["mask_x"], tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
        ens_t = linear_blend(_t("gru_next6_with_today"), linear_blend(_t("gru_only6_with_today"), _t("gru_x6_with_today"), ow), nw)
        gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
        mlp_t = linear_blend(_t(mlp6n), _t(mlpn), mlp_mix)
        blender = FusionModel({"weight_grid": [mw]})
        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
        out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_mlp_y2cols.npy")
        print("wrote task2_fusion_mlp_y2cols.npy")
    print("VALID_RANKIC", f"{best[0]:.6f}")
    return float(best[0])


def search_x6_bag(summary, best_ic: float) -> float:
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    x42, x43 = _v("gru_x6_with_today"), _v("gru_x6_with_today_s43")
    bag = 0.5 * (x42 + x43)
    print(f"[x6_bag] s42={mean_rank_ic(x42, y, my):.6f} s43={mean_rank_ic(x43, y, my):.6f} bag={mean_rank_ic(bag, y, my):.6f}")

    ow, nw, mw = summary["ens"]["only6_w"], summary["ens"]["next6_w"], float(summary["mlp_w"])
    tp, gp = summary["tree"], summary["gate"]
    tree = coverage_gate_blend(_v("hist_lgbm_n200"), _v("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens = linear_blend(_v("gru_next6_with_today"), linear_blend(_v("gru_only6_with_today"), bag, ow), nw)
    gated = coverage_gate_blend(ens, tree, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp = linear_blend(_v("cs_mlp_only6"), _v("cs_mlp"), 0.4)
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    out = blender.predict(mlp, gated, mx)
    ic = mean_rank_ic(out, y, my)
    print(f"bag fusion ic={ic:.6f} prev={best_ic:.6f}")
    if ic > best_ic + 1e-6:
        tree_t = coverage_gate_blend(_t("hist_lgbm_n200"), _t("baseline"), test["mask_x"], tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
        bag_t = 0.5 * (_t("gru_x6_with_today") + _t("gru_x6_with_today_s43"))
        ens_t = linear_blend(_t("gru_next6_with_today"), linear_blend(_t("gru_only6_with_today"), bag_t, ow), nw)
        gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
        mlp_t = linear_blend(_t("cs_mlp_only6"), _t("cs_mlp"), 0.4)
        out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_x6_bag.npy")
        print("wrote task2_fusion_x6_bag.npy")
    print("VALID_RANKIC", f"{ic:.6f}")
    return float(ic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("next6_y2left", "mlp_y2cols", "x6_bag"), required=True)
    args = parser.parse_args()
    summary = _load_summary()
    best_ic = float(summary["ic"])
    if args.mode == "next6_y2left":
        search_next6(summary, "gru_next6_y2left", best_ic, "next6_y2left")
    elif args.mode == "mlp_y2cols":
        search_mlp_y2cols(summary, best_ic)
    else:
        search_x6_bag(summary, best_ic)


if __name__ == "__main__":
    main()
