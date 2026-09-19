"""3-seed x6 bag (42+43+44) in locked fusion_best recipe. Valid-only."""

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
BEST = 0.088817


def _v(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_valid.npy")


def _t(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_test.npy")


def _x6_bag(split: str, mx, seeds: list[int]) -> np.ndarray:
    load = _v if split == "valid" else _t
    arrs = [load("gru_x6_with_today") if s == 42 else load(f"gru_x6_with_today_s{s}") for s in seeds]
    return np.mean(arrs, axis=0).astype(np.float32)


def main() -> None:
    summary = json.loads((OUT / "fusion_best_summary.json").read_text(encoding="utf-8"))
    tp, gp = summary["tree"], summary["gate"]
    ow, nw = summary["ens"]["only6_w"], summary["ens"]["next6_w"]
    mm = summary.get("mlp6_mix", 0.3)
    sw = summary["mlp_w"]

    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]

    for seeds in ([42, 43], [42, 43, 44]):
        missing = [s for s in seeds if s != 42 and not (OUT / f"gru_x6_with_today_s{s}_valid.npy").is_file()]
        if missing:
            print(f"skip seeds {seeds}, missing {missing}")
            continue
        x6 = _x6_bag("valid", mx, seeds)
        print(f"seeds {seeds} x6 bag single {mean_rank_ic(x6, y, my):.6f}")

        tree = coverage_gate_blend(_v("hist_lgbm_n200"), _v("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
        ens = linear_blend(_v("gru_next6_with_today"), linear_blend(_v("gru_only6_with_today"), x6, ow), nw)
        gated = coverage_gate_blend(ens, tree, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
        mlp_ens = linear_blend(_v("cs_mlp_only6"), _v("cs_mlp"), mm)
        blender = FusionModel({"weight_grid": [sw]})
        blender.locked = {"name": "rank_blend", "weight": sw, "space": "rank", "ic": float("nan")}
        out = blender.predict(mlp_ens, gated, mx)
        ic = mean_rank_ic(out, y, my)
        print(f"  fusion ic={ic:.6f} prev={BEST:.6f}")

        if ic > BEST + 1e-6 and seeds == [42, 43, 44]:
            tree_t = coverage_gate_blend(_t("hist_lgbm_n200"), _t("baseline"), test["mask_x"], tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
            x6t = _x6_bag("test", test["mask_x"], seeds)
            ens_t = linear_blend(_t("gru_next6_with_today"), linear_blend(_t("gru_only6_with_today"), x6t, ow), nw)
            gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
            mlp_t = linear_blend(_t("cs_mlp_only6"), _t("cs_mlp"), mm)
            out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
            save_submission(out_t, ROOT / "submissions" / "task2_fusion_best.npy")
            summary["ic"] = ic
            summary["x6_bag_seeds"] = seeds
            (OUT / "fusion_best_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print("wrote task2_fusion_best.npy")
        print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
