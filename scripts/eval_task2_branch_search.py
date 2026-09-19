"""Search GRU branch variants under locked tree/gate/mlp from fusion_search."""

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
MW = 0.05


def _load_summary() -> dict:
    p = OUT / "fusion_gate_resync_summary.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads((OUT / "fusion_search_summary.json").read_text(encoding="utf-8"))


def _v(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_valid.npy")


def _t(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_test.npy")


def _recipe(summary, x6n, o6n, n6n, valid, test, y, my, mx, ow, nw):
    tp = summary["best_tree"]["params"]
    gp = summary["best_gate"]["params"]
    tree_v = coverage_gate_blend(_v("hist_lgbm_n200"), _v("baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    tree_t = coverage_gate_blend(_t("hist_lgbm_n200"), _t("baseline"), test["mask_x"], tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens_v = linear_blend(_v(n6n), linear_blend(_v(o6n), _v(x6n), ow), nw)
    ens_t = linear_blend(_t(n6n), linear_blend(_t(o6n), _t(x6n), ow), nw)
    gated_v = coverage_gate_blend(ens_v, tree_v, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp_v = linear_blend(_v("cs_mlp_only6"), _v("cs_mlp"), 0.4)
    mlp_t = linear_blend(_t("cs_mlp_only6"), _t("cs_mlp"), 0.4)
    blender = FusionModel({"weight_grid": [MW]})
    blender.locked = {"name": "rank_blend", "weight": MW, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_v, gated_v, mx)
    out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
    return float(mean_rank_ic(out_v, y, my)), out_v, out_t, f"x6={x6n} o6={o6n} n6={n6n} ow={ow} nw={nw}"


def main() -> None:
    summary = _load_summary()
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]

    x6_opts = ["gru_x6_with_today"]
    if (OUT / "gru_x6_d600_valid.npy").is_file():
        x6_opts.append("gru_x6_d600")
    if (OUT / "gru_y2cols_d600_valid.npy").is_file():
        x6_opts.append("gru_y2cols_d600")
    o6_opts = ["gru_only6_with_today"]
    if (OUT / "gru_only6_d600_valid.npy").is_file():
        o6_opts.append("gru_only6_d600")
    n6n = "gru_next6_with_today"

    best = (-1.0, None, None, None)
    for x6n in x6_opts:
        for o6n in o6_opts:
            for ow in (0.25, 0.4, 0.5, 0.6):
                for nw in (0.0, 0.1, 0.15, 0.25):
                    ic, ov, ot, tag = _recipe(summary, x6n, o6n, n6n, valid, test, y, my, mx, ow, nw)
                    if ic > best[0]:
                        best = (ic, ov, ot, tag)
                        print(f"  * {tag}  {ic:.6f}")
                    elif ic >= BEST - 0.0002:
                        print(f"    {tag}  {ic:.6f}")

    print(f"BEST {best[3]} ic={best[0]:.6f} locked={BEST:.6f}")
    if best[0] > BEST - 1e-9:
        np.save(OUT / "fusion_branch_valid.npy", best[1])
        np.save(OUT / "fusion_branch_test.npy", best[2])
        save_submission(best[2], ROOT / "submissions" / "task2_fusion_branch.npy")
        print("wrote task2_fusion_branch.npy")
    print("VALID_RANKIC", f"{best[0]:.6f}")


if __name__ == "__main__":
    main()
