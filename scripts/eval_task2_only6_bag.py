"""Only6 seed bag (42+43) under high31-optimized or best recipe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
COV_TAU = 4670.0


def _load(split: str, name: str) -> np.ndarray:
    sfx = "valid" if split == "valid" else "test"
    return np.load(OUT / f"{name}_{sfx}.npy")


def _ic_high(pred, y, my, mx) -> float:
    nx = np.asarray(mx).sum(axis=1)
    s = rank_ic_series(pred, y, my)
    s[nx < COV_TAU] = np.nan
    return float(np.nanmean(s))


def main() -> None:
    summary = json.loads((OUT / "fusion_best_summary.json").read_text(encoding="utf-8"))
    # use hicov params if present
    gp = summary.get("gate") or {"tau": 4670, "w_lo": 0.28, "w_hi": 0.92}
    ens_p = summary.get("ens") or {"only6_w": 0.7, "next6_w": 0.34}
    mm = summary.get("mlp6_mix", 0.4)
    sw = summary.get("mlp_w", 0.06)
    tp = summary["tree"]

    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]

    seeds = [42]
    if (OUT / "gru_only6_with_today_s43_valid.npy").is_file():
        seeds.append(43)

    o6_list = [_load("valid", "gru_only6_with_today" if s == 42 else f"gru_only6_with_today_s{s}") for s in seeds]
    o6v = np.mean(o6_list, axis=0).astype(np.float32)
    x6v = _load("valid", "gru_x6_with_today")
    n6v = _load("valid", "gru_next6_with_today")
    ow, nw = ens_p["only6_w"], ens_p["next6_w"]

    tree_v = coverage_gate_blend(_load("valid", "hist_lgbm_n200"), _load("valid", "baseline"), mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens_v = linear_blend(n6v, linear_blend(o6v, x6v, ow), nw)
    gated_v = coverage_gate_blend(ens_v, tree_v, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp_v = linear_blend(_load("valid", "cs_mlp_only6"), _load("valid", "cs_mlp"), mm)
    blender = FusionModel({"weight_grid": [sw]})
    blender.locked = {"name": "rank_blend", "weight": sw, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_v, gated_v, mx)

    ic = mean_rank_ic(out_v, y, my)
    ich = _ic_high(out_v, y, my, mx)
    print(f"only6 bag seeds={seeds} full={ic:.6f} hi31={ich:.6f}")

    o6t = np.mean([_load("test", "gru_only6_with_today" if s == 42 else f"gru_only6_with_today_s{s}") for s in seeds], 0).astype(np.float32)
    tree_t = coverage_gate_blend(_load("test", "hist_lgbm_n200"), _load("test", "baseline"), test["mask_x"], tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens_t = linear_blend(_load("test", "gru_next6_with_today"), linear_blend(o6t, _load("test", "gru_x6_with_today"), ow), nw)
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp_t = linear_blend(_load("test", "cs_mlp_only6"), _load("test", "cs_mlp"), mm)
    out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
    save_submission(out_t, ROOT / "submissions" / "task2_fusion_only6_bag.npy")
    print("wrote task2_fusion_only6_bag.npy")
    print("VALID_RANKIC", f"{ic:.6f}", f"HIGH31={ich:.6f}")


if __name__ == "__main__":
    main()
