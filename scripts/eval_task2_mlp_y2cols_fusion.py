"""Swap cs_mlp to y2 Top21 inside the rank-gate recipe. Does not overwrite mains."""

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
TAU = 4546.0
W_LOW = 0.25
W_HIGH = 0.6


def _q4(pred, y, my, nx):
    s = rank_ic_series(pred, y, my).copy()
    s[nx < float(np.quantile(nx, 0.75))] = np.nan
    return float(np.nanmean(s))


def main() -> None:
    splits, src = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y = split_label_array(valid, "y2")
    my, mx = valid["mask_y"], valid["mask_x"]
    nx = np.asarray(mx).sum(axis=1)
    print(f"cache={src}")

    x6_v = np.load(OUT / "gru_x6_with_today_valid.npy")
    x6_t = np.load(OUT / "gru_x6_with_today_test.npy")
    o6_v = np.load(OUT / "gru_only6_with_today_valid.npy")
    o6_t = np.load(OUT / "gru_only6_with_today_test.npy")
    n6_v = np.load(OUT / "gru_next6_with_today_valid.npy")
    n6_t = np.load(OUT / "gru_next6_with_today_test.npy")
    tree_v = np.load(OUT / "fusion_valid.npy")
    tree_t = np.load(OUT / "fusion_test.npy")
    old_mlp_v = np.load(OUT / "cs_mlp_valid.npy")
    mlp6_v = np.load(OUT / "cs_mlp_only6_valid.npy")
    mlp6_t = np.load(OUT / "cs_mlp_only6_test.npy")
    new_mlp_v = np.load(OUT / "cs_mlp_y2cols_valid.npy")
    new_mlp_t = np.load(OUT / "cs_mlp_y2cols_test.npy")
    locked = np.load(OUT / "fusion_rank_gate_valid.npy")

    print(f"old cs_mlp {mean_rank_ic(old_mlp_v, y, my):.6f}  Q4 {_q4(old_mlp_v, y, my, nx):.6f}")
    print(f"y2cols mlp {mean_rank_ic(new_mlp_v, y, my):.6f}  Q4 {_q4(new_mlp_v, y, my, nx):.6f}")

    ens_v = linear_blend(n6_v, linear_blend(o6_v, x6_v, 0.4), 0.15)
    ens_t = linear_blend(n6_t, linear_blend(o6_t, x6_t, 0.4), 0.15)
    gated_v = coverage_gate_blend(ens_v, tree_v, mx, TAU, W_LOW, W_HIGH, "rank")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], TAU, W_LOW, W_HIGH, "rank")

    blender = FusionModel({"weight_grid": [0.15]})
    blender.locked = {"name": "rank_blend", "weight": 0.15, "space": "rank", "ic": float("nan")}

    # replace full mlp with y2cols; keep only6 mlp
    mlp_ens_v = linear_blend(mlp6_v, new_mlp_v, 0.4)
    mlp_ens_t = linear_blend(mlp6_t, new_mlp_t, 0.4)
    out_v = blender.predict(mlp_ens_v, gated_v, mx)
    out_t = blender.predict(mlp_ens_t, gated_t, test["mask_x"])
    ic = mean_rank_ic(out_v, y, my)
    print(
        f"rank-gate + y2cols-mlp {ic:.6f}  Q4 {_q4(out_v, y, my, nx):.6f}  "
        f"locked {mean_rank_ic(locked, y, my):.6f}"
    )

    # y2cols mlp alone as the mlp ens
    out2 = blender.predict(new_mlp_v, gated_v, mx)
    print(f"rank-gate + y2cols-mlp only {mean_rank_ic(out2, y, my):.6f}  Q4 {_q4(out2, y, my, nx):.6f}")

    summary = {
        "y2cols_mlp": float(mean_rank_ic(new_mlp_v, y, my)),
        "fusion": float(ic),
        "locked_rank_gate": float(mean_rank_ic(locked, y, my)),
    }
    (OUT / "mlp_y2cols_fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if ic > float(mean_rank_ic(locked, y, my)) + 1e-6:
        np.save(OUT / "fusion_rank_gate_mlpy2_valid.npy", out_v)
        np.save(OUT / "fusion_rank_gate_mlpy2_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions/task2_fusion_rank_gate_mlpy2.npy")
        print("wrote new submission (beat locked rank-gate)")
    else:
        print("no fusion lift; not writing over a new main")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
