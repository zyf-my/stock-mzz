"""Replace next6 with y2 leftover GRU inside rank-gate. No main overwrite unless lift."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402

OUT = ROOT / "outputs" / "task2"


def main() -> None:
    splits, _ = load_eval_splits(splits=("valid",), dump_if_missing=False)
    valid = splits["valid"]
    y = split_label_array(valid, "y2")
    my, mx = valid["mask_y"], valid["mask_x"]

    x6 = np.load(OUT / "gru_x6_with_today_valid.npy")
    o6 = np.load(OUT / "gru_only6_with_today_valid.npy")
    old_n6 = np.load(OUT / "gru_next6_with_today_valid.npy")
    new_n6 = np.load(OUT / "gru_next6_y2left_valid.npy")
    tree = np.load(OUT / "fusion_valid.npy")
    mlp = np.load(OUT / "cs_mlp_valid.npy")
    mlp6 = np.load(OUT / "cs_mlp_only6_valid.npy")
    locked = np.load(OUT / "fusion_rank_gate_valid.npy")

    print(f"old next6 {mean_rank_ic(old_n6, y, my):.6f}")
    print(f"y2left    {mean_rank_ic(new_n6, y, my):.6f}")

    blender = FusionModel({"weight_grid": [0.15]})
    blender.locked = {"name": "rank_blend", "weight": 0.15, "space": "rank", "ic": float("nan")}
    mlp_ens = linear_blend(mlp6, mlp, 0.4)

    ens = linear_blend(new_n6, linear_blend(o6, x6, 0.4), 0.15)
    gated = coverage_gate_blend(ens, tree, mx, 4546.0, 0.25, 0.6, "rank")
    out = blender.predict(mlp_ens, gated, mx)
    ic = mean_rank_ic(out, y, my)
    print(f"rank-gate + y2left next6 {ic:.6f}  locked {mean_rank_ic(locked, y, my):.6f}")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
