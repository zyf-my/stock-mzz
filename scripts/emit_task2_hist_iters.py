"""Emit hist@N predictions from main hist checkpoint for bag search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.dataset import split_label_array  # noqa: E402

OUT = ROOT / "outputs" / "task2"


def emit(n: int) -> None:
    stem = f"hist_lgbm_n{n}"
    cfg = load_config("configs/task2/hist_lgbm.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y2")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(cfg.get("features") or {}),
        seed=int(cfg.get("seed", 42)),
    )
    model.load(ROOT / (cfg.get("paths") or {})["checkpoint"])
    hist_v = model.predict_panel(valid, fill_invalid=0.0, num_iteration=n)
    hist_t = model.predict_panel(test, fill_invalid=0.0, num_iteration=n)
    y, my = split_label_array(valid, "y2"), valid["mask_y"]
    ic = mean_rank_ic(hist_v, y, my)
    hi = valid["mask_x"].sum(1) >= 4670
    ic_hi = mean_rank_ic(hist_v[hi], y[hi], my[hi])
    np.save(OUT / f"{stem}_valid.npy", hist_v)
    np.save(OUT / f"{stem}_test.npy", hist_t)
    print(f"{stem}: valid={ic:.6f} hi31={ic_hi:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", default="210,240,250", help="comma-separated tree counts")
    args = parser.parse_args()
    for part in args.iters.split(","):
        emit(int(part.strip()))


if __name__ == "__main__":
    main()
