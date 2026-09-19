"""Scan n_estimators for hist_lgbm_n1200 on valid."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402

ITERS = [50, 100, 150, 200, 250, 300, 350, 400]


def main() -> None:
    cfg = load_config("configs/task2/hist_lgbm_n1200.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y2")
    valid = slice_split(data, "valid")
    y, my = split_label_array(valid, "y2"), valid["mask_y"]
    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(cfg.get("features") or {}),
        seed=int(cfg.get("seed", 42)),
    )
    model.load(ROOT / (cfg.get("paths") or {})["checkpoint"])
    preds = model.predict_panel_iters(valid, ITERS, fill_invalid=0.0)
    best = (-1.0, None)
    for n in ITERS:
        ic = mean_rank_ic(preds[n], y, my)
        print(f"n={n:3d}  {ic:.6f}")
        if ic > best[0]:
            best = (ic, n)
    print(f"best n={best[1]} {best[0]:.6f}  (800/day @200 was 0.077658)")
    print("VALID_RANKIC", f"{best[0]:.6f}")


if __name__ == "__main__":
    main()
