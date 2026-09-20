"""Measure lag-1 y1 persistence over several historical time blocks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_y1_history_signal import build_history  # noqa: E402
from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import panel_cs_rank  # noqa: E402


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    y = np.asarray(data["y1"], dtype=np.float32)
    my = np.asarray(data["mask_y"], dtype=bool)
    mx = np.asarray(data["mask_x"], dtype=bool)
    h = build_history(y, my, [1])[1]
    # Evaluate 243-day blocks that end before the official valid split.
    ends = [1700, 1943, 2186, 2429, 2672, 2915, 3161]
    for end in ends:
        start = max(486, end - 243)
        p = panel_cs_rank(np.nan_to_num(h[start:end], nan=0.0), mx[start:end])
        ic = mean_rank_ic(p, y[start:end], my[start:end])
        print(start, end, f"lag1_rankic={ic:.9f}")


if __name__ == "__main__":
    main()
