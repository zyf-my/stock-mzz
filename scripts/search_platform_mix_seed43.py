"""Test whether the platform mix benefits from a small seed=43 full-fusion signal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import panel_cs_rank  # noqa: E402

OUT = ROOT / "outputs"


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    a_v = np.load(OUT / "fusion_alpha_platform_mix_valid.npy")
    a_t = np.load(OUT / "fusion_alpha_platform_mix_test.npy")
    b_v = np.load(OUT / "fusion_alpha_x_s43_valid.npy")
    b_t = np.load(OUT / "fusion_alpha_x_s43_test.npy")
    ar_v, ar_t = panel_cs_rank(a_v, valid["mask_x"]), panel_cs_rank(a_t, test["mask_x"])
    br_v, br_t = panel_cs_rank(b_v, valid["mask_x"]), panel_cs_rank(b_t, test["mask_x"])
    rows = []
    for ws in np.arange(0.0, .301, .025):
        out = (ws * br_v + (1.0 - ws) * ar_v).astype(np.float32)
        s = rank_ic_series(out, valid["y1"], valid["mask_y"])
        rows.append((float(mean_rank_ic(out, valid["y1"], valid["mask_y"])), float(np.nanmean(s[-60:])), float(ws)))
    rows.sort(reverse=True)
    print("BEST", rows[:30])
    (OUT / "platform_mix_seed43_grid.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
