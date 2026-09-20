"""Search a second, low-weight rank blend around the promoted candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import panel_cs_rank  # noqa: E402

OUT = ROOT / "outputs"


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    a_v = np.load(OUT / "fusion_alpha_diverse_blend_valid.npy")
    a_t = np.load(OUT / "fusion_alpha_diverse_blend_test.npy")
    ar_v, ar_t = panel_cs_rank(a_v, valid["mask_x"]), panel_cs_rank(a_t, test["mask_x"])
    rows = []
    names = [
        "fusion_next6_wt_mlp6", "fusion_x6_cov_mlp", "fusion_wt_mlp6",
        "fusion_x6_only6_mlp6", "fusion_nfull_cov_mlp", "fusion_recent_gru_n2000_cov_mlp",
    ]
    for name in names:
        vp, tp = OUT / f"{name}_valid.npy", OUT / f"{name}_test.npy"
        if not vp.exists() or not tp.exists():
            continue
        bv = panel_cs_rank(np.load(vp), valid["mask_x"])
        for w in np.linspace(.70, .99, 30):
            out = (w * ar_v + (1.0 - w) * bv).astype(np.float32)
            rows.append((float(mean_rank_ic(out, valid["y1"], valid["mask_y"])), name, float(w)))
    rows.sort(reverse=True)
    print("BEST", rows[:15])
    (OUT / "diverse_second_blend_grid.json").write_text(json.dumps(rows[:50], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
