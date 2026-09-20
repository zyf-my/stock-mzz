"""Search branch blends using recent validation stability plus full-period score."""

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
    valid = slice_split(data, "valid")
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    anchor = np.load(OUT / "fusion_alpha_diverse_blend_valid.npy")
    ar = panel_cs_rank(anchor, mx)
    names = [
        "fusion_x6_only6_mlp6", "fusion_next6_wt_mlp6", "fusion_wt_mlp6",
        "fusion_x6_cov_mlp", "fusion_nfull_cov_mlp", "fusion_recent_gru_n2000_cov_mlp",
        "fusion_alpha_disagree",
    ]
    rows = []
    for name in names:
        p = OUT / f"{name}_valid.npy"
        if not p.exists():
            continue
        br = panel_cs_rank(np.load(p), mx)
        for w in np.arange(.50, 1.001, .025):
            out = (w * ar + (1.0 - w) * br).astype(np.float32)
            s = rank_ic_series(out, y, my)
            full = float(np.nanmean(s))
            last60 = float(np.nanmean(s[-60:]))
            q4 = float(np.nanmean(s[3 * len(s) // 4:]))
            # Favor recent performance, but reject candidates that lose too much overall.
            robust = 0.50 * full + 0.30 * last60 + 0.20 * q4
            rows.append((robust, full, last60, q4, name, float(w)))
    rows.sort(reverse=True)
    print("ROBUST_BEST", rows[:30])
    print("FULL_BEST", sorted(rows, key=lambda x: x[1], reverse=True)[:15])
    (OUT / "recent_stable_blend_grid.json").write_text(json.dumps(rows[:100], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
