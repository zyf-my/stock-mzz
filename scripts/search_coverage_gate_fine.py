"""Fine search the coverage-conditioned final blend around the current best."""

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

OUT = ROOT / "outputs"


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid = slice_split(data, "valid")
    a = np.load(OUT / "fusion_alpha_diverse_blend_valid.npy")
    b = np.load(OUT / "fusion_x6_only6_mlp6_valid.npy")
    cov = np.asarray(valid["mask_x"]).sum(axis=1).astype(float)
    rows = []
    for q in np.arange(.12, .281, .01):
        cut = float(np.quantile(cov, q))
        high = cov >= cut
        for wh in np.arange(.45, .701, .025):
            for wl in np.arange(.90, 1.001, .025):
                w = np.where(high, wh, wl)
                out = (w[:, None] * a + (1.0 - w[:, None]) * b).astype(np.float32)
                ic = mean_rank_ic(out, valid["y1"], valid["mask_y"])
                rows.append((float(ic), float(q), float(cut), float(wh), float(wl)))
    rows.sort(reverse=True)
    print("BEST", rows[:30])
    (OUT / "coverage_gate_fine_grid.json").write_text(json.dumps(rows[:100], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
