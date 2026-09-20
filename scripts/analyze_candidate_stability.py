"""Inspect time-window stability of candidate blends before another submission."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split  # noqa: E402
from src.metrics import rank_ic_series  # noqa: E402

OUT = ROOT / "outputs"


def window_stats(series: np.ndarray, n: int) -> list[float]:
    cuts = [0, n // 4, n // 2, 3 * n // 4, n]
    return [float(np.nanmean(series[cuts[i]:cuts[i + 1]])) for i in range(4)]


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid = slice_split(data, "valid")
    names = {
        "alpha_diverse": "fusion_alpha_diverse_blend_valid.npy",
        "coverage_opt": "fusion_alpha_coverage_gate_opt_valid.npy",
        "alpha_no_mlp": "fusion_alpha_no_mlp_valid.npy",
        "alpha_disagree": "fusion_alpha_disagree_valid.npy",
        "x6_only6": "fusion_x6_only6_mlp6_valid.npy",
        "alpha_x": "fusion_alpha_x_valid.npy",
        "baseline_fusion": "fusion_valid.npy",
    }
    result = {}
    for name, fn in names.items():
        p = OUT / fn
        if not p.exists():
            continue
        s = rank_ic_series(np.load(p), valid["y1"], valid["mask_y"])
        result[name] = {
            "full": float(np.nanmean(s)),
            "quarters": window_stats(s, len(s)),
            "last_60": float(np.nanmean(s[-60:])),
            "negative_days": int(np.sum(s < 0)),
        }
    print(json.dumps(result, indent=2))
    (OUT / "candidate_stability.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
