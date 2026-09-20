"""Promote the stable rank blend of the alpha-disagreement candidate and x6 branch."""

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
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs"
W = 0.82


def blend(a: np.ndarray, b: np.ndarray, w: float, mask: np.ndarray) -> np.ndarray:
    ar = panel_cs_rank(a, mask)
    br = panel_cs_rank(b, mask)
    return (w * ar + (1.0 - w) * br).astype(np.float32)


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    a_v = np.load(OUT / "fusion_alpha_disagree_valid.npy")
    a_t = np.load(OUT / "fusion_alpha_disagree_test.npy")
    b_v = np.load(OUT / "fusion_x6_only6_mlp6_valid.npy")
    b_t = np.load(OUT / "fusion_x6_only6_mlp6_test.npy")
    out_v = blend(a_v, b_v, W, valid["mask_x"])
    out_t = blend(a_t, b_t, W, test["mask_x"])
    ic = mean_rank_ic(out_v, valid["y1"], valid["mask_y"])
    series = rank_ic_series(out_v, valid["y1"], valid["mask_y"])
    hi = series[valid["mask_y"].sum(axis=1) >= 31]
    np.save(OUT / "fusion_alpha_diverse_blend_valid.npy", out_v)
    np.save(OUT / "fusion_alpha_diverse_blend_test.npy", out_t)
    save_submission(out_t, ROOT / "submissions/task1_fusion_alpha_diverse_blend.npy")
    (OUT / "fusion_alpha_diverse_blend_lock.json").write_text(
        json.dumps(
            {
                "valid_ic": float(ic),
                "high_coverage_31_ic": float(np.nanmean(hi)),
                "alpha_disagree_weight": W,
                "x6_only6_branch_weight": 1.0 - W,
                "note": "candidate only; platform test not run",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("VALID_RANKIC", f"{ic:.9f}")
    print("HIGH_COVERAGE_31_RANKIC", f"{float(np.nanmean(hi)):.9f}")
    print("NEGATIVE_DAYS", int(np.sum(series < 0)))
    print("WROTE", "submissions/task1_fusion_alpha_diverse_blend.npy")


if __name__ == "__main__":
    main()
