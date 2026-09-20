"""Promote the stable optimum from the extended coverage-gate search."""

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
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs"
CUT_Q = 0.28
CUT = 4367.04
W_HIGH = 0.325
W_LOW = 1.0


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    av = np.load(OUT / "fusion_alpha_diverse_blend_valid.npy")
    at = np.load(OUT / "fusion_alpha_diverse_blend_test.npy")
    bv = np.load(OUT / "fusion_x6_only6_mlp6_valid.npy")
    bt = np.load(OUT / "fusion_x6_only6_mlp6_test.npy")
    cov_v = np.asarray(valid["mask_x"]).sum(axis=1)
    cov_t = np.asarray(test["mask_x"]).sum(axis=1)
    wv = np.where(cov_v >= CUT, W_HIGH, W_LOW)
    wt = np.where(cov_t >= CUT, W_HIGH, W_LOW)
    out_v = (wv[:, None] * av + (1.0 - wv[:, None]) * bv).astype(np.float32)
    out_t = (wt[:, None] * at + (1.0 - wt[:, None]) * bt).astype(np.float32)
    ic = mean_rank_ic(out_v, valid["y1"], valid["mask_y"])
    series = rank_ic_series(out_v, valid["y1"], valid["mask_y"])
    hi31 = series[valid["mask_y"].sum(axis=1) >= 31]
    np.save(OUT / "fusion_alpha_coverage_gate_opt_valid.npy", out_v)
    np.save(OUT / "fusion_alpha_coverage_gate_opt_test.npy", out_t)
    save_submission(out_t, ROOT / "submissions/task1_fusion_alpha_coverage_gate_opt.npy")
    (OUT / "fusion_alpha_coverage_gate_opt_lock.json").write_text(
        json.dumps(
            {
                "valid_ic": float(ic),
                "high_coverage_31_ic": float(np.nanmean(hi31)),
                "coverage_quantile": CUT_Q,
                "coverage_cut": CUT,
                "current_candidate_weight_high_coverage": W_HIGH,
                "current_candidate_weight_low_coverage": W_LOW,
                "other_branch": "fusion_x6_only6_mlp6",
                "note": "candidate only; platform test not run",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("VALID_RANKIC", f"{ic:.9f}")
    print("HIGH_COVERAGE_31_RANKIC", f"{float(np.nanmean(hi31)):.9f}")
    print("HIGH_COVERAGE_DAYS", int(np.sum(cov_v >= CUT)), "of", int(cov_v.size))
    print("WROTE", "submissions/task1_fusion_alpha_coverage_gate_opt.npy")


if __name__ == "__main__":
    main()
