"""Create the conservative platform-mix + seed=43 candidate."""

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
W_SEED43 = 0.10


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
    out_v = (W_SEED43 * br_v + (1.0 - W_SEED43) * ar_v).astype(np.float32)
    out_t = (W_SEED43 * br_t + (1.0 - W_SEED43) * ar_t).astype(np.float32)
    ic = mean_rank_ic(out_v, valid["y1"], valid["mask_y"])
    series = rank_ic_series(out_v, valid["y1"], valid["mask_y"])
    np.save(OUT / "fusion_alpha_platform_mix_seed43_valid.npy", out_v)
    np.save(OUT / "fusion_alpha_platform_mix_seed43_test.npy", out_t)
    save_submission(out_t, ROOT / "submissions/task1_fusion_alpha_platform_mix_seed43.npy")
    (OUT / "fusion_alpha_platform_mix_seed43_lock.json").write_text(
        json.dumps(
            {
                "valid_ic": float(ic),
                "last60_ic": float(np.nanmean(series[-60:])),
                "seed43_weight": W_SEED43,
                "platform_mix_weight": 1.0 - W_SEED43,
                "note": "candidate only; platform test not run",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("VALID_RANKIC", f"{ic:.9f}")
    print("LAST60_RANKIC", f"{float(np.nanmean(series[-60:])):.9f}")
    print("WROTE", "submissions/task1_fusion_alpha_platform_mix_seed43.npy")


if __name__ == "__main__":
    main()
