"""Create a conservative rank mix of the two platform-tested candidates."""

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
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs"
W_NO_MLP = 0.825


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    a_v = np.load(OUT / "fusion_alpha_diverse_blend_valid.npy")
    a_t = np.load(OUT / "fusion_alpha_diverse_blend_test.npy")
    b_v = np.load(OUT / "fusion_alpha_no_mlp_valid.npy")
    b_t = np.load(OUT / "fusion_alpha_no_mlp_test.npy")
    ar_v, ar_t = panel_cs_rank(a_v, valid["mask_x"]), panel_cs_rank(a_t, test["mask_x"])
    br_v, br_t = panel_cs_rank(b_v, valid["mask_x"]), panel_cs_rank(b_t, test["mask_x"])
    out_v = (W_NO_MLP * br_v + (1.0 - W_NO_MLP) * ar_v).astype(np.float32)
    out_t = (W_NO_MLP * br_t + (1.0 - W_NO_MLP) * ar_t).astype(np.float32)
    ic = mean_rank_ic(out_v, valid["y1"], valid["mask_y"])
    np.save(OUT / "fusion_alpha_platform_mix_valid.npy", out_v)
    np.save(OUT / "fusion_alpha_platform_mix_test.npy", out_t)
    save_submission(out_t, ROOT / "submissions/task1_fusion_alpha_platform_mix.npy")
    (OUT / "fusion_alpha_platform_mix_lock.json").write_text(
        json.dumps(
            {
                "valid_ic": float(ic),
                "no_mlp_weight": W_NO_MLP,
                "alpha_diverse_weight": 1.0 - W_NO_MLP,
                "platform_tested_inputs": {
                    "alpha_diverse": 0.126891,
                    "alpha_no_mlp": 0.126852,
                },
                "note": "candidate only; platform test not run",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("VALID_RANKIC", f"{ic:.9f}")
    print("WROTE", "submissions/task1_fusion_alpha_platform_mix.npy")


if __name__ == "__main__":
    main()
