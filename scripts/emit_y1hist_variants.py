"""Emit isolated y1-history weight variants without overwriting the submitted baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_y1_history_signal import build_history  # noqa: E402
from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import panel_cs_rank  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs"


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    hist = build_history(np.asarray(data["y1"], dtype=np.float32), np.asarray(data["mask_y"], dtype=bool), [1])
    hv = hist[1][valid["start"]:valid["start"] + len(valid["mask_x"])]
    ht = hist[1][test["start"]:test["start"] + len(test["mask_x"])]
    hr_v = panel_cs_rank(np.nan_to_num(hv, nan=0.0), valid["mask_x"])
    hr_t = panel_cs_rank(np.nan_to_num(ht, nan=0.0), test["mask_x"])
    cur_v = panel_cs_rank(np.load(OUT / "fusion_alpha_platform_mix_valid.npy"), valid["mask_x"])
    cur_t = panel_cs_rank(np.load(OUT / "fusion_alpha_platform_mix_test.npy"), test["mask_x"])
    for wh in [.4, .5, .6, .8, .9, 1.0]:
        out_v = ((1 - wh) * cur_v + wh * hr_v).astype(np.float32)
        out_t = ((1 - wh) * cur_t + wh * hr_t).astype(np.float32)
        tag = f"w{int(wh * 100):02d}"
        np.save(OUT / f"fusion_alpha_y1hist_{tag}_valid.npy", out_v)
        np.save(OUT / f"fusion_alpha_y1hist_{tag}_test.npy", out_t)
        save_submission(out_t, ROOT / f"submissions/task1_fusion_alpha_y1hist_{tag}.npy")
        ic = mean_rank_ic(out_v, valid["y1"], valid["mask_y"])
        s = rank_ic_series(out_v, valid["y1"], valid["mask_y"])
        print(tag, "valid", f"{ic:.9f}", "last60", f"{float(np.nanmean(s[-60:])):.9f}")
        if wh == .4:
            # Restore the exact file that produced the already submitted 0.127559 score.
            save_submission(out_t, ROOT / "submissions/task1_fusion_alpha_y1hist.npy")
            np.save(OUT / "fusion_alpha_y1hist_valid.npy", out_v)
            np.save(OUT / "fusion_alpha_y1hist_test.npy", out_t)


if __name__ == "__main__":
    main()
