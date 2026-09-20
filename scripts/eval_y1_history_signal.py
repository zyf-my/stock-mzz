"""Evaluate causal same-stock historical-y1 signals and conservative blends."""

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


def build_history(y, my, windows):
    tlen, stocks = y.shape
    outs = {w: np.full((tlen, stocks), np.nan, dtype=np.float32) for w in windows}
    # Prefix sums/counts make each value use only days strictly before t.
    for w in windows:
        ss = np.zeros(stocks, dtype=np.float64)
        cc = np.zeros(stocks, dtype=np.float64)
        hist_s = []
        hist_c = []
        outs[w].fill(np.nan)
        for t in range(tlen):
            good = cc > 0
            outs[w][t, good] = (ss[good] / cc[good]).astype(np.float32)
            yt = np.asarray(y[t], dtype=np.float64)
            mt = np.asarray(my[t], dtype=bool) & np.isfinite(yt)
            add_s = np.zeros(stocks, dtype=np.float64); add_c = np.zeros(stocks, dtype=np.float64)
            add_s[mt], add_c[mt] = yt[mt], 1.0
            hist_s.append(add_s); hist_c.append(add_c)
            ss += add_s; cc += add_c
            if len(hist_s) > w:
                ss -= hist_s.pop(0); cc -= hist_c.pop(0)
    return outs


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    y_all = np.asarray(data["y1"], dtype=np.float32)
    my_all = np.asarray(data["mask_y"], dtype=bool)
    hist = build_history(y_all, my_all, [1, 2, 3, 5, 10, 20, 60, 120])
    # Convert each causal history signal to daily ranks before blending.
    cur_v = np.load(OUT / "fusion_alpha_platform_mix_valid.npy")
    cur_t = np.load(OUT / "fusion_alpha_platform_mix_test.npy")
    cur_vr = panel_cs_rank(cur_v, valid["mask_x"])
    cur_tr = panel_cs_rank(cur_t, test["mask_x"])
    rows = []
    for w, h in hist.items():
        hv = h[valid["start"]:valid["start"] + len(valid["mask_x"])]
        ht = h[test["start"]:test["start"] + len(test["mask_x"])]
        hr_v = panel_cs_rank(np.nan_to_num(hv, nan=0.0), valid["mask_x"])
        hr_t = panel_cs_rank(np.nan_to_num(ht, nan=0.0), test["mask_x"])
        h_ic = mean_rank_ic(hr_v, valid["y1"], valid["mask_y"])
        for wh in [0.02, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8]:
            out = (1.0 - wh) * cur_vr + wh * hr_v
            ic = mean_rank_ic(out, valid["y1"], valid["mask_y"])
            s = rank_ic_series(out, valid["y1"], valid["mask_y"])
            rows.append((float(ic), float(np.nanmean(s[-60:])), int(w), float(wh), float(h_ic), hr_t))
    rows.sort(key=lambda x: x[:5], reverse=True)
    print("BEST", [(r[0], r[1], r[2], r[3], r[4]) for r in rows[:30]])
    (OUT / "y1_history_signal_grid.json").write_text(
        json.dumps([(r[0], r[1], r[2], r[3], r[4]) for r in rows], indent=2), encoding="utf-8"
    )
    best = rows[0]
    if best[0] > 0.12370 and best[1] >= 0.1103:
        _, _, w, wh, _, hr_t = best
        out_t = ((1.0 - wh) * cur_tr + wh * hr_t).astype(np.float32)
        np.save(OUT / "fusion_alpha_y1hist_valid.npy", (1.0 - wh) * cur_vr + wh * panel_cs_rank(np.nan_to_num(hist[w][valid["start"]:valid["start"] + len(valid["mask_x"])], nan=0.0), valid["mask_x"]))
        np.save(OUT / "fusion_alpha_y1hist_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions/task1_fusion_alpha_y1hist.npy")
        (OUT / "fusion_alpha_y1hist_lock.json").write_text(
            json.dumps(
                {
                    "valid_ic": float(best[0]),
                    "last60_ic": float(best[1]),
                    "history_window": int(w),
                    "history_weight": float(wh),
                    "current_candidate_weight": float(1.0 - wh),
                    "causal": "same-stock y1 labels strictly before prediction day",
                    "note": "candidate only; platform test not run",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("WROTE", "outputs/fusion_alpha_y1hist_test.npy")
        print("WROTE", "submissions/task1_fusion_alpha_y1hist.npy")


if __name__ == "__main__":
    main()
