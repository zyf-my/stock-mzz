"""Compare rank, z-score and robust-z-score final fusion shapes."""

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
from src.models.fusion import coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402

OUT = ROOT / "outputs"


def day_norm(x, mask, mode):
    out = np.zeros_like(x, dtype=np.float32)
    for t in range(x.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if m.sum() < 2:
            continue
        v = np.asarray(x[t, m], dtype=np.float64)
        if mode == "rank":
            out[t, m] = panel_cs_rank(x[t:t + 1], mask[t:t + 1])[0, m]
        elif mode == "zscore":
            out[t, m] = ((v - v.mean()) / max(v.std(), 1e-8)).astype(np.float32)
        elif mode == "robust":
            med = np.median(v)
            scale = np.median(np.abs(v - med)) * 1.4826
            out[t, m] = ((v - med) / max(scale, 1e-8)).astype(np.float32)
        else:
            raise ValueError(mode)
    return out


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid = slice_split(data, "valid")
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    x6 = np.load(OUT / "gru_x6_with_today_valid.npy")
    o6 = np.load(OUT / "gru_only6_with_today_valid.npy")
    n6 = np.load(OUT / "gru_next6_with_today_valid.npy")
    mlp = np.load(OUT / "cs_mlp_valid.npy")
    mlp6 = np.load(OUT / "cs_mlp_only6_valid.npy")
    base = np.load(OUT / "baseline_valid.npy")
    av = np.load(OUT / "hist_lgbm_alpha_x_base_valid.npy")
    hv = np.load(OUT / "hist_lgbm_alpha_x_hard_valid.npy")
    d = np.zeros(av.shape[0])
    ar, hr = panel_cs_rank(av, mx), panel_cs_rank(hv, mx)
    for t in range(av.shape[0]):
        m = np.asarray(mx[t], dtype=bool)
        d[t] = np.mean(np.abs(ar[t, m] - hr[t, m])) if m.sum() > 2 else 0
    cut = float(np.quantile(d, .75))
    day_w = np.where(d >= cut, .3, .5)[:, None]
    tree = linear_blend(day_w * av + (1 - day_w) * hv, base, .7)
    temporal = linear_blend(n6, linear_blend(o6, x6, .4), .15)
    gated = coverage_gate_blend(temporal, tree, mx, 4546.0, .25, .6, "raw")
    mlp_raw = linear_blend(mlp6, mlp, .4)
    branch = panel_cs_rank(np.load(OUT / "fusion_x6_only6_mlp6_valid.npy"), mx)
    rows = []
    for shape in ["rank", "zscore", "robust"]:
        gr = day_norm(gated, mx, shape)
        mr = day_norm(mlp_raw, mx, shape)
        for mw in [0.0, .05, .10, .15, .20, .25, .30]:
            alpha = (mw * mr + (1 - mw) * gr).astype(np.float32)
            alpha_rank = panel_cs_rank(alpha, mx)
            for aw in [.70, .75, .80, .82, .85, .90, .95, 1.0]:
                out = (aw * alpha_rank + (1 - aw) * branch).astype(np.float32)
                rows.append((float(mean_rank_ic(out, y, my)), shape, float(mw), float(aw)))
    rows.sort(reverse=True)
    print("BEST", rows[:40])
    (OUT / "shape_fusion_grid.json").write_text(json.dumps(rows[:100], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
