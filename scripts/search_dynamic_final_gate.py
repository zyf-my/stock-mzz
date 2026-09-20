"""Search observable day-level gates between the current candidate and a diverse branch."""

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

OUT = ROOT / "outputs"


def day_disagreement(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ar = panel_cs_rank(a, mask)
    br = panel_cs_rank(b, mask)
    out = np.zeros(a.shape[0], dtype=np.float64)
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if m.sum() > 2:
            out[t] = float(np.mean(np.abs(ar[t, m] - br[t, m])))
    return out


def evaluate(av, bv, at, bt, valid, test, y, my, label):
    rows = []
    d_v = day_disagreement(av, bv, valid["mask_x"])
    d_t = day_disagreement(at, bt, test["mask_x"])
    cov_v = np.asarray(valid["mask_x"]).sum(axis=1).astype(float)
    cov_t = np.asarray(test["mask_x"]).sum(axis=1).astype(float)
    # Freeze all thresholds on valid and apply unchanged to test.
    for signal_name, sv, st in [("disagreement", d_v, d_t), ("coverage", cov_v, cov_t)]:
        for q in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
            cut = float(np.quantile(sv, q))
            high_v = sv >= cut
            high_t = st >= cut
            for wh in [.55, .65, .75, .85, .95]:
                for wl in [.55, .65, .75, .85, .95]:
                    wv = np.where(high_v, wh, wl)
                    wt = np.where(high_t, wh, wl)
                    out = (wv[:, None] * av + (1 - wv[:, None]) * bv).astype(np.float32)
                    ic = mean_rank_ic(out, y, my)
                    rows.append((float(ic), signal_name, float(q), float(wh), float(wl), float(cut)))
    return rows


def main() -> None:
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    av = np.load(OUT / "fusion_alpha_diverse_blend_valid.npy")
    at = np.load(OUT / "fusion_alpha_diverse_blend_test.npy")
    # The branch used in the current candidate; use the original output, not a re-ranked copy.
    bv = np.load(OUT / "fusion_x6_only6_mlp6_valid.npy")
    bt = np.load(OUT / "fusion_x6_only6_mlp6_test.npy")
    rows = evaluate(av, bv, at, bt, valid, test, valid["y1"], valid["mask_y"], "branch")
    rows.sort(reverse=True)
    print("BEST", rows[:30])
    (OUT / "dynamic_final_gate_grid.json").write_text(json.dumps(rows[:100], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
