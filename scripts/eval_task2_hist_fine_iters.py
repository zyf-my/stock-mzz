"""Fine-scan hist n_estimators around 200 and emit best@n preds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402

OUT = ROOT / "outputs" / "task2"
ITERS = [150, 175, 200, 225, 250]


def main() -> None:
    cfg = load_config("configs/task2/hist_lgbm.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y2")
    valid = slice_split(data, "valid")
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(cfg.get("features") or {}),
        seed=int(cfg.get("seed", 42)),
    )
    model.load(ROOT / (cfg.get("paths") or {})["checkpoint"])
    preds = model.predict_panel_iters(valid, ITERS, fill_invalid=0.0)
    base = np.load(OUT / "baseline_valid.npy")
    ens = linear_blend(
        np.load(OUT / "gru_next6_with_today_valid.npy"),
        linear_blend(np.load(OUT / "gru_only6_with_today_valid.npy"), np.load(OUT / "gru_x6_with_today_valid.npy"), 0.4),
        0.15,
    )
    mlp = linear_blend(np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_valid.npy"), 0.4)
    blender = FusionModel({"weight_grid": [0.15]})
    blender.locked = {"name": "rank_blend", "weight": 0.15, "space": "rank", "ic": float("nan")}
    best = (-1.0, None)
    for n in ITERS:
        tree = coverage_gate_blend(preds[n], base, mx, 4491.0, 0.5, 0.75, "rank")
        gated = coverage_gate_blend(ens, tree, mx, 4546.0, 0.25, 0.6, "rank")
        out = blender.predict(mlp, gated, mx)
        ic = mean_rank_ic(out, y, my)
        print(f"n={n} hist={mean_rank_ic(preds[n], y, my):.6f} full={ic:.6f}")
        if ic > best[0]:
            best = (ic, n)
    print(f"best n={best[1]} ic={best[0]:.6f}")


if __name__ == "__main__":
    main()
