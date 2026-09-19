"""y2-native: how many LightGBM trees? Reuse task2 checkpoints, no refit."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.fusion import coverage_gate_blend  # noqa: E402

OUT = ROOT / "outputs" / "task2"
ITERS = [50, 100, 150, 200, 250, 300, 350, 400]


def _scan(cfg_path: str, split) -> dict[int, np.ndarray]:
    cfg = load_config(cfg_path)
    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(cfg.get("features") or {}),
        seed=int(cfg.get("seed", 42)),
    )
    model.load(ROOT / (cfg.get("paths") or {})["checkpoint"])
    print(f"scan {cfg_path}", flush=True)
    t0 = time.perf_counter()
    preds = model.predict_panel_iters(split, ITERS, fill_invalid=0.0)
    print(f"  predict {time.perf_counter() - t0:.1f}s", flush=True)
    return preds


def main() -> None:
    cfg = load_config("configs/task2/baseline.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y2")
    valid = slice_split(data, "valid")
    y = split_label_array(valid, "y2")
    my, mx = valid["mask_y"], valid["mask_x"]
    print(f"valid days={valid['mask_x'].shape[0]}")

    base = _scan("configs/task2/baseline.yaml", valid)
    hist = _scan("configs/task2/hist_lgbm.yaml", valid)
    print("single-model valid by n_estimators")
    best_b, best_h = (-1.0, None), (-1.0, None)
    for n in ITERS:
        ib = mean_rank_ic(base[n], y, my)
        ih = mean_rank_ic(hist[n], y, my)
        print(f"  n={n:3d}  baseline={ib:.6f}  hist={ih:.6f}")
        if ib > best_b[0]:
            best_b = (ib, n)
        if ih > best_h[0]:
            best_h = (ih, n)
    print(f"best baseline n={best_b[1]} {best_b[0]:.6f}  (400 was 0.074372)")
    print(f"best hist     n={best_h[1]} {best_h[0]:.6f}  (400 was 0.075879)")

    print("tree fusion locked rule at those n")
    tree = coverage_gate_blend(hist[best_h[1]], base[best_b[1]], mx, 4491.0, 0.5, 0.75, "rank")
    print(f"  fusion {mean_rank_ic(tree, y, my):.6f}  (old 0.078735)")
    print("VALID_RANKIC", f"{mean_rank_ic(tree, y, my):.6f}")


if __name__ == "__main__":
    main()
