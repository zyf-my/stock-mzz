"""Locked fusion after adding next-tier GRU and 6-col MLP mix.

GRU: 0.15 * next6_today + 0.85 * (0.4 * only6_today + 0.6 * x6)
coverage raw tau=4546 w_low=0.25 w_high=0.6
MLP mix: 0.4 * cs_mlp_only6 + 0.6 * cs_mlp, then rank-blend 0.15

Does not overwrite the 0.120168 file.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_next6_fusion.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import load_eval_splits  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

ONLY6_W = 0.4
NEXT6_W = 0.15
MLP6_W = 0.4
MLP_STACK_W = 0.15
TAU = 4546.0
W_LOW = 0.25
W_HIGH = 0.6
PREV_BEST = 0.120168


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    cache_dir = ROOT / "outputs" / "split_cache"
    data_path = None
    if not (cache_dir / "valid.npz").is_file() or not (cache_dir / "test.npz").is_file():
        data_path = resolve_data_path(cfg, None)
    splits, src = load_eval_splits(cache_dir=cache_dir, data_path=data_path, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    print(f"loaded {src} in {time.perf_counter() - t0:.1f}s")

    x6_v = np.load(ROOT / "outputs/gru_no_today_recent_n2000_x6_valid.npy")
    x6_t = np.load(ROOT / "outputs/gru_no_today_recent_n2000_x6_test.npy")
    o6_v = np.load(ROOT / "outputs/gru_only6_with_today_valid.npy")
    o6_t = np.load(ROOT / "outputs/gru_only6_with_today_test.npy")
    n6_v = np.load(ROOT / "outputs/gru_next6_with_today_valid.npy")
    n6_t = np.load(ROOT / "outputs/gru_next6_with_today_test.npy")
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
    mlp6_v = np.load(ROOT / "outputs/cs_mlp_only6_valid.npy")
    mlp6_t = np.load(ROOT / "outputs/cs_mlp_only6_test.npy")
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]

    ens_v = linear_blend(n6_v, linear_blend(o6_v, x6_v, ONLY6_W), NEXT6_W)
    ens_t = linear_blend(n6_t, linear_blend(o6_t, x6_t, ONLY6_W), NEXT6_W)
    print(f"GRU ensemble RankIC={mean_rank_ic(ens_v, y, my):.6f}")

    gated_v = coverage_gate_blend(ens_v, tree_v, mx, TAU, W_LOW, W_HIGH, "raw")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], TAU, W_LOW, W_HIGH, "raw")
    print(f"coverage gate RankIC={mean_rank_ic(gated_v, y, my):.6f}")

    mlp_ens_v = linear_blend(mlp6_v, mlp_v, MLP6_W)
    mlp_ens_t = linear_blend(mlp6_t, mlp_t, MLP6_W)
    blender = FusionModel({"weight_grid": [MLP_STACK_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_STACK_W, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_ens_v, gated_v, mx)
    out_t = blender.predict(mlp_ens_t, gated_t, test["mask_x"])
    ic = mean_rank_ic(out_v, y, my)
    series = rank_ic_series(out_v, y, my)
    print(
        f"stack RankIC={ic:.6f} min={np.nanmin(series):.4f} "
        f"median={np.nanmedian(series):.4f} neg={int((series < 0).sum())}"
    )

    np.save(ROOT / "outputs/fusion_next6_wt_mlp6_valid.npy", out_v)
    np.save(ROOT / "outputs/fusion_next6_wt_mlp6_test.npy", out_t)
    sub = ROOT / "submissions/task1_fusion_next6_wt_mlp6.npy"
    save_submission(out_t, sub)
    meta = {
        "valid_ic": ic,
        "gru": {"only6_w": ONLY6_W, "next6_w": NEXT6_W},
        "gate": {"tau": TAU, "weight_low": W_LOW, "weight_high": W_HIGH, "space": "raw"},
        "mlp6_w": MLP6_W,
        "mlp_stack_w": MLP_STACK_W,
        "prev_best": PREV_BEST,
        "beat_prev": bool(ic > PREV_BEST + 1e-4),
    }
    (ROOT / "outputs/fusion_next6_wt_mlp6_lock.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {sub}")
    print("does not overwrite submissions/task1_fusion_x6_only6_w06_cov_mlp.npy")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
