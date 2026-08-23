"""Swap locked only6 GRU for input-decay variant. Fusion weights stay frozen.

Does not overwrite task1_fusion_next6_wt_mlp6.npy.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_only6_decay_fusion.py
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

LOCKED_VALID = 0.120542
ONLY6_W = 0.4
NEXT6_W = 0.15
MLP6_W = 0.4
MLP_STACK_W = 0.15
TAU = 4546.0
W_LOW = 0.25
W_HIGH = 0.6
UPLOAD_BAR = 0.001


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

    o6_v = np.load(ROOT / "outputs/gru_only6_input_decay_valid.npy")
    o6_t = np.load(ROOT / "outputs/gru_only6_input_decay_test.npy")
    x6_v = np.load(ROOT / "outputs/gru_no_today_recent_n2000_x6_valid.npy")
    x6_t = np.load(ROOT / "outputs/gru_no_today_recent_n2000_x6_test.npy")
    n6_v = np.load(ROOT / "outputs/gru_next6_with_today_valid.npy")
    n6_t = np.load(ROOT / "outputs/gru_next6_with_today_test.npy")
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
    mlp6_v = np.load(ROOT / "outputs/cs_mlp_only6_valid.npy")
    mlp6_t = np.load(ROOT / "outputs/cs_mlp_only6_test.npy")
    old_o6_v = np.load(ROOT / "outputs/gru_only6_with_today_valid.npy")
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]

    print(f"only6_decay RankIC={mean_rank_ic(o6_v, y, my):.6f}")
    print(f"only6_locked RankIC={mean_rank_ic(old_o6_v, y, my):.6f}")

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
    print(f"locked fusion RankIC={ic:.6f} (bar={LOCKED_VALID + UPLOAD_BAR:.6f})")

    out_dir = ROOT / "outputs" / "only6_decay_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "valid.npy", out_v)
    np.save(out_dir / "test.npy", out_t)
    summary = {
        "only6_decay_ic": float(mean_rank_ic(o6_v, y, my)),
        "only6_locked_ic": float(mean_rank_ic(old_o6_v, y, my)),
        "gru_ens_ic": float(mean_rank_ic(ens_v, y, my)),
        "gate_ic": float(mean_rank_ic(gated_v, y, my)),
        "fusion_ic": float(ic),
        "locked_valid": LOCKED_VALID,
        "delta_vs_locked": float(ic - LOCKED_VALID),
        "passes_bar": bool(ic >= LOCKED_VALID + UPLOAD_BAR),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if summary["passes_bar"]:
        save_submission(out_t, ROOT / "submissions/task1_fusion_only6_decay.npy")
        print("saved submissions/task1_fusion_only6_decay.npy")
    else:
        print("below upload bar — locked submission unchanged")


if __name__ == "__main__":
    main()
