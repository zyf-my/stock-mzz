"""Locked next6 fusion with x6_today GRU + pre-locked 3-level coverage gate.

GRU: 0.15 * next6 + 0.85 * (0.4 * only6_today + 0.6 * x6_today)
Gate lock from fusion-gate3-001: tau_mid=4546 w_low=0.25, mid w=0.4, tau_high=4650 w_high=1.0
MLP mix: 0.4 * cs_mlp_only6 + 0.6 * cs_mlp, rank-blend 0.15

Does not overwrite task1_fusion_next6_wt_mlp6.npy.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_x6_today_gate3_fusion.py
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
from src.models.fusion import FusionModel, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

ONLY6_W = 0.4
NEXT6_W = 0.15
MLP6_W = 0.4
MLP_STACK_W = 0.15
TAU_MID = 4546.0
TAU_HIGH = 4650.0
W_LOW = 0.25
W_MID = 0.4
W_HIGH = 1.0
UPLOAD_BAR = 0.121542
LOCKED_VALID = 0.120542


def coverage_gate3(
    temporal: np.ndarray,
    cs: np.ndarray,
    mask_x: np.ndarray,
    tau_mid: float,
    tau_high: float,
    w_low: float,
    w_mid: float,
    w_high: float,
) -> np.ndarray:
    n_x = np.asarray(mask_x).sum(axis=1)
    pred = np.empty_like(temporal, dtype=np.float32)
    low = n_x < float(tau_mid)
    high = n_x >= float(tau_high)
    mid = ~low & ~high
    pred[low] = linear_blend(temporal[low], cs[low], w_low)
    pred[mid] = linear_blend(temporal[mid], cs[mid], w_mid)
    pred[high] = linear_blend(temporal[high], cs[high], w_high)
    return pred


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    cache_dir = ROOT / "outputs" / "split_cache"
    data_path = None
    if not (cache_dir / "valid.npz").is_file() or not (cache_dir / "test.npz").is_file():
        data_path = resolve_data_path(cfg, None)
    splits, src = load_eval_splits(cache_dir=cache_dir, data_path=data_path, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    print(f"loaded {src} in {time.perf_counter() - t0:.1f}s")

    x6_v = np.load(ROOT / "outputs/gru_x6_with_today_valid.npy")
    x6_t = np.load(ROOT / "outputs/gru_x6_with_today_test.npy")
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

    print(f"x6_today RankIC={mean_rank_ic(x6_v, y, my):.6f}")

    ens_v = linear_blend(n6_v, linear_blend(o6_v, x6_v, ONLY6_W), NEXT6_W)
    ens_t = linear_blend(n6_t, linear_blend(o6_t, x6_t, ONLY6_W), NEXT6_W)
    print(f"GRU ensemble RankIC={mean_rank_ic(ens_v, y, my):.6f}")

    gated_v = coverage_gate3(ens_v, tree_v, mx, TAU_MID, TAU_HIGH, W_LOW, W_MID, W_HIGH)
    gated_t = coverage_gate3(ens_t, tree_t, test["mask_x"], TAU_MID, TAU_HIGH, W_LOW, W_MID, W_HIGH)
    print(f"gate3 RankIC={mean_rank_ic(gated_v, y, my):.6f}")

    mlp_ens_v = linear_blend(mlp6_v, mlp_v, MLP6_W)
    mlp_ens_t = linear_blend(mlp6_t, mlp_t, MLP6_W)
    blender = FusionModel({"weight_grid": [MLP_STACK_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_STACK_W, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_ens_v, gated_v, mx)
    out_t = blender.predict(mlp_ens_t, gated_t, test["mask_x"])
    ic = mean_rank_ic(out_v, y, my)
    series = rank_ic_series(out_v, y, my)
    print(
        f"fusion RankIC={ic:.6f} (bar={UPLOAD_BAR:.6f}) "
        f"min={np.nanmin(series):.4f} neg={int((series < 0).sum())}"
    )

    out_dir = ROOT / "outputs" / "x6_today_gate3_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "valid.npy", out_v)
    np.save(out_dir / "test.npy", out_t)
    summary = {
        "x6_today_ic": float(mean_rank_ic(x6_v, y, my)),
        "gru_ens_ic": float(mean_rank_ic(ens_v, y, my)),
        "gate3_ic": float(mean_rank_ic(gated_v, y, my)),
        "fusion_ic": float(ic),
        "upload_bar": UPLOAD_BAR,
        "locked_valid": LOCKED_VALID,
        "delta_vs_bar": float(ic - UPLOAD_BAR),
        "passes_bar": bool(ic >= UPLOAD_BAR),
        "note": "gate3 valid high but prior x6_only6 gate3 test=0.117855; verify on platform",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if summary["passes_bar"]:
        save_submission(out_t, ROOT / "submissions/task1_fusion_x6_today_gate3.npy")
        print("saved submissions/task1_fusion_x6_today_gate3.npy")
    print("does not overwrite submissions/task1_fusion_next6_wt_mlp6.npy")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
