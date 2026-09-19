"""Apply locked 3-level coverage gate to x6 + only6-today GRU. One change: GRU branch.

Gate lock from fusion-gate3-001: tau_mid=4546 w_low=0.25, mid w=0.4, tau_high=4650 w_high=1.0
GRU mix from x6_only6: 0.4 * only6_today + 0.6 * x6
then rank-blend cs_mlp 0.15

Does not overwrite gate3 / n2000 files.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_x6_gate3.py
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
TAU_MID = 4546.0
TAU_HIGH = 4650.0
W_LOW = 0.25
W_MID = 0.4
W_HIGH = 1.0
MLP_W = 0.15
PREV_BEST = 0.122413


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

    x6_v = np.load(ROOT / "outputs/gru_no_today_recent_n2000_x6_valid.npy")
    x6_t = np.load(ROOT / "outputs/gru_no_today_recent_n2000_x6_test.npy")
    o6_v = np.load(ROOT / "outputs/gru_only6_with_today_valid.npy")
    o6_t = np.load(ROOT / "outputs/gru_only6_with_today_test.npy")
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")

    print(f"x6 RankIC={mean_rank_ic(x6_v, y, my):.6f}")
    print(f"only6-today RankIC={mean_rank_ic(o6_v, y, my):.6f}")

    for name, ens_v, ens_t in (
        ("x6", x6_v, x6_t),
        ("only6", o6_v, o6_t),
        ("0.4*only6+0.6*x6", linear_blend(o6_v, x6_v, ONLY6_W), linear_blend(o6_t, x6_t, ONLY6_W)),
    ):
        print(f"\n== {name} ==")
        print(f"  GRU ens RankIC={mean_rank_ic(ens_v, y, my):.6f}")
        gated_v = coverage_gate3(ens_v, tree_v, mx, TAU_MID, TAU_HIGH, W_LOW, W_MID, W_HIGH)
        gated_t = coverage_gate3(ens_t, tree_t, test["mask_x"], TAU_MID, TAU_HIGH, W_LOW, W_MID, W_HIGH)
        print(f"  gate3 RankIC={mean_rank_ic(gated_v, y, my):.6f}")
        blender = FusionModel({"weight_grid": [MLP_W]})
        blender.locked = {"name": "rank_blend", "weight": MLP_W, "space": "rank", "ic": float("nan")}
        out_v = blender.predict(mlp_v, gated_v, mx)
        ic = mean_rank_ic(out_v, y, my)
        series = rank_ic_series(out_v, y, my)
        print(
            f"  stack mlp {MLP_W} RankIC={ic:.6f} min={np.nanmin(series):.4f} "
            f"neg={int((series < 0).sum())}"
        )
        if name.startswith("0.4") and ic > PREV_BEST + 1e-4:
            out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
            np.save(ROOT / "outputs/fusion_x6_only6_gate3_valid.npy", out_v)
            np.save(ROOT / "outputs/fusion_x6_only6_gate3_test.npy", out_t)
            save_submission(out_t, ROOT / "submissions/task1_fusion_x6_only6_gate3.npy")
            meta = {
                "valid_ic": ic,
                "prev_best": PREV_BEST,
                "gate": {
                    "tau_mid": TAU_MID,
                    "tau_high": TAU_HIGH,
                    "w_low": W_LOW,
                    "w_mid": W_MID,
                    "w_high": W_HIGH,
                },
                "only6_w": ONLY6_W,
                "mlp_w": MLP_W,
            }
            (ROOT / "outputs/fusion_x6_only6_gate3_lock.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            print("wrote submissions/task1_fusion_x6_only6_gate3.npy")

    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
