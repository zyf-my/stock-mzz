"""Task2 (y2) main fusion — same locked recipe as task1 x6_today.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_task2_x6_today_fusion.py
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
from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

ONLY6_W = 0.4
NEXT6_W = 0.15
MLP6_W = 0.4
MLP_STACK_W = 0.15
TAU = 4546.0
W_LOW = 0.25
W_HIGH = 0.6
LABEL_KEY = "y2"
OUT = ROOT / "outputs" / "task2"


def _load(name: str) -> np.ndarray:
    return np.load(OUT / name)


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    cache_dir = ROOT / "outputs" / "split_cache"
    data_path = None
    if not (cache_dir / "valid.npz").is_file():
        data_path = resolve_data_path(cfg, None)
    splits, src = load_eval_splits(cache_dir=cache_dir, data_path=data_path, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    y = split_label_array(valid, LABEL_KEY)
    my, mx = valid["mask_y"], valid["mask_x"]
    print(f"loaded {src} label={LABEL_KEY} in {time.perf_counter() - t0:.1f}s")

    x6_v = _load("gru_x6_with_today_valid.npy")
    x6_t = _load("gru_x6_with_today_test.npy")
    o6_v = _load("gru_only6_with_today_valid.npy")
    o6_t = _load("gru_only6_with_today_test.npy")
    n6_v = _load("gru_next6_with_today_valid.npy")
    n6_t = _load("gru_next6_with_today_test.npy")
    tree_v = _load("fusion_valid.npy")
    tree_t = _load("fusion_test.npy")
    mlp_v = _load("cs_mlp_valid.npy")
    mlp_t = _load("cs_mlp_test.npy")
    mlp6_v = _load("cs_mlp_only6_valid.npy")
    mlp6_t = _load("cs_mlp_only6_test.npy")

    print(f"x6_today RankIC={mean_rank_ic(x6_v, y, my):.6f}")

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
    print(f"task2 fusion RankIC={ic:.6f}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "fusion_x6_today_valid.npy", out_v)
    np.save(OUT / "fusion_x6_today_test.npy", out_t)
    summary = {
        "label_key": LABEL_KEY,
        "x6_today_ic": float(mean_rank_ic(x6_v, y, my)),
        "gru_ens_ic": float(mean_rank_ic(ens_v, y, my)),
        "gate_ic": float(mean_rank_ic(gated_v, y, my)),
        "fusion_valid_ic": float(ic),
        "recipe": "same as task1 x6_today gate2 fusion",
    }
    (OUT / "fusion_x6_today_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    sub = ROOT / "submissions/task2_fusion_x6_today.npy"
    save_submission(out_t, sub)
    print(f"wrote {sub}")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
