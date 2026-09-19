"""Task2 fusion after swapping hist/GRU columns to y2 Top 21.

Tree blend: raw 0.7 (task1 lock) plus the searched fusion_y2cols if present.
Gate / MLP weights stay at the task1 x6_today constants.
Does not overwrite task2_fusion_x6_today.npy.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_task2_y2cols_fusion.py
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
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

LOCKED_VALID = 0.080643
TREE_W = 0.7
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

    hist_v = _load("hist_lgbm_y2cols_valid.npy")
    hist_t = _load("hist_lgbm_y2cols_test.npy")
    old_hist_v = _load("hist_lgbm_valid.npy")
    base_v = _load("baseline_valid.npy")
    base_t = _load("baseline_test.npy")
    print(f"hist_y2cols RankIC={mean_rank_ic(hist_v, y, my):.6f}")
    print(f"hist_task1cols RankIC={mean_rank_ic(old_hist_v, y, my):.6f}")
    print(f"baseline RankIC={mean_rank_ic(base_v, y, my):.6f}")

    raw_v = linear_blend(hist_v, base_v, TREE_W)
    raw_t = linear_blend(hist_t, base_t, TREE_W)
    old_tree_v = _load("fusion_valid.npy")
    old_tree_t = _load("fusion_test.npy")
    print(f"tree raw0.7 y2cols RankIC={mean_rank_ic(raw_v, y, my):.6f}")
    print(f"tree searched task1cols RankIC={mean_rank_ic(old_tree_v, y, my):.6f}")
    tree_v, tree_t, tree_tag = old_tree_v, old_tree_t, "old_searched"
    if (OUT / "fusion_y2cols_valid.npy").is_file():
        searched = _load("fusion_y2cols_valid.npy")
        searched_ic = mean_rank_ic(searched, y, my)
        print(f"tree searched y2cols RankIC={searched_ic:.6f}")
        if searched_ic > mean_rank_ic(tree_v, y, my) + 1e-4:
            tree_v = searched
            tree_t = _load("fusion_y2cols_test.npy")
            tree_tag = "y2cols_searched"
    print(f"gate tree={tree_tag}")

    gru_v = _load("gru_y2cols_valid.npy")
    gru_t = _load("gru_y2cols_test.npy")
    print(f"gru_y2cols RankIC={mean_rank_ic(gru_v, y, my):.6f}")

    o6_path = OUT / "gru_only6_with_today_valid.npy"
    n6_path = OUT / "gru_next6_with_today_valid.npy"
    if o6_path.is_file() and n6_path.is_file():
        o6_v = _load("gru_only6_with_today_valid.npy")
        o6_t = _load("gru_only6_with_today_test.npy")
        n6_v = _load("gru_next6_with_today_valid.npy")
        n6_t = _load("gru_next6_with_today_test.npy")
        ens_v = linear_blend(n6_v, linear_blend(o6_v, gru_v, ONLY6_W), NEXT6_W)
        ens_t = linear_blend(n6_t, linear_blend(o6_t, gru_t, ONLY6_W), NEXT6_W)
        print(f"GRU ens (+old only6/next6) RankIC={mean_rank_ic(ens_v, y, my):.6f}")
    else:
        ens_v, ens_t = gru_v, gru_t
        print("no only6/next6; gate uses gru_y2cols alone")

    gated_v = coverage_gate_blend(ens_v, tree_v, mx, TAU, W_LOW, W_HIGH, "raw")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], TAU, W_LOW, W_HIGH, "raw")
    print(f"coverage gate RankIC={mean_rank_ic(gated_v, y, my):.6f}")

    mlp_v = _load("cs_mlp_valid.npy")
    mlp_t = _load("cs_mlp_test.npy")
    mlp6_v = _load("cs_mlp_only6_valid.npy")
    mlp6_t = _load("cs_mlp_only6_test.npy")
    mlp_ens_v = linear_blend(mlp6_v, mlp_v, MLP6_W)
    mlp_ens_t = linear_blend(mlp6_t, mlp_t, MLP6_W)
    blender = FusionModel({"weight_grid": [MLP_STACK_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_STACK_W, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_ens_v, gated_v, mx)
    out_t = blender.predict(mlp_ens_t, gated_t, test["mask_x"])
    ic = mean_rank_ic(out_v, y, my)
    series = rank_ic_series(out_v, y, my)
    print(
        f"fusion RankIC={ic:.6f} delta={ic - LOCKED_VALID:+.6f} "
        f"min={float(np.nanmin(series)):.4f} neg={int(np.nansum(series < 0))}"
    )

    meta = {
        "locked": LOCKED_VALID,
        "valid_ic": float(ic),
        "hist_y2cols": float(mean_rank_ic(hist_v, y, my)),
        "gru_y2cols": float(mean_rank_ic(gru_v, y, my)),
        "gate": float(mean_rank_ic(gated_v, y, my)),
        "delta": float(ic - LOCKED_VALID),
        "beat_012": bool(ic >= 0.12),
        "recipe": "y2 Top21 hist+GRU; tree search if present else raw 0.7; gate2; old mlp",
    }
    (OUT / "fusion_y2cols_main_lock.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(OUT / "fusion_y2cols_main_valid.npy", out_v)
    np.save(OUT / "fusion_y2cols_main_test.npy", out_t)
    save_submission(out_t, ROOT / "submissions/task2_fusion_y2cols.npy")
    print("wrote submissions/task2_fusion_y2cols.npy")
    print("does not overwrite submissions/task2_fusion_x6_today.npy")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
