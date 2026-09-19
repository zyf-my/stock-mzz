"""Swap task2 only6 GRU for y2-native cols (66 -> 55). Other weights frozen.

Does not overwrite task2_fusion_x6_today.npy unless valid beats the locked recipe.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\train_gru.py --config configs/task2/gru_only6_y2cols.yaml
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_task2_only6_y2cols.py
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
ONLY6_W = 0.4
NEXT6_W = 0.15
MLP6_W = 0.4
MLP_STACK_W = 0.15
TAU = 4546.0
W_LOW = 0.25
W_HIGH = 0.6
LABEL_KEY = "y2"
OUT = ROOT / "outputs" / "task2"
UPLOAD_BAR = 0.001


def _load(name: str) -> np.ndarray:
    return np.load(OUT / name)


def main() -> None:
    t0 = time.perf_counter()
    new_v_path = OUT / "gru_only6_y2cols_valid.npy"
    new_t_path = OUT / "gru_only6_y2cols_test.npy"
    if not new_v_path.is_file() or not new_t_path.is_file():
        raise FileNotFoundError(
            "missing gru_only6_y2cols preds; train first:\n"
            "  .\\.venv\\Scripts\\python.exe -u scripts\\train_gru.py "
            "--config configs/task2/gru_only6_y2cols.yaml"
        )

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
    old_o6_v = _load("gru_only6_with_today_valid.npy")
    o6_v = np.load(new_v_path)
    o6_t = np.load(new_t_path)
    n6_v = _load("gru_next6_with_today_valid.npy")
    n6_t = _load("gru_next6_with_today_test.npy")
    tree_v = _load("fusion_valid.npy")
    tree_t = _load("fusion_test.npy")
    mlp_v = _load("cs_mlp_valid.npy")
    mlp_t = _load("cs_mlp_test.npy")
    mlp6_v = _load("cs_mlp_only6_valid.npy")
    mlp6_t = _load("cs_mlp_only6_test.npy")

    print(f"only6_y2cols RankIC={mean_rank_ic(o6_v, y, my):.6f}")
    print(f"only6_task1cols RankIC={mean_rank_ic(old_o6_v, y, my):.6f}")

    ens_v = linear_blend(n6_v, linear_blend(o6_v, x6_v, ONLY6_W), NEXT6_W)
    ens_t = linear_blend(n6_t, linear_blend(o6_t, x6_t, ONLY6_W), NEXT6_W)
    old_ens_v = linear_blend(n6_v, linear_blend(old_o6_v, x6_v, ONLY6_W), NEXT6_W)
    print(f"GRU ens y2cols RankIC={mean_rank_ic(ens_v, y, my):.6f}")
    print(f"GRU ens task1cols RankIC={mean_rank_ic(old_ens_v, y, my):.6f}")

    gated_v = coverage_gate_blend(ens_v, tree_v, mx, TAU, W_LOW, W_HIGH, "raw")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], TAU, W_LOW, W_HIGH, "raw")
    old_gated_v = coverage_gate_blend(old_ens_v, tree_v, mx, TAU, W_LOW, W_HIGH, "raw")
    print(f"coverage gate y2cols RankIC={mean_rank_ic(gated_v, y, my):.6f}")
    print(f"coverage gate task1cols RankIC={mean_rank_ic(old_gated_v, y, my):.6f}")

    mlp_ens_v = linear_blend(mlp6_v, mlp_v, MLP6_W)
    mlp_ens_t = linear_blend(mlp6_t, mlp_t, MLP6_W)
    blender = FusionModel({"weight_grid": [MLP_STACK_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_STACK_W, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_ens_v, gated_v, mx)
    out_t = blender.predict(mlp_ens_t, gated_t, test["mask_x"])
    old_out_v = blender.predict(mlp_ens_v, old_gated_v, mx)
    ic = mean_rank_ic(out_v, y, my)
    old_ic = mean_rank_ic(old_out_v, y, my)
    series = rank_ic_series(out_v, y, my)
    print(
        f"locked fusion RankIC={ic:.6f} delta={ic - LOCKED_VALID:+.6f} "
        f"(old only6 recipe {old_ic:.6f}) "
        f"min={float(np.nanmin(series)):.4f} neg={int(np.nansum(series < 0))}"
    )

    meta = {
        "locked": LOCKED_VALID,
        "valid_ic": float(ic),
        "old_recipe_ic": float(old_ic),
        "only6_y2cols": float(mean_rank_ic(o6_v, y, my)),
        "only6_task1cols": float(mean_rank_ic(old_o6_v, y, my)),
        "gru_ens": float(mean_rank_ic(ens_v, y, my)),
        "gate": float(mean_rank_ic(gated_v, y, my)),
        "delta": float(ic - LOCKED_VALID),
        "beat": bool(ic > LOCKED_VALID + 1e-4),
        "upload_ok": bool(ic > LOCKED_VALID + UPLOAD_BAR),
        "cols": [42, 58, 55, 69, 73, 74],
        "recipe": "only6 66->55; x6/next6/tree/mlp/gate frozen",
    }
    (OUT / "fusion_only6_y2cols_lock.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(OUT / "fusion_only6_y2cols_valid.npy", out_v)
    np.save(OUT / "fusion_only6_y2cols_test.npy", out_t)
    if ic > LOCKED_VALID + 1e-4:
        save_submission(out_t, ROOT / "submissions/task2_fusion_only6_y2cols.npy")
        print("wrote submissions/task2_fusion_only6_y2cols.npy")
        print("does not overwrite submissions/task2_fusion_x6_today.npy")
    else:
        print(f"did not beat locked {LOCKED_VALID:.6f}; not writing a new main submission")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
