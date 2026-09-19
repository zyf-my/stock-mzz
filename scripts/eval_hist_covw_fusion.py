"""Swap hist_lgbm-002 for coverage-weighted full-train hist inside locked x6_today fusion.

Tree blend stays raw 0.7. Gate / GRU / MLP weights stay frozen.
Does not overwrite fusion_valid.npy or task1_fusion_x6_today.npy.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\train_baseline.py --config configs/hist_lgbm_covw.yaml
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_hist_covw_fusion.py
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

LOCKED_VALID = 0.120869
TREE_W = 0.7
ONLY6_W = 0.4
NEXT6_W = 0.15
MLP6_W = 0.4
MLP_STACK_W = 0.15
TAU = 4546.0
W_LOW = 0.25
W_HIGH = 0.6
UPLOAD_BAR = 0.001


def _ic_on(pred, y, my, day_mask) -> float:
    series = rank_ic_series(pred, y, my)
    out = series.copy()
    out[~np.asarray(day_mask, dtype=bool)] = np.nan
    return float(np.nanmean(out))


def main() -> None:
    t0 = time.perf_counter()
    hist_v_path = ROOT / "outputs/hist_lgbm_covw_valid.npy"
    hist_t_path = ROOT / "outputs/hist_lgbm_covw_test.npy"
    if not hist_v_path.is_file() or not hist_t_path.is_file():
        raise FileNotFoundError(
            "missing hist_lgbm_covw preds; train first:\n"
            "  .\\.venv\\Scripts\\python.exe -u scripts\\train_baseline.py "
            "--config configs/hist_lgbm_covw.yaml"
        )

    cfg = load_config("configs/default.yaml")
    cache_dir = ROOT / "outputs" / "split_cache"
    data_path = None
    if not (cache_dir / "valid.npz").is_file() or not (cache_dir / "test.npz").is_file():
        data_path = resolve_data_path(cfg, None)
    splits, src = load_eval_splits(cache_dir=cache_dir, data_path=data_path, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    print(f"loaded {src} in {time.perf_counter() - t0:.1f}s")

    hist_v = np.load(hist_v_path)
    hist_t = np.load(hist_t_path)
    old_hist_v = np.load(ROOT / "outputs/hist_lgbm_valid.npy")
    base_v = np.load(ROOT / "outputs/baseline_valid.npy")
    base_t = np.load(ROOT / "outputs/baseline_test.npy")
    old_tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    nx = np.asarray(mx).sum(axis=1)

    print(f"hist_lgbm_covw RankIC={mean_rank_ic(hist_v, y, my):.6f}")
    print(f"hist_lgbm_002 RankIC={mean_rank_ic(old_hist_v, y, my):.6f}")
    print(
        f"hist highcov>=4546 covw={_ic_on(hist_v, y, my, nx >= TAU):.6f} "
        f"002={_ic_on(old_hist_v, y, my, nx >= TAU):.6f}"
    )

    tree_v = linear_blend(hist_v, base_v, TREE_W)
    tree_t = linear_blend(hist_t, base_t, TREE_W)
    print(f"tree blend covw RankIC={mean_rank_ic(tree_v, y, my):.6f}")
    print(f"tree blend 002 RankIC={mean_rank_ic(old_tree_v, y, my):.6f}")

    x6_v = np.load(ROOT / "outputs/gru_x6_with_today_valid.npy")
    x6_t = np.load(ROOT / "outputs/gru_x6_with_today_test.npy")
    o6_v = np.load(ROOT / "outputs/gru_only6_with_today_valid.npy")
    o6_t = np.load(ROOT / "outputs/gru_only6_with_today_test.npy")
    n6_v = np.load(ROOT / "outputs/gru_next6_with_today_valid.npy")
    n6_t = np.load(ROOT / "outputs/gru_next6_with_today_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
    mlp6_v = np.load(ROOT / "outputs/cs_mlp_only6_valid.npy")
    mlp6_t = np.load(ROOT / "outputs/cs_mlp_only6_test.npy")

    ens_v = linear_blend(n6_v, linear_blend(o6_v, x6_v, ONLY6_W), NEXT6_W)
    ens_t = linear_blend(n6_t, linear_blend(o6_t, x6_t, ONLY6_W), NEXT6_W)
    gated_v = coverage_gate_blend(ens_v, tree_v, mx, TAU, W_LOW, W_HIGH, "raw")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], TAU, W_LOW, W_HIGH, "raw")
    old_gated_v = coverage_gate_blend(ens_v, old_tree_v, mx, TAU, W_LOW, W_HIGH, "raw")
    print(f"coverage gate covw RankIC={mean_rank_ic(gated_v, y, my):.6f}")
    print(f"coverage gate 002 RankIC={mean_rank_ic(old_gated_v, y, my):.6f}")

    mlp_ens_v = linear_blend(mlp6_v, mlp_v, MLP6_W)
    mlp_ens_t = linear_blend(mlp6_t, mlp_t, MLP6_W)
    blender = FusionModel({"weight_grid": [MLP_STACK_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_STACK_W, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_ens_v, gated_v, mx)
    out_t = blender.predict(mlp_ens_t, gated_t, test["mask_x"])
    old_out_v = blender.predict(mlp_ens_v, old_gated_v, mx)
    ic = mean_rank_ic(out_v, y, my)
    old_ic = mean_rank_ic(old_out_v, y, my)
    high_new = _ic_on(out_v, y, my, nx >= TAU)
    high_old = _ic_on(old_out_v, y, my, nx >= TAU)
    series = rank_ic_series(out_v, y, my)
    print(
        f"locked fusion RankIC={ic:.6f} delta={ic - LOCKED_VALID:+.6f} "
        f"(002-tree recipe {old_ic:.6f}) highcov {high_new:.6f} vs {high_old:.6f} "
        f"min={float(np.nanmin(series)):.4f} neg={int(np.nansum(series < 0))}"
    )

    meta = {
        "locked": LOCKED_VALID,
        "valid_ic": float(ic),
        "old_recipe_ic": float(old_ic),
        "hist_covw": float(mean_rank_ic(hist_v, y, my)),
        "hist_002": float(mean_rank_ic(old_hist_v, y, my)),
        "tree_blend_covw": float(mean_rank_ic(tree_v, y, my)),
        "gate_covw": float(mean_rank_ic(gated_v, y, my)),
        "highcov_fusion_covw": high_new,
        "highcov_fusion_002": high_old,
        "delta": float(ic - LOCKED_VALID),
        "beat": bool(ic > LOCKED_VALID + 1e-4),
        "upload_ok": bool(ic > LOCKED_VALID + UPLOAD_BAR and high_new >= high_old - 1e-4),
        "recipe": "hist_lgbm_covw + baseline raw 0.7; x6_today gate2; other weights frozen",
    }
    (ROOT / "outputs/fusion_hist_covw_lock.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.save(ROOT / "outputs/fusion_hist_covw_valid.npy", out_v)
    np.save(ROOT / "outputs/fusion_hist_covw_test.npy", out_t)
    if meta["upload_ok"]:
        save_submission(out_t, ROOT / "submissions/task1_fusion_hist_covw.npy")
        print("wrote submissions/task1_fusion_hist_covw.npy")
    elif ic > LOCKED_VALID + 1e-4:
        print("valid up but high-cov valid down; not writing upload candidate")
    else:
        print(f"did not beat locked {LOCKED_VALID:.6f}; not writing a new main submission")
    print("does not overwrite submissions/task1_fusion_x6_today.npy")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
