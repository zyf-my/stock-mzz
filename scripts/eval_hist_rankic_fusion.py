"""Train hist_lgbm with LambdaRankIC, blend with baseline, eval locked fusion.

Does not overwrite task1_fusion_next6_wt_mlp6.npy.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_hist_rankic_fusion.py
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_hist_rankic_fusion.py --skip-train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import (  # noqa: E402
    drop_task2_label,
    flatten_masked_rows,
    group_sizes,
    load_panel,
    resolve_cat_indices,
    slice_split,
)
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

LOCKED_VALID = 0.120542
TREE_W = 0.7
ONLY6_W = 0.4
NEXT6_W = 0.15
MLP6_W = 0.4
MLP_STACK_W = 0.15
TAU = 4546.0
W_LOW = 0.25
W_HIGH = 0.6
UPLOAD_BAR = 0.001


def _train_rankic_hist() -> float:
    cfg = load_config("configs/hist_lgbm_rankic.yaml")
    seed = int(cfg.get("seed", 42))
    feature_cfg = dict(cfg.get("features") or {})
    cat_indices = resolve_cat_indices(feature_cfg)
    paths = cfg.get("paths") or {}
    fill_invalid = float((cfg.get("train") or {}).get("fill_invalid", 0.0))

    data_path = resolve_data_path(cfg, None)
    print(f"data={data_path}")
    t0 = time.perf_counter()
    data = load_panel(str(data_path))
    drop_task2_label(data)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    train = slice_split(data, "train")
    valid = slice_split(data, "valid")
    test = slice_split(data, "test")
    x_train, y_train, coords = flatten_masked_rows(
        train,
        cat_indices=cat_indices,
        require_label=True,
        feature_cfg=feature_cfg,
        seed=seed,
    )
    group = group_sizes(coords)
    print(f"train rows={x_train.shape[0]} groups={group.size} mean_group={float(group.mean()):.1f}")

    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=feature_cfg,
        seed=seed,
    )
    t1 = time.perf_counter()
    model.fit(x_train, y_train, group=group)
    print(f"fit {time.perf_counter() - t1:.1f}s")
    del x_train, y_train

    valid_pred = model.predict_panel(valid, fill_invalid=fill_invalid)
    valid_ic = mean_rank_ic(valid_pred, valid["y1"], valid["mask_y"])
    print(f"hist_lgbm_rankic valid RankIC={valid_ic:.6f}")

    out_v = ROOT / paths.get("valid_pred", "outputs/hist_lgbm_rankic_valid.npy")
    out_t = ROOT / paths.get("test_pred", "outputs/hist_lgbm_rankic_test.npy")
    out_v.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_v, valid_pred)
    test_pred = model.predict_panel(test, fill_invalid=fill_invalid)
    np.save(out_t, test_pred)
    ckpt = ROOT / paths.get("checkpoint", "checkpoints/hist_lgbm_rankic.txt")
    model.save(ckpt)
    print(f"wrote {out_v} {out_t} {ckpt}")
    return float(valid_ic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if not args.skip_train:
        hist_ic = _train_rankic_hist()
        print(f"single-model valid={hist_ic:.6f} (hist_lgbm MSE baseline ~0.104)")
    else:
        print("skip train; load existing hist_lgbm_rankic preds")

    from src.dataset import load_eval_splits  # noqa: E402

    splits, src = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    print(f"eval cache={src}")

    hist_v = np.load(ROOT / "outputs/hist_lgbm_rankic_valid.npy")
    hist_t = np.load(ROOT / "outputs/hist_lgbm_rankic_test.npy")
    base_v = np.load(ROOT / "outputs/baseline_valid.npy")
    base_t = np.load(ROOT / "outputs/baseline_test.npy")
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]

    print(f"hist_rankic RankIC={mean_rank_ic(hist_v, y, my):.6f}")
    print(f"baseline RankIC={mean_rank_ic(base_v, y, my):.6f}")
    old_hist_v = np.load(ROOT / "outputs/hist_lgbm_valid.npy")
    print(f"hist_mse RankIC={mean_rank_ic(old_hist_v, y, my):.6f}")

    tree_v = linear_blend(hist_v, base_v, TREE_W)
    tree_t = linear_blend(hist_t, base_t, TREE_W)
    tree_ic = mean_rank_ic(tree_v, y, my)
    old_tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    print(f"tree blend (rankic hist) RankIC={tree_ic:.6f}")
    print(f"tree blend (mse hist) RankIC={mean_rank_ic(old_tree_v, y, my):.6f}")

    x6_v = np.load(ROOT / "outputs/gru_no_today_recent_n2000_x6_valid.npy")
    o6_v = np.load(ROOT / "outputs/gru_only6_with_today_valid.npy")
    n6_v = np.load(ROOT / "outputs/gru_next6_with_today_valid.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp6_v = np.load(ROOT / "outputs/cs_mlp_only6_valid.npy")
    x6_t = np.load(ROOT / "outputs/gru_no_today_recent_n2000_x6_test.npy")
    o6_t = np.load(ROOT / "outputs/gru_only6_with_today_test.npy")
    n6_t = np.load(ROOT / "outputs/gru_next6_with_today_test.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
    mlp6_t = np.load(ROOT / "outputs/cs_mlp_only6_test.npy")

    ens_v = linear_blend(n6_v, linear_blend(o6_v, x6_v, ONLY6_W), NEXT6_W)
    ens_t = linear_blend(n6_t, linear_blend(o6_t, x6_t, ONLY6_W), NEXT6_W)
    gated_v = coverage_gate_blend(ens_v, tree_v, mx, TAU, W_LOW, W_HIGH, "raw")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], TAU, W_LOW, W_HIGH, "raw")
    mlp_ens_v = linear_blend(mlp6_v, mlp_v, MLP6_W)
    mlp_ens_t = linear_blend(mlp6_t, mlp_t, MLP6_W)
    blender = FusionModel({"weight_grid": [MLP_STACK_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_STACK_W, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_ens_v, gated_v, mx)
    out_t = blender.predict(mlp_ens_t, gated_t, test["mask_x"])
    ic = mean_rank_ic(out_v, y, my)
    series = rank_ic_series(out_v, y, my)
    print(
        f"locked fusion RankIC={ic:.6f} delta={ic - LOCKED_VALID:+.6f} "
        f"min={np.nanmin(series):.4f} neg={int((series < 0).sum())}"
    )

    meta = {
        "locked": LOCKED_VALID,
        "valid_ic": ic,
        "hist_rankic": float(mean_rank_ic(hist_v, y, my)),
        "tree_blend": float(tree_ic),
        "beat": bool(ic > LOCKED_VALID + 1e-4),
        "upload_ok": bool(ic > LOCKED_VALID + UPLOAD_BAR),
    }
    (ROOT / "outputs/fusion_hist_rankic_lock.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if ic > LOCKED_VALID + 1e-4:
        np.save(ROOT / "outputs/fusion_hist_rankic_valid.npy", out_v)
        np.save(ROOT / "outputs/fusion_hist_rankic_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions/task1_fusion_hist_rankic.npy")
        print("wrote submissions/task1_fusion_hist_rankic.npy")
        if ic <= LOCKED_VALID + UPLOAD_BAR:
            print(f"delta < {UPLOAD_BAR}; do not upload")
    else:
        print("did not beat locked 0.120542; not writing a new main submission")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
