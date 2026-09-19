"""Train y2 tree-residual cs_mlp (hist@200 OLS residual). No y1 features."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split, split_bounds, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.cs_mlp import CSMLPModel  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"


def _drop_heavy(*bags) -> None:
    for bag in bags:
        for k in ("num_x", "cat_x", "panel_num_x"):
            bag.pop(k, None)
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task2/cs_mlp_y2resid.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    label_key = "y2"
    feat = dict(cfg.get("features") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    paths = cfg.get("paths") or {}
    tree_spec = dict(cfg.get("tree") or {})
    mlp_cfg = {**feat, **model_cfg, **train_cfg, "label_key": label_key}

    data = load_panel(str(resolve_data_path(cfg, args.data)))
    drop_other_label(data, label_key)
    train_start, train_end = split_bounds(data, "train")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")

    # hist@200 preds for tree_ols residual (train fill only; valid/test from saved if exist)
    n_iter = int(tree_spec.get("num_iteration", 200))
    hist_v_path = OUT / "hist_lgbm_n200_valid.npy"
    hist_t_path = OUT / "hist_lgbm_n200_test.npy"
    if hist_v_path.is_file() and hist_t_path.is_file():
        print("reuse hist_lgbm_n200 valid/test preds")
        hist_v = np.load(hist_v_path)
        hist_t = np.load(hist_t_path)
        tree_cfg = load_config(tree_spec.get("config", "configs/task2/hist_lgbm.yaml"))
        tree = LightGBMBaseline(
            params=(tree_cfg.get("model") or {}).get("params") or {},
            feature_cfg=dict(tree_cfg.get("features") or {}),
            seed=seed,
        )
        tree.load(ROOT / tree_spec["checkpoint"])
        train = slice_split(data, "train")
        train_pred = tree.predict_panel(train, num_iteration=n_iter)
        del tree
    else:
        tree_cfg = load_config(tree_spec.get("config", "configs/task2/hist_lgbm.yaml"))
        tree = LightGBMBaseline(
            params=(tree_cfg.get("model") or {}).get("params") or {},
            feature_cfg=dict(tree_cfg.get("features") or {}),
            seed=seed,
        )
        tree.load(ROOT / tree_spec["checkpoint"])
        train = slice_split(data, "train")
        train_pred = tree.predict_panel(train, num_iteration=n_iter)
        hist_v = tree.predict_panel(valid, num_iteration=n_iter)
        hist_t = tree.predict_panel(test, num_iteration=n_iter)
        del tree

    aux = np.zeros(data["y2"].shape, dtype=np.float32)
    aux[train_start:train_end] = train_pred
    va0, va1 = split_bounds(data, "valid")
    te0, te1 = split_bounds(data, "test")
    aux[va0:va1] = hist_v
    aux[te0:te1] = hist_t

    model = CSMLPModel(mlp_cfg, seed=seed)
    model.aux_pred = aux
    model.prepare_features(data)
    _drop_heavy(data, valid, test)
    t0 = time.perf_counter()
    fit_info = model.fit(data, train_start, train_end, valid=valid)
    print(f"fit {time.perf_counter() - t0:.1f}s best_valid={fit_info.get('best_valid_ic')}")
    model.aux_pred = None

    pred_v = model.predict_panel(valid, data)
    pred_t = model.predict_panel(test, data)
    y, my = split_label_array(valid, label_key), valid["mask_y"]
    ic = mean_rank_ic(pred_v, y, my)
    print(f"residual-mlp valid RankIC={ic:.6f}")

    np.save(ROOT / paths["valid_pred"], pred_v)
    np.save(ROOT / paths["test_pred"], pred_t)
    model.save(ROOT / paths["checkpoint"])
    save_submission(pred_t, ROOT / paths["submission"])
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
