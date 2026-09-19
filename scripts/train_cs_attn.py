"""Industry self-attention CS model. Does not overwrite cs_mlp or n2000 fusion files.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\train_cs_attn.py --config configs/cs_attn.yaml
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_task2_label, load_panel, slice_split, split_bounds  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.cs_attn import CSAttnModel  # noqa: E402
from src.models.fusion import FusionModel  # noqa: E402
from src.submit import save_submission  # noqa: E402

PREV_BEST = 0.117177
BLEND_VALID = ROOT / "outputs/fusion_recent_gru_n2000_cov_mlp_valid.npy"
BLEND_TEST = ROOT / "outputs/fusion_recent_gru_n2000_cov_mlp_test.npy"


def _drop_heavy_panels(*bags: dict) -> None:
    for bag in bags:
        for key in ("num_x", "cat_x", "panel_num_x"):
            bag.pop(key, None)
    gc.collect()


def _pairwise_pred_ic(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(a[t]) & np.isfinite(b[t])
        if int(m.sum()) < 8:
            continue
        r = spearmanr(a[t, m], b[t, m]).statistic
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train industry self-attention CS model")
    parser.add_argument("--config", default="configs/cs_attn.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    feat = dict(cfg.get("features") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    paths = cfg.get("paths") or {}
    fill_invalid = float(train_cfg.get("fill_invalid", 0.0))
    attn_cfg = {**feat, **model_cfg, **train_cfg}

    try:
        import torch

        n_threads = int(train_cfg.get("num_threads") or os.cpu_count() or 4)
        torch.set_num_threads(max(1, n_threads))
    except Exception:
        pass

    print(f"config={args.config}")
    print(
        f"CSAttn cols={len(feat.get('num_indices') or [])} hidden={model_cfg.get('hidden_size')} "
        f"heads={model_cfg.get('n_heads', 2)} loss={model_cfg.get('loss', 'pearson_ic')}"
    )

    data_path = resolve_data_path(cfg, args.data)
    print(f"data={data_path}")
    t0 = time.perf_counter()
    data = load_panel(str(data_path))
    drop_task2_label(data)
    print(f"loaded in {time.perf_counter() - t0:.1f}s  num_x={tuple(data['num_x'].shape)}")

    train_start, train_end = split_bounds(data, "train")
    valid = slice_split(data, "valid")
    test = slice_split(data, "test")
    print(f"days train={train_end - train_start} valid={valid['mask_x'].shape[0]}")

    model = CSAttnModel(attn_cfg, seed=seed)
    t1 = time.perf_counter()
    model.prepare_features(data)
    _drop_heavy_panels(data, valid, test)
    print(f"ind cache {time.perf_counter() - t1:.1f}s")

    t2 = time.perf_counter()
    fit_info = model.fit(data, train_start, train_end, valid=valid)
    print(f"fit {time.perf_counter() - t2:.1f}s best_valid_ic={fit_info.get('best_valid_ic')}")

    valid_pred = model.predict_panel(valid, data, fill_invalid=fill_invalid)
    valid_ic = mean_rank_ic(valid_pred, valid["y1"], valid["mask_y"])
    print(f"valid mean RankIC={valid_ic:.6f}")

    valid_pred_path = ROOT / paths.get("valid_pred", "outputs/cs_attn_valid.npy")
    valid_pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(valid_pred_path, valid_pred)

    peers = {
        "cs_mlp": ROOT / "outputs/cs_mlp_valid.npy",
        "tree_fusion": ROOT / "outputs/fusion_valid.npy",
        "gru_n2000": ROOT / "outputs/gru_no_today_recent_n2000_valid.npy",
        "locked_n2000": BLEND_VALID,
    }
    print("mean daily Spearman vs peers:")
    for name, path in peers.items():
        if not path.is_file():
            continue
        print(f"  vs {name:18s}  {_pairwise_pred_ic(valid_pred, np.load(path), valid['mask_x']):.4f}")

    test_pred = model.predict_panel(test, data, fill_invalid=fill_invalid)
    np.save(ROOT / paths.get("test_pred", "outputs/cs_attn_test.npy"), test_pred)
    save_submission(test_pred, ROOT / paths.get("submission", "submissions/task1_cs_attn.npy"))
    model.save(ROOT / paths.get("checkpoint", "checkpoints/cs_attn.pt"))

    if BLEND_VALID.is_file() and BLEND_TEST.is_file():
        blender = FusionModel({"weight_grid": [0.0, 0.15, 0.25, 0.4, 0.5]})
        locked = blender.fit(valid_pred, np.load(BLEND_VALID), valid["y1"], valid["mask_y"], valid["mask_x"])
        print("blend with locked n2000 fusion:")
        for row in (locked.get("valid_leaderboard") or [])[:6]:
            extra = {k: v for k, v in row.items() if k not in {"valid_leaderboard", "name", "ic"}}
            print(f"  {row['ic']:.6f}  {row['name']}  {extra}")
        print(f"BLEND_LOCKED {locked['name']} ic={locked['ic']:.6f}")
        if float(locked["ic"]) > PREV_BEST + 1e-4:
            out_v = blender.predict(valid_pred, np.load(BLEND_VALID), valid["mask_x"])
            out_t = blender.predict(test_pred, np.load(BLEND_TEST), test["mask_x"])
            np.save(ROOT / "outputs/fusion_n2000_cs_attn_valid.npy", out_v)
            np.save(ROOT / "outputs/fusion_n2000_cs_attn_test.npy", out_t)
            save_submission(out_t, ROOT / "submissions/task1_fusion_n2000_cs_attn.npy")
            print("wrote submissions/task1_fusion_n2000_cs_attn.npy (did not overwrite n2000 file)")
        else:
            print(f"blend did not beat {PREV_BEST:.6f}; not writing a new main submission")

    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{valid_ic:.6f}")


if __name__ == "__main__":
    main()
