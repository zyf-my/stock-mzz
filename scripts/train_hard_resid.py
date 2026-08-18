"""Train-hard-day residual MLP. Hard days come from hist_lgbm train RankIC only.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\train_hard_resid.py --config configs/hard_resid.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
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
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.cs_mlp import CSMLPModel  # noqa: E402
from src.models.fusion import FusionModel, linear_blend, panel_cs_rank  # noqa: E402
from src.submit import save_submission  # noqa: E402


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


def _load_base_blend(cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    spec = dict(cfg.get("base_blend") or {})
    valid_path = ROOT / spec.get("valid", "outputs/fusion_cs_mlp_valid.npy")
    test_path = ROOT / spec.get("test", "outputs/fusion_cs_mlp_test.npy")
    if not valid_path.is_file() or not test_path.is_file():
        valid_path = ROOT / spec.get("fallback_valid", "outputs/fusion_gru_blend_valid.npy")
        test_path = ROOT / spec.get("fallback_test", "outputs/fusion_gru_blend_test.npy")
    if not valid_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("need fusion_cs_mlp or fusion_gru_blend valid/test npy")
    baseline_ic = float(spec.get("baseline_ic", 0.111264))
    print(f"base blend valid={valid_path.name} test={test_path.name}")
    return np.load(valid_path), np.load(test_path), baseline_ic


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual MLP on train-hard days")
    parser.add_argument("--config", default="configs/hard_resid.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    feat = dict(cfg.get("features") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    paths = cfg.get("paths") or {}
    tree_spec = dict(cfg.get("tree") or {})
    fill_invalid = float(train_cfg.get("fill_invalid", 0.0))
    hard_q = float(feat.get("hard_quantile", 0.25))
    mlp_cfg = {**feat, **model_cfg, **train_cfg}

    try:
        import torch

        n_threads = int(train_cfg.get("num_threads") or os.cpu_count() or 4)
        torch.set_num_threads(max(1, n_threads))
    except Exception:
        pass

    print(f"config={args.config}")
    print(f"seed={seed} target_mode={feat.get('target_mode')} hard_quantile={hard_q} hard_repeat={feat.get('hard_repeat')}")

    t0 = time.perf_counter()
    data = load_panel(str(resolve_data_path(cfg, args.data)))
    drop_task2_label(data)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    train_start, train_end = split_bounds(data, "train")
    train = slice_split(data, "train")
    valid = slice_split(data, "valid")
    test = slice_split(data, "test")

    tree_cfg = load_config(tree_spec.get("config", "configs/hist_lgbm.yaml"))
    tree = LightGBMBaseline(
        params=(tree_cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(tree_cfg.get("features") or {}),
        seed=int(tree_cfg.get("seed", seed)),
    )
    ckpt = ROOT / tree_spec.get("checkpoint", "checkpoints/hist_lgbm.txt")
    print(f"load tree {ckpt}")
    tree.load(ckpt)
    t1 = time.perf_counter()
    print("predict hist_lgbm on train (hard days are defined here only)")
    train_pred = tree.predict_panel(train, fill_invalid=fill_invalid)
    print(f"train tree predict {time.perf_counter() - t1:.1f}s")
    del tree

    series = rank_ic_series(train_pred, train["y1"], train["mask_y"])
    thr = float(np.nanpercentile(series, 100.0 * hard_q))
    hard_local = np.flatnonzero(np.asarray(series) <= thr)
    hard_days = {int(train_start + i) for i in hard_local.tolist()}
    easy = series[np.asarray(series) > thr]
    print(
        f"train days={series.size} hard={len(hard_days)} thr={thr:.4f} "
        f"hard_mean_RankIC={float(np.nanmean(series[hard_local])):.4f} "
        f"easy_mean_RankIC={float(np.nanmean(easy)) if easy.size else float('nan'):.4f}"
    )
    hard_meta_path = ROOT / "outputs/hard_resid_days.json"
    hard_meta_path.parent.mkdir(parents=True, exist_ok=True)
    hard_meta_path.write_text(
        json.dumps(
            {
                "hard_quantile": hard_q,
                "n_train_days": int(series.size),
                "n_hard": len(hard_days),
                "thr": thr,
                "hard_mean_rankic": float(np.nanmean(series[hard_local])),
                "easy_mean_rankic": float(np.nanmean(easy)) if easy.size else None,
                "hard_local_idx": hard_local.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    aux = np.zeros(data["y1"].shape, dtype=np.float32)
    aux[train_start:train_end] = train_pred
    del train_pred

    base_valid, base_test, baseline_ic = _load_base_blend(cfg)

    def blend_scorer(pred, label, mask_y):
        t_rank = panel_cs_rank(pred, valid["mask_x"])
        c_rank = panel_cs_rank(base_valid, valid["mask_x"])
        blended = linear_blend(t_rank, c_rank, 0.25)
        return mean_rank_ic(blended, label, mask_y)

    model = CSMLPModel(mlp_cfg, seed=seed)
    model.hard_days = hard_days
    model.aux_pred = aux
    t2 = time.perf_counter()
    model.prepare_features(data)
    _drop_heavy_panels(data, train, valid, test)
    print(f"ind cache {time.perf_counter() - t2:.1f}s")

    t3 = time.perf_counter()
    fit_info = model.fit(data, train_start, train_end, valid=valid, scorer=blend_scorer)
    print(f"fit {time.perf_counter() - t3:.1f}s best_blend_valid={fit_info.get('best_valid_ic')}")
    del aux
    model.aux_pred = None

    resid_valid = model.predict_panel(valid, data, fill_invalid=fill_invalid)
    resid_ic = mean_rank_ic(resid_valid, valid["y1"], valid["mask_y"])
    print(f"residual-only valid RankIC={resid_ic:.6f}")
    print(
        f"vs fusion_cs_mlp Spearman={_pairwise_pred_ic(resid_valid, base_valid, valid['mask_x']):.4f}"
    )

    valid_pred_path = ROOT / paths.get("valid_pred", "outputs/hard_resid_valid.npy")
    valid_pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(valid_pred_path, resid_valid)

    resid_test = model.predict_panel(test, data, fill_invalid=fill_invalid)
    np.save(ROOT / paths.get("test_pred", "outputs/hard_resid_test.npy"), resid_test)
    save_submission(resid_test, ROOT / paths.get("submission", "submissions/task1_hard_resid.npy"))
    model.save(ROOT / paths.get("checkpoint", "checkpoints/hard_resid.pt"))

    blender = FusionModel({"weight_grid": [0.0, 0.15, 0.25, 0.4, 0.5, 0.7, 1.0]})
    locked = blender.fit(
        resid_valid,
        base_valid,
        valid["y1"],
        valid["mask_y"],
        valid["mask_x"],
        industry=model.industry[valid["start"] : valid["end"]] if model.industry is not None else None,
    )
    print("blend with 0.111 base:")
    for row in (locked.get("valid_leaderboard") or [])[:6]:
        extra = {k: v for k, v in row.items() if k not in {"valid_leaderboard", "name", "ic"}}
        print(f"  {row['ic']:.6f}  {row['name']}  {extra}")
    print(f"BLEND_LOCKED {locked['name']} ic={locked['ic']:.6f}")

    if float(locked["ic"]) > baseline_ic + 1e-4:
        out_valid = blender.predict(resid_valid, base_valid, valid["mask_x"])
        out_test = blender.predict(resid_test, base_test, test["mask_x"])
        np.save(ROOT / "outputs/fusion_hard_resid_valid.npy", out_valid)
        np.save(ROOT / "outputs/fusion_hard_resid_test.npy", out_test)
        save_submission(out_test, ROOT / "submissions/task1_fusion_hard_resid.npy")
        lock_path = ROOT / "outputs/fusion_hard_resid_lock.json"
        meta = {k: v for k, v in locked.items() if k != "valid_leaderboard"}
        meta["leaderboard"] = locked.get("valid_leaderboard")
        lock_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print("wrote submissions/task1_fusion_hard_resid.npy (did not overwrite 0.111 file)")
    else:
        print(f"blend did not beat {baseline_ic:.6f}; not writing a new main submission")

    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{float(locked['ic']):.6f}")
    print("does not overwrite submissions/task1_fusion_cs_mlp.npy")


if __name__ == "__main__":
    main()
