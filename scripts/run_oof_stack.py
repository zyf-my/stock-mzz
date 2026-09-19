"""OOF RankIC stacking of locked branches. Does not fit on valid.

Walk-forward: trees + MLPs trained on train prefix; frozen GRUs score the next
400 days (before GRU recency). Search a global 3-way mix on that OOF window,
then apply the locked full-model scores on valid/test.

Never overwrites task1_fusion_next6_wt_mlp6.npy.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\run_oof_stack.py
"""

from __future__ import annotations

import gc
import json
import os
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
    load_panel,
    resolve_cat_indices,
    slice_days,
    slice_split,
)
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.cs_mlp import CSMLPModel  # noqa: E402
from src.models.fusion import linear_blend, panel_cs_rank  # noqa: E402
from src.models.gru_ts import GRUModel  # noqa: E402
from src.submit import save_submission  # noqa: E402

LOCKED_VALID = 0.120542
ONLY6_W = 0.4
NEXT6_W = 0.15
MLP6_W = 0.4
TREE_W = 0.7
GRU_RECENT = 800
OOF_DAYS = 400
UPLOAD_BAR = 0.001
WEIGHT_STEP = 0.05

GRU_SPECS = (
    ("x6", "configs/gru_no_today_recent_n2000_x6.yaml", "checkpoints/gru_no_today_recent_n2000_x6.pt"),
    ("only6", "configs/gru_only6_with_today.yaml", "checkpoints/gru_only6_with_today.pt"),
    ("next6", "configs/gru_next6_with_today.yaml", "checkpoints/gru_next6_with_today.pt"),
)
MLP_SPECS = (
    ("mlp", "configs/cs_mlp.yaml"),
    ("mlp6", "configs/cs_mlp_only6.yaml"),
)


def _gru_ens(x6, only6, next6):
    return linear_blend(next6, linear_blend(only6, x6, ONLY6_W), NEXT6_W)


def _mlp_ens(mlp, mlp6):
    return linear_blend(mlp6, mlp, MLP6_W)


def _simplex(step: float = WEIGHT_STEP):
    n = int(round(1.0 / step))
    out = []
    for i in range(n + 1):
        for j in range(n - i + 1):
            wg = i * step
            wt = j * step
            wm = 1.0 - wg - wt
            out.append((float(wg), float(wt), float(round(wm, 10))))
    return out


def _blend3(gru, tree, mlp, wg, wt, wm, mask, space: str):
    if space == "rank":
        gru = panel_cs_rank(gru, mask)
        tree = panel_cs_rank(tree, mask)
        mlp = panel_cs_rank(mlp, mask)
    elif space != "raw":
        raise ValueError(space)
    return (wg * gru + wt * tree + wm * mlp).astype(np.float32)


def _predict_gru_window(data, split, spec) -> np.ndarray:
    tag, cfg_path, ckpt = spec
    cfg = load_config(cfg_path)
    feat = dict(cfg.get("features") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    gru = GRUModel({**feat, **model_cfg, **train_cfg}, seed=int(cfg.get("seed", 42)))
    gru.load(ROOT / ckpt)
    gru.prepare_features(data)
    pred = gru.predict_panel(split, data, fill_invalid=0.0)
    print(f"  {tag} OOF RankIC={mean_rank_ic(pred, split['y1'], split['mask_y']):.6f}", flush=True)
    gru.cs_sel = None
    gru.net = None
    gc.collect()
    return pred


def _train_tree(data, fit_split, pred_split, cfg_path: str, tag: str) -> np.ndarray:
    cfg = load_config(cfg_path)
    feature_cfg = dict(cfg.get("features") or {})
    cat_indices = resolve_cat_indices(feature_cfg)
    seed = int(cfg.get("seed", 42))
    print(f"flatten {tag} days={fit_split['num_x'].shape[0]}", flush=True)
    x, y, _ = flatten_masked_rows(
        fit_split,
        cat_indices=cat_indices,
        require_label=True,
        feature_cfg=feature_cfg,
        seed=seed,
    )
    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=feature_cfg,
        seed=seed,
    )
    t0 = time.perf_counter()
    model.fit(x, y)
    print(f"  {tag} fit {time.perf_counter() - t0:.1f}s rows={x.shape[0]}", flush=True)
    del x, y
    gc.collect()
    pred = model.predict_panel(pred_split, fill_invalid=0.0)
    print(f"  {tag} OOF RankIC={mean_rank_ic(pred, pred_split['y1'], pred_split['mask_y']):.6f}", flush=True)
    return pred


def _train_mlp_oof(data, prefix_start: int, prefix_end: int, oof, cfg_path: str, tag: str) -> np.ndarray:
    cfg = load_config(cfg_path)
    feat = dict(cfg.get("features") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    mlp = CSMLPModel({**feat, **model_cfg, **train_cfg}, seed=int(cfg.get("seed", 42)))
    mlp.prepare_features(data)
    t0 = time.perf_counter()
    # No valid split: do not early-stop or select epochs on valid / OOF labels.
    mlp.fit(data, prefix_start, prefix_end, valid=None)
    print(f"  {tag} fit {time.perf_counter() - t0:.1f}s", flush=True)
    pred = mlp.predict_panel(oof, data, fill_invalid=0.0)
    print(f"  {tag} OOF RankIC={mean_rank_ic(pred, oof['y1'], oof['mask_y']):.6f}", flush=True)
    mlp.ind_sel = None
    mlp.net = None
    gc.collect()
    return pred


def _search(gru, tree, mlp, y, my, mx) -> dict:
    best = {"ic": -1e9}
    for space in ("rank", "raw"):
        if space == "rank":
            g, t, m = panel_cs_rank(gru, mx), panel_cs_rank(tree, mx), panel_cs_rank(mlp, mx)
        else:
            g, t, m = gru, tree, mlp
        for wg, wt, wm in _simplex():
            pred = (wg * g + wt * t + wm * m).astype(np.float32)
            ic = mean_rank_ic(pred, y, my)
            if np.isfinite(ic) and ic > best["ic"]:
                best = {"ic": float(ic), "wg": wg, "wt": wt, "wm": wm, "space": space}
        print(f"  best so far after {space}: ic={best['ic']:.6f} space={best['space']}", flush=True)
    return best


def main() -> None:
    try:
        import torch

        torch.set_num_threads(max(1, os.cpu_count() or 4))
    except Exception:
        pass

    t0 = time.perf_counter()
    dest = ROOT / "outputs" / "oof"
    dest.mkdir(parents=True, exist_ok=True)

    gru_path = dest / "gru_ens_oof.npy"
    tree_path = dest / "tree_oof.npy"
    mlp_path = dest / "mlp_ens_oof.npy"

    cfg = load_config("configs/default.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_task2_label(data)
    train = slice_split(data, "train")
    n_train = int(train["num_x"].shape[0])
    gru_fit_local = n_train - GRU_RECENT
    oof_local0 = gru_fit_local - OOF_DAYS
    print(
        f"train days={n_train} prefix=[0,{oof_local0}) oof=[{oof_local0},{gru_fit_local}) "
        f"gru_fit=[{gru_fit_local},{n_train})"
    )
    prefix = slice_days(train, 0, oof_local0)
    oof = slice_days(train, oof_local0, gru_fit_local)
    prefix_start = int(prefix["start"])
    prefix_end = prefix_start + int(prefix["num_x"].shape[0])
    print(f"prefix global=[{prefix_start},{prefix_end}) oof global=[{oof['start']},{oof['start'] + oof['num_x'].shape[0]})")

    if gru_path.is_file():
        gru_oof = np.load(gru_path)
        print(f"reuse {gru_path.name} RankIC={mean_rank_ic(gru_oof, oof['y1'], oof['mask_y']):.6f}")
    else:
        print("GRU OOF preds (frozen, out of recency window)", flush=True)
        gru_preds = {}
        for spec in GRU_SPECS:
            gru_preds[spec[0]] = _predict_gru_window(data, oof, spec)
            np.save(dest / f"gru_{spec[0]}_oof.npy", gru_preds[spec[0]])
        gru_oof = _gru_ens(gru_preds["x6"], gru_preds["only6"], gru_preds["next6"])
        np.save(gru_path, gru_oof)
        print(f"GRU ens OOF RankIC={mean_rank_ic(gru_oof, oof['y1'], oof['mask_y']):.6f}")
        del gru_preds
        gc.collect()

    if tree_path.is_file():
        tree_oof = np.load(tree_path)
        print(f"reuse {tree_path.name} RankIC={mean_rank_ic(tree_oof, oof['y1'], oof['mask_y']):.6f}")
    else:
        print("retrain trees on prefix only", flush=True)
        hist_oof = _train_tree(data, prefix, oof, "configs/hist_lgbm.yaml", "hist_lgbm")
        base_oof = _train_tree(data, prefix, oof, "configs/baseline.yaml", "baseline")
        tree_oof = linear_blend(hist_oof, base_oof, TREE_W)
        np.save(tree_path, tree_oof)
        print(f"tree blend OOF RankIC={mean_rank_ic(tree_oof, oof['y1'], oof['mask_y']):.6f}")
        del hist_oof, base_oof
        gc.collect()

    if mlp_path.is_file():
        mlp_oof = np.load(mlp_path)
        print(f"reuse {mlp_path.name} RankIC={mean_rank_ic(mlp_oof, oof['y1'], oof['mask_y']):.6f}")
    else:
        print("retrain MLPs on prefix only (no valid early-stop)", flush=True)
        mlp_preds = {}
        for tag, cfg_path in MLP_SPECS:
            mlp_preds[tag] = _train_mlp_oof(data, prefix_start, prefix_end, oof, cfg_path, tag)
            np.save(dest / f"{tag}_oof.npy", mlp_preds[tag])
        mlp_oof = _mlp_ens(mlp_preds["mlp"], mlp_preds["mlp6"])
        np.save(mlp_path, mlp_oof)
        print(f"MLP ens OOF RankIC={mean_rank_ic(mlp_oof, oof['y1'], oof['mask_y']):.6f}")
        del mlp_preds
        gc.collect()

    y, my, mx = oof["y1"], oof["mask_y"], oof["mask_x"]
    print(
        f"OOF singles gru={mean_rank_ic(gru_oof, y, my):.6f} "
        f"tree={mean_rank_ic(tree_oof, y, my):.6f} "
        f"mlp={mean_rank_ic(mlp_oof, y, my):.6f}"
    )
    print("search 3-way mix on OOF (not valid)", flush=True)
    best = _search(gru_oof, tree_oof, mlp_oof, y, my, mx)
    print(
        f"OOF best stack RankIC={best['ic']:.6f} "
        f"w=({best['wg']:.2f},{best['wt']:.2f},{best['wm']:.2f}) space={best['space']}"
    )

    del data, train, prefix, oof, gru_oof, tree_oof, mlp_oof
    gc.collect()

    from src.dataset import load_eval_splits  # noqa: E402

    splits, src = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    print(f"eval cache={src}")
    x6_v = np.load(ROOT / "outputs/gru_no_today_recent_n2000_x6_valid.npy")
    o6_v = np.load(ROOT / "outputs/gru_only6_with_today_valid.npy")
    n6_v = np.load(ROOT / "outputs/gru_next6_with_today_valid.npy")
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp6_v = np.load(ROOT / "outputs/cs_mlp_only6_valid.npy")
    x6_t = np.load(ROOT / "outputs/gru_no_today_recent_n2000_x6_test.npy")
    o6_t = np.load(ROOT / "outputs/gru_only6_with_today_test.npy")
    n6_t = np.load(ROOT / "outputs/gru_next6_with_today_test.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
    mlp6_t = np.load(ROOT / "outputs/cs_mlp_only6_test.npy")

    gru_v = _gru_ens(x6_v, o6_v, n6_v)
    gru_t = _gru_ens(x6_t, o6_t, n6_t)
    mlp_ens_v = _mlp_ens(mlp_v, mlp6_v)
    mlp_ens_t = _mlp_ens(mlp_t, mlp6_t)
    out_v = _blend3(gru_v, tree_v, mlp_ens_v, best["wg"], best["wt"], best["wm"], valid["mask_x"], best["space"])
    out_t = _blend3(gru_t, tree_t, mlp_ens_t, best["wg"], best["wt"], best["wm"], test["mask_x"], best["space"])
    ic = mean_rank_ic(out_v, valid["y1"], valid["mask_y"])
    series = rank_ic_series(out_v, valid["y1"], valid["mask_y"])
    print(
        f"valid stack RankIC={ic:.6f} delta={ic - LOCKED_VALID:+.6f} "
        f"min={np.nanmin(series):.4f} neg={int((series < 0).sum())}"
    )

    meta = {
        "locked": LOCKED_VALID,
        "valid_ic": ic,
        "oof_ic": best["ic"],
        "weights": {"gru": best["wg"], "tree": best["wt"], "mlp": best["wm"]},
        "space": best["space"],
        "oof_days": OOF_DAYS,
        "gru_recent": GRU_RECENT,
        "beat": bool(ic > LOCKED_VALID + 1e-4),
        "upload_ok": bool(ic > LOCKED_VALID + UPLOAD_BAR),
    }
    (dest / "stack_lock.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if ic > LOCKED_VALID + 1e-4:
        np.save(ROOT / "outputs/fusion_next6_oof_stack_valid.npy", out_v)
        np.save(ROOT / "outputs/fusion_next6_oof_stack_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions/task1_fusion_next6_oof_stack.npy")
        print("wrote submissions/task1_fusion_next6_oof_stack.npy")
        print("does not overwrite submissions/task1_fusion_next6_wt_mlp6.npy")
        if ic <= LOCKED_VALID + UPLOAD_BAR:
            print(f"delta < {UPLOAD_BAR}; do not upload")
    else:
        print("did not beat locked 0.120542; not writing a new main submission")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
