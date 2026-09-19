"""OOF day router on top of the locked next6 fusion. Does not fit on valid.

Train trees on train prefix (before GRU recency). Fit the router on the next
400 days, where GRUs (trained on last 800) are out-of-sample.
At inference, apply the router to the locked full models. Never overwrites
task1_fusion_next6_wt_mlp6.npy.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\run_oof_router.py
"""

from __future__ import annotations

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
from src.models.fusion import FusionModel, linear_blend  # noqa: E402
from src.models.gru_ts import GRUModel  # noqa: E402
from src.submit import save_submission  # noqa: E402

LOCKED_VALID = 0.120542
ONLY6_W = 0.4
NEXT6_W = 0.15
MLP6_W = 0.4
MLP_STACK_W = 0.15
TREE_W = 0.7
GRU_RECENT = 800
OOF_DAYS = 400
W_GRID = [0.0, 0.15, 0.25, 0.4, 0.6, 0.85, 1.0]
UPLOAD_BAR = 0.001

GRU_SPECS = (
    ("x6", "configs/gru_no_today_recent_n2000_x6.yaml", "checkpoints/gru_no_today_recent_n2000_x6.pt"),
    ("only6", "configs/gru_only6_with_today.yaml", "checkpoints/gru_only6_with_today.pt"),
    ("next6", "configs/gru_next6_with_today.yaml", "checkpoints/gru_next6_with_today.pt"),
)


def _gru_ens(x6, only6, next6):
    return linear_blend(next6, linear_blend(only6, x6, ONLY6_W), NEXT6_W)


def _day_disagree(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape[0], dtype=np.float64)
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(a[t]) & np.isfinite(b[t])
        if int(m.sum()) < 8:
            continue
        r = spearmanr(a[t, m], b[t, m]).statistic
        out[t] = 1.0 - float(r) if np.isfinite(r) else 1.0
    return out


def _features(coverage: np.ndarray, disagree: np.ndarray) -> np.ndarray:
    cov = (coverage - coverage.mean()) / (coverage.std() + 1e-8)
    dis = (disagree - disagree.mean()) / (disagree.std() + 1e-8)
    return np.column_stack([np.ones(coverage.shape[0]), cov, dis, cov * dis])


def _best_w_series(gru, tree, y, mask) -> np.ndarray:
    n = gru.shape[0]
    best = np.zeros(n, dtype=np.float64)
    for t in range(n):
        scores = []
        for w in W_GRID:
            pred = w * gru[t] + (1.0 - w) * tree[t]
            ic = mean_rank_ic(pred[None, :], y[t : t + 1], mask[t : t + 1])
            scores.append(ic if np.isfinite(ic) else -1.0)
        best[t] = W_GRID[int(np.argmax(scores))]
    return best


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
    ic = mean_rank_ic(pred, split["y1"], split["mask_y"])
    print(f"  {tag} OOF RankIC={ic:.6f}", flush=True)
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


def main() -> None:
    try:
        import torch

        torch.set_num_threads(max(1, os.cpu_count() or 4))
    except Exception:
        pass

    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_task2_label(data)
    train = slice_split(data, "train")
    n_train = int(train["num_x"].shape[0])
    gru_fit_local = n_train - GRU_RECENT
    oof_local0 = gru_fit_local - OOF_DAYS
    print(f"train days={n_train} prefix=[0,{oof_local0}) oof=[{oof_local0},{gru_fit_local}) gru_fit=[{gru_fit_local},{n_train})")
    prefix = slice_days(train, 0, oof_local0)
    oof = slice_days(train, oof_local0, gru_fit_local)
    print(f"prefix global=[{prefix['start']},{prefix['start'] + prefix['num_x'].shape[0]})")
    print(f"oof global=[{oof['start']},{oof['start'] + oof['num_x'].shape[0]})")

    dest = ROOT / "outputs" / "oof"
    dest.mkdir(parents=True, exist_ok=True)

    print("GRU OOF preds (frozen, out of recency window)", flush=True)
    gru_preds = {}
    for spec in GRU_SPECS:
        gru_preds[spec[0]] = _predict_gru_window(data, oof, spec)
        np.save(dest / f"gru_{spec[0]}_oof.npy", gru_preds[spec[0]])
    gru_oof = _gru_ens(gru_preds["x6"], gru_preds["only6"], gru_preds["next6"])
    np.save(dest / "gru_ens_oof.npy", gru_oof)
    print(f"GRU ens OOF RankIC={mean_rank_ic(gru_oof, oof['y1'], oof['mask_y']):.6f}")

    print("retrain trees on prefix only", flush=True)
    hist_oof = _train_tree(data, prefix, oof, "configs/hist_lgbm.yaml", "hist_lgbm")
    base_oof = _train_tree(data, prefix, oof, "configs/baseline.yaml", "baseline")
    tree_oof = linear_blend(hist_oof, base_oof, TREE_W)
    np.save(dest / "tree_oof.npy", tree_oof)
    print(f"tree blend OOF RankIC={mean_rank_ic(tree_oof, oof['y1'], oof['mask_y']):.6f}")

    y, my, mx = oof["y1"], oof["mask_y"], oof["mask_x"]
    coverage = np.asarray(mx).sum(axis=1).astype(np.float64)
    disagree = _day_disagree(gru_oof, tree_oof, mx)
    best_w = _best_w_series(gru_oof, tree_oof, y, my)
    x_oof = _features(coverage, disagree)
    beta, *_ = np.linalg.lstsq(x_oof, best_w, rcond=None)
    w_hat_oof = np.clip(x_oof @ beta, 0.0, 1.0)
    routed_oof = np.empty_like(gru_oof)
    for t in range(gru_oof.shape[0]):
        routed_oof[t] = linear_blend(gru_oof[t], tree_oof[t], float(w_hat_oof[t]))
    oracle_oof = np.empty_like(gru_oof)
    for t in range(gru_oof.shape[0]):
        oracle_oof[t] = linear_blend(gru_oof[t], tree_oof[t], float(best_w[t]))
    print(f"OOF oracle-w RankIC={mean_rank_ic(oracle_oof, y, my):.6f}")
    print(f"OOF router RankIC={mean_rank_ic(routed_oof, y, my):.6f}")
    print(f"OOF mean predicted w={float(w_hat_oof.mean()):.3f}  mean oracle w={float(best_w.mean()):.3f}")
    print(f"router beta={beta.tolist()}")

    del data, train, prefix, oof, gru_preds, hist_oof, base_oof
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

    def _apply(gru, tree, mlp, mlp6, mask):
        cov = np.asarray(mask).sum(axis=1).astype(np.float64)
        dis = _day_disagree(gru, tree, mask)
        # freeze OOF feature scaling
        cov_z = (cov - coverage.mean()) / (coverage.std() + 1e-8)
        dis_z = (dis - disagree.mean()) / (disagree.std() + 1e-8)
        x = np.column_stack([np.ones(cov.shape[0]), cov_z, dis_z, cov_z * dis_z])
        w = np.clip(x @ beta, 0.0, 1.0)
        gated = np.empty_like(gru)
        for t in range(gru.shape[0]):
            gated[t] = linear_blend(gru[t], tree[t], float(w[t]))
        mlp_ens = linear_blend(mlp6, mlp, MLP6_W)
        blender = FusionModel({"weight_grid": [MLP_STACK_W]})
        blender.locked = {"name": "rank_blend", "weight": MLP_STACK_W, "space": "rank", "ic": float("nan")}
        return blender.predict(mlp_ens, gated, mask), w

    gru_v = _gru_ens(x6_v, o6_v, n6_v)
    gru_t = _gru_ens(x6_t, o6_t, n6_t)
    out_v, w_v = _apply(gru_v, tree_v, mlp_v, mlp6_v, valid["mask_x"])
    out_t, _ = _apply(gru_t, tree_t, mlp_t, mlp6_t, test["mask_x"])
    ic = mean_rank_ic(out_v, valid["y1"], valid["mask_y"])
    series = rank_ic_series(out_v, valid["y1"], valid["mask_y"])
    print(
        f"valid router RankIC={ic:.6f} delta={ic - LOCKED_VALID:+.6f} "
        f"min={np.nanmin(series):.4f} neg={int((series < 0).sum())} "
        f"mean_w={float(w_v.mean()):.3f}"
    )

    meta = {
        "locked": LOCKED_VALID,
        "valid_ic": ic,
        "beta": [float(x) for x in beta],
        "oof_days": OOF_DAYS,
        "gru_recent": GRU_RECENT,
        "mean_w_valid": float(w_v.mean()),
        "beat": bool(ic > LOCKED_VALID + 1e-4),
        "upload_ok": bool(ic > LOCKED_VALID + UPLOAD_BAR),
    }
    (dest / "router_lock.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if ic > LOCKED_VALID + 1e-4:
        np.save(ROOT / "outputs/fusion_next6_oof_router_valid.npy", out_v)
        np.save(ROOT / "outputs/fusion_next6_oof_router_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions/task1_fusion_next6_oof_router.npy")
        print("wrote submissions/task1_fusion_next6_oof_router.npy")
        print("does not overwrite submissions/task1_fusion_next6_wt_mlp6.npy")
        if ic <= LOCKED_VALID + UPLOAD_BAR:
            print(f"delta < {UPLOAD_BAR}; do not upload")
    else:
        print("did not beat locked 0.120542; not writing a new main submission")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
