"""Train-only coverage quartile bucket fusion (Plan B). Valid never used for fitting.

1. Quartile edges from train coverage (all days or recent window).
2. Per-quartile GRU weight w_q chosen on train holdout days only (grid search).
3. Apply locked next6 GRU ens + mlp6 stack on valid/test.

Does not overwrite task1_fusion_next6_wt_mlp6.npy.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_regime_bucket_train.py
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_regime_bucket_train.py --skip-train-pred
"""

from __future__ import annotations

import argparse
import gc
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
    load_panel,
    load_split_cache,
    slice_days,
    slice_split,
    split_cache_dir,
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
FIT_RECENT = 800
W_GRID = [0.0, 0.15, 0.25, 0.4, 0.6, 0.85, 1.0]
UPLOAD_BAR = 0.001

GRU_SPECS = (
    ("x6", "configs/gru_no_today_recent_n2000_x6.yaml", "checkpoints/gru_no_today_recent_n2000_x6.pt"),
    ("only6", "configs/gru_only6_with_today.yaml", "checkpoints/gru_only6_with_today.pt"),
    ("next6", "configs/gru_next6_with_today.yaml", "checkpoints/gru_next6_with_today.pt"),
)


def _gru_ens(x6, o6, n6):
    return linear_blend(n6, linear_blend(o6, x6, ONLY6_W), NEXT6_W)


def _quartile_edges(coverage: np.ndarray) -> np.ndarray:
    return np.percentile(coverage, [0, 25, 50, 75, 100])


def _quartile_index(coverage: np.ndarray, edges: np.ndarray) -> np.ndarray:
    q = np.zeros(coverage.shape[0], dtype=np.int32)
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        if i < 3:
            q[(coverage >= lo) & (coverage < hi)] = i
        else:
            q[coverage >= lo] = i  # top bucket includes all crowded days above Q4 lo
    return q


def _apply_bucket_blend(gru, tree, mask_x, edges: np.ndarray, weights: list[float]) -> np.ndarray:
    cov = np.asarray(mask_x).sum(axis=1).astype(np.float64)
    q = _quartile_index(cov, edges)
    out = np.empty_like(gru, dtype=np.float32)
    for t in range(gru.shape[0]):
        w = float(weights[int(q[t])])
        out[t] = linear_blend(gru[t], tree[t], w)
    return out


def _fit_weights_per_quartile(
    gru,
    tree,
    y,
    my,
    mx,
    edges: np.ndarray,
) -> tuple[list[float], list[float]]:
    cov = np.asarray(mx).sum(axis=1).astype(np.float64)
    q = _quartile_index(cov, edges)
    weights: list[float] = []
    train_ics: list[float] = []
    for qi in range(4):
        days = q == qi
        if int(days.sum()) < 8:
            weights.append(0.4)
            train_ics.append(float("nan"))
            continue
        best_w, best_ic = W_GRID[0], -1e9
        for w in W_GRID:
            pred = np.empty_like(gru)
            for t in np.flatnonzero(days):
                pred[t] = linear_blend(gru[t], tree[t], w)
            ic = mean_rank_ic(pred[days], y[days], my[days])
            if np.isfinite(ic) and ic > best_ic:
                best_ic, best_w = float(ic), float(w)
        weights.append(best_w)
        train_ics.append(best_ic)
    return weights, train_ics


def _predict_gru(data, split, spec) -> np.ndarray:
    tag, cfg_path, ckpt = spec
    from src.config import load_config

    cfg = load_config(cfg_path)
    feat = dict(cfg.get("features") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    gru = GRUModel({**feat, **model_cfg, **train_cfg}, seed=int(cfg.get("seed", 42)))
    gru.load(ROOT / ckpt)
    gru.prepare_features(data)
    pred = gru.predict_panel(split, data, fill_invalid=0.0)
    print(f"  {tag} train-fit RankIC={mean_rank_ic(pred, split['y1'], split['mask_y']):.6f}", flush=True)
    gru.cs_sel = None
    gru.net = None
    gc.collect()
    return pred


def _predict_tree_blend(data, split) -> np.ndarray:
    hist = LightGBMBaseline({}, feature_cfg=dict(load_config("configs/hist_lgbm.yaml").get("features") or {}))
    hist.load(ROOT / "checkpoints/hist_lgbm.txt")
    base = LightGBMBaseline({}, feature_cfg=dict(load_config("configs/baseline.yaml").get("features") or {}))
    base.load(ROOT / "checkpoints/baseline.txt")
    fill = 0.0
    h = hist.predict_panel(split, fill_invalid=fill)
    b = base.predict_panel(split, fill_invalid=fill)
    out = linear_blend(h, b, TREE_W)
    print(f"  tree blend train-fit RankIC={mean_rank_ic(out, split['y1'], split['mask_y']):.6f}", flush=True)
    return out


def _ensure_train_preds(force: bool = False) -> Path:
    dest = ROOT / "outputs" / "regime_bucket"
    dest.mkdir(parents=True, exist_ok=True)
    tag = dest / "train_fit_800.npz"
    if tag.is_file() and not force:
        return tag

    cfg = load_config("configs/default.yaml")
    t0 = time.perf_counter()
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_task2_label(data)
    train = slice_split(data, "train")
    n = int(train["num_x"].shape[0])
    fit = slice_days(train, max(0, n - FIT_RECENT), n)
    print(f"train fit window days={fit['num_x'].shape[0]} global=[{fit['start']},{fit['start'] + fit['num_x'].shape[0]})")

    tree = _predict_tree_blend(data, fit)
    x6 = _predict_gru(data, fit, GRU_SPECS[0])
    o6 = _predict_gru(data, fit, GRU_SPECS[1])
    n6 = _predict_gru(data, fit, GRU_SPECS[2])
    gru = _gru_ens(x6, o6, n6)
    np.savez(
        tag,
        gru=gru,
        tree=tree,
        y1=fit["y1"],
        mask_y=fit["mask_y"],
        mask_x=fit["mask_x"],
        edges_source="train_all",
    )
    print(f"wrote {tag} in {time.perf_counter() - t0:.1f}s")
    del data, train, fit, x6, o6, n6
    gc.collect()
    return tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train-pred", action="store_true")
    parser.add_argument("--force-train-pred", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache = split_cache_dir(ROOT)
    train_npz = cache / "train.npz"
    if not train_npz.is_file():
        print("train cache missing; loading panel once to dump train labels", flush=True)
        data = load_panel(str(resolve_data_path(load_config("configs/default.yaml"), None)))
        drop_task2_label(data)
        from src.dataset import dump_split_cache

        dump_split_cache(data, cache, splits=("train",))
        del data
        gc.collect()

    train = load_split_cache("train", cache)
    train_cov = np.asarray(train["mask_x"]).sum(axis=1).astype(np.float64)
    edges = _quartile_edges(train_cov)
    print(f"train quartile edges (all {train_cov.size} days): {edges.astype(int).tolist()}")

    if args.skip_train_pred:
        fit_path = ROOT / "outputs/regime_bucket/train_fit_800.npz"
        if not fit_path.is_file():
            raise FileNotFoundError("run without --skip-train-pred first")
    else:
        fit_path = _ensure_train_preds(force=args.force_train_pred)

    fit = dict(np.load(fit_path))
    gru_f = fit["gru"]
    tree_f = fit["tree"]
    y_f, my_f, mx_f = fit["y1"], fit["mask_y"], fit["mask_x"]
    fit_cov = np.asarray(mx_f).sum(axis=1).astype(np.float64)
    edges_fit = _quartile_edges(fit_cov)
    print(f"quartile edges from train-fit {fit_cov.size} days: {edges_fit.astype(int).tolist()}")
    print(f"quartile edges from all-train {train_cov.size} days: {edges.astype(int).tolist()}")

    for label, ed in (("fit_window", edges_fit), ("all_train", edges)):
        weights, train_q_ic = _fit_weights_per_quartile(gru_f, tree_f, y_f, my_f, mx_f, ed)
        print(f"\n=== bucket weights fitted on train-fit 800d, edges={label} ===")
        for i, (w, ic) in enumerate(zip(weights, train_q_ic)):
            lo, hi = ed[i], ed[i + 1]
            print(f"  Q{i + 1} [{lo:.0f},{hi:.0f}] w={w:.2f} train_RankIC={ic}")

    # Use fit-window edges (same regime as GRU recency); top bucket absorbs valid's higher coverage.
    edges = edges_fit
    weights, train_q_ic = _fit_weights_per_quartile(gru_f, tree_f, y_f, my_f, mx_f, edges)

    from src.dataset import load_eval_splits  # noqa: E402

    splits, src = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    print(f"eval {src}")

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
    gated_v = _apply_bucket_blend(gru_v, tree_v, mx, edges, weights)
    gated_t = _apply_bucket_blend(gru_t, tree_t, test["mask_x"], edges, weights)
    gated_ic = mean_rank_ic(gated_v, y, my)
    print(f"bucket gate valid RankIC={gated_ic:.6f}")

    cov_v = np.asarray(mx).sum(axis=1).astype(np.float64)
    q_v = _quartile_index(cov_v, edges)
    s = rank_ic_series(gated_v, y, my)
    print("valid per-quartile RankIC after bucket gate:")
    for qi in range(4):
        sel = q_v == qi
        if int(sel.sum()) == 0:
            continue
        print(f"  Q{qi + 1} n={int(sel.sum())} ic={float(np.nanmean(s[sel])):.4f} w={weights[qi]:.2f}")

    mlp_ens_v = linear_blend(mlp6_v, mlp_v, MLP6_W)
    mlp_ens_t = linear_blend(mlp6_t, mlp_t, MLP6_W)
    blender = FusionModel({"weight_grid": [MLP_STACK_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_STACK_W, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_ens_v, gated_v, mx)
    out_t = blender.predict(mlp_ens_t, gated_t, test["mask_x"])
    ic = mean_rank_ic(out_v, y, my)
    series = rank_ic_series(out_v, y, my)
    print(
        f"stack valid RankIC={ic:.6f} delta={ic - LOCKED_VALID:+.6f} "
        f"min={np.nanmin(series):.4f} neg={int((series < 0).sum())}"
    )
    print(f"locked 2-level gate ref=0.120542")

    meta = {
        "locked": LOCKED_VALID,
        "valid_ic": ic,
        "edges": [float(x) for x in edges],
        "weights": weights,
        "train_q_ic": train_q_ic,
        "fit_recent": FIT_RECENT,
        "beat": bool(ic > LOCKED_VALID + 1e-4),
        "upload_ok": bool(ic > LOCKED_VALID + UPLOAD_BAR),
    }
    lock_path = ROOT / "outputs/regime_bucket/lock.json"
    lock_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if ic > LOCKED_VALID + 1e-4:
        np.save(ROOT / "outputs/fusion_regime_bucket_valid.npy", out_v)
        np.save(ROOT / "outputs/fusion_regime_bucket_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions/task1_fusion_regime_bucket.npy")
        print("wrote submissions/task1_fusion_regime_bucket.npy")
        if ic <= LOCKED_VALID + UPLOAD_BAR:
            print(f"delta < {UPLOAD_BAR}; do not upload")
    else:
        print("did not beat locked 0.120542")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
