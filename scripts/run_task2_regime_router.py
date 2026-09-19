"""Route tree vs GRU by day-constant cols 17-25. No coverage. Fit on train.

Does not overwrite task2_fusion_rank_gate.npy unless valid beats it.
"""

from __future__ import annotations

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
    drop_other_label,
    load_eval_splits,
    load_panel,
    load_split_cache,
    slice_split,
    split_bounds,
    split_label_array,
)
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402
from src.models.gru_ts import GRUModel  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
REGIME_PATH = OUT / "day_regime.npy"
LOCKED_IC = 0.086225
REGIME_COLS = list(range(17, 26))
TREE_TAU, TREE_W_LO, TREE_W_HI = 4491.0, 0.5, 0.75
MLP_W = 0.15


def _extract_regime(num_x: np.ndarray, mask_x: np.ndarray) -> np.ndarray:
    t_len = int(num_x.shape[0])
    out = np.zeros((t_len, len(REGIME_COLS)), dtype=np.float32)
    for t in range(t_len):
        m = np.asarray(mask_x[t], dtype=bool)
        if not m.any():
            continue
        out[t] = np.asarray(num_x[t, m][:, REGIME_COLS], dtype=np.float32).mean(axis=0)
    return out


def _rank_blend_daily(temporal, cs, weights, mask):
    rt = panel_cs_rank(temporal, mask)
    rc = panel_cs_rank(cs, mask)
    w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
    return (w * rt + (1.0 - w) * rc).astype(np.float32)


def _daily_ic(pred, y, my):
    return np.asarray(rank_ic_series(pred, y, my), dtype=np.float64)


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """y is (n, 2). Returns (n_feat+1, 2) including intercept."""
    n = x.shape[0]
    xb = np.concatenate([np.ones((n, 1), dtype=np.float64), x], axis=1)
    xtx = xb.T @ xb
    xtx[np.diag_indices_from(xtx)] += float(l2)
    xtx[0, 0] -= float(l2)
    return np.linalg.solve(xtx, xb.T @ y)


def _predict_ridge(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return xb @ coef


def _standardize(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return mu.astype(np.float64), sd.astype(np.float64)


def _finite_days(*series: np.ndarray) -> np.ndarray:
    ok = np.ones(series[0].shape[0], dtype=bool)
    for s in series:
        ok &= np.isfinite(s)
    return ok


def _eval_weights(tree, gru, w, y, my, mx, mlp_ens=None):
    routed = _rank_blend_daily(gru, tree, w, mx)
    if mlp_ens is None:
        return float(mean_rank_ic(routed, y, my)), routed
    blender = FusionModel({"weight_grid": [MLP_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_W, "space": "rank", "ic": float("nan")}
    out = blender.predict(mlp_ens, routed, mx)
    return float(mean_rank_ic(out, y, my)), out


def _make_weights(delta: np.ndarray, kind: str, temp: float) -> np.ndarray:
    if kind == "hard":
        return (delta > 0).astype(np.float32)
    z = np.clip(delta / max(temp, 1e-6), -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _dump_regime_and_train_preds() -> None:
    need_regime = not REGIME_PATH.is_file()
    need_train = not all(
        (OUT / name).is_file()
        for name in (
            "baseline_train.npy",
            "hist_lgbm_train.npy",
            "gru_x6_with_today_train.npy",
            "gru_only6_with_today_train.npy",
            "gru_next6_with_today_train.npy",
        )
    )
    if not need_regime and not need_train:
        print("regime + train preds already on disk")
        return

    cfg = load_config("configs/task2/baseline.yaml")
    data_path = resolve_data_path(cfg, None)
    print(f"data={data_path}")
    t0 = time.perf_counter()
    data = load_panel(str(data_path))
    drop_other_label(data, "y2")
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    if need_regime:
        regime = _extract_regime(data["num_x"], data["mask_x"])
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(REGIME_PATH, regime)
        print(f"wrote {REGIME_PATH} {regime.shape}")
    else:
        print(f"reuse {REGIME_PATH}")

    if not need_train:
        del data
        gc.collect()
        return

    train = slice_split(data, "train")
    print(f"predict train days={train['mask_x'].shape[0]}")

    tree_cfgs = [
        ("configs/task2/baseline.yaml", "baseline_train.npy"),
        ("configs/task2/hist_lgbm.yaml", "hist_lgbm_train.npy"),
    ]
    for cfg_path, out_name in tree_cfgs:
        dest = OUT / out_name
        if dest.is_file():
            print(f"reuse {dest}")
            continue
        tcfg = load_config(cfg_path)
        model = LightGBMBaseline(
            params=(tcfg.get("model") or {}).get("params") or {},
            feature_cfg=dict(tcfg.get("features") or {}),
            seed=int(tcfg.get("seed", 42)),
        )
        ckpt = ROOT / (tcfg.get("paths") or {})["checkpoint"]
        print(f"tree predict {ckpt}")
        t1 = time.perf_counter()
        model.load(ckpt)
        pred = model.predict_panel(train, fill_invalid=0.0)
        np.save(dest, pred)
        print(f"  wrote {dest} {pred.shape} in {time.perf_counter() - t1:.1f}s")
        del model, pred
        gc.collect()

    gru_cfgs = [
        "configs/task2/gru_x6_with_today.yaml",
        "configs/task2/gru_only6_with_today.yaml",
        "configs/task2/gru_next6_with_today.yaml",
    ]
    for cfg_path in gru_cfgs:
        tcfg = load_config(cfg_path)
        dest = ROOT / (tcfg.get("paths") or {})["valid_pred"]
        dest = OUT / dest.name.replace("_valid.npy", "_train.npy")
        if dest.is_file():
            print(f"reuse {dest}")
            continue
        feat = dict(tcfg.get("features") or {})
        model_cfg = dict(tcfg.get("model") or {})
        train_cfg = dict(tcfg.get("train") or {})
        gru_cfg = {**feat, **model_cfg, **train_cfg, "label_key": "y2"}
        model = GRUModel(gru_cfg, seed=int(tcfg.get("seed", 42)))
        ckpt = ROOT / (tcfg.get("paths") or {})["checkpoint"]
        print(f"gru load {ckpt}")
        t1 = time.perf_counter()
        model.load(ckpt)
        model.prepare_features(data)
        pred = model.predict_panel(train, data, fill_invalid=0.0)
        np.save(dest, pred)
        print(f"  wrote {dest} {pred.shape} in {time.perf_counter() - t1:.1f}s")
        model.cs_sel = None
        del model, pred
        gc.collect()

    del data, train
    gc.collect()


def _assemble_tree_gru(prefix: str, mask_x):
    hist = np.load(OUT / f"hist_lgbm_{prefix}.npy")
    base = np.load(OUT / f"baseline_{prefix}.npy")
    x6 = np.load(OUT / f"gru_x6_with_today_{prefix}.npy")
    o6 = np.load(OUT / f"gru_only6_with_today_{prefix}.npy")
    n6 = np.load(OUT / f"gru_next6_with_today_{prefix}.npy")
    tree = coverage_gate_blend(hist, base, mask_x, TREE_TAU, TREE_W_LO, TREE_W_HI, "rank")
    ens = linear_blend(n6, linear_blend(o6, x6, 0.4), 0.15)
    return tree, ens


def main() -> None:
    t0 = time.perf_counter()
    _dump_regime_and_train_preds()
    regime = np.load(REGIME_PATH)
    print(f"regime {regime.shape}")

    splits, src = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    train_cache = load_split_cache("train")
    print(f"labels {src} + train cache")

    y_tr = split_label_array(train_cache, "y2")
    y_va = split_label_array(valid, "y2")
    tree_tr, gru_tr = _assemble_tree_gru("train", train_cache["mask_x"])
    tree_va, gru_va = _assemble_tree_gru("valid", valid["mask_x"])
    tree_te, gru_te = _assemble_tree_gru("test", test["mask_x"])
    mlp_va = linear_blend(np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_valid.npy"), 0.4)
    mlp_te = linear_blend(np.load(OUT / "cs_mlp_only6_test.npy"), np.load(OUT / "cs_mlp_test.npy"), 0.4)
    locked_va = np.load(OUT / "fusion_rank_gate_valid.npy")

    tr0, tr1 = int(train_cache["start"]), int(train_cache["end"])
    va0, va1 = int(valid["start"]), int(valid["end"])
    te0, te1 = int(test["start"]), int(test["end"])

    ic_tree_tr = _daily_ic(tree_tr, y_tr, train_cache["mask_y"])
    ic_gru_tr = _daily_ic(gru_tr, y_tr, train_cache["mask_y"])
    ic_tree_va = _daily_ic(tree_va, y_va, valid["mask_y"])
    ic_gru_va = _daily_ic(gru_va, y_va, valid["mask_y"])
    print(
        f"train mean IC tree={np.nanmean(ic_tree_tr):.6f} gru={np.nanmean(ic_gru_tr):.6f} "
        f"gru_wins={float(np.nanmean(ic_gru_tr > ic_tree_tr)):.3f}"
    )
    print(
        f"valid mean IC tree={np.nanmean(ic_tree_va):.6f} gru={np.nanmean(ic_gru_va):.6f} "
        f"gru_wins={float(np.nanmean(ic_gru_va > ic_tree_va)):.3f}"
    )

    x_tr = np.asarray(regime[tr0:tr1], dtype=np.float64)
    x_va = np.asarray(regime[va0:va1], dtype=np.float64)
    x_te = np.asarray(regime[te0:te1], dtype=np.float64)
    ok = _finite_days(ic_tree_tr, ic_gru_tr)
    mu, sd = _standardize(x_tr[ok])
    xs_tr = (x_tr - mu) / sd
    xs_va = (x_va - mu) / sd
    xs_te = (x_te - mu) / sd
    y_fit = np.stack([ic_tree_tr, ic_gru_tr], axis=1)
    coef = _fit_ridge(xs_tr[ok], y_fit[ok], l2=1.0)
    hat_tr = _predict_ridge(xs_tr, coef)
    hat_va = _predict_ridge(xs_va, coef)
    hat_te = _predict_ridge(xs_te, coef)
    d_tr = hat_tr[:, 1] - hat_tr[:, 0]
    d_va = hat_va[:, 1] - hat_va[:, 0]
    d_te = hat_te[:, 1] - hat_te[:, 0]

    # valid-half robustness (not used to lock)
    n_va = x_va.shape[0]
    mid = n_va // 2
    ok_va = _finite_days(ic_tree_va, ic_gru_va)
    fit_m = ok_va.copy()
    fit_m[mid:] = False
    hold_m = ok_va.copy()
    hold_m[:mid] = False
    if int(fit_m.sum()) >= 20 and int(hold_m.sum()) >= 20:
        mu_h, sd_h = _standardize(x_va[fit_m])
        coef_h = _fit_ridge((x_va[fit_m] - mu_h) / sd_h, np.stack([ic_tree_va, ic_gru_va], axis=1)[fit_m])
        hat_h = _predict_ridge((x_va - mu_h) / sd_h, coef_h)
        d_h = hat_h[:, 1] - hat_h[:, 0]
        w_h = _make_weights(d_h, "hard", 1.0)
        ic_hold, _ = _eval_weights(tree_va, gru_va, w_h, y_va, valid["mask_y"], valid["mask_x"])
        # compare locked vs router on holdout days only
        hold_true = np.zeros(n_va, dtype=bool)
        hold_true[mid:] = True
        from src.metrics import rank_ic_series as _ric

        def _mean_on(pred, day_mask):
            s = _ric(pred, y_va, valid["mask_y"]).copy()
            s[~day_mask] = np.nan
            return float(np.nanmean(s))

        routed_h, _ = None, None
        routed_hold = _rank_blend_daily(gru_va, tree_va, w_h, valid["mask_x"])
        print(
            f"valid-half hold router(hard)={_mean_on(routed_hold, hold_true):.6f} "
            f"locked={_mean_on(locked_va, hold_true):.6f} "
            f"(fit days={int(fit_m.sum())} hold={int(hold_true.sum())})"
        )

    print("train-fit router (lock on train blend IC)")
    best = (-1.0, None, None)
    for kind, temp in (("hard", 1.0), ("soft", 0.02), ("soft", 0.05), ("soft", 0.10), ("soft", 0.20)):
        w_tr = _make_weights(d_tr, kind, temp)
        ic_tr, _ = _eval_weights(tree_tr, gru_tr, w_tr, y_tr, train_cache["mask_y"], train_cache["mask_x"])
        w_va = _make_weights(d_va, kind, temp)
        ic_va, out_va = _eval_weights(tree_va, gru_va, w_va, y_va, valid["mask_y"], valid["mask_x"], mlp_va)
        print(f"  {kind} temp={temp:.2f}  train={ic_tr:.6f}  valid+mlp={ic_va:.6f}")
        if ic_tr > best[0]:
            best = (ic_tr, (kind, temp), (w_va, out_va, ic_va))

    kind, temp = best[1]
    w_va, out_va, ic_va = best[2]
    print(f"LOCKED by train {kind} temp={temp} valid={ic_va:.6f} vs rank-gate {LOCKED_IC:.6f}")

    w_te = _make_weights(d_te, kind, temp)
    routed_te = _rank_blend_daily(gru_te, tree_te, w_te, test["mask_x"])
    blender = FusionModel({"weight_grid": [MLP_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_W, "space": "rank", "ic": float("nan")}
    out_te = blender.predict(mlp_te, routed_te, test["mask_x"])

    summary = {
        "fit": "train daily IC of tree vs gru_ens ~ cols 17-25 ridge",
        "lock": {"kind": kind, "temp": temp, "train_ic": best[0]},
        "valid_ic": ic_va,
        "locked_rank_gate": LOCKED_IC,
        "gru_win_rate_train": float(np.nanmean(ic_gru_tr > ic_tree_tr)),
        "gru_win_rate_valid": float(np.nanmean(ic_gru_va > ic_tree_va)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "regime_router_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if ic_va > LOCKED_IC + 1e-6:
        np.save(OUT / "fusion_regime_router_valid.npy", out_va)
        np.save(OUT / "fusion_regime_router_test.npy", out_te)
        save_submission(out_te, ROOT / "submissions" / "task2_fusion_regime_router.npy")
        print("wrote submissions/task2_fusion_regime_router.npy")
    else:
        print("no valid lift vs rank-gate; not writing a new main")
    print("VALID_RANKIC", f"{ic_va:.6f}")
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
