"""Overnight task1 batch: linear axes, confidence shrink, walk-forward, then wide GRU.

Never overwrites submissions/task1_fusion_x6_today.npy.
Writes outputs/task1_overnight_report.md for morning review.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\run_task1_overnight.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_eval_splits, load_panel, precompute_cs_cols, slice_split, split_bounds  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import (  # noqa: E402
    FusionModel,
    coverage_gate_blend,
    linear_blend,
    panel_cs_rank,
    shrink_to_day_mean,
)
from src.models.linear_cs import fit_online_ridge, fit_ridge, predict_panel  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs"
SUB = ROOT / "submissions"
REPORT = OUT / "task1_overnight_report.md"
SUMMARY = OUT / "task1_overnight_summary.json"

HIGH_TAU = 4670.0
BASE_VALID = 0.120869
BASE_HI31 = 0.211839
PLATFORM = 0.125679
TRAIN_COV_MAX = 4239.0
ONLY6_W, NEXT6_W, MLP6_W, MLP_W = 0.4, 0.15, 0.4, 0.15
TAU, W_LO, W_HI = 4546.0, 0.25, 0.6
X6_COLS = [5, 6, 7, 8, 10, 11, 17, 18, 19, 20, 21, 22, 24, 25, 39, 40, 41, 42, 57, 58, 66, 68, 69, 70, 73, 74, 90]


def _v(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_valid.npy")


def _t(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_test.npy")


def _metrics(pred: np.ndarray, y, my, hi) -> dict:
    s = rank_ic_series(pred, y, my)
    s_hi = s.copy()
    s_hi[~hi] = np.nan
    n = len(s)
    mid = n // 2
    return {
        "valid_ic": float(np.nanmean(s)),
        "hi31_ic": float(np.nanmean(s_hi)),
        "neg_days": int(np.nansum(s < 0)),
        "valid_first_half": float(np.nanmean(s[:mid])),
        "valid_second_half": float(np.nanmean(s[mid:])),
    }


def _locked(x6, o6, n6, tree, mlp, mlp6, mx):
    ens = linear_blend(n6, linear_blend(o6, x6, ONLY6_W), NEXT6_W)
    gated = coverage_gate_blend(ens, tree, mx, TAU, W_LO, W_HI, "raw")
    mlp_ens = linear_blend(mlp6, mlp, MLP6_W)
    blender = FusionModel({"weight_grid": [MLP_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_W, "space": "rank", "ic": float("nan")}
    fused = blender.predict(mlp_ens, gated, mx)
    return ens, gated, fused


def _day_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.ones(a.shape[0], dtype=np.float64)
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if int(m.sum()) < 8:
            continue
        r = spearmanr(a[t, m], b[t, m]).correlation
        if np.isfinite(r):
            out[t] = float(r)
    return out


def _log(lines: list[str], msg: str) -> None:
    print(msg, flush=True)
    lines.append(msg)


def main() -> None:
    t_all = time.perf_counter()
    lines: list[str] = []
    rows: list[dict] = []
    _log(lines, "# Task1 overnight report")
    _log(lines, "")
    _log(lines, f"locked valid={BASE_VALID:.6f} hi31={BASE_HI31:.6f} platform_test={PLATFORM:.6f}")
    _log(lines, "main file NOT overwritten. Candidates go to submissions/task1_fusion_overnight_*.npy")
    _log(lines, "")

    splits, src = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    mxt = test["mask_x"]
    cov = np.asarray(mx).sum(axis=1).astype(np.float64)
    hi = cov >= HIGH_TAU
    _log(lines, f"cache={src} valid_days={len(cov)} cov={cov.min():.0f}-{cov.max():.0f} hi31={int(hi.sum())}")

    x6_v, x6_t = _v("gru_x6_with_today"), _t("gru_x6_with_today")
    o6_v, o6_t = _v("gru_only6_with_today"), _t("gru_only6_with_today")
    n6_v, n6_t = _v("gru_next6_with_today"), _t("gru_next6_with_today")
    tree_v, tree_t = _v("fusion"), _t("fusion")
    mlp_v, mlp_t = _v("cs_mlp"), _t("cs_mlp")
    mlp6_v, mlp6_t = _v("cs_mlp_only6"), _t("cs_mlp_only6")
    ens_v, gated_v, fused_v = _locked(x6_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)
    ens_t, gated_t, fused_t = _locked(x6_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)
    base_m = _metrics(fused_v, y, my, hi)
    _log(lines, f"rebuilt locked valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f} "
         f"half={base_m['valid_first_half']:.6f}/{base_m['valid_second_half']:.6f}")
    rows.append({"name": "locked_x6_today", **base_m, "write": False})

    def consider(name: str, pred_v: np.ndarray, pred_t: np.ndarray | None, write: bool) -> dict:
        m = _metrics(pred_v, y, my, hi)
        dv = m["valid_ic"] - BASE_VALID
        dh = m["hi31_ic"] - BASE_HI31
        ok = m["hi31_ic"] >= BASE_HI31 - 1e-6 and m["valid_ic"] >= BASE_VALID - 0.0005
        _log(
            lines,
            f"  {name:40s} valid={m['valid_ic']:.6f} ({dv:+.6f})  "
            f"hi31={m['hi31_ic']:.6f} ({dh:+.6f})  half={m['valid_first_half']:.4f}/{m['valid_second_half']:.4f}  "
            f"{'PASS' if ok else 'fail'}",
        )
        rec = {"name": name, **m, "delta_valid": dv, "delta_hi31": dh, "pass": ok, "write": False}
        if write and ok and (dv > 1e-6 or dh > 1e-6) and pred_t is not None:
            path = SUB / f"task1_fusion_overnight_{name}.npy"
            save_submission(pred_t, path)
            np.save(OUT / f"fusion_overnight_{name}_valid.npy", pred_v)
            rec["write"] = True
            rec["path"] = str(path)
            _log(lines, f"    wrote {path.name}")
        rows.append(rec)
        return rec

    # --- 1. walk-forward already in metrics; parameter perturbation ---
    _log(lines, "")
    _log(lines, "## 1. gate w_hi perturbation (report only, raw space)")
    for w_hi in (0.50, 0.55, 0.60, 0.65, 0.70):
        g = coverage_gate_blend(ens_v, tree_v, mx, TAU, W_LO, w_hi, "raw")
        blender = FusionModel({"weight_grid": [MLP_W]})
        blender.locked = {"name": "rank_blend", "weight": MLP_W, "space": "rank", "ic": float("nan")}
        mlp_ens = linear_blend(mlp6_v, mlp_v, MLP6_W)
        pred = blender.predict(mlp_ens, g, mx)
        consider(f"gate_whi_{w_hi:.2f}", pred, None, write=False)

    # --- 2. confidence shrink (RankIC meta-labeling analog) ---
    _log(lines, "")
    _log(lines, "## 2. confidence shrink toward day-mean")
    disagree = 1.0 - _day_corr(ens_v, tree_v, mx)
    cov_extrap = np.clip((cov - TRAIN_COV_MAX) / 800.0, 0.0, 1.0)
    for smax in (0.05, 0.10, 0.20, 0.30):
        consider(f"shrink_const_{smax:.2f}", shrink_to_day_mean(fused_v, mx, smax), None, False)
        consider(
            f"shrink_disagree_{smax:.2f}",
            shrink_to_day_mean(fused_v, mx, smax * np.clip(disagree, 0, 1)),
            None,
            False,
        )
        consider(
            f"shrink_covextrap_{smax:.2f}",
            shrink_to_day_mean(fused_v, mx, smax * cov_extrap),
            None,
            False,
        )

    # apply best-looking cov-extrap to test as candidate only if pass (won't, usually)
    # also write a conservative test file for shrink_covextrap_0.10 regardless for upload option
    s_t = np.clip((np.asarray(mxt).sum(1).astype(np.float64) - TRAIN_COV_MAX) / 800.0, 0.0, 1.0)
    pred_t_shrink = shrink_to_day_mean(fused_t, mxt, 0.10 * s_t)
    save_submission(pred_t_shrink, SUB / "task1_fusion_overnight_shrink_cov10.npy")
    np.save(OUT / "fusion_overnight_shrink_cov10_valid.npy", shrink_to_day_mean(fused_v, mx, 0.10 * cov_extrap))
    _log(lines, "  wrote task1_fusion_overnight_shrink_cov10.npy (heuristic, may be worse; for optional upload)")

    # --- 3. ridge + online linear ---
    _log(lines, "")
    _log(lines, "## 3. ridge / online linear (CS z-score x6 cols)")
    try:
        cfg = load_config("configs/default.yaml")
        data = load_panel(str(resolve_data_path(cfg, None)))
        drop_other_label(data, "y1")
        tr0, tr1 = split_bounds(data, "train")
        _log(lines, f"  panel loaded train=[{tr0},{tr1}) n_days={tr1-tr0}")
        cs_sel = precompute_cs_cols(data["num_x"], data["mask_x"], X6_COLS)
        np.clip(cs_sel, -5.0, 5.0, out=cs_sel)
        mx_all = np.asarray(data["mask_x"])
        n_stocks = mx_all.shape[1]

        def extras_cov(g: int) -> np.ndarray:
            n = float(np.asarray(mx_all[g], dtype=bool).sum())
            return np.asarray([n / max(n_stocks, 1), np.log1p(n)], dtype=np.float64)

        valid_p = slice_split(data, "valid")
        test_p = slice_split(data, "test")

        for tag, extras in (("nocov", None), ("cov", extras_cov)):
            w_ridge = fit_ridge(cs_sel, data, tr0, tr1, lam=10.0, extras_fn=extras)
            ridge_v = predict_panel(cs_sel, valid_p, w_ridge, extras_fn=extras)
            ridge_t = predict_panel(cs_sel, test_p, w_ridge, extras_fn=extras)
            np.save(OUT / f"linear_ridge_{tag}_valid.npy", ridge_v)
            np.save(OUT / f"linear_ridge_{tag}_test.npy", ridge_t)
            ic = mean_rank_ic(ridge_v, y, my)
            ic_hi = mean_rank_ic(ridge_v[hi], y[hi], my[hi])
            _log(lines, f"  ridge_{tag} single valid={ic:.6f} hi31={ic_hi:.6f}")
            for w in (0.05, 0.10, 0.15, 0.25):
                fv = linear_blend(panel_cs_rank(ridge_v, mx), panel_cs_rank(fused_v, mx), w)
                ft = linear_blend(panel_cs_rank(ridge_t, mxt), panel_cs_rank(fused_t, mxt), w)
                consider(f"ridge_{tag}_w{w:.2f}", fv, ft, write=True)

            w_on = fit_online_ridge(cs_sel, data, tr0, tr1, lam=10.0, forget=0.997, extras_fn=extras)
            on_v = predict_panel(cs_sel, valid_p, w_on, extras_fn=extras)
            on_t = predict_panel(cs_sel, test_p, w_on, extras_fn=extras)
            np.save(OUT / f"linear_online_{tag}_valid.npy", on_v)
            np.save(OUT / f"linear_online_{tag}_test.npy", on_t)
            ic = mean_rank_ic(on_v, y, my)
            ic_hi = mean_rank_ic(on_v[hi], y[hi], my[hi])
            _log(lines, f"  online_{tag} single valid={ic:.6f} hi31={ic_hi:.6f} forget=0.997")
            for w in (0.05, 0.10, 0.15, 0.25):
                fv = linear_blend(panel_cs_rank(on_v, mx), panel_cs_rank(fused_v, mx), w)
                ft = linear_blend(panel_cs_rank(on_t, mxt), panel_cs_rank(fused_t, mxt), w)
                consider(f"online_{tag}_w{w:.2f}", fv, ft, write=True)

        # stronger forgetting (closer to recent 800d)
        w_on2 = fit_online_ridge(cs_sel, data, tr0, tr1, lam=10.0, forget=0.99, extras_fn=None)
        on2_v = predict_panel(cs_sel, valid_p, w_on2, extras_fn=None)
        on2_t = predict_panel(cs_sel, test_p, w_on2, extras_fn=None)
        np.save(OUT / "linear_online_forget99_valid.npy", on2_v)
        np.save(OUT / "linear_online_forget99_test.npy", on2_t)
        _log(lines, f"  online_forget0.99 single valid={mean_rank_ic(on2_v, y, my):.6f}")
        for w in (0.10, 0.15):
            fv = linear_blend(panel_cs_rank(on2_v, mx), panel_cs_rank(fused_v, mx), w)
            ft = linear_blend(panel_cs_rank(on2_t, mxt), panel_cs_rank(fused_t, mxt), w)
            consider(f"online_forget99_w{w:.2f}", fv, ft, write=True)

        del cs_sel, data
    except Exception:
        _log(lines, "  LINEAR FAILED:")
        _log(lines, traceback.format_exc())

    # --- 4. bag existing extra GRU stems if present ---
    _log(lines, "")
    _log(lines, "## 4. raw-mean bag extra GRU stems into x6 slot")
    extra_x6 = [
        s
        for s in ("gru_x6_attn", "gru_x6_listnet", "gru_x6_full_hicov", "gru_x6_wide")
        if (OUT / f"{s}_valid.npy").is_file() and (OUT / f"{s}_test.npy").is_file()
    ]
    for s in extra_x6:
        try:
            xv, xt = _v(s), _t(s)
            bag_v = np.mean([x6_v, xv], axis=0).astype(np.float32)
            bag_t = np.mean([x6_t, xt], axis=0).astype(np.float32)
            fv = _locked(bag_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)[2]
            ft = _locked(bag_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)[2]
            consider(f"x6_bag_{s}", fv, ft, write=True)
        except Exception:
            _log(lines, f"  bag {s} failed:\n{traceback.format_exc()}")

    # --- 5. wide GRU (subprocess, last) ---
    _log(lines, "")
    _log(lines, "## 5. wide GRU hidden=128 layers=2")
    wide_cfg = ROOT / "configs" / "gru_x6_wide.yaml"
    if wide_cfg.is_file() and not (OUT / "gru_x6_wide_valid.npy").is_file():
        _log(lines, "  training gru_x6_wide (CPU, several minutes)...")
        try:
            proc = subprocess.run(
                [sys.executable, "-u", str(ROOT / "scripts" / "train_gru.py"), "--config", str(wide_cfg)],
                cwd=str(ROOT),
                capture_output=False,
                check=False,
            )
            _log(lines, f"  train_gru exit={proc.returncode}")
        except Exception:
            _log(lines, traceback.format_exc())
    if (OUT / "gru_x6_wide_valid.npy").is_file():
        xv, xt = _v("gru_x6_wide"), _t("gru_x6_wide")
        _log(lines, f"  wide single valid={mean_rank_ic(xv, y, my):.6f} hi31={mean_rank_ic(xv[hi], y[hi], my[hi]):.6f}")
        fv = _locked(xv, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)[2]
        ft = _locked(xt, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)[2]
        consider("x6_wide_replace", fv, ft, write=True)
        bag_v = np.mean([x6_v, xv], axis=0).astype(np.float32)
        bag_t = np.mean([x6_t, xt], axis=0).astype(np.float32)
        fv = _locked(bag_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)[2]
        ft = _locked(bag_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)[2]
        consider("x6_bag_wide", fv, ft, write=True)

    passing = [r for r in rows if r.get("pass") and r["name"] != "locked_x6_today"]
    lifts = [r for r in passing if r.get("delta_valid", 0) > 1e-6 or r.get("delta_hi31", 0) > 1e-6]
    _log(lines, "")
    _log(lines, "## summary")
    _log(lines, f"elapsed_s={time.perf_counter() - t_all:.1f}")
    if lifts:
        best = max(lifts, key=lambda r: (r["hi31_ic"], r["valid_ic"]))
        _log(lines, f"best PASS lift: {best['name']} valid={best['valid_ic']:.6f} hi31={best['hi31_ic']:.6f}")
    else:
        _log(lines, "no experiment beat locked on (hi31 not down AND valid not -0.0005). keep task1_fusion_x6_today.npy")
    _log(lines, "")
    _log(lines, "Do not upload shrink/gate_whi files unless PASS. Platform test of locked remains 0.125679.")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SUMMARY.write_text(json.dumps({"rows": rows, "base": base_m}, indent=2), encoding="utf-8")
    print(f"wrote {REPORT}")
    print("VALID_RANKIC", f"{base_m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
