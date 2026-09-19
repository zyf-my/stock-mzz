"""Second overnight batch: fusion diagnostics + extra GRU stems.

Runs after scripts/run_task1_overnight.py (wide GRU). Sequential, 16GB RAM.
Never overwrites submissions/task1_fusion_x6_today.npy.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\run_task1_overnight_more.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import (  # noqa: E402
    FusionModel,
    coverage_gate_blend,
    linear_blend,
    neutralize_industry,
    panel_cs_rank,
)
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs"
SUB = ROOT / "submissions"
REPORT = OUT / "task1_overnight_report2.md"
SUMMARY = OUT / "task1_overnight_summary2.json"
MASTER = OUT / "task1_overnight_report.md"

HIGH_TAU = 4670.0
BASE_VALID = 0.120869
BASE_HI31 = 0.211839
PLATFORM = 0.125679
ONLY6_W, NEXT6_W, MLP6_W, MLP_W = 0.4, 0.15, 0.4, 0.15
TAU, W_LO, W_HI = 4546.0, 0.25, 0.6

GRU_JOBS = [
    "configs/gru_x6_with_today_s43.yaml",
    "configs/gru_x6_with_today_s44.yaml",
    "configs/gru_x6_h96.yaml",
    "configs/gru_x6_l2.yaml",
    "configs/gru_x6_pairwise.yaml",
    "configs/gru_only6_with_today_s43.yaml",
]


def _v(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_valid.npy")


def _t(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_test.npy")


def _has(name: str) -> bool:
    return (OUT / f"{name}_valid.npy").is_file() and (OUT / f"{name}_test.npy").is_file()


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


def _locked_weights(x6, o6, n6, tree, mlp, mlp6, mx, *, only6_w, next6_w, mlp_w):
    ens = linear_blend(n6, linear_blend(o6, x6, only6_w), next6_w)
    gated = coverage_gate_blend(ens, tree, mx, TAU, W_LO, W_HI, "raw")
    mlp_ens = linear_blend(mlp6, mlp, MLP6_W)
    blender = FusionModel({"weight_grid": [mlp_w]})
    blender.locked = {"name": "rank_blend", "weight": mlp_w, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp_ens, gated, mx)


def _ortho_against(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float32)
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if int(m.sum()) < 8:
            continue
        x = np.asarray(a[t, m], dtype=np.float64)
        z = np.asarray(b[t, m], dtype=np.float64)
        x = x - x.mean()
        z = z - z.mean()
        denom = float(np.dot(z, z)) + 1e-8
        beta = float(np.dot(x, z)) / denom
        resid = np.zeros(a.shape[1], dtype=np.float64)
        resid[m] = x - beta * z
        out[t] = resid.astype(np.float32)
    return out


def _winsor(pred: np.ndarray, mask: np.ndarray, p: float) -> np.ndarray:
    out = np.array(pred, dtype=np.float32, copy=True)
    for t in range(pred.shape[0]):
        m = np.asarray(mask[t], dtype=bool)
        if int(m.sum()) < 16:
            continue
        lo, hi = np.quantile(out[t, m], [p, 1.0 - p])
        out[t, m] = np.clip(out[t, m], lo, hi)
    return out


def _stem_from_cfg(cfg_path: str) -> str:
    return Path(cfg_path).stem


def _log(lines: list[str], msg: str) -> None:
    print(msg, flush=True)
    lines.append(msg)


def _train_gru(cfg_path: str, lines: list[str]) -> int:
    stem = _stem_from_cfg(cfg_path)
    if _has(stem):
        _log(lines, f"  skip train {stem} (npy exists)")
        return 0
    _log(lines, f"  training {cfg_path} ...")
    proc = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts" / "train_gru.py"), "--config", str(ROOT / cfg_path)],
        cwd=str(ROOT),
        check=False,
    )
    _log(lines, f"  train_gru {stem} exit={proc.returncode}")
    return int(proc.returncode)


def main() -> None:
    t_all = time.perf_counter()
    lines: list[str] = []
    rows: list[dict] = []
    _log(lines, "# Task1 overnight report 2")
    _log(lines, "")
    _log(lines, f"locked valid={BASE_VALID:.6f} hi31={BASE_HI31:.6f} platform_test={PLATFORM:.6f}")
    _log(lines, "main file NOT overwritten. Candidates go to submissions/task1_fusion_overnight_*.npy")
    _log(lines, "")

    splits, src = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    mxt = test["mask_x"]
    industry_v = valid["industry"]
    industry_t = test["industry"]
    cov = np.asarray(mx).sum(axis=1).astype(np.float64)
    hi = cov >= HIGH_TAU
    _log(lines, f"cache={src} valid_days={len(cov)} cov={cov.min():.0f}-{cov.max():.0f} hi31={int(hi.sum())}")

    x6_v, x6_t = _v("gru_x6_with_today"), _t("gru_x6_with_today")
    o6_v, o6_t = _v("gru_only6_with_today"), _t("gru_only6_with_today")
    n6_v, n6_t = _v("gru_next6_with_today"), _t("gru_next6_with_today")
    tree_v, tree_t = _v("fusion"), _t("fusion")
    mlp_v, mlp_t = _v("cs_mlp"), _t("cs_mlp")
    mlp6_v, mlp6_t = _v("cs_mlp_only6"), _t("cs_mlp_only6")
    ens_v, _, fused_v = _locked(x6_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)
    ens_t, _, fused_t = _locked(x6_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)
    base_m = _metrics(fused_v, y, my, hi)
    _log(
        lines,
        f"rebuilt locked valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f} "
        f"half={base_m['valid_first_half']:.6f}/{base_m['valid_second_half']:.6f}",
    )
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

    # --- 6. fusion weight micro-grid (report only; valid search is leaky) ---
    _log(lines, "")
    _log(lines, "## 6. fusion weight micro-grid (report only)")
    for w in (0.30, 0.35, 0.40, 0.45, 0.50):
        pred = _locked_weights(x6_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx, only6_w=w, next6_w=NEXT6_W, mlp_w=MLP_W)
        consider(f"only6w_{w:.2f}", pred, None, write=False)
    for w in (0.10, 0.15, 0.20, 0.25):
        pred = _locked_weights(x6_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx, only6_w=ONLY6_W, next6_w=w, mlp_w=MLP_W)
        consider(f"next6w_{w:.2f}", pred, None, write=False)
    for w in (0.10, 0.12, 0.15, 0.18, 0.20):
        pred = _locked_weights(x6_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx, only6_w=ONLY6_W, next6_w=NEXT6_W, mlp_w=w)
        consider(f"mlpw_{w:.2f}", pred, None, write=False)

    # --- 7. industry neutralize / winsor / GRU-vs-tree ortho ---
    _log(lines, "")
    _log(lines, "## 7. industry neutralize, winsor, GRU-vs-tree ortho")
    neu_v = neutralize_industry(fused_v, industry_v, mx)
    neu_t = neutralize_industry(fused_t, industry_t, mxt)
    consider("ind_neu_fused", neu_v, neu_t, write=True)
    for w in (0.25, 0.50, 0.75, 1.00):
        fv = linear_blend(panel_cs_rank(neu_v, mx), panel_cs_rank(fused_v, mx), w)
        ft = linear_blend(panel_cs_rank(neu_t, mxt), panel_cs_rank(fused_t, mxt), w)
        consider(f"ind_neu_w{w:.2f}", fv, ft, write=True)
    for p in (0.005, 0.01, 0.02):
        consider(f"winsor_{p:.3f}", _winsor(fused_v, mx, p), _winsor(fused_t, mxt, p), write=True)

    ortho_v = _ortho_against(ens_v, tree_v, mx)
    ortho_t = _ortho_against(ens_t, tree_t, mxt)
    _log(lines, f"  gru_ortho_tree single valid={mean_rank_ic(ortho_v, y, my):.6f}")
    for w in (0.05, 0.10, 0.15, 0.25):
        fv = linear_blend(panel_cs_rank(ortho_v, mx), panel_cs_rank(fused_v, mx), w)
        ft = linear_blend(panel_cs_rank(ortho_t, mxt), panel_cs_rank(fused_t, mxt), w)
        consider(f"gru_ortho_w{w:.2f}", fv, ft, write=True)

    # --- 8. train extra GRU stems ---
    _log(lines, "")
    _log(lines, "## 8. extra GRU stems (seeds / width / pairwise / only6 seed)")
    for cfg in GRU_JOBS:
        try:
            _train_gru(cfg, lines)
        except Exception:
            _log(lines, f"  TRAIN FAILED {cfg}:\n{traceback.format_exc()}")
        stem = _stem_from_cfg(cfg)
        if not _has(stem):
            _log(lines, f"  missing preds {stem}")
            continue
        xv, xt = _v(stem), _t(stem)
        _log(lines, f"  {stem} single valid={mean_rank_ic(xv, y, my):.6f} hi31={mean_rank_ic(xv[hi], y[hi], my[hi]):.6f}")
        if stem.startswith("gru_only6"):
            bag_v = np.mean([o6_v, xv], axis=0).astype(np.float32)
            bag_t = np.mean([o6_t, xt], axis=0).astype(np.float32)
            fv = _locked(x6_v, bag_v, n6_v, tree_v, mlp_v, mlp6_v, mx)[2]
            ft = _locked(x6_t, bag_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)[2]
            consider(f"only6_bag_{stem}", fv, ft, write=True)
            fv = _locked(x6_v, xv, n6_v, tree_v, mlp_v, mlp6_v, mx)[2]
            ft = _locked(x6_t, xt, n6_t, tree_t, mlp_t, mlp6_t, mxt)[2]
            consider(f"only6_replace_{stem}", fv, ft, write=True)
        else:
            fv = _locked(xv, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)[2]
            ft = _locked(xt, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)[2]
            consider(f"x6_replace_{stem}", fv, ft, write=True)
            bag_v = np.mean([x6_v, xv], axis=0).astype(np.float32)
            bag_t = np.mean([x6_t, xt], axis=0).astype(np.float32)
            fv = _locked(bag_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)[2]
            ft = _locked(bag_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)[2]
            consider(f"x6_bag_{stem}", fv, ft, write=True)

    seed_names = [s for s in ("gru_x6_with_today_s43", "gru_x6_with_today_s44") if _has(s)]
    if len(seed_names) >= 1:
        parts_v = [x6_v] + [_v(s) for s in seed_names]
        parts_t = [x6_t] + [_t(s) for s in seed_names]
        bag_v = np.mean(parts_v, axis=0).astype(np.float32)
        bag_t = np.mean(parts_t, axis=0).astype(np.float32)
        fv = _locked(bag_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)[2]
        ft = _locked(bag_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)[2]
        consider(f"x6_bag_seeds_{len(parts_v)}", fv, ft, write=True)

    extra = [s for s in ("gru_x6_wide", "gru_x6_h96", "gru_x6_l2", "gru_x6_pairwise") if _has(s)]
    if extra:
        parts_v = [x6_v] + [_v(s) for s in extra]
        parts_t = [x6_t] + [_t(s) for s in extra]
        bag_v = np.mean(parts_v, axis=0).astype(np.float32)
        bag_t = np.mean(parts_t, axis=0).astype(np.float32)
        fv = _locked(bag_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)[2]
        ft = _locked(bag_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)[2]
        consider("x6_bag_capacity", fv, ft, write=True)

    passing = [r for r in rows if r.get("pass") and r["name"] != "locked_x6_today"]
    lifts = [r for r in passing if r.get("delta_valid", 0) > 1e-6 or r.get("delta_hi31", 0) > 1e-6]
    _log(lines, "")
    _log(lines, "## summary")
    _log(lines, f"elapsed_s={time.perf_counter() - t_all:.1f}")
    if lifts:
        best = max(lifts, key=lambda r: (r["hi31_ic"], r["valid_ic"]))
        _log(lines, f"best PASS lift: {best['name']} valid={best['valid_ic']:.6f} hi31={best['hi31_ic']:.6f}")
        _log(lines, "Do not overwrite task1_fusion_x6_today.npy until platform test of a PASS candidate.")
    else:
        _log(lines, "no experiment beat locked on (hi31 not down AND valid not -0.0005). keep task1_fusion_x6_today.npy")
    _log(lines, f"Platform test of locked remains {PLATFORM:.6f}.")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SUMMARY.write_text(json.dumps({"rows": rows, "base": base_m}, indent=2), encoding="utf-8")
    if MASTER.is_file():
        extra_txt = "\n\n" + "\n".join(lines) + "\n"
        MASTER.write_text(MASTER.read_text(encoding="utf-8") + extra_txt, encoding="utf-8")
    print(f"wrote {REPORT}")
    print("VALID_RANKIC", f"{base_m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
