"""3-level coverage gate + day-oracle by quartile. No retrain.

Q4 (most crowded) tree 0.058 vs GRU 0.107; the 2-level tau=4546 gate mixes Q3
(tree 0.125) with Q4, so w_high cannot go to 1. Split Q4 off.

Does not overwrite the 0.117 n2000 file.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_regime_gate.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import load_eval_splits  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

PREV_BEST = 0.117177
TAU_MID = 4546.0
W_LOW = 0.25
W_MID = 0.4


def coverage_gate3(
    temporal: np.ndarray,
    cs: np.ndarray,
    mask_x: np.ndarray,
    tau_mid: float,
    tau_high: float,
    w_low: float,
    w_mid: float,
    w_high: float,
) -> np.ndarray:
    n_x = np.asarray(mask_x).sum(axis=1)
    pred = np.empty_like(temporal, dtype=np.float32)
    low = n_x < float(tau_mid)
    high = n_x >= float(tau_high)
    mid = ~low & ~high
    pred[low] = linear_blend(temporal[low], cs[low], w_low)
    pred[mid] = linear_blend(temporal[mid], cs[mid], w_mid)
    pred[high] = linear_blend(temporal[high], cs[high], w_high)
    return pred


def day_disagree(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape[0], dtype=np.float64)
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(a[t]) & np.isfinite(b[t])
        if int(m.sum()) < 8:
            out[t] = 0.0
            continue
        r = spearmanr(a[t, m], b[t, m]).statistic
        out[t] = 1.0 - float(r) if np.isfinite(r) else 1.0
    return out


def stack_mlp(gated, gated_t, mlp_v, mlp_t, y, my, mx, mx_t):
    blender = FusionModel({"weight_grid": [0.15, 0.25, 0.4]})
    locked = blender.fit(mlp_v, gated, y, my, mx)
    out_v = blender.predict(mlp_v, gated, mx)
    out_t = blender.predict(mlp_t, gated_t, mx_t)
    return locked, out_v, out_t


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    cache_dir = ROOT / "outputs" / "split_cache"
    data_path = None
    if not (cache_dir / "valid.npz").is_file() or not (cache_dir / "test.npz").is_file():
        data_path = resolve_data_path(cfg, None)
    splits, src = load_eval_splits(cache_dir=cache_dir, data_path=data_path, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    n_x = np.asarray(mx).sum(axis=1).astype(np.float64)
    print(f"loaded {src} in {time.perf_counter() - t0:.1f}s")

    gru_v = np.load(ROOT / "outputs/gru_no_today_recent_n2000_valid.npy")
    gru_t = np.load(ROOT / "outputs/gru_no_today_recent_n2000_test.npy")
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")

    s_tree = rank_ic_series(tree_v, y, my)
    s_gru = rank_ic_series(gru_v, y, my)
    s_mlp = rank_ic_series(mlp_v, y, my)
    edges = np.nanpercentile(n_x, [0, 25, 50, 75, 100])
    print("== per-quartile day-oracle (tree / GRU / MLP) ==")
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        sel = (n_x >= lo) & (n_x <= hi) if i == 3 else (n_x >= lo) & (n_x < hi)
        stack = np.vstack([s_tree[sel], s_gru[sel], s_mlp[sel]])
        oracle = np.nanmax(stack, axis=0)
        print(
            f"  Q{i + 1} [{lo:.0f},{hi:.0f}] n={int(sel.sum())}  "
            f"tree={float(np.nanmean(s_tree[sel])):.4f}  "
            f"gru={float(np.nanmean(s_gru[sel])):.4f}  "
            f"mlp={float(np.nanmean(s_mlp[sel])):.4f}  "
            f"oracle={float(np.nanmean(oracle)):.4f}"
        )

    print("\n== 3-level coverage grid ==")
    results = []
    for tau_high in (4580.0, 4614.0, 4650.0):
        for w_high in (0.7, 0.85, 1.0):
            gated = coverage_gate3(gru_v, tree_v, mx, TAU_MID, tau_high, W_LOW, W_MID, w_high)
            gated_ic = mean_rank_ic(gated, y, my)
            locked, _, _ = stack_mlp(gated, gated, mlp_v, mlp_v, y, my, mx, mx)
            row = {
                "tau_high": tau_high,
                "w_high": w_high,
                "gated": gated_ic,
                "stacked": float(locked["ic"]),
                "stack_name": locked["name"],
                "stack_w": float(locked.get("weight", np.nan)),
            }
            results.append(row)
            print(
                f"  tau_high={tau_high:.0f} w_high={w_high:.2f}  "
                f"gated={gated_ic:.6f}  +mlp {locked['name']} {locked['ic']:.6f}"
            )

    print("\n== disagreement x coverage (high-disagree -> more GRU on crowded days) ==")
    disagree = day_disagree(gru_v, tree_v, mx)
    d_cut = float(np.nanmedian(disagree))
    print(f"  disagree median={d_cut:.3f}  corr(coverage, disagree)={float(np.corrcoef(n_x, disagree)[0, 1]):.3f}")
    for w_agree, w_dis in ((0.25, 0.7), (0.25, 0.85), (0.4, 0.85)):
        pred = np.empty_like(gru_v, dtype=np.float32)
        for t in range(gru_v.shape[0]):
            crowded = n_x[t] >= TAU_MID
            w = w_dis if (crowded and disagree[t] >= d_cut) else (W_MID if crowded else W_LOW)
            if crowded and disagree[t] < d_cut:
                w = w_agree
            pred[t] = linear_blend(gru_v[t], tree_v[t], w)
        ic = mean_rank_ic(pred, y, my)
        locked, _, _ = stack_mlp(pred, pred, mlp_v, mlp_v, y, my, mx, mx)
        print(f"  w_agree={w_agree:.2f} w_dis={w_dis:.2f}  gated={ic:.6f}  +mlp {locked['ic']:.6f}")
        results.append(
            {
                "name": "disagree",
                "w_agree": w_agree,
                "w_dis": w_dis,
                "gated": ic,
                "stacked": float(locked["ic"]),
            }
        )

    best = max(results, key=lambda r: r["stacked"])
    print(f"\nBEST stacked={best['stacked']:.6f}  spec={best}")

    if best["stacked"] > PREV_BEST + 1e-4 and "tau_high" in best:
        gated_v = coverage_gate3(
            gru_v, tree_v, mx, TAU_MID, best["tau_high"], W_LOW, W_MID, best["w_high"]
        )
        gated_t = coverage_gate3(
            gru_t, tree_t, test["mask_x"], TAU_MID, best["tau_high"], W_LOW, W_MID, best["w_high"]
        )
        locked, out_v, out_t = stack_mlp(gated_v, gated_t, mlp_v, mlp_t, y, my, mx, test["mask_x"])
        series = rank_ic_series(out_v, y, my)
        print(
            f"write stack RankIC={locked['ic']:.6f} min={np.nanmin(series):.4f} "
            f"neg={int((series < 0).sum())}"
        )
        np.save(ROOT / "outputs/fusion_n2000_gate3_valid.npy", out_v)
        np.save(ROOT / "outputs/fusion_n2000_gate3_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions/task1_fusion_n2000_gate3.npy")
        meta = {"best": best, "prev_best": PREV_BEST, "valid_ic": float(locked["ic"])}
        (ROOT / "outputs/fusion_n2000_gate3_lock.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print("wrote submissions/task1_fusion_n2000_gate3.npy (did not overwrite n2000 file)")
    else:
        print(f"no write: best {best['stacked']:.6f} vs prev {PREV_BEST:.6f}")

    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{best['stacked']:.6f}")


if __name__ == "__main__":
    main()
