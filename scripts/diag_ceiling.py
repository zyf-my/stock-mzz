"""Upper bound of current branches. No retrain, no overwrite.

Answers: if we could pick the best branch each day, how close is 0.14?
Also re-checks coverage-gate w_high on the local n2000 GRU (lock was 0.4; 0.12 recipe used 0.6).

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\diag_ceiling.py
"""

from __future__ import annotations

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
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402

CANDIDATES = {
    "baseline": "outputs/baseline_valid.npy",
    "hist_lgbm": "outputs/hist_lgbm_valid.npy",
    "tree_fusion": "outputs/fusion_valid.npy",
    "gru": "outputs/gru_valid.npy",
    "gru_no_today": "outputs/gru_no_today_valid.npy",
    "gru_recent": "outputs/gru_no_today_recent_valid.npy",
    "gru_n2000": "outputs/gru_no_today_recent_n2000_valid.npy",
    "gru_x6": "outputs/gru_no_today_recent_n2000_x6_valid.npy",
    "gru_only6": "outputs/gru_no_today_recent_n2000_only6_valid.npy",
    "gru_only6_today": "outputs/gru_only6_with_today_valid.npy",
    "gru_next6": "outputs/gru_next6_with_today_valid.npy",
    "cs_mlp": "outputs/cs_mlp_valid.npy",
    "cs_mlp_only6": "outputs/cs_mlp_only6_valid.npy",
    "locked_n2000": "outputs/fusion_recent_gru_n2000_cov_mlp_valid.npy",
    "locked_next6": "outputs/fusion_next6_wt_mlp6_valid.npy",
}


def _daily_spearman(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(a.shape[0], np.nan)
    for t in range(a.shape[0]):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(a[t]) & np.isfinite(b[t])
        if int(m.sum()) < 8:
            continue
        r = spearmanr(a[t, m], b[t, m]).statistic
        if np.isfinite(r):
            out[t] = float(r)
    return out


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    cache_dir = ROOT / "outputs" / "split_cache"
    data_path = None
    if not (cache_dir / "valid.npz").is_file():
        data_path = resolve_data_path(cfg, None)
    splits, src = load_eval_splits(cache_dir=cache_dir, data_path=data_path, splits=("valid",))
    valid = splits["valid"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    n_x = np.asarray(mx).sum(axis=1).astype(np.float64)
    print(f"loaded {src} valid={y.shape} in {time.perf_counter() - t0:.1f}s")
    print(
        f"coverage min={n_x.min():.0f} p50={np.median(n_x):.0f} "
        f"p60={np.quantile(n_x, 0.6):.0f} max={n_x.max():.0f}"
    )

    loaded: dict[str, np.ndarray] = {}
    series: dict[str, np.ndarray] = {}
    print("\n== single-branch valid RankIC ==")
    for name, rel in CANDIDATES.items():
        path = ROOT / rel
        if not path.is_file():
            print(f"  {name:18s}  MISSING  {rel}")
            continue
        pred = np.load(path)
        loaded[name] = pred
        s = rank_ic_series(pred, y, my)
        series[name] = s
        print(
            f"  {name:18s}  {mean_rank_ic(pred, y, my):.6f}  "
            f"min={np.nanmin(s):.3f} med={np.nanmedian(s):.3f} "
            f"neg={int((s < 0).sum())}"
        )

    if len(loaded) < 2:
        print("need at least two prediction files")
        return

    names = list(loaded)
    print("\n== mean daily Spearman between branches ==")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = float(np.nanmean(_daily_spearman(loaded[a], loaded[b], mx)))
            print(f"  {a:18s} vs {b:18s}  {r:.3f}")

    print("\n== day-oracle: pick the max RankIC branch that day ==")
    stack = np.vstack([series[n] for n in names])
    oracle = np.nanmax(stack, axis=0)
    print(
        f"  all loaded branches: {float(np.nanmean(oracle)):.6f}  "
        f"neg={int((oracle < 0).sum())}"
    )
    core = [n for n in ("tree_fusion", "gru_n2000", "cs_mlp", "locked_n2000", "locked_next6") if n in series]
    if core:
        core_oracle = np.nanmax(np.vstack([series[n] for n in core]), axis=0)
        print(f"  core {core}: {float(np.nanmean(core_oracle)):.6f}")

    print("\n== coverage quartile RankIC ==")
    edges = np.nanpercentile(n_x, [0, 25, 50, 75, 100])
    show = [n for n in ("tree_fusion", "gru_n2000", "cs_mlp", "locked_n2000", "locked_next6") if n in series]
    header = "  Q     n_days  " + "  ".join(f"{n:>14s}" for n in show)
    print(header)
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        sel = (n_x >= lo) & (n_x <= hi) if i == 3 else (n_x >= lo) & (n_x < hi)
        bits = [f"{float(np.nanmean(series[n][sel])):.4f}" for n in show]
        print(f"  Q{i + 1} [{lo:.0f},{hi:.0f}] {int(sel.sum()):5d}  " + "  ".join(f"{b:>14s}" for b in bits))

    if "gru_n2000" in loaded and "tree_fusion" in loaded:
        print("\n== n2000 coverage gate w_high sweep (tau=4546, w_low=0.25, raw) ==")
        gru, tree = loaded["gru_n2000"], loaded["tree_fusion"]
        mlp = loaded.get("cs_mlp")
        for w_high in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
            gated = coverage_gate_blend(gru, tree, mx, 4546.0, 0.25, w_high, "raw")
            gic = mean_rank_ic(gated, y, my)
            extra = ""
            if mlp is not None:
                blender = FusionModel({"weight_grid": [0.15, 0.25]})
                locked = blender.fit(mlp, gated, y, my, mx)
                extra = f"  +mlp {locked['name']} {locked['ic']:.6f}"
            print(f"  w_high={w_high:.1f}  gated={gic:.6f}{extra}")

    print("\n== 0.14 gap ==")
    best_name = max(loaded, key=lambda n: float(np.nanmean(series[n])))
    best_ic = float(np.nanmean(series[best_name]))
    print(f"  local best {best_name}={best_ic:.6f}")
    print(f"  day-oracle={float(np.nanmean(oracle)):.6f}")
    print(f"  gap to 0.14 from best={0.14 - best_ic:.6f}  from oracle={0.14 - float(np.nanmean(oracle)):.6f}")
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
