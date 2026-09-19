"""Scan all 99 numeric columns on high-coverage TRAIN days only.

Does not read valid/test labels. Writes outputs/hicov_col_ic.json and
patches num_indices in the hicov configs.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\scan_hicov_cols.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, split_bounds  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402

COV_TAU = 4000
RECENT_DAYS = 800
ABS_IC_MIN = 0.03
MIN_COLS = 12
MAX_COLS = 27


def _patch_indices(path: Path, cols: list[int], hist: bool) -> None:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    feat = dict(cfg.get("features") or {})
    if hist:
        hist_cfg = dict(feat.get("history") or {})
        hist_cfg["num_indices"] = cols
        feat["history"] = hist_cfg
    else:
        feat["num_indices"] = cols
    cfg["features"] = feat
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"  patched {path} cols={cols}")


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y1")
    tr0, tr1 = split_bounds(data, "train")
    start = max(int(tr0), int(tr1) - RECENT_DAYS)
    cov = np.asarray(data["mask_x"][start:tr1], dtype=bool).sum(axis=1)
    keep = cov >= COV_TAU
    print(
        f"train window [{start},{tr1}) n={tr1 - start} "
        f"cov={int(cov.min())}-{int(cov.max())} "
        f"hicov>={COV_TAU}: {int(keep.sum())} days"
    )
    if int(keep.sum()) < 40:
        raise RuntimeError("too few high-coverage train days; lower COV_TAU in this script")

    num = np.asarray(data["num_x"][start:tr1][keep])
    y = np.asarray(data["y1"][start:tr1][keep])
    my = np.asarray(data["mask_y"][start:tr1][keep])
    n_cols = int(num.shape[-1])
    rows = []
    for col in range(n_cols):
        ic = mean_rank_ic(num[:, :, col], y, my)
        rows.append((col, float(ic), abs(float(ic)) if np.isfinite(ic) else 0.0))
        if (col + 1) % 20 == 0:
            print(f"  scanned {col + 1}/{n_cols}", flush=True)
    rows.sort(key=lambda r: r[2], reverse=True)
    print("top 20 |RankIC| on high-cov train days (no valid):")
    for col, ic, ab in rows[:20]:
        print(f"  col {col:3d}  RankIC={ic:+.6f}  abs={ab:.6f}")

    chosen = [int(c) for c, _, ab in rows if ab >= ABS_IC_MIN][:MAX_COLS]
    if len(chosen) < MIN_COLS:
        chosen = [int(c) for c, _, _ in rows[:MIN_COLS]]
    dest = ROOT / "outputs/hicov_col_ic.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "cov_tau": COV_TAU,
                "recent_days": RECENT_DAYS,
                "n_hicov_days": int(keep.sum()),
                "abs_ic_min": ABS_IC_MIN,
                "top": [{"col": int(c), "ic": float(ic), "abs": float(ab)} for c, ic, ab in rows],
                "chosen": chosen,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {dest} chosen={chosen}")
    _patch_indices(ROOT / "configs/hist_lgbm_hicov.yaml", chosen, hist=True)
    _patch_indices(ROOT / "configs/cs_mlp_hicov.yaml", chosen, hist=False)
    _patch_indices(ROOT / "configs/gru_hicov.yaml", chosen, hist=False)
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
