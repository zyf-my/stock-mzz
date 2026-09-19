"""Scan all 99 numeric columns' mean RankIC vs y2. Does not train.

Train is the ranking source (same as 计划.md 4.5 for y1). Valid is only reported.
Does not use y1 as a feature. Writes outputs/task2/col_rankic.json.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\scan_task2_cols.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import load_panel, split_bounds  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402

TASK1_TOP15 = [8, 11, 7, 41, 57, 90, 68, 40, 69, 73, 74, 39, 42, 66, 58]
TASK1_HIST21 = [5, 6, 7, 8, 10, 11, 17, 18, 19, 20, 21, 22, 24, 25, 39, 40, 41, 57, 68, 70, 90]
TASK1_ONLY6 = [42, 58, 66, 69, 73, 74]
TASK1_NEXT6 = [38, 72, 47, 3, 1, 4]


def _panel_rankic(num: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Official Spearman RankIC, all columns, ranking y once per day."""
    n_days, _, n_cols = num.shape
    acc = np.zeros(n_cols, dtype=np.float64)
    cnt = np.zeros(n_cols, dtype=np.int32)
    t0 = time.perf_counter()
    for t in range(n_days):
        m = np.asarray(mask[t], dtype=bool) & np.isfinite(y[t])
        if int(m.sum()) < 3:
            continue
        yy = np.asarray(y[t, m], dtype=np.float64)
        ry = rankdata(yy, method="average")
        ry_std = float(np.std(ry))
        if ry_std == 0:
            continue
        xx = np.asarray(num[t, m], dtype=np.float64)
        rx = rankdata(xx, axis=0, method="average")
        rx_c = rx - rx.mean(axis=0)
        ry_c = ry - ry.mean()
        denom = np.sqrt((rx_c * rx_c).sum(axis=0)) * float(np.sqrt((ry_c * ry_c).sum()))
        ok = denom > 1e-12
        acc[ok] += (rx_c[:, ok] * ry_c[:, None]).sum(axis=0) / denom[ok]
        cnt[ok] += 1
        if t > 0 and t % 400 == 0:
            print(f"  day {t}/{n_days} {time.perf_counter() - t0:.1f}s", flush=True)
    out = np.full(n_cols, np.nan, dtype=np.float64)
    ok = cnt > 0
    out[ok] = acc[ok] / cnt[ok]
    return out


def _top(ics: np.ndarray, k: int) -> list[tuple[int, float]]:
    order = np.argsort(-np.abs(ics))
    return [(int(i), float(ics[i])) for i in order[:k] if np.isfinite(ics[i])]


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    path = resolve_data_path(cfg, None)
    print(f"data={path}")
    data = load_panel(str(path))
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    tr0, tr1 = split_bounds(data, "train")
    va0, va1 = split_bounds(data, "valid")
    num = data["num_x"]
    y2 = data["y2"]
    y1 = data["y1"]
    my = data["mask_y"]
    n_cols = int(num.shape[-1])

    y1y2 = mean_rank_ic(y1[tr0:tr1], y2[tr0:tr1], my[tr0:tr1])
    print(f"train y1 vs y2 RankIC={y1y2:.6f}")

    print(f"scan train days [{tr0}, {tr1}) cols={n_cols}")
    train_ic = _panel_rankic(num[tr0:tr1], y2[tr0:tr1], my[tr0:tr1])
    print(f"scan valid days [{va0}, {va1})")
    valid_ic = _panel_rankic(num[va0:va1], y2[va0:va1], my[va0:va1])

    rows = []
    for i in range(n_cols):
        rows.append(
            {
                "col": i,
                "train": float(train_ic[i]) if np.isfinite(train_ic[i]) else None,
                "valid": float(valid_ic[i]) if np.isfinite(valid_ic[i]) else None,
                "train_abs": float(abs(train_ic[i])) if np.isfinite(train_ic[i]) else 0.0,
            }
        )
    rows.sort(key=lambda r: r["train_abs"], reverse=True)

    top15 = _top(train_ic, 15)
    print("y2 train |RankIC| Top 15:")
    for col, ic in top15:
        print(f"  col {col:3d}  train={ic:+.6f}  valid={valid_ic[col]:+.6f}")

    def _set_stats(name: str, cols: list[int]) -> dict:
        ics = [float(train_ic[c]) for c in cols]
        return {
            "cols": cols,
            "train": ics,
            "train_abs_mean": float(np.mean(np.abs(ics))),
            "train_abs_max": float(np.max(np.abs(ics))),
        }

    dest = ROOT / "outputs" / "task2" / "col_rankic.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label_key": "y2",
        "n_cols": n_cols,
        "y1_y2_train_rankic": float(y1y2),
        "top15": [{"col": c, "train": ic, "valid": float(valid_ic[c])} for c, ic in top15],
        "task1_top15_on_y2": _set_stats("task1_top15", TASK1_TOP15),
        "task1_only6_on_y2": _set_stats("only6", TASK1_ONLY6),
        "task1_next6_on_y2": _set_stats("next6", TASK1_NEXT6),
        "task1_hist21_on_y2": _set_stats("hist21", TASK1_HIST21),
        "all": rows,
    }
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("TOP1", top15[0][0], f"{top15[0][1]:+.6f}")


if __name__ == "__main__":
    main()
