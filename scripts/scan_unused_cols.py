"""Pick the next 6 unused numeric columns by |RankIC| on train last 800 days.

Does not look at valid/test labels. Writes configs/gru_rest6_with_today.yaml.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\scan_unused_cols.py
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
from src.dataset import drop_task2_label, load_panel, split_bounds  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402

USED = {
    1, 3, 4, 5, 6, 7, 8, 10, 11, 17, 18, 19, 20, 21, 22, 24, 25,
    38, 39, 40, 41, 42, 47, 57, 58, 66, 68, 69, 70, 72, 73, 74, 90,
}
RECENT_DAYS = 800
TOP_K = 6


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    path = resolve_data_path(cfg, None)
    data = load_panel(str(path))
    drop_task2_label(data)
    train_start, train_end = split_bounds(data, "train")
    start = max(int(train_start), int(train_end) - RECENT_DAYS)
    num = data["num_x"][start:train_end]
    y = data["y1"][start:train_end]
    mask = data["mask_y"][start:train_end]
    n_cols = int(num.shape[-1])
    unused = [c for c in range(n_cols) if c not in USED]
    print(f"train days [{start}, {train_end}) unused={len(unused)}/{n_cols}")

    rows = []
    for col in unused:
        ic = mean_rank_ic(num[:, :, col], y, mask)
        rows.append((col, ic, abs(ic)))
        if len(rows) % 20 == 0:
            print(f"  scanned {len(rows)}/{len(unused)}", flush=True)
    rows.sort(key=lambda r: r[2], reverse=True)
    top = rows[:TOP_K]
    print("top unused |RankIC| on last 800 train days:")
    for col, ic, ab in top:
        print(f"  col {col:3d}  RankIC={ic:+.6f}  abs={ab:.6f}")
    chosen = [int(c) for c, _, _ in top]
    out = {
        "recent_days": RECENT_DAYS,
        "used": sorted(USED),
        "top": [{"col": int(c), "ic": float(ic)} for c, ic, _ in rows[:20]],
        "chosen": chosen,
    }
    dest = ROOT / "outputs/unused_col_ic_recent800.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    yaml_path = ROOT / "configs/gru_rest6_with_today.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "include": "configs/gru_next6_with_today.yaml",
                "features": {
                    "num_indices": chosen,
                    "include_current_day": True,
                },
                "paths": {
                    "checkpoint": "checkpoints/gru_rest6_with_today.pt",
                    "valid_pred": "outputs/gru_rest6_with_today_valid.npy",
                    "test_pred": "outputs/gru_rest6_with_today_test.npy",
                    "submission": "submissions/task1_gru_rest6_with_today.npy",
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {dest}")
    print(f"wrote {yaml_path} cols={chosen}")
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
