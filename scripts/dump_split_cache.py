"""Dump valid/test/train labels and masks so fusion scripts skip data.z.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\dump_split_cache.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import (  # noqa: E402
    drop_task2_label,
    dump_split_cache,
    load_panel,
    split_cache_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache split labels/masks without num_x")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--industry-col", type=int, default=6)
    args = parser.parse_args()

    dest = split_cache_dir(ROOT)
    names = ("train", "valid", "test")
    if not args.force and all((dest / f"{name}.npz").is_file() for name in names):
        print(f"cache already at {dest} (pass --force to rebuild)")
        for name in names:
            path = dest / f"{name}.npz"
            print(f"  {path.name} {path.stat().st_size / 1e6:.1f}MB")
        return

    cfg = load_config(args.config)
    data_path = resolve_data_path(cfg, args.data)
    t0 = time.perf_counter()
    data = load_panel(str(data_path))
    drop_task2_label(data)
    print(f"loaded panel in {time.perf_counter() - t0:.1f}s")
    dump_split_cache(data, dest, industry_col=args.industry_col, splits=names)
    print(f"wrote {dest} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
