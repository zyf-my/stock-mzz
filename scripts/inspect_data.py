"""Print shapes, split sizes, and mask rates. Does not train anything.

Usage:
    python scripts/inspect_data.py --file data/xxx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import read_zstd, split_ranges  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402


def _slice(arr: np.ndarray, start: int, end: int | None) -> np.ndarray:
    return arr[start:end]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="local contest data file")
    args = parser.parse_args()

    data = read_zstd(args.file)
    print("keys:", sorted(data.keys()))

    num_x = data["num_x"]
    cat_x = data["cat_x"]
    y1 = data["y1"]
    mask_x = data["mask_x"]
    mask_y = data["mask_y"]

    print(f"num_x: {num_x.dtype} {num_x.shape}")
    print(f"cat_x: {cat_x.dtype} {cat_x.shape}")
    print(f"y1   : {y1.dtype} {y1.shape}")
    print(f"mask_x False rate: {(~mask_x).mean():.4f}")
    print(f"mask_y False rate: {(~mask_y).mean():.4f}")
    print(
        "idx: train_start=%s valid_start=%s test_start=%s"
        % (data["train_start_idx"], data["valid_start_idx"], data["test_start_idx"])
    )

    ranges = split_ranges(data)
    for name, (start, end) in ranges.items():
        n_t = (end if end is not None else y1.shape[0]) - start
        print(f"{name}: t=[{start}:{end if end is not None else ''}] days={n_t}")

    train_y = _slice(y1, *ranges["train"])
    print(
        "y1 train: mean=%.6f std=%.6f nan=%.4f"
        % (
            np.nanmean(train_y),
            np.nanstd(train_y),
            np.isnan(train_y).mean(),
        )
    )

    # Cheap sanity: each numeric feature's RankIC vs y1 on train (first 8).
    n_feat = min(8, num_x.shape[-1])
    train_mask = _slice(mask_y, *ranges["train"])
    print("single-feature train RankIC (first 8 numeric features):")
    for i in range(n_feat):
        feat = _slice(num_x[..., i], *ranges["train"])
        score = mean_rank_ic(feat, train_y, train_mask)
        print(f"  num_x[..., {i}]: {score:.6f}")


if __name__ == "__main__":
    main()
