"""Panel slices and sample construction.

Stage 2 should flatten (t, stock) rows here. Do not shuffle across time.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal

import numpy as np

from .io import read_zstd, split_ranges

SplitName = Literal["train", "valid", "test", "history"]


def load_panel(path: str) -> dict[str, Any]:
    return read_zstd(path)


def split_bounds(data: dict[str, Any], split: SplitName) -> tuple[int, int]:
    t_len = int(data["num_x"].shape[0])
    ranges = split_ranges(data)
    if split == "history":
        return 0, int(data["train_start_idx"])
    start, end = ranges[split]
    return start, t_len if end is None else end


def slice_split(data: dict[str, Any], split: SplitName) -> dict[str, Any]:
    start, end = split_bounds(data, split)
    out = {
        "start": start,
        "end": end,
        "num_x": data["num_x"][start:end],
        "cat_x": data["cat_x"][start:end],
        "y1": data["y1"][start:end],
        "mask_x": data["mask_x"][start:end],
        "mask_y": data["mask_y"][start:end],
    }
    if "y2" in data:
        out["y2"] = data["y2"][start:end]
    return out


def iter_days(split_data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one trading day at a time to avoid holding extra copies."""
    n_days = split_data["num_x"].shape[0]
    for t in range(n_days):
        item = {
            "t": t,
            "num_x": split_data["num_x"][t],
            "cat_x": split_data["cat_x"][t],
            "y1": split_data["y1"][t],
            "mask_x": split_data["mask_x"][t],
            "mask_y": split_data["mask_y"][t],
        }
        if "y2" in split_data:
            item["y2"] = split_data["y2"][t]
        yield item


def flatten_masked_rows(
    split_data: dict[str, Any],
    cat_indices: list[int],
    require_label: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a split into LightGBM-style rows.

    Returns
    -------
    x : (N, Nn + len(cat_indices))
    y : (N,)
    coords : (N, 2) with columns [local_t, stock]
    """
    raise NotImplementedError("阶段 2：在此把 mask 后的 (t, stock) 摊成二维表")
