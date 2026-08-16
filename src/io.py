"""Load the Jingge contest dict.

Official snippet:
    pickle.loads(zstd.loads(pd.read_pickle(file)))

The published file is a pickle-wrapped zstd blob, so pickle.load is enough;
pandas is only a fallback. Keep the data on an encrypted local disk.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def read_zstd(file: str | Path) -> dict[str, Any]:
    with open(file, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj
    if not isinstance(obj, (bytes, bytearray, memoryview)):
        obj = _read_outer_pickle(file)
    raw = _zstd_decompress(bytes(obj))
    data = pickle.loads(raw)
    if not isinstance(data, dict):
        raise TypeError(f"expected dict after decode, got {type(data)!r}")
    return data


def _read_outer_pickle(file: str | Path) -> bytes:
    with open(file, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    try:
        import pandas as pd

        payload = pd.read_pickle(file)
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"outer pickle is not bytes: {type(payload)!r}") from exc
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"outer pickle is not bytes: {type(payload)!r}")
    return bytes(payload)


def _zstd_decompress(payload: bytes) -> bytes:
    try:
        import zstd

        if hasattr(zstd, "loads"):
            return zstd.loads(payload)
        if hasattr(zstd, "decompress"):
            return zstd.decompress(payload)
    except Exception:
        pass

    import zstandard

    return zstandard.ZstdDecompressor().decompress(payload)


def split_ranges(data: dict[str, Any]) -> dict[str, tuple[int, int | None]]:
    train_start = int(data["train_start_idx"])
    valid_start = int(data["valid_start_idx"])
    test_start = int(data["test_start_idx"])
    return {
        "train": (train_start, valid_start),
        "valid": (valid_start, test_start),
        "test": (test_start, None),
    }
