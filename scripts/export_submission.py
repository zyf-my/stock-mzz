"""Validate and copy a test prediction into submission format.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\export_submission.py --pred outputs/baseline_test.npy --out submissions/task1.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.submit import save_submission  # noqa: E402

EXPECTED_SHAPE = (442, 5282)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", default="submissions/task1.npy")
    parser.add_argument("--expect-shape", default="442,5282")
    args = parser.parse_args()

    pred = np.load(args.pred)
    expect = tuple(int(x) for x in args.expect_shape.split(","))
    if pred.shape != expect:
        print(f"warning: shape {pred.shape} != expected {expect}")
    path = save_submission(pred, args.out)
    loaded = np.load(path)
    print(f"wrote {path} shape={loaded.shape} dtype={loaded.dtype}")
    if loaded.shape != EXPECTED_SHAPE:
        print(f"current dataset expects {EXPECTED_SHAPE}; re-check test_start_idx if this differs")


if __name__ == "__main__":
    main()
