"""Train cov4k + rank800 x6 variants."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
CONFIGS = [
    "configs/task2/gru_x6_cov4k.yaml",
    "configs/task2/gru_x6_rank800.yaml",
]


def main() -> None:
    t0 = time.perf_counter()
    for cfg in CONFIGS:
        print(f"\n=== {cfg} ===", flush=True)
        rc = subprocess.call([str(PY), "-u", "scripts/train_gru.py", "--config", cfg], cwd=str(ROOT))
        if rc != 0:
            raise SystemExit(rc)
    print(f"done {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
