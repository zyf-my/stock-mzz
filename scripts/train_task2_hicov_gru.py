"""Train task2 hicov GRU trio sequentially (x6 / only6 / next6)."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

CONFIGS = [
    "configs/task2/gru_x6_hicov.yaml",
    "configs/task2/gru_only6_hicov.yaml",
    "configs/task2/gru_next6_hicov.yaml",
]


def main() -> None:
    t0 = time.perf_counter()
    for cfg in CONFIGS:
        print(f"\n=== train {cfg} ===", flush=True)
        rc = subprocess.call([str(PY), "-u", "scripts/train_gru.py", "--config", cfg], cwd=str(ROOT))
        if rc != 0:
            raise SystemExit(f"failed {cfg} exit={rc}")
    print(f"\nall hicov GRU done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
