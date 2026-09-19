"""Train full task2 (y2) pipeline mirroring task1 x6_today main recipe.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\run_task2_main.py
    .\\.venv\\Scripts\\python.exe -u scripts\\run_task2_main.py --from gru_x6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

STEPS: list[tuple[str, list[str]]] = [
    ("cache", [str(PY), "-u", "scripts/dump_split_cache.py", "--force"]),
    ("baseline", [str(PY), "-u", "scripts/train_baseline.py", "--config", "configs/task2/baseline.yaml"]),
    ("hist_lgbm", [str(PY), "-u", "scripts/train_baseline.py", "--config", "configs/task2/hist_lgbm.yaml"]),
    ("hist_n200", [str(PY), "-u", "scripts/eval_task2_hist_n200.py"]),
    ("fusion_trees", [str(PY), "-u", "scripts/train_fusion.py", "--config", "configs/task2/fusion.yaml"]),    ("gru_x6", [str(PY), "-u", "scripts/train_gru.py", "--config", "configs/task2/gru_x6_with_today.yaml"]),
    ("gru_only6", [str(PY), "-u", "scripts/train_gru.py", "--config", "configs/task2/gru_only6_with_today.yaml"]),
    ("gru_next6", [str(PY), "-u", "scripts/train_gru.py", "--config", "configs/task2/gru_next6_with_today.yaml"]),
    ("cs_mlp", [str(PY), "-u", "scripts/train_cs_mlp.py", "--config", "configs/task2/cs_mlp.yaml"]),
    ("cs_mlp_only6", [str(PY), "-u", "scripts/train_cs_mlp.py", "--config", "configs/task2/cs_mlp_only6.yaml"]),
    ("eval_fusion", [str(PY), "-u", "scripts/eval_task2_fusion_search.py"]),
    ("eval_locked", [str(PY), "-u", "scripts/eval_task2_locked_best.py"]),]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task2 y2 training pipeline")
    parser.add_argument("--from", dest="start", default=None, help="step name to start from")
    args = parser.parse_args()

    names = [n for n, _ in STEPS]
    start_idx = 0
    if args.start:
        if args.start not in names:
            raise SystemExit(f"unknown step {args.start!r}; choose from {names}")
        start_idx = names.index(args.start)

    t0 = time.perf_counter()
    for name, cmd in STEPS[start_idx:]:
        print(f"\n=== task2 step: {name} ===", flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            raise SystemExit(f"step {name} failed with exit code {rc}")
    print(f"\ntask2 pipeline done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
