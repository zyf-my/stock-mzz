"""Reproduce the locked task-1 submission.

Daily entry. Do not run two full-panel jobs at once (panel is ~8.5GB).

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\reproduce.py
    .\\.venv\\Scripts\\python.exe scripts\\reproduce.py --from-preds
    .\\.venv\\Scripts\\python.exe scripts\\reproduce.py --force
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS: list[dict[str, object]] = [
    {
        "name": "baseline",
        "argv": [sys.executable, "scripts/train_baseline.py", "--config", "configs/baseline.yaml"],
        "outputs": ["outputs/baseline_valid.npy", "outputs/baseline_test.npy"],
    },
    {
        "name": "hist_lgbm",
        "argv": [sys.executable, "scripts/train_baseline.py", "--config", "configs/hist_lgbm.yaml"],
        "outputs": ["outputs/hist_lgbm_valid.npy", "outputs/hist_lgbm_test.npy"],
    },
    {
        "name": "tree_fusion",
        "argv": [sys.executable, "scripts/train_fusion.py", "--config", "configs/fusion.yaml"],
        "outputs": ["outputs/fusion_valid.npy", "outputs/fusion_test.npy"],
    },
    {
        "name": "gru_x6",
        "argv": [
            sys.executable,
            "scripts/train_gru.py",
            "--config",
            "configs/gru_no_today_recent_n2000_x6.yaml",
        ],
        "outputs": [
            "outputs/gru_no_today_recent_n2000_x6_valid.npy",
            "outputs/gru_no_today_recent_n2000_x6_test.npy",
        ],
    },
    {
        "name": "gru_only6",
        "argv": [
            sys.executable,
            "scripts/train_gru.py",
            "--config",
            "configs/gru_only6_with_today.yaml",
        ],
        "outputs": [
            "outputs/gru_only6_with_today_valid.npy",
            "outputs/gru_only6_with_today_test.npy",
        ],
    },
    {
        "name": "gru_next6",
        "argv": [
            sys.executable,
            "scripts/train_gru.py",
            "--config",
            "configs/gru_next6_with_today.yaml",
        ],
        "outputs": [
            "outputs/gru_next6_with_today_valid.npy",
            "outputs/gru_next6_with_today_test.npy",
        ],
    },
    {
        "name": "cs_mlp",
        "argv": [sys.executable, "scripts/train_cs_mlp.py", "--config", "configs/cs_mlp.yaml"],
        "outputs": ["outputs/cs_mlp_valid.npy", "outputs/cs_mlp_test.npy"],
    },
    {
        "name": "cs_mlp_only6",
        "argv": [sys.executable, "scripts/train_cs_mlp.py", "--config", "configs/cs_mlp_only6.yaml"],
        "outputs": ["outputs/cs_mlp_only6_valid.npy", "outputs/cs_mlp_only6_test.npy"],
    },
]

FUSE = {
    "name": "lock_fusion",
    "argv": [sys.executable, "scripts/eval_next6_fusion.py"],
    "outputs": ["submissions/task1_fusion_next6_wt_mlp6.npy"],
}

EXPECTED_VALID = 0.120542
SUBMISSION = "submissions/task1_fusion_next6_wt_mlp6.npy"


def _ready(rel_paths: list[str]) -> bool:
    return all((ROOT / p).is_file() for p in rel_paths)


def _run(name: str, argv: list[str]) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(argv))
    subprocess.run(argv, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce locked task-1 fusion")
    parser.add_argument("--force", action="store_true", help="retrain every branch")
    parser.add_argument(
        "--from-preds",
        action="store_true",
        help="skip training; only fuse existing valid/test npy",
    )
    args = parser.parse_args()

    if args.from_preds:
        missing = []
        for step in STEPS:
            for path in step["outputs"]:  # type: ignore[union-attr]
                if not (ROOT / str(path)).is_file():
                    missing.append(str(path))
        if missing:
            raise SystemExit("missing predictions for --from-preds:\n  " + "\n  ".join(missing))
        _run(str(FUSE["name"]), list(FUSE["argv"]))  # type: ignore[arg-type]
        print(f"\nexpected valid RankIC ~ {EXPECTED_VALID:.6f}")
        print(f"submission {ROOT / SUBMISSION}")
        return

    for step in STEPS:
        name = str(step["name"])
        outputs = [str(p) for p in step["outputs"]]  # type: ignore[union-attr]
        if not args.force and _ready(outputs):
            print(f"skip {name} (outputs exist)")
            continue
        _run(name, list(step["argv"]))  # type: ignore[arg-type]

    _run(str(FUSE["name"]), list(FUSE["argv"]))  # type: ignore[arg-type]
    print(f"\nexpected valid RankIC ~ {EXPECTED_VALID:.6f}")
    print(f"submission {ROOT / SUBMISSION}")
    print("do not run a second full-panel job in parallel")


if __name__ == "__main__":
    main()
