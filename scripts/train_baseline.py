"""Stage 2: train the cross-section LightGBM baseline.

Usage (after implementation):
    .\\.venv\\Scripts\\python.exe scripts\\train_baseline.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM cross-section baseline")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data", default=None, help="overrides JINGGE_DATA and config data.path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"config={args.config}")
    print("seed=", cfg.get("seed"))
    try:
        data_path = resolve_data_path(cfg, args.data)
        print(f"data={data_path}")
    except FileNotFoundError as exc:
        print(f"data path not set yet: {exc}")
    raise SystemExit(
        "阶段 2 训练尚未实现。下一步填 src/dataset.py.flatten_masked_rows "
        "与 src/models/baseline.py，验收标准见 计划.md。"
    )


if __name__ == "__main__":
    main()
