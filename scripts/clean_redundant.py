"""Delete abandoned experiment artifacts. Keeps locked task1 + task2 live branches.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\clean_redundant.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEEP_OUTPUTS = {
    ".gitkeep",
    "baseline_importance.csv",
    "baseline_test.npy",
    "baseline_valid.npy",
    "cs_mlp_only6_test.npy",
    "cs_mlp_only6_valid.npy",
    "cs_mlp_test.npy",
    "cs_mlp_valid.npy",
    "fusion_lock.json",
    "fusion_x6_today_lock.json",
    "fusion_x6_today_test.npy",
    "fusion_x6_today_valid.npy",
    "fusion_next6_wt_mlp6_lock.json",
    "fusion_next6_wt_mlp6_test.npy",
    "fusion_next6_wt_mlp6_valid.npy",
    "fusion_test.npy",
    "fusion_valid.npy",
    "gru_next6_with_today_test.npy",
    "gru_next6_with_today_valid.npy",
    "gru_no_today_recent_n2000_x6_test.npy",
    "gru_no_today_recent_n2000_x6_valid.npy",
    "gru_only6_with_today_test.npy",
    "gru_only6_with_today_valid.npy",
    "gru_x6_with_today_test.npy",
    "gru_x6_with_today_valid.npy",
    "hist_lgbm_importance.csv",
    "hist_lgbm_test.npy",
    "hist_lgbm_valid.npy",
}
KEEP_CKPT = {
    ".gitkeep",
    "baseline.txt",
    "baseline.txt.meta.json",
    "cs_mlp.pt",
    "cs_mlp.pt.meta.json",
    "cs_mlp_only6.pt",
    "cs_mlp_only6.pt.meta.json",
    "gru_next6_with_today.pt",
    "gru_next6_with_today.pt.meta.json",
    "gru_no_today_recent_n2000_x6.pt",
    "gru_no_today_recent_n2000_x6.pt.meta.json",
    "gru_only6_with_today.pt",
    "gru_only6_with_today.pt.meta.json",
    "gru_x6_with_today.pt",
    "gru_x6_with_today.pt.meta.json",
    "hist_lgbm.txt",
    "hist_lgbm.txt.meta.json",
}
KEEP_SUB = {
    ".gitkeep",
    "task1_fusion_next6_wt_mlp6.npy",
    "task1_fusion_x6_today.npy",
    "task1_fusion_alpha_tree.npy",
    "task1_fusion_alpha_x.npy",
}
KEEP_SUB_PREFIX = ("task2_",)
KEEP_OUTPUT_DIRS = {"split_cache", "task2", "x6_today_fusion"}
KEEP_CKPT_DIRS = {"task2"}


def _clean_dir(folder: Path, keep: set[str], keep_prefix: tuple[str, ...] = ()) -> list[str]:
    removed: list[str] = []
    if not folder.is_dir():
        return removed
    for path in folder.iterdir():
        if path.is_dir():
            continue
        if path.name in keep or path.name.startswith(keep_prefix):
            continue
        path.unlink()
        removed.append(str(path.relative_to(ROOT)))
    return removed


def _clean_subdirs(folder: Path, keep_dirs: set[str]) -> list[str]:
    removed: list[str] = []
    if not folder.is_dir():
        return removed
    for path in folder.iterdir():
        if not path.is_dir() or path.name in keep_dirs:
            continue
        shutil.rmtree(path)
        removed.append(str(path.relative_to(ROOT)) + "/")
    return removed


def main() -> None:
    removed = []
    removed.extend(_clean_dir(ROOT / "outputs", KEEP_OUTPUTS))
    removed.extend(_clean_subdirs(ROOT / "outputs", KEEP_OUTPUT_DIRS))
    removed.extend(_clean_dir(ROOT / "checkpoints", KEEP_CKPT))
    removed.extend(_clean_subdirs(ROOT / "checkpoints", KEEP_CKPT_DIRS))
    removed.extend(_clean_dir(ROOT / "submissions", KEEP_SUB, KEEP_SUB_PREFIX))
    print(f"removed {len(removed)} files")
    for name in removed:
        print(f"  {name}")
    print("kept task1 locked + previous main + live branches + split_cache/ + task2/")


if __name__ == "__main__":
    main()
