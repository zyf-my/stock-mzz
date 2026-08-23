"""Delete abandoned experiment artifacts. Keeps the locked next6 fusion and its branches.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\clean_redundant.py
"""

from __future__ import annotations

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
}


def _clean_dir(folder: Path, keep: set[str]) -> list[str]:
    removed: list[str] = []
    if not folder.is_dir():
        return removed
    for path in folder.iterdir():
        if path.is_dir():
            continue
        if path.name in keep:
            continue
        path.unlink()
        removed.append(str(path.relative_to(ROOT)))
    return removed


def main() -> None:
    removed = []
    removed.extend(_clean_dir(ROOT / "outputs", KEEP_OUTPUTS))
    removed.extend(_clean_dir(ROOT / "checkpoints", KEEP_CKPT))
    removed.extend(_clean_dir(ROOT / "submissions", KEEP_SUB))
    print(f"removed {len(removed)} files")
    for name in removed:
        print(f"  {name}")
    print("kept locked fusion + x6/only6/next6/tree/mlp branches + split_cache/")


if __name__ == "__main__":
    main()
