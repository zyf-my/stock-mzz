"""Report labeled days we do not fit on, and cat_5 coverage. No training."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import load_panel  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402


def _block(name: str, y, my, mx, a: int, b: int) -> None:
    yy, m, x = y[a:b], my[a:b], mx[a:b]
    nlab = int(m.sum())
    nfin = int(np.isfinite(yy[m]).sum()) if nlab else 0
    mean = float(np.nanmean(yy[m])) if nlab else float("nan")
    print(
        f"{name:6s} days={b-a} mask_y={nlab} finite_y={nfin} "
        f"mask_x={int(x.sum())} ymean={mean:.3f} "
        f"cov_x={int(x.sum(1).min())}-{int(x.sum(1).max())}"
    )


def main() -> None:
    cfg = load_config("configs/default.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    print("keys", sorted(data.keys()))
    tr = int(data["train_start_idx"])
    va = int(data["valid_start_idx"])
    te = int(data["test_start_idx"])
    y = np.asarray(data["y1"])
    my = np.asarray(data["mask_y"])
    mx = np.asarray(data["mask_x"])
    print("idx", tr, va, te, "T", y.shape[0], "S", y.shape[1])
    _block("buffer", y, my, mx, 0, tr)
    _block("train", y, my, mx, tr, va)
    _block("valid", y, my, mx, va, te)
    _block("test", y, my, mx, te, y.shape[0])
    if my[:tr].any():
        print("buffer RankIC col8", mean_rank_ic(data["num_x"][:tr, :, 8], y[:tr], my[:tr]))
    cat5 = np.asarray(data["cat_x"][..., 5])
    print("cat5 unique buffer", len(np.unique(cat5[:tr][mx[:tr]])))
    print("cat5 unique train", len(np.unique(cat5[tr:va][mx[tr:va]])))
    print("cat5 unique valid", len(np.unique(cat5[va:te][mx[va:te]])))
    print("cat5 unique test", len(np.unique(cat5[te:][mx[te:]])))


if __name__ == "__main__":
    main()
