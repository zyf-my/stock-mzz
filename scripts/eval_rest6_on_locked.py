"""Stack a new GRU onto the locked next6 fusion. Does not change the 2-level gate.

Weights follow the next6 protocol: {0.10, 0.15, 0.25} in raw and rank.
Never overwrites submissions/task1_fusion_next6_wt_mlp6.npy.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_rest6_on_locked.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import load_eval_splits  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

LOCKED_VALID = 0.120542
WEIGHTS = [0.10, 0.15, 0.25]
UPLOAD_BAR = 0.001


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    cache_dir = ROOT / "outputs" / "split_cache"
    data_path = None
    if not (cache_dir / "valid.npz").is_file() or not (cache_dir / "test.npz").is_file():
        data_path = resolve_data_path(cfg, None)
    splits, src = load_eval_splits(cache_dir=cache_dir, data_path=data_path, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    print(f"loaded {src} in {time.perf_counter() - t0:.1f}s")

    locked_v = np.load(ROOT / "outputs/fusion_next6_wt_mlp6_valid.npy")
    locked_t = np.load(ROOT / "outputs/fusion_next6_wt_mlp6_test.npy")
    new_v = np.load(ROOT / "outputs/gru_rest6_with_today_valid.npy")
    new_t = np.load(ROOT / "outputs/gru_rest6_with_today_test.npy")
    print(f"locked RankIC={mean_rank_ic(locked_v, y, my):.6f}")
    print(f"rest6 RankIC={mean_rank_ic(new_v, y, my):.6f}")

    results = []
    for space in ("raw", "rank"):
        for w in WEIGHTS:
            if space == "rank":
                blender = FusionModel({"weight_grid": [w]})
                blender.locked = {"name": "rank_blend", "weight": w, "space": "rank", "ic": float("nan")}
                pred = blender.predict(new_v, locked_v, mx)
            else:
                pred = linear_blend(new_v, locked_v, w)
            ic = mean_rank_ic(pred, y, my)
            series = rank_ic_series(pred, y, my)
            row = {
                "space": space,
                "weight": w,
                "ic": ic,
                "neg": int((series < 0).sum()),
                "min": float(np.nanmin(series)),
            }
            results.append(row)
            print(
                f"  {space:4s} w={w:.2f}  RankIC={ic:.6f}  "
                f"delta={ic - LOCKED_VALID:+.6f}  neg={row['neg']}"
            )

    best = max(results, key=lambda r: r["ic"])
    print(f"BEST {best['space']} w={best['weight']:.2f} ic={best['ic']:.6f}")
    beat = best["ic"] > LOCKED_VALID + 1e-4
    upload_ok = best["ic"] > LOCKED_VALID + UPLOAD_BAR
    if beat:
        w = float(best["weight"])
        if best["space"] == "rank":
            blender = FusionModel({"weight_grid": [w]})
            blender.locked = {"name": "rank_blend", "weight": w, "space": "rank", "ic": float("nan")}
            out_v = blender.predict(new_v, locked_v, mx)
            out_t = blender.predict(new_t, locked_t, test["mask_x"])
        else:
            out_v = linear_blend(new_v, locked_v, w)
            out_t = linear_blend(new_t, locked_t, w)
        np.save(ROOT / "outputs/fusion_next6_rest6_valid.npy", out_v)
        np.save(ROOT / "outputs/fusion_next6_rest6_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions/task1_fusion_next6_rest6.npy")
        print("wrote submissions/task1_fusion_next6_rest6.npy")
        print("does not overwrite submissions/task1_fusion_next6_wt_mlp6.npy")
        if not upload_ok:
            print(f"delta < {UPLOAD_BAR}; do not upload (gate3 lesson)")
    else:
        print("did not beat locked 0.120542; not writing a new submission")

    meta = {
        "locked": LOCKED_VALID,
        "best": best,
        "results": results,
        "upload_ok": bool(upload_ok),
    }
    (ROOT / "outputs/fusion_next6_rest6_lock.json").write_text(
        json.dumps(meta, indent=2, default=float), encoding="utf-8"
    )
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{best['ic']:.6f}")


if __name__ == "__main__":
    main()
