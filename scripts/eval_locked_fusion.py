"""Apply the locked 0.117 fusion recipe to a new GRU branch. Do not re-search tau.

Locked from fusion-gru-cov-001 / gru-no-today-recent-n2000-001:
  coverage raw, tau=4546, GRU weight 0.25 (low cov) / 0.4 (high cov)
  then rank-blend original cs_mlp at 0.15 and 0.25 (report both, write the better)

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_locked_fusion.py --gru-valid outputs/gru_no_today_recent_nfull_valid.npy --gru-test outputs/gru_no_today_recent_nfull_test.npy --tag nfull
"""

from __future__ import annotations

import argparse
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
from src.models.fusion import FusionModel, coverage_gate_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

LOCK_TAU = 4546.0
LOCK_W_LOW = 0.25
LOCK_W_HIGH = 0.4
LOCK_SPACE = "raw"
PREV_BEST = 0.117177


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gru-valid", required=True)
    parser.add_argument("--gru-test", required=True)
    parser.add_argument("--tree-valid", default="outputs/fusion_valid.npy")
    parser.add_argument("--tree-test", default="outputs/fusion_test.npy")
    parser.add_argument("--mlp-valid", default="outputs/cs_mlp_valid.npy")
    parser.add_argument("--mlp-test", default="outputs/cs_mlp_test.npy")
    parser.add_argument("--tag", default="nfull")
    parser.add_argument("--data", default=None)
    parser.add_argument("--w-high", type=float, default=LOCK_W_HIGH)
    parser.add_argument("--w-low", type=float, default=LOCK_W_LOW)
    parser.add_argument("--tau", type=float, default=LOCK_TAU)
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    cache_dir = ROOT / "outputs" / "split_cache"
    data_path = None
    if not (cache_dir / "valid.npz").is_file() or not (cache_dir / "test.npz").is_file():
        data_path = resolve_data_path(cfg, args.data)
    splits, src = load_eval_splits(cache_dir=cache_dir, data_path=data_path, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    print(f"loaded {src} in {time.perf_counter() - t0:.1f}s")

    gru_v = np.load(ROOT / args.gru_valid)
    gru_t = np.load(ROOT / args.gru_test)
    tree_v = np.load(ROOT / args.tree_valid)
    tree_t = np.load(ROOT / args.tree_test)
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]

    gru_ic = mean_rank_ic(gru_v, y, my)
    tree_ic = mean_rank_ic(tree_v, y, my)
    print(f"GRU valid RankIC={gru_ic:.6f}")
    print(f"tree fusion valid RankIC={tree_ic:.6f}")

    tau = float(args.tau)
    w_low = float(args.w_low)
    w_high = float(args.w_high)
    gated_v = coverage_gate_blend(gru_v, tree_v, mx, tau, w_low, w_high, LOCK_SPACE)
    gated_t = coverage_gate_blend(gru_t, tree_t, test["mask_x"], tau, w_low, w_high, LOCK_SPACE)
    gated_ic = mean_rank_ic(gated_v, y, my)
    print(f"locked coverage gate RankIC={gated_ic:.6f} tau={tau} w={w_low}/{w_high} {LOCK_SPACE}")

    series = rank_ic_series(gated_v, y, my)
    print(f"gated daily RankIC min={np.nanmin(series):.4f} median={np.nanmedian(series):.4f} max={np.nanmax(series):.4f}")

    tag = args.tag
    gate_sub = ROOT / f"submissions/task1_fusion_{tag}_cov.npy"
    np.save(ROOT / f"outputs/fusion_{tag}_cov_valid.npy", gated_v)
    np.save(ROOT / f"outputs/fusion_{tag}_cov_test.npy", gated_t)
    save_submission(gated_t, gate_sub)

    mlp_v_path = ROOT / args.mlp_valid
    stacked_ic = None
    stacked_name = None
    if mlp_v_path.is_file():
        cs_v = np.load(mlp_v_path)
        cs_t = np.load(ROOT / args.mlp_test)
        blender = FusionModel({"weight_grid": [0.0, 0.15, 0.25, 0.4]})
        stacked = blender.fit(cs_v, gated_v, y, my, mx)
        print("stack cs_mlp on gated GRU:")
        for row in (stacked.get("valid_leaderboard") or [])[:6]:
            extra = {k: v for k, v in row.items() if k not in {"valid_leaderboard", "name", "ic"}}
            print(f"  {row['ic']:.6f}  {row['name']}  {extra}")
        stacked_ic = float(stacked["ic"])
        stacked_name = stacked["name"]
        print(f"STACK_LOCKED {stacked_name} ic={stacked_ic:.6f}")
        out_v = blender.predict(cs_v, gated_v, mx)
        out_t = blender.predict(cs_t, gated_t, test["mask_x"])
        np.save(ROOT / f"outputs/fusion_{tag}_cov_mlp_valid.npy", out_v)
        np.save(ROOT / f"outputs/fusion_{tag}_cov_mlp_test.npy", out_t)
        save_submission(out_t, ROOT / f"submissions/task1_fusion_{tag}_cov_mlp.npy")
        print(f"wrote submissions/task1_fusion_{tag}_cov_mlp.npy")
    else:
        print(f"no {mlp_v_path}, skip cs_mlp stack")

    best = stacked_ic if stacked_ic is not None else gated_ic
    meta = {
        "gru_ic": gru_ic,
        "tree_ic": tree_ic,
        "gated_ic": gated_ic,
        "stacked_ic": stacked_ic,
        "stacked_name": stacked_name,
        "lock": {"tau": tau, "weight_low": w_low, "weight_high": w_high, "space": LOCK_SPACE},
        "prev_best": PREV_BEST,
        "beat_prev": bool(best is not None and best > PREV_BEST + 1e-4),
    }
    lock_path = ROOT / f"outputs/fusion_{tag}_lock.json"
    lock_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {lock_path}")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{best:.6f}")
    if best <= PREV_BEST + 1e-4:
        print(f"did not beat prev best {PREV_BEST:.6f}")


if __name__ == "__main__":
    main()
