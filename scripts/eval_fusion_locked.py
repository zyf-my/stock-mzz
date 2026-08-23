"""Generic locked next6 fusion eval; swap GRU x6/o6/n6 pred paths via CLI.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_fusion_locked.py \\
        --x6 outputs/gru_x6_today_ic_valid.npy --x6-test outputs/gru_x6_today_ic_test.npy \\
        --tag x6_today_ic
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
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

ONLY6_W = 0.4
NEXT6_W = 0.15
MLP6_W = 0.4
MLP_STACK_W = 0.15
TAU = 4546.0
W_LOW = 0.25
W_HIGH = 0.6
UPLOAD_BAR = 0.121542


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--x6", default="outputs/gru_no_today_recent_n2000_x6_valid.npy")
    p.add_argument("--x6-test", default="outputs/gru_no_today_recent_n2000_x6_test.npy")
    p.add_argument("--o6", default="outputs/gru_only6_with_today_valid.npy")
    p.add_argument("--o6-test", default="outputs/gru_only6_with_today_test.npy")
    p.add_argument("--n6", default="outputs/gru_next6_with_today_valid.npy")
    p.add_argument("--n6-test", default="outputs/gru_next6_with_today_test.npy")
    p.add_argument("--tag", default="custom")
    args = p.parse_args()

    t0 = time.perf_counter()
    cfg = load_config("configs/default.yaml")
    cache_dir = ROOT / "outputs" / "split_cache"
    data_path = None
    if not (cache_dir / "valid.npz").is_file():
        data_path = resolve_data_path(cfg, None)
    splits, src = load_eval_splits(cache_dir=cache_dir, data_path=data_path, splits=("valid", "test"))
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    print(f"loaded {src} tag={args.tag} in {time.perf_counter() - t0:.1f}s")

    x6_v = np.load(ROOT / args.x6)
    x6_t = np.load(ROOT / args.x6_test)
    o6_v = np.load(ROOT / args.o6)
    o6_t = np.load(ROOT / args.o6_test)
    n6_v = np.load(ROOT / args.n6)
    n6_t = np.load(ROOT / args.n6_test)
    tree_v = np.load(ROOT / "outputs/fusion_valid.npy")
    tree_t = np.load(ROOT / "outputs/fusion_test.npy")
    mlp_v = np.load(ROOT / "outputs/cs_mlp_valid.npy")
    mlp_t = np.load(ROOT / "outputs/cs_mlp_test.npy")
    mlp6_v = np.load(ROOT / "outputs/cs_mlp_only6_valid.npy")
    mlp6_t = np.load(ROOT / "outputs/cs_mlp_only6_test.npy")

    ens_v = linear_blend(n6_v, linear_blend(o6_v, x6_v, ONLY6_W), NEXT6_W)
    ens_t = linear_blend(n6_t, linear_blend(o6_t, x6_t, ONLY6_W), NEXT6_W)
    gated_v = coverage_gate_blend(ens_v, tree_v, mx, TAU, W_LOW, W_HIGH, "raw")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], TAU, W_LOW, W_HIGH, "raw")
    mlp_ens_v = linear_blend(mlp6_v, mlp_v, MLP6_W)
    mlp_ens_t = linear_blend(mlp6_t, mlp_t, MLP6_W)
    blender = FusionModel({"weight_grid": [MLP_STACK_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_STACK_W, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_ens_v, gated_v, mx)
    out_t = blender.predict(mlp_ens_t, gated_t, test["mask_x"])
    ic = mean_rank_ic(out_v, y, my)
    print(f"x6={mean_rank_ic(x6_v, y, my):.6f} gru={mean_rank_ic(ens_v, y, my):.6f} "
          f"gate={mean_rank_ic(gated_v, y, my):.6f} fusion={ic:.6f} bar={UPLOAD_BAR:.6f}")

    out_dir = ROOT / "outputs" / f"fusion_locked_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "valid.npy", out_v)
    np.save(out_dir / "test.npy", out_t)
    summary = {"tag": args.tag, "fusion_ic": float(ic), "passes_bar": bool(ic >= UPLOAD_BAR)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary["passes_bar"]:
        save_submission(out_t, ROOT / f"submissions/task1_fusion_{args.tag}.npy")
        print(f"saved submissions/task1_fusion_{args.tag}.npy")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
