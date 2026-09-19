"""Swap a new x6 (or GRU ens) into locked task1 alpha_x fusion. Does not overwrite main."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs"
HIGH_TAU = 4670.0
BASE_VALID = 0.122846
BASE_HI31 = 0.218269
ONLY6_W, NEXT6_W, MLP6_W, MLP_W = 0.4, 0.15, 0.4, 0.15
TAU, W_LO, W_HI = 4546.0, 0.25, 0.6
TREE_W = 0.7


def _v(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_valid.npy")


def _t(name: str) -> np.ndarray:
    return np.load(OUT / f"{name}_test.npy")


def _metrics(pred: np.ndarray, y, my, hi) -> dict:
    s = rank_ic_series(pred, y, my)
    s_hi = s.copy()
    s_hi[~hi] = np.nan
    return {
        "valid_ic": float(mean_rank_ic(pred, y, my)),
        "hi31_ic": float(np.nanmean(s_hi)),
        "neg_days": int(np.nansum(s < 0)),
    }


def _build(x6, o6, n6, tree, mlp, mlp6, mx):
    ens = linear_blend(n6, linear_blend(o6, x6, ONLY6_W), NEXT6_W)
    gated = coverage_gate_blend(ens, tree, mx, TAU, W_LO, W_HI, "raw")
    mlp_ens = linear_blend(mlp6, mlp, MLP6_W)
    blender = FusionModel({"weight_grid": [MLP_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_W, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp_ens, gated, mx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x6", default="gru_x6_full_hicov")
    parser.add_argument("--only6", default="gru_only6_with_today")
    parser.add_argument("--next6", default="gru_next6_with_today")
    parser.add_argument("--bag-x6", nargs="*", default=[], help="rank-bag these stems as x6")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    hi = np.asarray(mx).sum(axis=1) >= HIGH_TAU
    tree_v = linear_blend(_v("hist_lgbm_alpha_x"), _v("baseline"), TREE_W)
    tree_t = linear_blend(_t("hist_lgbm_alpha_x"), _t("baseline"), TREE_W)
    mlp_v, mlp_t = _v("cs_mlp"), _t("cs_mlp")
    mlp6_v, mlp6_t = _v("cs_mlp_only6"), _t("cs_mlp_only6")
    o6_v, o6_t = _v(args.only6), _t(args.only6)
    n6_v, n6_t = _v(args.next6), _t(args.next6)

    if args.bag_x6:
        arrs_v = [_v(s) for s in args.bag_x6]
        arrs_t = [_t(s) for s in args.bag_x6]
        x6_v = np.mean(arrs_v, axis=0).astype(np.float32)
        x6_t = np.mean(arrs_t, axis=0).astype(np.float32)
        tag = args.tag or ("bag_" + "+".join(s.replace("gru_", "") for s in args.bag_x6))
        print(f"x6 raw-mean bag {args.bag_x6}")
    else:
        x6_v, x6_t = _v(args.x6), _t(args.x6)
        tag = args.tag or args.x6
        print(f"x6={args.x6} only6={args.only6} next6={args.next6}")
        print(f"  x6 single valid={mean_rank_ic(x6_v, y, my):.6f} hi31={mean_rank_ic(x6_v[hi], y[hi], my[hi]):.6f}")

    out_v = _build(x6_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)
    m = _metrics(out_v, y, my, hi)
    print(f"fusion valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f} neg={m['neg_days']}")
    print(f"locked     valid={BASE_VALID:.6f} hi31={BASE_HI31:.6f}")
    d_v, d_h = m["valid_ic"] - BASE_VALID, m["hi31_ic"] - BASE_HI31
    print(f"delta valid={d_v:+.6f} hi31={d_h:+.6f}")

    if args.write:
        out_t = _build(x6_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, test["mask_x"])
        sub = ROOT / "submissions" / f"task1_fusion_{tag}.npy"
        if sub.name == "task1_fusion_alpha_x.npy":
            print("skip overwrite task1_fusion_alpha_x.npy")
            print("VALID_RANKIC", f"{m['valid_ic']:.6f}")
            return
        save_submission(out_t, sub)
        np.save(OUT / f"fusion_{tag}_valid.npy", out_v)
        np.save(OUT / f"fusion_{tag}_test.npy", out_t)
        meta = {
            "name": tag,
            "x6": args.x6,
            "bag_x6": args.bag_x6,
            **m,
            "baseline_valid": BASE_VALID,
            "baseline_hi31": BASE_HI31,
            "platform_test_prev": 0.126487,
        }
        (OUT / f"fusion_{tag}_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"wrote {sub}")
    print("VALID_RANKIC", f"{m['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
