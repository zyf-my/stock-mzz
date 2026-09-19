"""Plug an Alpha-style tree into locked task1 fusion.

Default stem is the current main tree. Never overwrites
submissions/task1_fusion_alpha_x.npy unless --replace-main.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_alpha_fusion.py
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_alpha_fusion.py --stem hist_lgbm_alpha_x --tag alpha_x
"""

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
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs"
SUB = ROOT / "submissions"
HIGH_TAU = 4670.0
BASE_VALID = 0.122846
BASE_HI31 = 0.218269
ONLY6_W, NEXT6_W, MLP6_W, MLP_W = 0.4, 0.15, 0.4, 0.15
TAU, W_LO, W_HI = 4546.0, 0.25, 0.6
TREE_W = 0.7


def _m(pred, y, my, hi):
    s = rank_ic_series(pred, y, my)
    s_hi = s.copy()
    s_hi[~hi] = np.nan
    return float(np.nanmean(s)), float(np.nanmean(s_hi))


def _locked(x6, o6, n6, tree, mlp, mlp6, mx):
    ens = linear_blend(n6, linear_blend(o6, x6, ONLY6_W), NEXT6_W)
    gated = coverage_gate_blend(ens, tree, mx, TAU, W_LO, W_HI, "raw")
    mlp_ens = linear_blend(mlp6, mlp, MLP6_W)
    blender = FusionModel({"weight_grid": [MLP_W]})
    blender.locked = {"name": "rank_blend", "weight": MLP_W, "space": "rank", "ic": float("nan")}
    return blender.predict(mlp_ens, gated, mx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default="hist_lgbm_alpha_x")
    parser.add_argument("--tag", default="")
    parser.add_argument("--replace-main", action="store_true")
    args = parser.parse_args()
    tag = args.tag or args.stem.replace("hist_lgbm_", "")
    main_name = "task1_fusion_alpha_x.npy"

    splits, src = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    mxt = test["mask_x"]
    hi = np.asarray(mx).sum(1) >= HIGH_TAU
    print(f"cache={src} hi31={int(hi.sum())} stem={args.stem} tag={tag}")

    x6_v, x6_t = np.load(OUT / "gru_x6_with_today_valid.npy"), np.load(OUT / "gru_x6_with_today_test.npy")
    o6_v, o6_t = np.load(OUT / "gru_only6_with_today_valid.npy"), np.load(OUT / "gru_only6_with_today_test.npy")
    n6_v, n6_t = np.load(OUT / "gru_next6_with_today_valid.npy"), np.load(OUT / "gru_next6_with_today_test.npy")
    mlp_v, mlp_t = np.load(OUT / "cs_mlp_valid.npy"), np.load(OUT / "cs_mlp_test.npy")
    mlp6_v, mlp6_t = np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_only6_test.npy")
    base_v, base_t = np.load(OUT / "baseline_valid.npy"), np.load(OUT / "baseline_test.npy")
    main_alpha_v = np.load(OUT / "hist_lgbm_alpha_x_valid.npy")
    main_alpha_t = np.load(OUT / "hist_lgbm_alpha_x_test.npy")
    alpha_v = np.load(OUT / f"{args.stem}_valid.npy")
    alpha_t = np.load(OUT / f"{args.stem}_test.npy")

    main_tree_v = linear_blend(main_alpha_v, base_v, TREE_W)
    tree_v = linear_blend(alpha_v, base_v, TREE_W)
    tree_t = linear_blend(alpha_t, base_t, TREE_W)
    av, ah = _m(alpha_v, y, my, hi)
    tv, th = _m(tree_v, y, my, hi)
    print(f"{args.stem} single valid={av:.6f} hi31={ah:.6f}")
    print(f"{args.stem}+baseline0.3 valid={tv:.6f} hi31={th:.6f}")

    locked_v = _locked(x6_v, o6_v, n6_v, main_tree_v, mlp_v, mlp6_v, mx)
    lv, lh = _m(locked_v, y, my, hi)
    print(f"rebuilt locked valid={lv:.6f} hi31={lh:.6f}")

    pred_v = _locked(x6_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)
    pred_t = _locked(x6_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)
    fv, fh = _m(pred_v, y, my, hi)
    print(
        f"replace_tree valid={fv:.6f} ({fv - BASE_VALID:+.6f})  "
        f"hi31={fh:.6f} ({fh - BASE_HI31:+.6f})"
    )
    np.save(OUT / f"fusion_{tag}_valid.npy", pred_v)
    np.save(OUT / f"fusion_{tag}_test.npy", pred_t)
    sub = SUB / f"task1_fusion_{tag}.npy"
    if sub.name == main_name and not args.replace_main:
        print(f"skip overwrite {main_name}")
    else:
        save_submission(pred_t, sub)
        print(f"wrote {sub.as_posix()}")
    ok = fh >= BASE_HI31 - 1e-6 and fv >= BASE_VALID - 0.0005 and (fv > BASE_VALID + 1e-6 or fh > BASE_HI31 + 1e-6)
    print("VERDICT", "candidate" if ok else "keep_main")
    (OUT / f"{tag}_fusion_summary.json").write_text(
        json.dumps(
            {
                "stem": args.stem,
                "tag": tag,
                "alpha_single": {"valid": av, "hi31": ah},
                "alpha_tree": {"valid": tv, "hi31": th},
                "locked": {"valid": lv, "hi31": lh},
                "replace_tree": {"valid": fv, "hi31": fh},
                "verdict": "candidate" if ok else "keep_main",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("VALID_RANKIC", f"{fv:.6f}")


if __name__ == "__main__":
    main()
