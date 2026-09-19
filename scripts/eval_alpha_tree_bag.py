"""Bag already-trained alpha trees into locked fusion. Does not overwrite main."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits  # noqa: E402
from src.metrics import rank_ic_series  # noqa: E402
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
MAIN = "task1_fusion_alpha_x.npy"


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
    splits, src = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = valid["y1"], valid["mask_y"], valid["mask_x"]
    mxt = test["mask_x"]
    hi = np.asarray(mx).sum(1) >= HIGH_TAU
    print(f"cache={src} hi31={int(hi.sum())}")

    x6_v, x6_t = np.load(OUT / "gru_x6_with_today_valid.npy"), np.load(OUT / "gru_x6_with_today_test.npy")
    o6_v, o6_t = np.load(OUT / "gru_only6_with_today_valid.npy"), np.load(OUT / "gru_only6_with_today_test.npy")
    n6_v, n6_t = np.load(OUT / "gru_next6_with_today_valid.npy"), np.load(OUT / "gru_next6_with_today_test.npy")
    mlp_v, mlp_t = np.load(OUT / "cs_mlp_valid.npy"), np.load(OUT / "cs_mlp_test.npy")
    mlp6_v, mlp6_t = np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_only6_test.npy")
    base_v, base_t = np.load(OUT / "baseline_valid.npy"), np.load(OUT / "baseline_test.npy")
    ax_v, ax_t = np.load(OUT / "hist_lgbm_alpha_x_valid.npy"), np.load(OUT / "hist_lgbm_alpha_x_test.npy")
    a_v, a_t = np.load(OUT / "hist_lgbm_alpha_valid.npy"), np.load(OUT / "hist_lgbm_alpha_test.npy")

    rows = []
    for w, tag in ((1.0, "alpha_x_only"), (0.7, "alpha_bag70"), (0.5, "alpha_bag50")):
        mix_v = linear_blend(ax_v, a_v, w)
        mix_t = linear_blend(ax_t, a_t, w)
        tree_v = linear_blend(mix_v, base_v, TREE_W)
        tree_t = linear_blend(mix_t, base_t, TREE_W)
        pred_v = _locked(x6_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)
        fv, fh = _m(pred_v, y, my, hi)
        print(f"{tag} w_x={w:.1f} valid={fv:.6f} ({fv - BASE_VALID:+.6f}) hi31={fh:.6f} ({fh - BASE_HI31:+.6f})")
        rows.append({"tag": tag, "w_x": w, "valid": fv, "hi31": fh})
        if tag == "alpha_x_only":
            continue
        pred_t = _locked(x6_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)
        np.save(OUT / f"fusion_{tag}_valid.npy", pred_v)
        np.save(OUT / f"fusion_{tag}_test.npy", pred_t)
        sub = SUB / f"task1_fusion_{tag}.npy"
        if sub.name != MAIN:
            save_submission(pred_t, sub)
            print(f"wrote {sub.name}")

    (OUT / "alpha_tree_bag_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    best = max(rows, key=lambda r: r["valid"])
    print("VALID_RANKIC", f"{best['valid']:.6f}")


if __name__ == "__main__":
    main()
