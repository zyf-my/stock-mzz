"""Evaluate high-coverage specialist stems vs locked x6_today fusion.

Never overwrites submissions/task1_fusion_x6_today.npy.
Writes submissions/task1_fusion_hicov_*.npy only if hi31 does not drop
and valid is not down by more than 0.0005.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_hicov_specialist.py
"""

from __future__ import annotations

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
SUB = ROOT / "submissions"
HIGH_TAU = 4670.0
BASE_VALID = 0.120869
BASE_HI31 = 0.211839
ONLY6_W, NEXT6_W, MLP6_W, MLP_W = 0.4, 0.15, 0.4, 0.15
TAU, W_LO, W_HI = 4546.0, 0.25, 0.6

STEMS = ("hist_lgbm_hicov", "cs_mlp_hicov", "gru_hicov")


def _has(name: str) -> bool:
    return (OUT / f"{name}_valid.npy").is_file() and (OUT / f"{name}_test.npy").is_file()


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(OUT / f"{name}_valid.npy"), np.load(OUT / f"{name}_test.npy")


def _metrics(pred, y, my, hi) -> dict:
    s = rank_ic_series(pred, y, my)
    s_hi = s.copy()
    s_hi[~hi] = np.nan
    n = len(s)
    mid = n // 2
    return {
        "valid_ic": float(np.nanmean(s)),
        "hi31_ic": float(np.nanmean(s_hi)),
        "valid_first_half": float(np.nanmean(s[:mid])),
        "valid_second_half": float(np.nanmean(s[mid:])),
    }


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
    hi = np.asarray(mx).sum(axis=1) >= HIGH_TAU
    print(f"cache={src} hi31={int(hi.sum())}")
    print(f"locked valid={BASE_VALID:.6f} hi31={BASE_HI31:.6f}")
    print("does NOT overwrite submissions/task1_fusion_x6_today.npy")

    x6_v, x6_t = _load("gru_x6_with_today")
    o6_v, o6_t = _load("gru_only6_with_today")
    n6_v, n6_t = _load("gru_next6_with_today")
    tree_v, tree_t = _load("fusion")
    mlp_v, mlp_t = _load("cs_mlp")
    mlp6_v, mlp6_t = _load("cs_mlp_only6")
    locked_v = _locked(x6_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx)
    locked_t = _locked(x6_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt)
    base = _metrics(locked_v, y, my, hi)
    print(f"rebuilt locked valid={base['valid_ic']:.6f} hi31={base['hi31_ic']:.6f}")

    rows = []
    stems_v: list[np.ndarray] = []
    stems_t: list[np.ndarray] = []
    print("\n== hicov stems (single) ==")
    for name in STEMS:
        if not _has(name):
            print(f"  missing {name}")
            continue
        pv, pt = _load(name)
        m = _metrics(pv, y, my, hi)
        print(
            f"  {name:20s} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f} "
            f"half={m['valid_first_half']:.4f}/{m['valid_second_half']:.4f}"
        )
        rows.append({"name": name, **m})
        stems_v.append(pv)
        stems_t.append(pt)

    def consider(name: str, pred_v: np.ndarray, pred_t: np.ndarray) -> None:
        m = _metrics(pred_v, y, my, hi)
        dv = m["valid_ic"] - BASE_VALID
        dh = m["hi31_ic"] - BASE_HI31
        ok = m["hi31_ic"] >= BASE_HI31 - 1e-6 and m["valid_ic"] >= BASE_VALID - 0.0005
        print(
            f"  {name:20s} valid={m['valid_ic']:.6f} ({dv:+.6f})  "
            f"hi31={m['hi31_ic']:.6f} ({dh:+.6f})  {'PASS' if ok else 'fail'}"
        )
        rec = {"name": name, **m, "delta_valid": dv, "delta_hi31": dh, "pass": ok, "write": False}
        if ok and (dv > 1e-6 or dh > 1e-6):
            path = SUB / f"task1_fusion_hicov_{name}.npy"
            save_submission(pred_t, path)
            np.save(OUT / f"fusion_hicov_{name}_valid.npy", pred_v)
            rec["write"] = True
            print(f"    wrote {path.name}  (do not replace main until platform test)")
        rows.append(rec)

    print("\n== vs locked (frozen weights, no valid search) ==")
    if _has("gru_hicov"):
        gv, gt = _load("gru_hicov")
        consider("replace_x6", _locked(gv, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx),
                 _locked(gt, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt))
        bag_v = np.mean([x6_v, gv], axis=0).astype(np.float32)
        bag_t = np.mean([x6_t, gt], axis=0).astype(np.float32)
        consider("bag_x6", _locked(bag_v, o6_v, n6_v, tree_v, mlp_v, mlp6_v, mx),
                 _locked(bag_t, o6_t, n6_t, tree_t, mlp_t, mlp6_t, mxt))
    if _has("hist_lgbm_hicov"):
        hv, ht = _load("hist_lgbm_hicov")
        consider("replace_tree", _locked(x6_v, o6_v, n6_v, hv, mlp_v, mlp6_v, mx),
                 _locked(x6_t, o6_t, n6_t, ht, mlp_t, mlp6_t, mxt))
    if _has("cs_mlp_hicov"):
        mv, mt = _load("cs_mlp_hicov")
        consider("replace_mlp", _locked(x6_v, o6_v, n6_v, tree_v, mv, mlp6_v, mx),
                 _locked(x6_t, o6_t, n6_t, tree_t, mt, mlp6_t, mxt))
    if stems_v:
        eq_v = np.mean([panel_cs_rank(p, mx) for p in stems_v], axis=0).astype(np.float32)
        eq_t = np.mean([panel_cs_rank(p, mxt) for p in stems_t], axis=0).astype(np.float32)
        consider("equal_rank_stems", eq_v, eq_t)

    (OUT / "hicov_eval.json").write_text(json.dumps({"base": base, "rows": rows}, indent=2), encoding="utf-8")
    print("\nwrote outputs/hicov_eval.json")
    print("KEEP main submissions/task1_fusion_x6_today.npy unless a PASS file also wins platform test.")
    print("VALID_RANKIC", f"{base['valid_ic']:.6f}")


if __name__ == "__main__":
    main()
