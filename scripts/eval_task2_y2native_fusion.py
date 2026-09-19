"""Full y2-native GRU trio swap + weight re-search (not one-at-a-time).

Replaces x6/only6/next6 with gru_y2cols / gru_only6_y2cols / gru_next6_y2left.
Tree stays hist@200 + baseline (rank gate). Optional y2cols / y2resid MLP stack.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\eval_task2_y2native_fusion.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
BEST_IC = 0.088799


def _load(stem: str) -> np.ndarray:
    p = OUT / f"{stem}_valid.npy"
    if not p.is_file():
        raise FileNotFoundError(p)
    return np.load(p)


def _load_t(stem: str) -> np.ndarray:
    return np.load(OUT / f"{stem}_test.npy")


def _require(*stems: str) -> None:
    missing = [s for s in stems if not (OUT / f"{s}_valid.npy").is_file()]
    if missing:
        raise FileNotFoundError(
            "missing preds: " + ", ".join(missing) + "\n"
            "train with configs/task2/gru_y2cols.yaml etc."
        )


def main() -> None:
    _require(
        "gru_y2cols",
        "gru_only6_y2cols",
        "gru_next6_y2left",
        "hist_lgbm_n200",
        "baseline",
        "cs_mlp",
        "cs_mlp_only6",
    )
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y = split_label_array(valid, "y2")
    my, mx = valid["mask_y"], valid["mask_x"]

    # singles
    x6_v = _load("gru_y2cols")
    o6_v = _load("gru_only6_y2cols")
    n6_v = _load("gru_next6_y2left")
    print(f"gru_y2cols       {mean_rank_ic(x6_v, y, my):.6f}")
    print(f"gru_only6_y2cols {mean_rank_ic(o6_v, y, my):.6f}")
    print(f"gru_next6_y2left {mean_rank_ic(n6_v, y, my):.6f}")
    old_x6 = _load("gru_x6_with_today")
    old_o6 = _load("gru_only6_with_today")
    old_n6 = _load("gru_next6_with_today")
    print(f"old x6/only6/next6 {mean_rank_ic(old_x6, y, my):.6f} / "
          f"{mean_rank_ic(old_o6, y, my):.6f} / {mean_rank_ic(old_n6, y, my):.6f}")

    hist_v = _load("hist_lgbm_n200")
    base_v = _load("baseline")
    best_tree = (-1.0, None, None)
    for tau in (4491.0, 4546.0, 4614.0, 4670.0):
        for w_lo in (0.35, 0.45, 0.55):
            for w_hi in (0.85, 0.95, 1.0):
                tree = coverage_gate_blend(hist_v, base_v, mx, tau, w_lo, w_hi, "rank")
                ic = mean_rank_ic(tree, y, my)
                if ic > best_tree[0]:
                    best_tree = (ic, (tau, w_lo, w_hi), tree)
    print(f"tree best {best_tree[1]} ic={best_tree[0]:.6f}")

    best_ens = (-1.0, None)
    for ow in (0.4, 0.5, 0.6, 0.7, 0.8):
        for nw in (0.0, 0.15, 0.25, 0.34, 0.45):
            ens = linear_blend(n6_v, linear_blend(o6_v, x6_v, ow), nw)
            ic = mean_rank_ic(ens, y, my)
            if ic > best_ens[0]:
                best_ens = (ic, (ow, nw))
    ow, nw = best_ens[1]
    ens_v = linear_blend(n6_v, linear_blend(o6_v, x6_v, ow), nw)
    print(f"ens y2native best ({ow},{nw}) ic={best_ens[0]:.6f}")

    tree_v = best_tree[2]
    best_gate = (-1.0, None, None)
    for tau in (4546.0, 4614.0, 4670.0, 4720.0):
        for w_lo in (0.15, 0.28, 0.4):
            for w_hi in (0.75, 0.85, 0.92, 1.0):
                gated = coverage_gate_blend(ens_v, tree_v, mx, tau, w_lo, w_hi, "rank")
                ic = mean_rank_ic(gated, y, my)
                if ic > best_gate[0]:
                    best_gate = (ic, (tau, w_lo, w_hi), gated)
    print(f"gate best {best_gate[1]} ic={best_gate[0]:.6f}")

    mlp_opts: list[tuple[str, np.ndarray]] = []
    mlp6_opts: list[tuple[str, np.ndarray]] = []
    for stem in ("cs_mlp", "cs_mlp_y2cols", "cs_mlp_y2resid"):
        p = OUT / f"{stem}_valid.npy"
        if p.is_file():
            mlp_opts.append((stem, np.load(p)))
    for stem in ("cs_mlp_only6", "cs_mlp_only6_y2cols"):
        p = OUT / f"{stem}_valid.npy"
        if p.is_file():
            mlp6_opts.append((stem, np.load(p)))
    if not mlp_opts:
        mlp_opts = [("cs_mlp", _load("cs_mlp"))]
    if not mlp6_opts:
        mlp6_opts = [("cs_mlp_only6", _load("cs_mlp_only6"))]

    blender = FusionModel({"weight_grid": [0.06]})
    best_full = (-1.0, None, None, None)
    for m_name, m_v in mlp_opts:
        for m6_name, m6_v in mlp6_opts:
            for mm in (0.0, 0.3, 0.4, 0.5, 0.6):
                mlp_ens = linear_blend(m6_v, m_v, mm)
                for mw in (0.0, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15):
                    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
                    out = blender.predict(mlp_ens, best_gate[2], mx)
                    ic = mean_rank_ic(out, y, my)
                    if ic > best_full[0]:
                        best_full = (ic, (m_name, m6_name, mm, mw), out, mlp_ens)
    m_name, m6_name, mm, mw = best_full[1]
    print(f"full best mlp=({m_name}+{m6_name} mix={mm}) stack_w={mw} ic={best_full[0]:.6f}")
    print(f"vs locked best {BEST_IC:.6f} delta={best_full[0] - BEST_IC:+.6f}")

    # hybrid: y2native ens + old x6 anchor (half blend)
    hybrid_ens = linear_blend(ens_v, old_x6, 0.5)
    hybrid_gate = coverage_gate_blend(hybrid_ens, tree_v, mx, *best_gate[1], "rank")
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    ic_h = mean_rank_ic(blender.predict(best_full[3], hybrid_gate, mx), y, my)
    print(f"hybrid ens50% old_x6 + same gate/mlp {ic_h:.6f}")

    summary = {
        "best_ic": best_full[0],
        "locked_ic": BEST_IC,
        "delta": float(best_full[0] - BEST_IC),
        "tree": {"ic": best_tree[0], "tau": best_tree[1][0], "w_lo": best_tree[1][1], "w_hi": best_tree[1][2]},
        "ens": {"ic": best_ens[0], "only6_w": ow, "next6_w": nw, "branches": ["gru_y2cols", "gru_only6_y2cols", "gru_next6_y2left"]},
        "gate": {"ic": best_gate[0], "tau": best_gate[1][0], "w_lo": best_gate[1][1], "w_hi": best_gate[1][2]},
        "mlp": {"mlp": m_name, "mlp6": m6_name, "mix": mm, "stack_w": mw},
        "hybrid_old_x6": float(ic_h),
        "singles": {
            "gru_y2cols": float(mean_rank_ic(x6_v, y, my)),
            "gru_only6_y2cols": float(mean_rank_ic(o6_v, y, my)),
            "gru_next6_y2left": float(mean_rank_ic(n6_v, y, my)),
        },
    }
    (OUT / "y2native_fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(OUT / "y2native_fusion_valid.npy", best_full[2])

    if best_full[0] > BEST_IC + 1e-6:
        tau_t, w_lo_t, w_hi_t = best_tree[1]
        tau_g, w_lo_g, w_hi_g = best_gate[1]
        mx_t = test["mask_x"]
        tree_t = coverage_gate_blend(
            _load_t("hist_lgbm_n200"), _load_t("baseline"), mx_t, tau_t, w_lo_t, w_hi_t, "rank"
        )
        ens_t = linear_blend(
            _load_t("gru_next6_y2left"),
            linear_blend(_load_t("gru_only6_y2cols"), _load_t("gru_y2cols"), ow),
            nw,
        )
        gated_t = coverage_gate_blend(ens_t, tree_t, mx_t, tau_g, w_lo_g, w_hi_g, "rank")
        mlp_t = linear_blend(_load_t(m6_name), _load_t(m_name), mm)
        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
        out_t = blender.predict(mlp_t, gated_t, mx_t)
        np.save(OUT / "y2native_fusion_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_y2native.npy")
        print("wrote submissions/task2_fusion_y2native.npy")
    else:
        print("no lift over fusion_best; keep task2_fusion_best.npy for upload")

    print("VALID_RANKIC", f"{best_full[0]:.6f}")


if __name__ == "__main__":
    main()
