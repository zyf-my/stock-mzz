"""Joint fine search from current best (mlp-only stack). Valid-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
START = 0.088897
HIGH_TAU = 4670.0


def _v(s: str) -> np.ndarray:
    return np.load(OUT / f"{s}_valid.npy")


def _t(s: str) -> np.ndarray:
    return np.load(OUT / f"{s}_test.npy")


def _hi_ic(pred, y, my, mx, tau=HIGH_TAU) -> float:
    nx = np.asarray(mx).sum(axis=1)
    s = rank_ic_series(pred, y, my)
    s[nx < tau] = np.nan
    return float(np.nanmean(s))


def main() -> None:
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]

    hist, base = _v("hist_lgbm_n200"), _v("baseline")
    x6, o6, n6 = _v("gru_x6_with_today"), _v("gru_only6_with_today"), _v("gru_next6_with_today")
    mlp = _v("cs_mlp")

    # optional x6 variants
    x6_opts = [("x6", x6)]
    for stem in ("gru_x6_d600", "gru_x6_d1200", "gru_x6_with_today_s43", "gru_x6_with_today_s44"):
        p = OUT / f"{stem}_valid.npy"
        if p.is_file():
            x6_opts.append((stem, np.load(p)))

    best = (-1.0, None)
    for x6_name, x6_arr in x6_opts:
        for tau_t in (4546.0, 4614.0, 4670.0):
            for w_lo_t in (0.40, 0.45, 0.50):
                for w_hi_t in (0.95, 1.0):
                    tree = coverage_gate_blend(hist, base, mx, tau_t, w_lo_t, w_hi_t, "rank")
                    for ow in (0.65, 0.70, 0.75, 0.80):
                        for nw in (0.30, 0.34, 0.38, 0.42):
                            ens = linear_blend(n6, linear_blend(o6, x6_arr, ow), nw)
                            for tau_g in (4614.0, 4670.0, 4720.0):
                                for w_lo_g in (0.20, 0.28, 0.35):
                                    for w_hi_g in (0.88, 0.92, 0.96, 1.0):
                                        gated = coverage_gate_blend(
                                            ens, tree, mx, tau_g, w_lo_g, w_hi_g, "rank"
                                        )
                                        for mw in (0.06, 0.08, 0.10, 0.12, 0.14, 0.16):
                                            blender = FusionModel({"weight_grid": [mw]})
                                            blender.locked = {
                                                "name": "rank_blend",
                                                "weight": mw,
                                                "space": "rank",
                                                "ic": float("nan"),
                                            }
                                            out = blender.predict(mlp, gated, mx)
                                            ic = mean_rank_ic(out, y, my)
                                            if ic > best[0]:
                                                best = (
                                                    ic,
                                                    {
                                                        "x6": x6_name,
                                                        "tree": (tau_t, w_lo_t, w_hi_t),
                                                        "ens": (ow, nw),
                                                        "gate": (tau_g, w_lo_g, w_hi_g),
                                                        "mlp_w": mw,
                                                        "hi_ic": _hi_ic(out, y, my, mx),
                                                    },
                                                    out,
                                                )
                                                print(
                                                    f"  * {ic:.6f} x6={x6_name} tree={tau_t}/{w_lo_t}/{w_hi_t} "
                                                    f"ens={ow}/{nw} gate={tau_g}/{w_lo_g}/{w_hi_g} mlp={mw} "
                                                    f"hi={best[1]['hi_ic']:.6f}"
                                                )

    ic, cfg, out_v = best
    print(f"BEST ic={ic:.6f} start={START:.6f} delta={ic-START:+.6f}")
    print(json.dumps(cfg, indent=2))

    summary = {
        "ic": float(ic),
        "platform_test": 0.091354,
        "x6": cfg["x6"],
        "tree": {"tau": cfg["tree"][0], "w_lo": cfg["tree"][1], "w_hi": cfg["tree"][2]},
        "ens": {"only6_w": cfg["ens"][0], "next6_w": cfg["ens"][1]},
        "gate": {"tau": cfg["gate"][0], "w_lo": cfg["gate"][1], "w_hi": cfg["gate"][2]},
        "mlp_w": cfg["mlp_w"],
        "mlp6_mix": 0.0,
        "hi31_ic": cfg["hi_ic"],
    }
    (OUT / "fusion_push_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.save(OUT / "fusion_push_valid.npy", out_v)

    if ic > START + 1e-6:
        x6_stem = cfg["x6"] if cfg["x6"] != "x6" else "gru_x6_with_today"
        tt, wlt, wht = cfg["tree"]
        ow, nw = cfg["ens"]
        tg, wlg, whg = cfg["gate"]
        mw = cfg["mlp_w"]
        mx_t = test["mask_x"]
        tree_t = coverage_gate_blend(_t("hist_lgbm_n200"), _t("baseline"), mx_t, tt, wlt, wht, "rank")
        ens_t = linear_blend(
            _t("gru_next6_with_today"),
            linear_blend(_t("gru_only6_with_today"), _t(x6_stem), ow),
            nw,
        )
        gated_t = coverage_gate_blend(ens_t, tree_t, mx_t, tg, wlg, whg, "rank")
        blender = FusionModel({"weight_grid": [mw]})
        blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
        out_t = blender.predict(_t("cs_mlp"), gated_t, mx_t)
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_best.npy")
        summary_path = OUT / "fusion_best_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("updated task2_fusion_best.npy")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
