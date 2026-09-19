"""hist@225 + locked search weights from fusion_search_summary.json"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, slice_split, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
N = 225
LOCKED = 0.086676


def main() -> None:
    summary = json.loads((OUT / "fusion_search_summary.json").read_text(encoding="utf-8"))
    tp = summary["best_tree"]["params"]
    gp = summary["best_gate"]["params"]
    ow, nw = summary["best_ens"]["only6_w"], summary["best_ens"]["next6_w"]
    mw = summary["best_full"]["mlp_w"]

    cfg = load_config("configs/task2/hist_lgbm.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, "y2")
    valid, test = slice_split(data, "valid"), slice_split(data, "test")
    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=dict(cfg.get("features") or {}),
        seed=int(cfg.get("seed", 42)),
    )
    model.load(ROOT / (cfg.get("paths") or {})["checkpoint"])
    hist_v = model.predict_panel(valid, num_iteration=N)
    hist_t = model.predict_panel(test, num_iteration=N)
    np.save(OUT / "hist_lgbm_n225_valid.npy", hist_v)
    np.save(OUT / "hist_lgbm_n225_test.npy", hist_t)

    y, my, mx = split_label_array(valid, "y2"), valid["mask_y"], valid["mask_x"]
    base_v, base_t = np.load(OUT / "baseline_valid.npy"), np.load(OUT / "baseline_test.npy")
    tree_v = coverage_gate_blend(hist_v, base_v, mx, tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    tree_t = coverage_gate_blend(hist_t, base_t, test["mask_x"], tp["tau"], tp["w_lo"], tp["w_hi"], "rank")
    ens_v = linear_blend(
        np.load(OUT / "gru_next6_with_today_valid.npy"),
        linear_blend(np.load(OUT / "gru_only6_with_today_valid.npy"), np.load(OUT / "gru_x6_with_today_valid.npy"), ow),
        nw,
    )
    ens_t = linear_blend(
        np.load(OUT / "gru_next6_with_today_test.npy"),
        linear_blend(np.load(OUT / "gru_only6_with_today_test.npy"), np.load(OUT / "gru_x6_with_today_test.npy"), ow),
        nw,
    )
    gated_v = coverage_gate_blend(ens_v, tree_v, mx, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    gated_t = coverage_gate_blend(ens_t, tree_t, test["mask_x"], gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    mlp_v = linear_blend(np.load(OUT / "cs_mlp_only6_valid.npy"), np.load(OUT / "cs_mlp_valid.npy"), 0.4)
    mlp_t = linear_blend(np.load(OUT / "cs_mlp_only6_test.npy"), np.load(OUT / "cs_mlp_test.npy"), 0.4)
    blender = FusionModel({"weight_grid": [mw]})
    blender.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    out_v = blender.predict(mlp_v, gated_v, mx)
    out_t = blender.predict(mlp_t, gated_t, test["mask_x"])
    ic = mean_rank_ic(out_v, y, my)
    print(f"hist@{N} + search weights ic={ic:.6f} locked={LOCKED:.6f}")
    if ic > LOCKED + 1e-6:
        np.save(OUT / "fusion_hist_n225_valid.npy", out_v)
        np.save(OUT / "fusion_hist_n225_test.npy", out_t)
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_hist_n225.npy")
        print("wrote task2_fusion_hist_n225.npy")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
