"""Weighted rankic multi-seed + extrap stems under TREE_BAG shell."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_task2_tree_bag_v2 import _fuse, _metrics, _rank_bag  # noqa: E402
from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.models.fusion import panel_cs_rank  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
RECIPE = OUT / "recipes" / "TREE_BAG_091467.json"
CORE = ["hist_lgbm_n240", "hist_lgbm_rankic", "hist_lgbm_mkt_rel"]
RANKIC_SEEDS = ["hist_lgbm_rankic", "hist_lgbm_rankic_s43", "hist_lgbm_rankic_s44"]


def _rankic_bag(split: str, mx: np.ndarray, weights: list[float]) -> np.ndarray:
    stems = [s for s in RANKIC_SEEDS if (OUT / f"{s}_{split}.npy").is_file()]
    if not stems:
        return panel_cs_rank(np.load(OUT / f"hist_lgbm_rankic_{split}.npy"), mx)
    ranks = [panel_cs_rank(np.load(OUT / f"{s}_{split}.npy"), mx) for s in stems]
    w = np.asarray(weights[: len(stems)], dtype=np.float64)
    w = w / w.sum()
    out = np.zeros_like(ranks[0], dtype=np.float32)
    for wi, r in zip(w, ranks):
        out += float(wi) * r
    return out


def _custom_hist(split: str, mx: np.ndarray, rankic_w: list[float]) -> np.ndarray:
    rb = _rankic_bag(split, mx, rankic_w)
    n240 = panel_cs_rank(np.load(OUT / f"hist_lgbm_n240_{split}.npy"), mx)
    mkt = panel_cs_rank(np.load(OUT / f"hist_lgbm_mkt_rel_{split}.npy"), mx)
    return np.mean([n240, rb, mkt], axis=0).astype(np.float32)


def main() -> None:
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    mx, mxt = valid["mask_x"], test["mask_x"]
    base_m = _metrics(_fuse(_rank_bag(CORE, "valid", mx), "valid", recipe), valid)
    print(f"base valid={base_m['valid_ic']:.6f} hi31={base_m['hi31_ic']:.6f}")

    best = (base_m["hi31_ic"], base_m["valid_ic"], None)
    for w0 in (0.2, 0.33, 0.5):
        for w1 in (0.2, 0.33, 0.5):
            w2 = 1.0 - w0 - w1
            if w2 < 0.1:
                continue
            hist_v = _custom_hist("valid", mx, [w0, w1, w2])
            m = _metrics(_fuse(hist_v, "valid", recipe), valid)
            ok = m["hi31_ic"] >= base_m["hi31_ic"] - 1e-6 and m["valid_ic"] >= base_m["valid_ic"] - 0.0005
            if ok and (m["hi31_ic"], m["valid_ic"]) > (best[0], best[1]):
                best = (m["hi31_ic"], m["valid_ic"], [w0, w1, w2])
                print(f"  PASS w={w0:.2f},{w1:.2f},{w2:.2f} valid={m['valid_ic']:.6f} hi31={m['hi31_ic']:.6f}")

    if best[2] is None:
        print("no weighted rankic beat base")
        return

    ws = best[2]
    out_t = _fuse(_custom_hist("test", mxt, ws), "test", recipe)
    save_submission(out_t, ROOT / "submissions" / "task2_fusion_rankic_bag.npy")
    meta = {
        "name": f"RANKIC_BAG_{ws[0]:.2f}_{ws[1]:.2f}_{ws[2]:.2f}",
        **recipe,
        "hist_bag": CORE,
        "rankic_seed_weights": ws,
        "valid_ic": best[1],
        "hi31_ic": best[0],
        "baseline_hi31": base_m["hi31_ic"],
        "platform_test_prev": 0.091467,
    }
    (OUT / "fusion_rankic_bag_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote task2_fusion_rankic_bag.npy hi31={best[0]:.6f} valid={best[1]:.6f}")


if __name__ == "__main__":
    main()
