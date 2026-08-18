"""Sweep boosting rounds on a frozen LightGBM. Does not retrain or overwrite 002.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\eval_lgbm_iters.py --config configs/hist_lgbm.yaml
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
from src.dataset import drop_task2_label, load_panel, slice_split  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.fusion import FusionModel  # noqa: E402
from src.submit import save_submission  # noqa: E402

ITERS = [50, 100, 150, 200, 250, 300, 350, 400]


def _predict_strided(model, split, iterations, stride: int, fill_invalid: float):
    """Predict every `stride`-th day, keeping global_t correct."""
    n_days, n_stocks, _ = split["num_x"].shape
    idx = list(range(0, n_days, stride))
    outs = {n: np.full((len(idx), n_stocks), float(fill_invalid), dtype=np.float32) for n in iterations}
    y = split["y1"][idx]
    mask_y = split["mask_y"][idx]
    for i, t in enumerate(idx):
        one = {
            "start": int(split["start"]) + t,
            "num_x": split["num_x"][t : t + 1],
            "cat_x": split["cat_x"][t : t + 1],
            "mask_x": split["mask_x"][t : t + 1],
            "panel_num_x": split.get("panel_num_x"),
            "panel_mask_x": split.get("panel_mask_x"),
        }
        day_out = model.predict_panel_iters(one, iterations, fill_invalid=fill_invalid)
        for n in iterations:
            outs[n][i] = day_out[n][0]
    return outs, y, mask_y


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep LightGBM boosting rounds on frozen model")
    parser.add_argument("--config", default="configs/hist_lgbm.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--train-stride", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    feature_cfg = dict(cfg.get("features") or {})
    fill_invalid = float((cfg.get("train") or {}).get("fill_invalid", 0.0))
    paths = cfg.get("paths") or {}
    ckpt = ROOT / paths.get("checkpoint", "checkpoints/hist_lgbm.txt")

    t0 = time.perf_counter()
    data = load_panel(str(resolve_data_path(cfg, args.data)))
    drop_task2_label(data)
    train = slice_split(data, "train")
    valid = slice_split(data, "valid")
    test = slice_split(data, "test")
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=feature_cfg,
        seed=seed,
    )
    print(f"load {ckpt}")
    model.load(ckpt)
    n_trees = int(model.booster.num_trees()) if model.booster is not None else 400
    iters = [n for n in ITERS if n <= n_trees]
    if n_trees not in iters:
        iters.append(n_trees)
    print(f"n_trees={n_trees} sweep={iters}")

    t1 = time.perf_counter()
    print("predict valid (all days, one feature pass)")
    valid_outs = model.predict_panel_iters(valid, iters, fill_invalid=fill_invalid)
    print(f"valid predict {time.perf_counter() - t1:.1f}s")

    t2 = time.perf_counter()
    print(f"predict train every {args.train_stride} days")
    train_outs, train_y, train_mask = _predict_strided(
        model, train, iters, args.train_stride, fill_invalid
    )
    print(f"train subset predict {time.perf_counter() - t2:.1f}s days={train_y.shape[0]}")

    rows = []
    print("iter  train_sub  valid")
    for n in iters:
        tr = mean_rank_ic(train_outs[n], train_y, train_mask)
        va = mean_rank_ic(valid_outs[n], valid["y1"], valid["mask_y"])
        rows.append({"num_iteration": int(n), "train_sub_ic": float(tr), "valid_ic": float(va)})
        print(f"  {n:4d}  {tr:.6f}  {va:.6f}")

    best = max(rows, key=lambda r: r["valid_ic"])
    full = next(r for r in rows if r["num_iteration"] == n_trees)
    print(f"BEST valid={best['valid_ic']:.6f} @ {best['num_iteration']} trees")
    print(f"FULL valid={full['valid_ic']:.6f} @ {full['num_iteration']} trees")

    out_json = ROOT / "outputs/hist_lgbm_iter_sweep.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"n_trees": n_trees, "rows": rows, "best": best}, indent=2), encoding="utf-8")

    baseline_ic = 0.104192
    if best["num_iteration"] < n_trees and best["valid_ic"] > baseline_ic + 1e-4:
        n_best = int(best["num_iteration"])
        print(f"earlier checkpoint beats 002; predicting test at {n_best}")
        test_pred = model.predict_panel(test, fill_invalid=fill_invalid, num_iteration=n_best)
        valid_pred = valid_outs[n_best]
        np.save(ROOT / "outputs/hist_lgbm_iter_valid.npy", valid_pred)
        np.save(ROOT / "outputs/hist_lgbm_iter_test.npy", test_pred)

        base_valid = np.load(ROOT / "outputs/baseline_valid.npy")
        base_test = np.load(ROOT / "outputs/baseline_test.npy")
        blender = FusionModel({"weight_grid": [0.0, 0.15, 0.25, 0.4, 0.5, 0.7, 0.75, 1.0]})
        locked = blender.fit(
            valid_pred,
            base_valid,
            valid["y1"],
            valid["mask_y"],
            valid["mask_x"],
            industry=valid["cat_x"][..., 6],
        )
        print("blend vs baseline:")
        for row in (locked.get("valid_leaderboard") or [])[:5]:
            extra = {k: v for k, v in row.items() if k not in {"valid_leaderboard", "name", "ic"}}
            print(f"  {row['ic']:.6f}  {row['name']}  {extra}")
        print(f"BLEND_LOCKED {locked['name']} ic={locked['ic']:.6f}")

        gru_valid = ROOT / "outputs/gru_valid.npy"
        if locked["ic"] > 0.106185 + 1e-4 and gru_valid.is_file():
            tree_valid = blender.predict(valid_pred, base_valid, valid["mask_x"], industry=valid["cat_x"][..., 6])
            tree_test = blender.predict(test_pred, base_test, test["mask_x"], industry=test["cat_x"][..., 6])
            gru_v = np.load(gru_valid)
            gru_t = np.load(ROOT / "outputs/gru_test.npy")
            gblend = FusionModel({"weight_grid": [0.0, 0.15, 0.25, 0.4, 0.5, 0.7, 1.0]})
            glock = gblend.fit(gru_v, tree_valid, valid["y1"], valid["mask_y"], valid["mask_x"])
            print(f"GRU blend {glock['name']} ic={glock['ic']:.6f}")
            if float(glock["ic"]) > 0.110069 + 1e-4:
                out_v = gblend.predict(gru_v, tree_valid, valid["mask_x"])
                out_t = gblend.predict(gru_t, tree_test, test["mask_x"])
                np.save(ROOT / "outputs/fusion_iter_gru_valid.npy", out_v)
                np.save(ROOT / "outputs/fusion_iter_gru_test.npy", out_t)
                save_submission(out_t, ROOT / "submissions/task1_fusion_iter_gru.npy")
                print("wrote submissions/task1_fusion_iter_gru.npy (did not overwrite 0.111)")
            else:
                print("GRU blend did not beat 0.110069")
        elif locked["ic"] > 0.106185 + 1e-4:
            out_test = blender.predict(test_pred, base_test, test["mask_x"], industry=test["cat_x"][..., 6])
            save_submission(out_test, ROOT / "submissions/task1_hist_lgbm_iter.npy")
            print("wrote submissions/task1_hist_lgbm_iter.npy")
        else:
            print("tree+baseline blend did not beat 0.106185")
    else:
        print("going back does not beat full 400 trees on valid; not writing a new submission")

    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{best['valid_ic']:.6f}")
    print("does not overwrite checkpoints/hist_lgbm.txt")


if __name__ == "__main__":
    main()
