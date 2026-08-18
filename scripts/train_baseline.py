"""Stage 2: train the cross-section LightGBM baseline.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\train_baseline.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import (  # noqa: E402
    drop_task2_label,
    flatten_masked_rows,
    group_sizes,
    load_panel,
    resolve_cat_indices,
    slice_split,
    within_day_relevance,
)
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.submit import save_submission  # noqa: E402


def _write_importance(path: Path, pairs: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        writer.writerows(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM cross-section baseline")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data", default=None, help="overrides JINGGE_DATA and config data.path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    feature_cfg = dict(cfg.get("features") or {})
    cat_indices = resolve_cat_indices(feature_cfg)
    fill_invalid = float((cfg.get("train") or {}).get("fill_invalid", 0.0))
    paths = cfg.get("paths") or {}

    print(f"config={args.config}")
    print(f"seed={seed}")
    print(f"cat_indices={cat_indices}")
    print(
        "features include_raw={include_raw} cs_zscore={cs_zscore} industry_zscore={industry_zscore} "
        "max_train_stocks_per_day={max_train_stocks_per_day}".format(**{
            "include_raw": feature_cfg.get("include_raw", True),
            "cs_zscore": feature_cfg.get("cs_zscore", True),
            "industry_zscore": feature_cfg.get("industry_zscore", True),
            "max_train_stocks_per_day": feature_cfg.get("max_train_stocks_per_day"),
        })
    )
    hist = feature_cfg.get("history") or {}
    if hist.get("enabled"):
        print(
            f"history length={hist.get('length')} short={hist.get('short_length')} "
            f"source={hist.get('source', 'cs_zscore')} cols={len(hist.get('num_indices') or [])} "
            f"stats={hist.get('stats')} short_stats={hist.get('short_stats')}"
        )
    mkt = feature_cfg.get("market_state") or {}
    if mkt.get("enabled"):
        print(
            f"market_state coverage={mkt.get('include_coverage', True)} "
            f"relative_length={mkt.get('relative_length')} "
            f"cols={len(mkt.get('num_indices') or [])} stats={mkt.get('stats')}"
        )

    data_path = resolve_data_path(cfg, args.data)
    print(f"data={data_path}")

    t0 = time.perf_counter()
    data = load_panel(str(data_path))
    drop_task2_label(data)
    print(f"loaded in {time.perf_counter() - t0:.1f}s  num_x={tuple(data['num_x'].shape)}")

    train = slice_split(data, "train")
    valid = slice_split(data, "valid")
    test = slice_split(data, "test")
    print(f"days train={train['num_x'].shape[0]} valid={valid['num_x'].shape[0]} test={test['num_x'].shape[0]}")

    t1 = time.perf_counter()
    x_train, y_train, coords = flatten_masked_rows(
        train,
        cat_indices=cat_indices,
        require_label=True,
        feature_cfg=feature_cfg,
        seed=seed,
    )
    print(
        f"train rows={x_train.shape[0]} cols={x_train.shape[1]} "
        f"flatten {time.perf_counter() - t1:.1f}s"
    )

    rank_label = (cfg.get("train") or {}).get("rank_label")
    group = None
    if rank_label:
        if rank_label == "within_day_rank":
            n_grades = int((cfg.get("train") or {}).get("rank_grades", 5))
            y_train = within_day_relevance(y_train, coords, n_grades=n_grades)
            print(
                f"relevance grades 0-{n_grades - 1}  "
                f"unique={int(np.unique(y_train).size)}"
            )
        group = group_sizes(coords)
        print(f"rank_label={rank_label} groups={group.size} mean_group={float(group.mean()):.1f}")

    model = LightGBMBaseline(params=(cfg.get("model") or {}).get("params") or {}, feature_cfg=feature_cfg, seed=seed)
    t2 = time.perf_counter()
    model.fit(x_train, y_train, group=group)
    print(f"fit {time.perf_counter() - t2:.1f}s")
    del x_train, y_train

    t3 = time.perf_counter()
    valid_pred = model.predict_panel(valid, fill_invalid=fill_invalid)
    valid_ic = mean_rank_ic(valid_pred, valid["y1"], valid["mask_y"])
    print(f"valid mean RankIC={valid_ic:.6f}  predict {time.perf_counter() - t3:.1f}s")

    importance = model.feature_importance()
    importance_path = ROOT / paths.get("importance", "outputs/baseline_importance.csv")
    _write_importance(importance_path, importance)
    print("importance top 20:")
    for name, gain in importance[:20]:
        print(f"  {name}\t{gain:.4f}")

    valid_pred_path = ROOT / paths.get("valid_pred", "outputs/baseline_valid.npy")
    valid_pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(valid_pred_path, valid_pred)

    t4 = time.perf_counter()
    test_pred = model.predict_panel(test, fill_invalid=fill_invalid)
    test_pred_path = ROOT / paths.get("test_pred", "outputs/baseline_test.npy")
    test_pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(test_pred_path, test_pred)
    submission = ROOT / paths.get("submission", "submissions/task1_baseline.npy")
    save_submission(test_pred, submission)
    print(
        f"test pred {test_pred.shape} {test_pred.dtype}  "
        f"predict+save {time.perf_counter() - t4:.1f}s"
    )

    checkpoint = ROOT / paths.get("checkpoint", "checkpoints/baseline.txt")
    model.save(checkpoint)
    print(f"wrote {checkpoint}")
    print(f"wrote {valid_pred_path}")
    print(f"wrote {submission}")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{valid_ic:.6f}")


if __name__ == "__main__":
    main()
