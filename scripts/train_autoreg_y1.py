"""Train a causal y1 autoregressive specialist from X plus lagged y1 ranks."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path
from src.dataset import (build_sample_features, cat_feature_col_indices, drop_other_label,
                         feature_names, flatten_masked_rows, load_panel, resolve_cat_indices,
                         slice_split)
from src.metrics import rank_ic_series
from src.models.fusion import panel_cs_rank
from src.submit import save_submission

OUT = ROOT / "outputs"


def pct_rank(x, mask):
    x = np.asarray(x, dtype=np.float32)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    out = np.zeros(x.shape, dtype=np.float32)
    if int(m.sum()) >= 2:
        r = rankdata(x[m], method="average").astype(np.float32)
        out[m] = (r - 1.0) / max(float(r.size - 1), 1.0)
    return out


def lag_panel(y, mask_y):
    t_len, n_stock = y.shape
    lag1 = np.zeros((t_len, n_stock), dtype=np.float32)
    lag2 = np.zeros((t_len, n_stock), dtype=np.float32)
    for t in range(t_len):
        if t >= 1:
            lag1[t] = pct_rank(y[t - 1], mask_y[t - 1])
        if t >= 2:
            lag2[t] = pct_rank(y[t - 2], mask_y[t - 2])
    return lag1, lag2


def train_features(split, lag1, lag2, features, cats, y):
    x, raw_y, coords = flatten_masked_rows(split, cat_indices=cats, feature_cfg=features,
                                            require_label=True, seed=42)
    global_t = int(split["start"]) + coords[:, 0]
    lag_x = np.column_stack((lag1[global_t, coords[:, 1]], lag2[global_t, coords[:, 1]])).astype(np.float32)
    return np.column_stack((x, lag_x)), raw_y, coords


def predict_recursive(model, split, lag1, lag2, features, cats, initial1, initial2):
    n_days, n_stock = split["mask_x"].shape
    out = np.zeros((n_days, n_stock), dtype=np.float32)
    p1 = np.asarray(initial1, dtype=np.float32).copy()
    p2 = np.asarray(initial2, dtype=np.float32).copy()
    for t in range(n_days):
        mask = np.asarray(split["mask_x"][t], dtype=bool)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        global_t = int(split["start"]) + t
        x = build_sample_features(split["num_x"][t], split["cat_x"][t], mask, idx, cats, features,
                                  global_t=global_t, panel_num_x=split.get("panel_num_x"),
                                  panel_mask_x=split.get("panel_mask_x"))
        lag_x = np.column_stack((pct_rank(p1, mask)[idx], pct_rank(p2, mask)[idx])).astype(np.float32)
        pred = model.predict(np.column_stack((x, lag_x)))
        out[t, idx] = pred.astype(np.float32)
        p2, p1 = p1, out[t]
    return out


def block_report(pred, split):
    s = rank_ic_series(pred, split["y1"], split["mask_y"])
    return {"full": float(np.nanmean(s)), "prefix": float(np.nanmean(s[:120])),
            "suffix": float(np.nanmean(s[120:])), "last60": float(np.nanmean(s[-60:])),
            "blocks": [float(np.nanmean(x)) for x in np.array_split(s, 4)]}


def main():
    started = time.time()
    cfg = load_config("configs/hist_lgbm_alpha_x.yaml")
    features = dict(cfg["features"])
    features["max_train_stocks_per_day"] = 800
    cats = resolve_cat_indices(features)
    data = load_panel(str(resolve_data_path(cfg)))
    drop_other_label(data, "y1")
    train, valid, test = [slice_split(data, s) for s in ("train", "valid", "test")]
    y = np.asarray(data["y1"], dtype=np.float32)
    mask_y = np.asarray(data["mask_y"], dtype=bool)
    lag1, lag2 = lag_panel(y, mask_y)
    print("BUILD TRAIN FEATURES", flush=True)
    x, target, coords = train_features(train, lag1, lag2, features, cats, y)
    print("TRAIN", x.shape, flush=True)
    n_base = x.shape[1] - 2
    cat_cols = cat_feature_col_indices(n_base, cats, features)
    names = feature_names(n_base, cats, features) + ["lag1_y1_rank", "lag2_y1_rank"]
    params = dict(objective="regression", learning_rate=.04, n_estimators=550, num_leaves=63,
                  min_child_samples=80, lambda_l2=3.0, subsample=.85, colsample_bytree=.85,
                  max_bin=127, n_jobs=6, verbosity=-1, random_state=123, deterministic=True,
                  force_col_wise=True)
    model = lgb.LGBMRegressor(**params)
    model.fit(x, target, categorical_feature=cat_cols)
    model.booster_.save_model(str(ROOT / "checkpoints/task1_autoreg_y1.txt"))
    print("PREDICT RECURSIVE", flush=True)
    pv = predict_recursive(model, valid, lag1, lag2, features, cats,
                           y[valid["start"] - 1], y[valid["start"] - 2])
    print("AUTOREG", json.dumps(block_report(pv, valid)), flush=True)
    base_v = np.load(OUT / "fusion_alpha_platform_mix_valid.npy")
    base_t = np.load(OUT / "fusion_alpha_platform_mix_test.npy")
    ar_v = panel_cs_rank(pv, valid["mask_x"])
    base_vr = panel_cs_rank(base_v, valid["mask_x"])
    base_s = rank_ic_series(base_vr, valid["y1"], valid["mask_y"])
    options = []
    for w in [0.15, .30, .50, .70, 1.0]:
        z = (1.0 - w) * base_vr + w * ar_v
        s = rank_ic_series(z, valid["y1"], valid["mask_y"])
        options.append((float(np.nanmean(s)), float(np.nanmean(s[:120])), float(np.nanmean(s[120:])), w, z, s))
    options.sort(reverse=True, key=lambda x: x[0])
    best = options[0]
    print("OPTIONS", [(a, b, c, w) for a, b, c, w, _, _ in options], flush=True)
    # Require both halves to improve before creating a submission.
    accepted = best[2] > float(np.nanmean(base_s[120:])) + .0003 and best[0] > float(np.nanmean(base_s)) + .0005
    report = {"autoreg": block_report(pv, valid), "base": block_report(base_v, valid),
              "options": [{"full": a, "prefix": b, "suffix": c, "weight": w} for a,b,c,w,_,_ in options],
              "accepted": bool(accepted), "seconds": time.time() - started}
    if accepted:
        pt = predict_recursive(model, test, lag1, lag2, features, cats,
                               y[test["start"] - 1], y[test["start"] - 2])
        zt = (1.0-best[3]) * panel_cs_rank(base_t, test["mask_x"]) + best[3] * panel_cs_rank(pt, test["mask_x"])
        tag = f"task1_fusion_autoreg_y1_w{int(best[3]*100):02d}"
        np.save(OUT / f"{tag}_valid.npy", best[4]); np.save(OUT / f"{tag}_test.npy", zt)
        save_submission(zt, ROOT / f"submissions/{tag}.npy")
        report["submission"] = f"submissions/{tag}.npy"
        print("WROTE", report["submission"], flush=True)
    else:
        print("REJECTED_SUFFIX", flush=True)
    (OUT / "task1_autoreg_y1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
