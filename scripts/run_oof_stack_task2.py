"""OOF 3-way stack for task2 (y2). Weights from train OOF, eval on locked full models.

Does NOT overwrite task2_fusion_best.npy.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\run_oof_stack_task2.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import (  # noqa: E402
    drop_other_label,
    flatten_masked_rows,
    load_eval_splits,
    load_panel,
    resolve_cat_indices,
    slice_days,
    slice_split,
    split_label_array,
)
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.baseline import LightGBMBaseline  # noqa: E402
from src.models.cs_mlp import CSMLPModel  # noqa: E402
from src.models.fusion import coverage_gate_blend, linear_blend, panel_cs_rank  # noqa: E402
from src.models.gru_ts import GRUModel  # noqa: E402
from src.submit import save_submission  # noqa: E402

# recipe builder for baseline hi31
import importlib.util

_spec = importlib.util.spec_from_file_location("eval_task2_recipe", ROOT / "scripts" / "eval_task2_recipe.py")
_etr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_etr)

OUT = ROOT / "outputs" / "task2"
OOF_DIR = OUT / "oof"
LABEL = "y2"
OOF_DAYS = 400
HIST_N = 200
WEIGHT_STEP = 0.05
HIGH_TAU = 4670.0
GRU_RECENT_TODAY = 800
GRU_RECENT_HICOV = 600

GRU_SPECS = (
    ("x6", "configs/task2/gru_x6_hicov.yaml", "checkpoints/task2/gru_x6_hicov.pt"),
    ("only6", "configs/task2/gru_only6_hicov.yaml", "checkpoints/task2/gru_only6_hicov.pt"),
    ("next6", "configs/task2/gru_next6_hicov.yaml", "checkpoints/task2/gru_next6_hicov.pt"),
)
FALLBACK_GRU = (
    ("x6", "configs/task2/gru_x6_with_today.yaml", "checkpoints/task2/gru_x6_with_today.pt"),
    ("only6", "configs/task2/gru_only6_with_today.yaml", "checkpoints/task2/gru_only6_with_today.pt"),
    ("next6", "configs/task2/gru_next6_with_today.yaml", "checkpoints/task2/gru_next6_with_today.pt"),
)


def _simplex(step: float = WEIGHT_STEP):
    n = int(round(1.0 / step))
    out = []
    for i in range(n + 1):
        for j in range(n - i + 1):
            wg, wt = i * step, j * step
            wm = 1.0 - wg - wt
            out.append((float(wg), float(wt), float(round(wm, 10))))
    return out


def _gru_specs(mode: str) -> tuple:
    if mode == "hicov":
        for spec in GRU_SPECS:
            if not (ROOT / spec[2]).is_file():
                raise FileNotFoundError(f"missing {spec[2]}")
        return GRU_SPECS
    return FALLBACK_GRU


def _predict_gru(data, split, spec) -> np.ndarray:
    tag, cfg_path, ckpt = spec
    cfg = load_config(cfg_path)
    feat = dict(cfg.get("features") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    gru = GRUModel({**feat, **model_cfg, **train_cfg, "label_key": LABEL}, seed=int(cfg.get("seed", 42)))
    gru.load(ROOT / ckpt)
    gru.prepare_features(data)
    pred = gru.predict_panel(split, data, fill_invalid=0.0)
    y = split_label_array(split, LABEL)
    print(f"  {tag} OOF RankIC={mean_rank_ic(pred, y, split['mask_y']):.6f}", flush=True)
    gru.cs_sel = gru.net = None
    gc.collect()
    return pred


def _train_tree(data, fit_split, pred_split, cfg_path: str, tag: str) -> np.ndarray:
    cfg = load_config(cfg_path)
    feature_cfg = dict(cfg.get("features") or {})
    cat_indices = resolve_cat_indices(feature_cfg)
    seed = int(cfg.get("seed", 42))
    x, y, _ = flatten_masked_rows(
        fit_split,
        cat_indices=cat_indices,
        require_label=True,
        feature_cfg=feature_cfg,
        seed=seed,
        label_key=LABEL,
    )
    model = LightGBMBaseline(
        params=(cfg.get("model") or {}).get("params") or {},
        feature_cfg=feature_cfg,
        seed=seed,
    )
    model.fit(x, y)
    del x, y
    gc.collect()
    if "hist" in tag:
        pred = model.predict_panel(pred_split, fill_invalid=0.0, num_iteration=HIST_N)
    else:
        pred = model.predict_panel(pred_split, fill_invalid=0.0)
    y = split_label_array(pred_split, LABEL)
    print(f"  {tag} OOF RankIC={mean_rank_ic(pred, y, pred_split['mask_y']):.6f}", flush=True)
    return pred


def _train_mlp_oof(data, prefix_start: int, prefix_end: int, oof, cfg_path: str, tag: str) -> np.ndarray:
    cfg = load_config(cfg_path)
    feat = dict(cfg.get("features") or {})
    model_cfg = dict(cfg.get("model") or {})
    train_cfg = dict(cfg.get("train") or {})
    mlp_cfg = {**feat, **model_cfg, **train_cfg, "label_key": LABEL}
    mlp = CSMLPModel(mlp_cfg, seed=int(cfg.get("seed", 42)))
    mlp.prepare_features(data)
    mlp.fit(data, prefix_start, prefix_end, valid=None)
    pred = mlp.predict_panel(oof, data, fill_invalid=0.0)
    y = split_label_array(oof, LABEL)
    print(f"  {tag} OOF RankIC={mean_rank_ic(pred, y, oof['mask_y']):.6f}", flush=True)
    mlp.ind_sel = mlp.net = None
    gc.collect()
    return pred


def _search(gru, tree, mlp, y, my, mx) -> dict:
    best = {"ic": -1e9}
    for space in ("rank",):
        g = panel_cs_rank(gru, mx)
        t = panel_cs_rank(tree, mx)
        m = panel_cs_rank(mlp, mx)
        for wg, wt, wm in _simplex():
            pred = (wg * g + wt * t + wm * m).astype(np.float32)
            ic = mean_rank_ic(pred, y, my)
            if np.isfinite(ic) and ic > best["ic"]:
                best = {"ic": float(ic), "wg": wg, "wt": wt, "wm": wm, "space": space}
    return best


def _hi31(pred, y, my, mx) -> float:
    s = rank_ic_series(pred, y, my)
    hi = np.asarray(mx).sum(axis=1) >= HIGH_TAU
    s[~hi] = np.nan
    return float(np.nanmean(s))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gru", choices=("today", "hicov"), default="today")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    gru_recent = GRU_RECENT_HICOV if args.gru == "hicov" else GRU_RECENT_TODAY
    oof_dir = OUT / "oof" / args.gru

    try:
        import torch

        torch.set_num_threads(max(1, os.cpu_count() or 4))
    except Exception:
        pass

    t0 = time.perf_counter()
    oof_dir.mkdir(parents=True, exist_ok=True)
    gru_path = oof_dir / "gru_ens_oof.npy"
    tree_path = oof_dir / "tree_oof.npy"
    mlp_path = oof_dir / "mlp_oof.npy"

    cfg = load_config("configs/task2/hist_lgbm.yaml")
    data = load_panel(str(resolve_data_path(cfg, None)))
    drop_other_label(data, LABEL)
    train = slice_split(data, "train")
    n_train = int(train["num_x"].shape[0])
    gru_fit_local = n_train - gru_recent
    oof_local0 = gru_fit_local - OOF_DAYS
    prefix = slice_days(train, 0, oof_local0)
    oof = slice_days(train, oof_local0, gru_fit_local)
    prefix_start = int(prefix["start"])
    prefix_end = prefix_start + int(prefix["num_x"].shape[0])
    print(f"OOF local [{oof_local0},{gru_fit_local}) prefix global [{prefix_start},{prefix_end})")

    specs = _gru_specs(args.gru)
    ow, nw = 0.7, 0.34

    if args.refresh:
        for p in (gru_path, tree_path, mlp_path):
            p.unlink(missing_ok=True)

    if gru_path.is_file():
        gru_oof = np.load(gru_path)
    else:
        print("GRU OOF (frozen)", flush=True)
        parts = {}
        for spec in specs:
            parts[spec[0]] = _predict_gru(data, oof, spec)
            np.save(oof_dir / f"gru_{spec[0]}_oof.npy", parts[spec[0]])
        gru_oof = linear_blend(parts["next6"], linear_blend(parts["only6"], parts["x6"], ow), nw)
        np.save(gru_path, gru_oof)
        del parts
        gc.collect()

    if tree_path.is_file():
        tree_oof = np.load(tree_path)
    else:
        print("trees on prefix", flush=True)
        hist_oof = _train_tree(data, prefix, oof, "configs/task2/hist_lgbm.yaml", "hist")
        base_oof = _train_tree(data, prefix, oof, "configs/task2/baseline.yaml", "baseline")
        mx_oof = oof["mask_x"]
        tree_oof = coverage_gate_blend(hist_oof, base_oof, mx_oof, 4614.0, 0.45, 1.0, "rank")
        np.save(tree_path, tree_oof)
        del hist_oof, base_oof
        gc.collect()

    if mlp_path.is_file():
        mlp_oof = np.load(mlp_path)
    else:
        print("MLP on prefix", flush=True)
        mlp_oof = _train_mlp_oof(data, prefix_start, prefix_end, oof, "configs/task2/cs_mlp.yaml", "cs_mlp")
        np.save(mlp_path, mlp_oof)

    y, my, mx = split_label_array(oof, LABEL), oof["mask_y"], oof["mask_x"]
    best = _search(gru_oof, tree_oof, mlp_oof, y, my, mx)
    print(f"OOF best ic={best['ic']:.6f} w=({best['wg']:.2f},{best['wt']:.2f},{best['wm']:.2f})")

    del data, train, prefix, oof
    gc.collect()

    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    yv = split_label_array(valid, LABEL)
    myv, mxv = valid["mask_y"], valid["mask_x"]

    def _full_gru(split: str) -> np.ndarray:
        lt = lambda s: np.load(OUT / f"{s}_{split}.npy")
        if args.gru == "hicov":
            x6_n, o6_n, n6_n = "gru_x6_hicov", "gru_only6_hicov", "gru_next6_hicov"
        else:
            x6_n, o6_n, n6_n = "gru_x6_with_today", "gru_only6_with_today", "gru_next6_with_today"
        return linear_blend(lt(n6_n), linear_blend(lt(o6_n), lt(x6_n), ow), nw)

    def _full_tree(split: str, mx) -> np.ndarray:
        lt = lambda s: np.load(OUT / f"{s}_{split}.npy")
        return coverage_gate_blend(lt("hist_lgbm_n200"), lt("baseline"), mx, 4614.0, 0.45, 1.0, "rank")

    def _blend(split: str, mx) -> np.ndarray:
        g, t, m = _full_gru(split), _full_tree(split, mx), np.load(OUT / f"cs_mlp_{split}.npy")
        if best["space"] == "rank":
            g, t, m = panel_cs_rank(g, mx), panel_cs_rank(t, mx), panel_cs_rank(m, mx)
        return (best["wg"] * g + best["wt"] * t + best["wm"] * m).astype(np.float32)

    out_v = _blend("valid", mxv)
    out_t = _blend("test", test["mask_x"])
    ic = mean_rank_ic(out_v, yv, myv)
    hi = _hi31(out_v, yv, myv, mxv)

    old_path = OUT / "recipes" / "OLD_091354.json"
    old_recipe = json.loads(old_path.read_text(encoding="utf-8"))
    old_hi = float(_etr.metrics(old_recipe)["hi31_ic"])

    meta = {
        "valid_ic": float(ic),
        "hi31_ic": float(hi),
        "oof_ic": best["ic"],
        "weights": {"gru": best["wg"], "tree": best["wt"], "mlp": best["wm"]},
        "gru_stems": [s[0] for s in specs],
        "baseline_hi31": old_hi,
        "pass_gate": bool(hi >= old_hi - 1e-6 and ic >= 0.088897 - 0.0005),
    }
    (oof_dir / "stack_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"valid={ic:.6f} hi31={hi:.6f} baseline_hi31={old_hi:.6f} gate={'PASS' if meta['pass_gate'] else 'FAIL'}")

    np.save(OUT / "oof_stack_valid.npy", out_v)
    np.save(OUT / "oof_stack_test.npy", out_t)
    if meta["pass_gate"]:
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_oof_stack.npy")
        print("wrote submissions/task2_fusion_oof_stack.npy")
    else:
        print("did not pass gate; keep task2_fusion_best.npy")
    print(f"total {time.perf_counter() - t0:.1f}s")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
