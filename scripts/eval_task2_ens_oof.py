"""Search GRU ens (ow,nw) on train OOF window; apply with OLD gate on full models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_eval_splits, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic, rank_ic_series  # noqa: E402
from src.models.fusion import FusionModel, coverage_gate_blend, linear_blend  # noqa: E402
from src.submit import save_submission  # noqa: E402

OUT = ROOT / "outputs" / "task2"
OOF_DAYS = 400
GRU_RECENT = 800
HIGH_TAU = 4670.0


def _hi31(pred, y, my, mx) -> float:
    s = rank_ic_series(pred, y, my)
    s[np.asarray(mx).sum(axis=1) < HIGH_TAU] = np.nan
    return float(np.nanmean(s))


def main() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("etr", ROOT / "scripts/eval_task2_recipe.py")
    etr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(etr)
    old = json.loads((OUT / "recipes" / "OLD_091354.json").read_text(encoding="utf-8"))
    old_hi = etr.metrics(old)["hi31_ic"]

    oof_dir = OUT / "oof" / "today"
    if not (oof_dir / "gru_x6_oof.npy").is_file():
        print("run run_oof_stack_task2.py --gru today first (partial oof)")
        return

    from src.config import load_config, resolve_data_path
    from src.dataset import drop_other_label, load_panel, slice_days, slice_split

    data = load_panel(str(resolve_data_path(load_config("configs/task2/hist_lgbm.yaml"), None)))
    drop_other_label(data, "y2")
    train = slice_split(data, "train")
    n = int(train["num_x"].shape[0])
    o0 = n - GRU_RECENT - OOF_DAYS
    o1 = n - GRU_RECENT
    oof = slice_days(train, o0, o1)
    y, my, mx = split_label_array(oof, "y2"), oof["mask_y"], oof["mask_x"]

    x6 = np.load(oof_dir / "gru_x6_oof.npy")
    o6 = np.load(oof_dir / "gru_only6_oof.npy")
    n6 = np.load(oof_dir / "gru_next6_oof.npy")
    tree = np.load(oof_dir / "tree_oof.npy")
    mlp = np.load(oof_dir / "mlp_oof.npy")

    tp, gp = old["tree"], old["gate"]
    mw = float(old["mlp_w"])

    splits, _ = load_eval_splits(splits=("valid", "test"), dump_if_missing=False)
    valid, test = splits["valid"], splits["test"]
    yv = split_label_array(valid, "y2")
    myv, mxv = valid["mask_y"], valid["mask_x"]
    lt = lambda s, sp: np.load(OUT / f"{s}_{sp}.npy")

    best = (-1.0, None, None)
    for ow in (0.60, 0.65, 0.70, 0.75, 0.80):
        for nw in (0.25, 0.30, 0.34, 0.38, 0.42):
            ens_oof = linear_blend(n6, linear_blend(o6, x6, ow), nw)
            ic_oof = mean_rank_ic(ens_oof, y, my)
            ens_v = linear_blend(
                lt("gru_next6_with_today", "valid"),
                linear_blend(lt("gru_only6_with_today", "valid"), lt("gru_x6_with_today", "valid"), ow),
                nw,
            )
            tree_v = coverage_gate_blend(
                lt("hist_lgbm_n200", "valid"), lt("baseline", "valid"), mxv,
                tp["tau"], tp["w_lo"], tp["w_hi"], "rank",
            )
            gated = coverage_gate_blend(ens_v, tree_v, mxv, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
            b = FusionModel({"weight_grid": [mw]})
            b.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
            out = b.predict(lt("cs_mlp", "valid"), gated, mxv)
            ic = mean_rank_ic(out, yv, myv)
            hi = _hi31(out, yv, myv, mxv)
            if hi >= old_hi - 1e-6 and ic > best[0]:
                best = (ic, (ow, nw, ic_oof, hi), out)

    if best[1] is None:
        print("no ens beat baseline hi31")
        print("VALID_RANKIC", f"{etr.metrics(old)['valid_ic']:.6f}")
        return

    ic, (ow, nw, ic_oof, hi), out_v = best[0], best[1], best[2]
    print(f"OOF ens best ow={ow} nw={nw} oof_ic={ic_oof:.6f} valid={ic:.6f} hi31={hi:.6f}")

    mx_t = test["mask_x"]
    ens_t = linear_blend(
        lt("gru_next6_with_today", "test"),
        linear_blend(lt("gru_only6_with_today", "test"), lt("gru_x6_with_today", "test"), ow),
        nw,
    )
    tree_t = coverage_gate_blend(
        lt("hist_lgbm_n200", "test"), lt("baseline", "test"), mx_t,
        tp["tau"], tp["w_lo"], tp["w_hi"], "rank",
    )
    gated_t = coverage_gate_blend(ens_t, tree_t, mx_t, gp["tau"], gp["w_lo"], gp["w_hi"], "rank")
    b = FusionModel({"weight_grid": [mw]})
    b.locked = {"name": "rank_blend", "weight": mw, "space": "rank", "ic": float("nan")}
    out_t = b.predict(lt("cs_mlp", "test"), gated_t, mx_t)

    if ic > old["valid_ic"] - 0.0005:
        save_submission(out_t, ROOT / "submissions" / "task2_fusion_ens_oof.npy")
        np.save(OUT / "fusion_ens_oof_valid.npy", out_v)
        recipe = {**old, "ens": {"only6_w": ow, "next6_w": nw}, "valid_ic": ic, "hi31_ic": hi, "name": "ens_oof"}
        (OUT / "recipes" / "ens_oof.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        print("wrote task2_fusion_ens_oof.npy")
    print("VALID_RANKIC", f"{ic:.6f}")


if __name__ == "__main__":
    main()
