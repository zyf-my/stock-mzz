"""Cross-section LightGBM baseline (stage 2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset import (
    build_sample_features,
    cat_feature_col_indices,
    feature_names,
    history_width,
    market_state_width,
    resolve_cat_indices,
)


class LightGBMBaseline:
    """Fit on train rows, predict a (T, S) score panel for a split."""

    def __init__(self, params: dict[str, Any], feature_cfg: dict[str, Any] | None = None, seed: int = 42):
        self.params = dict(params)
        self.feature_cfg = dict(feature_cfg or {})
        self.seed = int(seed)
        self.model = None
        self.booster = None
        self.booster2 = None
        self.feat_idx2: np.ndarray | None = None
        self.cat_indices: list[int] = resolve_cat_indices(self.feature_cfg)
        self.cat_feature_indices: list[int] = []
        self.feature_name_list: list[str] = []

    def fit(self, x, y, group=None, sample_weight=None, init_model=None) -> None:
        x = np.asarray(x)
        y = np.asarray(y)
        weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        n_num = self._infer_n_num(x.shape[1])
        self.cat_feature_indices = cat_feature_col_indices(n_num, self.cat_indices, self.feature_cfg)
        self.feature_name_list = feature_names(n_num, self.cat_indices, self.feature_cfg)

        params = dict(self.params)
        params.setdefault("random_state", self.seed)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", -1)
        objective = str(params.get("objective", "regression")).lower()
        fit_kw: dict[str, Any] = {}
        if self.cat_feature_indices:
            fit_kw["categorical_feature"] = self.cat_feature_indices
        lgb_weight_kw: dict[str, Any] = {}
        if weight is not None:
            lgb_weight_kw["weight"] = weight
        if objective == "lambdarank_ic":
            if group is None:
                raise ValueError("LambdaRankIC needs per-day group sizes")
            import lightgbm as lgb

            from src.models.lambdarank_ic import make_lambdarank_ic_objective

            lgb_params = self._to_lgb_params(params)
            lgb_params["metric"] = "None"
            max_pairs = int(params.get("max_pair_samples", 2048))
            train_set = lgb.Dataset(
                x, label=y, group=np.asarray(group, dtype=np.int32), **lgb_weight_kw, **fit_kw
            )
            rounds = int(params.get("n_estimators", 400))
            lgb_params["objective"] = make_lambdarank_ic_objective(max_pair_samples=max_pairs, seed=self.seed)

            def _log_period(env):
                if env.iteration == 0 or (env.iteration + 1) % 25 == 0:
                    print(f"  lgb iter {env.iteration + 1}/{rounds}", flush=True)

            self.booster = lgb.train(
                lgb_params,
                train_set,
                num_boost_round=rounds,
                callbacks=[_log_period],
            )
            self.model = None
        elif objective in {"lambdarank", "rank_xendcg"}:
            from lightgbm import LGBMRanker

            if group is None:
                raise ValueError("LambdaRank 需要按日 group")
            self.model = LGBMRanker(**params)
            sk_kw = dict(fit_kw)
            if weight is not None:
                sk_kw["sample_weight"] = weight
            self.model.fit(x, y, group=np.asarray(group, dtype=np.int32), **sk_kw)
            self.booster = self.model.booster_
        elif init_model is not None:
            import lightgbm as lgb

            train_set = lgb.Dataset(x, label=y, **lgb_weight_kw, **fit_kw)
            lgb_params = self._to_lgb_params(params)
            rounds = int(params.get("n_estimators", 400))
            print(f"finetune init_model rounds={rounds} rows={x.shape[0]}", flush=True)
            self.booster = lgb.train(
                lgb_params,
                train_set,
                num_boost_round=rounds,
                init_model=init_model,
            )
            self.model = None
        else:
            from lightgbm import LGBMRegressor

            self.model = LGBMRegressor(**params)
            sk_kw = dict(fit_kw)
            if weight is not None:
                sk_kw["sample_weight"] = weight
            self.model.fit(x, y, **sk_kw)
            self.booster = self.model.booster_

    def fit_double_ensemble(self, x, y, *, keep_frac: float = 0.6, sample_weight=None) -> None:
        """Qlib-style DoubleEnsemble: reweight hard rows + drop weak features, then bag."""
        self.fit(x, y, sample_weight=sample_weight)
        if self.booster is None:
            raise RuntimeError("first booster missing")
        p1 = np.asarray(self.booster.predict(x), dtype=np.float64)
        yv = np.asarray(y, dtype=np.float64)
        resid = np.abs(yv - p1)
        w = resid / (float(resid.mean()) + 1e-8)
        w = np.clip(w, 0.3, 3.0)
        if sample_weight is not None:
            w = w * np.asarray(sample_weight, dtype=np.float64)
            w = w / (float(w.mean()) + 1e-8)
        imp = np.asarray(self.booster.feature_importance(), dtype=np.float64)
        n_keep = max(int(imp.size * float(keep_frac)), min(32, imp.size))
        keep = np.sort(np.argsort(imp)[::-1][:n_keep])
        if self.cat_feature_indices:
            cats = np.asarray(self.cat_feature_indices, dtype=np.int64)
            keep = np.unique(np.concatenate([keep, cats[cats < imp.size]]))
        self.feat_idx2 = keep.astype(np.int32)
        print(
            f"double_ensemble keep={self.feat_idx2.size}/{imp.size} "
            f"w_mean={float(w.mean()):.3f} w_max={float(w.max()):.3f}",
            flush=True,
        )
        from lightgbm import LGBMRegressor

        params = dict(self.params)
        params.setdefault("random_state", self.seed + 1)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", -1)
        m2 = LGBMRegressor(**params)
        fit_kw: dict[str, Any] = {"sample_weight": w}
        pos = {int(v): i for i, v in enumerate(self.feat_idx2)}
        cat2 = [pos[c] for c in self.cat_feature_indices if c in pos]
        if cat2:
            fit_kw["categorical_feature"] = cat2
        m2.fit(np.asarray(x)[:, self.feat_idx2], yv, **fit_kw)
        self.booster2 = m2.booster_
        print("double_ensemble second booster fitted", flush=True)

    def predict_panel(
        self,
        split_data: dict[str, Any],
        fill_invalid: float = 0.0,
        num_iteration: int | None = None,
    ) -> np.ndarray:
        iters = [num_iteration]
        return self.predict_panel_iters(split_data, iters, fill_invalid=fill_invalid)[num_iteration]

    def predict_panel_delta(
        self,
        split_data: dict[str, Any],
        iter_hi: int,
        iter_lo: int,
        fill_invalid: float = 0.0,
    ) -> np.ndarray:
        """Score contribution of trees (iter_lo, iter_hi] — finetune delta only."""
        hi = self.predict_panel(split_data, fill_invalid=fill_invalid, num_iteration=iter_hi)
        lo = self.predict_panel(split_data, fill_invalid=fill_invalid, num_iteration=iter_lo)
        return (hi - lo).astype(np.float32)

    def predict_panel_iters(
        self,
        split_data: dict[str, Any],
        iterations: list[int | None],
        fill_invalid: float = 0.0,
    ) -> dict[int | None, np.ndarray]:
        if self.booster is None and self.model is None:
            raise RuntimeError("model is not fitted")
        n_days, n_stocks, _ = split_data["num_x"].shape
        outs = {
            n: np.full((n_days, n_stocks), float(fill_invalid), dtype=np.float32)
            for n in iterations
        }
        for t in range(n_days):
            if n_days > 400 and t > 0 and t % 400 == 0:
                print(f"  predict day {t}/{n_days}", flush=True)
            mask = np.asarray(split_data["mask_x"][t], dtype=bool)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            x = build_sample_features(
                split_data["num_x"][t],
                split_data["cat_x"][t],
                mask,
                idx,
                self.cat_indices,
                self.feature_cfg,
                global_t=int(split_data.get("start", 0)) + t,
                panel_num_x=split_data.get("panel_num_x"),
                panel_mask_x=split_data.get("panel_mask_x"),
            )
            for n in iterations:
                outs[n][t, idx] = self._predict_rows(x, num_iteration=n)
        return outs

    def feature_importance(self) -> list[tuple[str, float]]:
        if self.model is None and self.booster is None:
            raise RuntimeError("model is not fitted")
        gains = np.asarray(self.model.feature_importances_ if self.model is not None else self.booster.feature_importance(), dtype=np.float64)
        names = self.feature_name_list or [f"f{i}" for i in range(gains.size)]
        pairs = list(zip(names, gains.tolist()))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs

    def save(self, path: str | Path) -> None:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # LightGBM's C save_model does not accept non-ASCII Windows paths.
        path.write_text(self.booster.model_to_string(), encoding="utf-8")
        meta = {
            "params": self.params,
            "feature_cfg": self.feature_cfg,
            "cat_indices": self.cat_indices,
            "cat_feature_indices": self.cat_feature_indices,
            "feature_names": self.feature_name_list,
            "seed": self.seed,
        }
        if self.booster2 is not None and self.feat_idx2 is not None:
            ens2 = path.with_suffix(path.suffix + ".ens2.txt")
            ens2.write_text(self.booster2.model_to_string(), encoding="utf-8")
            meta["double_ensemble"] = True
            meta["feat_idx2"] = [int(i) for i in self.feat_idx2]
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        from lightgbm import Booster

        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.params = meta.get("params", self.params)
            self.feature_cfg = meta.get("feature_cfg", self.feature_cfg)
            self.cat_indices = list(meta.get("cat_indices", resolve_cat_indices(self.feature_cfg)))
            self.cat_feature_indices = list(meta.get("cat_feature_indices") or [])
            self.feature_name_list = list(meta.get("feature_names") or [])
            self.seed = int(meta.get("seed", self.seed))
        self.booster = Booster(model_str=path.read_text(encoding="utf-8"))
        self.model = None
        self.booster2 = None
        self.feat_idx2 = None
        if meta_path.is_file() and meta.get("double_ensemble") and meta.get("feat_idx2"):
            ens2 = path.with_suffix(path.suffix + ".ens2.txt")
            if ens2.is_file():
                self.booster2 = Booster(model_str=ens2.read_text(encoding="utf-8"))
                self.feat_idx2 = np.asarray(meta["feat_idx2"], dtype=np.int32)

    def _infer_n_num(self, n_feat: int) -> int:
        n_cat = len(self.cat_indices)
        n_blocks = numeric_block_count_safe(self.feature_cfg)
        hist_w = history_width(self.feature_cfg)
        mkt_w = market_state_width(self.feature_cfg)
        from src.dataset import stock_zscore_width

        stock_z_w = stock_zscore_width(self.feature_cfg)
        rest = n_feat - n_cat - hist_w - mkt_w - stock_z_w
        if n_blocks <= 0:
            if rest != 0:
                raise ValueError(
                    f"cannot infer numeric width from n_feat={n_feat} n_cat={n_cat} "
                    f"blocks={n_blocks} stock_z={stock_z_w}"
                )
            return 0
        n_num, rem = divmod(rest, n_blocks)
        if rem != 0 or n_num <= 0:
            raise ValueError(f"cannot infer numeric width from n_feat={n_feat} n_cat={n_cat} blocks={n_blocks}")
        return n_num

    def _to_lgb_params(self, params: dict[str, Any]) -> dict[str, Any]:
        p = dict(params)
        out: dict[str, Any] = {
            "learning_rate": float(p.get("learning_rate", 0.05)),
            "num_leaves": int(p.get("num_leaves", 31)),
            "verbosity": int(p.get("verbosity", -1)),
            "seed": int(p.get("random_state", self.seed)),
            "feature_pre_filter": False,
        }
        if "max_depth" in p:
            out["max_depth"] = int(p["max_depth"])
        if "min_child_samples" in p:
            out["min_data_in_leaf"] = int(p["min_child_samples"])
        if "min_split_gain" in p:
            out["min_split_gain"] = float(p["min_split_gain"])
        if "subsample" in p:
            out["bagging_fraction"] = float(p["subsample"])
            out["bagging_freq"] = int(p.get("bagging_freq", 1))
        if "colsample_bytree" in p:
            out["feature_fraction"] = float(p["colsample_bytree"])
        if "lambda_l2" in p:
            out["reg_lambda"] = float(p["lambda_l2"])
        if "lambda_l1" in p:
            out["reg_alpha"] = float(p["lambda_l1"])
        return out

    def _predict_rows(self, x: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
        if self.booster is not None:
            kw = {} if num_iteration is None else {"num_iteration": int(num_iteration)}
            p = np.asarray(self.booster.predict(x, **kw), dtype=np.float32)
            if self.booster2 is not None and self.feat_idx2 is not None:
                p2 = np.asarray(self.booster2.predict(x[:, self.feat_idx2], **kw), dtype=np.float32)
                p = (0.5 * p + 0.5 * p2).astype(np.float32)
            return p
        if self.model is not None:
            if num_iteration is not None:
                return np.asarray(self.model.predict(x, num_iteration=int(num_iteration)), dtype=np.float32)
            return np.asarray(self.model.predict(x), dtype=np.float32)
        raise RuntimeError("model is not fitted")


def numeric_block_count_safe(feature_cfg: dict[str, Any]) -> int:
    from src.dataset import numeric_block_count

    return numeric_block_count(feature_cfg)
