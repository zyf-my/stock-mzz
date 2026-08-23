"""Qlib-style per-stock GRU on a short window of cross-section z-scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from scipy.stats import rankdata

from src.dataset import gather_windows, precompute_cs_cols, precompute_ind_cols


class GRUNet(nn.Module):
    def __init__(
        self,
        d_feat: int,
        hidden: int,
        layers: int,
        dropout: float,
        cat_cardinalities: list[int] | None = None,
        cat_embed_dim: int = 8,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        cards = [int(c) for c in (cat_cardinalities or []) if int(c) > 0]
        self.embeds = nn.ModuleList(nn.Embedding(c, cat_embed_dim) for c in cards) if cards else None
        in_fc = hidden + (len(cards) * cat_embed_dim if cards else 0)
        self.fc = nn.Linear(in_fc, 1)
        self.head_drop = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor, cats: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.rnn(x)
        h = self.head_drop(out[:, -1, :])
        if self.embeds is not None:
            if cats is None:
                raise ValueError("GRUNet was built with category embeddings but cats is None")
            pieces = [h]
            for i, emb in enumerate(self.embeds):
                idx = cats[:, i].clamp(0, emb.num_embeddings - 1)
                pieces.append(emb(idx))
            h = torch.cat(pieces, dim=-1)
        return self.fc(h).squeeze(-1)


def pearson_ic_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - Pearson correlation on the current-day batch. Ranking-friendly, no future days."""
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = (pred.norm() * target.norm()).clamp_min(eps)
    return 1.0 - (pred * target).sum() / denom


class GRUModel:
    """One sequence per stock; last step is the prediction-day CS z-score (no future days)."""

    def __init__(self, cfg: dict[str, Any], seed: int = 42):
        self.cfg = dict(cfg)
        self.seed = int(seed)
        self.cols = [int(i) for i in (cfg.get("num_indices") or [])]
        if not self.cols:
            raise ValueError("GRU needs features.num_indices")
        self.length = int(cfg.get("length", 10))
        self.include_current = bool(cfg.get("include_current_day", True))
        hl = cfg.get("input_decay_halflife")
        self.input_decay_halflife = float(hl) if hl is not None else None
        self._input_decay_weights: np.ndarray | None = None
        if self.input_decay_halflife is not None and self.input_decay_halflife > 0:
            ages = np.arange(self.length - 1, -1, -1, dtype=np.float32)
            self._input_decay_weights = np.power(
                0.5, ages / max(self.input_decay_halflife, 1e-3)
            ).astype(np.float32)
        self.hidden_size = int(cfg.get("hidden_size", 64))
        self.num_layers = int(cfg.get("num_layers", 1))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.head_dropout = float(cfg.get("head_dropout", 0.0))
        self.target_mode = str(cfg.get("target_mode", "y1")).lower()
        self.accum_days = max(1, int(cfg.get("accum_days", 1)))
        self.lr_schedule = str(cfg.get("lr_schedule") or "").lower()
        self.clip = cfg.get("clip", 5.0)
        self.max_train_stocks = cfg.get("max_train_stocks_per_day")
        self.min_train_stocks = int(cfg.get("min_train_stocks", 8))
        self.log_every = int(cfg.get("log_every_days", 400))
        self.cat_indices = [int(i) for i in (cfg.get("cat_indices") or [])]
        self.cat_embed_dim = int(cfg.get("cat_embed_dim", 8))
        self.loss_name = str(cfg.get("loss", "mse")).lower()
        self.source = str(cfg.get("source", "cs_zscore"))
        self.industry_col = int(cfg.get("industry_col", 6))
        self.net: GRUNet | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cs_sel: np.ndarray | None = None
        self.cat_sel: np.ndarray | None = None
        self.cat_cardinalities: list[int] = []

    def prepare_features(self, data: dict[str, Any]) -> None:
        src = self.source
        print(f"precompute {src} cols={self.cols} T={data['num_x'].shape[0]}")
        if src in {"industry_zscore", "ind_zscore", "industry"}:
            self.cs_sel = precompute_ind_cols(
                data["num_x"],
                data["mask_x"],
                data["cat_x"],
                self.cols,
                self.industry_col,
            )
        elif src in {"cs_zscore", "cs"}:
            self.cs_sel = precompute_cs_cols(data["num_x"], data["mask_x"], self.cols)
        else:
            raise ValueError(f"unknown GRU feature source {src!r}")
        if self.clip is not None:
            np.clip(self.cs_sel, -float(self.clip), float(self.clip), out=self.cs_sel)
        print(f"cs_sel={self.cs_sel.shape} ~{self.cs_sel.nbytes / 1e9:.2f}GB source={src} device={self.device}")
        if self.cat_indices:
            self.cat_sel = np.asarray(data["cat_x"][..., self.cat_indices], dtype=np.int32)
            maxes = self.cat_sel.max(axis=(0, 1))
            self.cat_cardinalities = [int(m) + 1 for m in np.asarray(maxes).ravel()]
            print(f"cat_indices={self.cat_indices} cardinalities={self.cat_cardinalities}")
        else:
            self.cat_sel = None
            self.cat_cardinalities = []

    def _build_net(self) -> GRUNet:
        dropout = self.dropout if self.num_layers > 1 else 0.0
        return GRUNet(
            len(self.cols),
            self.hidden_size,
            self.num_layers,
            dropout,
            cat_cardinalities=self.cat_cardinalities,
            cat_embed_dim=self.cat_embed_dim,
            head_dropout=self.head_dropout,
        ).to(self.device)

    def _day_target(self, data: dict[str, Any], t: int, idx: np.ndarray) -> np.ndarray:
        y = np.asarray(data["y1"][t], dtype=np.float32)
        if self.target_mode not in {"rank", "cs_rank"}:
            return y[idx]
        labeled = np.asarray(data["mask_y"][t], dtype=bool) & np.isfinite(y)
        out = np.zeros_like(y)
        n = int(labeled.sum())
        if n >= 3:
            r = rankdata(y[labeled], method="average").astype(np.float32)
            r = (r - r.mean()) / (float(r.std()) + 1e-8)
            out[labeled] = r
        return out[idx]

    def _cats_tensor(self, global_t: int, idx: np.ndarray) -> torch.Tensor | None:
        if self.cat_sel is None:
            return None
        cats = np.ascontiguousarray(self.cat_sel[int(global_t), idx])
        return torch.from_numpy(cats.astype(np.int64, copy=False)).to(self.device)

    def _apply_input_decay(self, x: np.ndarray) -> np.ndarray:
        if self._input_decay_weights is None:
            return x
        w = self._input_decay_weights.reshape(1, self.length, 1)
        return np.ascontiguousarray(x * w)

    def _forward(self, x: np.ndarray, global_t: int, idx: np.ndarray) -> torch.Tensor:
        if self.net is None:
            raise RuntimeError("GRU is not fitted")
        xb = torch.from_numpy(self._apply_input_decay(x)).to(self.device)
        return self.net(xb, self._cats_tensor(global_t, idx))

    def _loss(self, pred: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        if self.loss_name in {"pearson_ic", "ic"}:
            return pearson_ic_loss(pred, yb)
        return nn.functional.mse_loss(pred, yb)

    def fit(
        self,
        data: dict[str, Any],
        train_start: int,
        train_end: int,
        valid: dict[str, Any] | None = None,
        scorer=None,
    ) -> dict[str, Any]:
        from src.metrics import mean_rank_ic

        score_fn = scorer or mean_rank_ic
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.cs_sel is None:
            self.prepare_features(data)

        self.net = self._build_net()
        lr = float(self.cfg.get("lr", 1e-3))
        decay = float(self.cfg.get("weight_decay", 1e-4))
        epochs = int(self.cfg.get("max_epochs", 8))
        patience = int(self.cfg.get("patience", 3))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=decay)
        scheduler = None
        if self.lr_schedule in {"cosine", "cosine_epoch"}:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(epochs, 1), eta_min=max(lr * 0.05, 1e-6)
            )
        rng = np.random.default_rng(self.seed)
        best_ic = -1e9
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        history: list[dict[str, Any]] = []
        n_params = sum(p.numel() for p in self.net.parameters())
        print(
            f"gru params={n_params} hidden={self.hidden_size} L={self.length} "
            f"include_t={self.include_current} source={self.source} "
            f"loss={self.loss_name} target={self.target_mode} accum={self.accum_days} "
            f"lr_sched={self.lr_schedule or 'const'} head_drop={self.head_dropout} "
            f"input_decay_hl={self.input_decay_halflife or '-'} "
            f"wd={decay} cats={self.cat_indices or '-'}"
        )

        days = list(range(int(train_start), int(train_end)))

        def _opt_step() -> None:
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        for epoch in range(epochs):
            self.net.train()
            rng.shuffle(days)
            losses: list[float] = []
            accum = 0
            opt.zero_grad(set_to_none=True)
            for i, t in enumerate(days):
                mask_y = np.asarray(data["mask_y"][t], dtype=bool)
                idx = np.flatnonzero(mask_y)
                if idx.size < self.min_train_stocks:
                    continue
                if self.max_train_stocks and idx.size > int(self.max_train_stocks):
                    idx = np.sort(rng.choice(idx, size=int(self.max_train_stocks), replace=False))
                x = gather_windows(
                    self.cs_sel,
                    data["mask_x"],
                    t,
                    idx,
                    self.length,
                    include_current=self.include_current,
                )
                y = self._day_target(data, t, idx)
                yb = torch.from_numpy(np.ascontiguousarray(y)).to(self.device)
                pred = self._forward(x, t, idx)
                loss = self._loss(pred, yb) / float(self.accum_days)
                loss.backward()
                accum += 1
                losses.append(float(loss.item()) * float(self.accum_days))
                if accum >= self.accum_days:
                    _opt_step()
                    accum = 0
                if self.log_every and (i + 1) % self.log_every == 0:
                    print(
                        f"  epoch {epoch + 1} day {i + 1}/{len(days)} "
                        f"loss={float(np.mean(losses[-self.log_every:])):.5f} "
                        f"lr={float(opt.param_groups[0]['lr']):.2e}",
                        flush=True,
                    )
            if accum > 0:
                _opt_step()
            if scheduler is not None:
                scheduler.step()
            valid_ic = None
            if valid is not None:
                pred_v = self.predict_panel(valid, data)
                valid_ic = float(score_fn(pred_v, valid["y1"], valid["mask_y"]))
                if np.isfinite(valid_ic) and valid_ic > best_ic:
                    best_ic = valid_ic
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
            mean_loss = float(np.mean(losses)) if losses else 0.0
            print(f"epoch {epoch + 1}/{epochs} loss={mean_loss:.5f} valid_RankIC={valid_ic}", flush=True)
            history.append({"epoch": epoch + 1, "loss": mean_loss, "valid_ic": valid_ic})
            if valid is not None and stale >= patience:
                print("early stop")
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return {"best_valid_ic": best_ic if best_ic > -1e8 else None, "history": history}

    def _mask_panel(self, split_data: dict[str, Any], full_data: dict[str, Any] | None) -> np.ndarray:
        if full_data is not None and "mask_x" in full_data:
            return full_data["mask_x"]
        if "panel_mask_x" in split_data:
            return split_data["panel_mask_x"]
        raise RuntimeError("need full-panel mask_x for lookback across split boundaries")

    def predict_panel(
        self,
        split_data: dict[str, Any],
        full_data: dict[str, Any] | None = None,
        fill_invalid: float = 0.0,
    ) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("GRU is not fitted")
        if self.cs_sel is None:
            if full_data is None:
                raise RuntimeError("call prepare_features first")
            self.prepare_features(full_data)
        self.net.eval()
        mask_panel = self._mask_panel(split_data, full_data)
        start = int(split_data.get("start", 0))
        n_days, n_stocks = split_data["mask_x"].shape
        out = np.full((n_days, n_stocks), float(fill_invalid), dtype=np.float32)
        with torch.no_grad():
            for local_t in range(n_days):
                mask = np.asarray(split_data["mask_x"][local_t], dtype=bool)
                idx = np.flatnonzero(mask)
                if idx.size == 0:
                    continue
                global_t = start + local_t
                x = gather_windows(
                    self.cs_sel,
                    mask_panel,
                    global_t,
                    idx,
                    self.length,
                    include_current=self.include_current,
                )
                pred = self._forward(x, global_t, idx)
                out[local_t, idx] = pred.detach().cpu().numpy().astype(np.float32)
        return out

    def save(self, path: str | Path) -> None:
        if self.net is None:
            raise RuntimeError("GRU is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        meta = {
            "cfg": self.cfg,
            "seed": self.seed,
            "cols": self.cols,
            "length": self.length,
            "include_current": self.include_current,
            "input_decay_halflife": self.input_decay_halflife,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "head_dropout": self.head_dropout,
            "target_mode": self.target_mode,
            "accum_days": self.accum_days,
            "lr_schedule": self.lr_schedule,
            "clip": self.clip,
            "cat_indices": self.cat_indices,
            "cat_embed_dim": self.cat_embed_dim,
            "cat_cardinalities": self.cat_cardinalities,
            "loss": self.loss_name,
            "source": self.source,
            "industry_col": self.industry_col,
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.cfg = meta.get("cfg", self.cfg)
            self.seed = int(meta.get("seed", self.seed))
            self.cols = [int(i) for i in meta.get("cols", self.cols)]
            self.length = int(meta.get("length", self.length))
            self.include_current = bool(meta.get("include_current", self.include_current))
            hl = meta.get("input_decay_halflife", self.cfg.get("input_decay_halflife"))
            self.input_decay_halflife = float(hl) if hl is not None else None
            self._input_decay_weights = None
            if self.input_decay_halflife is not None and self.input_decay_halflife > 0:
                ages = np.arange(self.length - 1, -1, -1, dtype=np.float32)
                self._input_decay_weights = np.power(
                    0.5, ages / max(self.input_decay_halflife, 1e-3)
                ).astype(np.float32)
            self.hidden_size = int(meta.get("hidden_size", self.hidden_size))
            self.num_layers = int(meta.get("num_layers", self.num_layers))
            self.dropout = float(meta.get("dropout", self.dropout))
            self.head_dropout = float(meta.get("head_dropout", self.cfg.get("head_dropout", 0.0)))
            self.target_mode = str(meta.get("target_mode", self.cfg.get("target_mode", "y1"))).lower()
            self.accum_days = max(1, int(meta.get("accum_days", self.cfg.get("accum_days", 1))))
            self.lr_schedule = str(meta.get("lr_schedule", self.cfg.get("lr_schedule") or "")).lower()
            self.clip = meta.get("clip", self.clip)
            self.cat_indices = [int(i) for i in meta.get("cat_indices", self.cfg.get("cat_indices") or [])]
            self.cat_embed_dim = int(meta.get("cat_embed_dim", self.cfg.get("cat_embed_dim", 8)))
            self.cat_cardinalities = [int(c) for c in (meta.get("cat_cardinalities") or [])]
            self.loss_name = str(meta.get("loss", self.cfg.get("loss", "mse"))).lower()
            self.source = str(meta.get("source", self.cfg.get("source", "cs_zscore")))
            self.industry_col = int(meta.get("industry_col", self.cfg.get("industry_col", 6)))
        self.net = self._build_net()
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state)
