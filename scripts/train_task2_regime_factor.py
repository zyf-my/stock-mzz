"""Regime-gated factor model for y2.

Day-constant cols 17-25 + coverage set the weights on same-day CS factors.
Not a tree split on market state, not a coverage gate on two models.

Usage:
    .\\.venv\\Scripts\\python.exe -u scripts\\train_task2_regime_factor.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_data_path  # noqa: E402
from src.dataset import drop_other_label, load_panel, precompute_cs_cols, slice_split, split_label_array  # noqa: E402
from src.metrics import mean_rank_ic  # noqa: E402
from src.models.gru_ts import pearson_ic_loss  # noqa: E402
from src.submit import save_submission  # noqa: E402

FACTOR_COLS = [8, 40, 7, 42, 11, 57, 41, 58, 90, 55]
REGIME_COLS = [17, 18, 19, 20, 21, 22, 23, 24, 25]
LABEL_KEY = "y2"
OUT = ROOT / "outputs" / "task2"


class RegimeFactorNet(nn.Module):
    def __init__(self, n_regime: int, n_factors: int, hidden: int = 16):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(n_factors))
        self.gate = nn.Sequential(
            nn.Linear(n_regime, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_factors),
        )

    def forward(self, regime: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        w = self.base + self.gate(regime)
        return (factors * w).sum(dim=-1)


def _day_regime(num_t: np.ndarray, mask_x_t: np.ndarray) -> np.ndarray:
    m = np.asarray(mask_x_t, dtype=bool)
    n_x = float(m.sum())
    n_s = float(mask_x_t.shape[0])
    if n_x < 1:
        raw = np.zeros(len(REGIME_COLS), dtype=np.float32)
    else:
        raw = np.asarray(num_t[m][:, REGIME_COLS], dtype=np.float32).mean(axis=0)
    cov = np.array([n_x / max(n_s, 1.0), np.log1p(n_x)], dtype=np.float32)
    return np.concatenate([raw, cov], axis=0)


def _standardize_regime(train_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_vec.mean(axis=0)
    sd = train_vec.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config("configs/task2/baseline.yaml")
    data_path = resolve_data_path(cfg, None)
    print(f"data={data_path}")
    data = load_panel(str(data_path))
    drop_other_label(data, LABEL_KEY)
    print(f"loaded in {time.perf_counter() - t0:.1f}s  num_x={tuple(data['num_x'].shape)}")

    print("precompute CS factors", FACTOR_COLS)
    factors = precompute_cs_cols(data["num_x"], data["mask_x"], FACTOR_COLS)
    np.clip(factors, -5.0, 5.0, out=factors)
    n_days = int(data["num_x"].shape[0])
    regime = np.stack(
        [_day_regime(data["num_x"][t], data["mask_x"][t]) for t in range(n_days)],
        axis=0,
    )
    train = slice_split(data, "train")
    valid = slice_split(data, "valid")
    test = slice_split(data, "test")
    tr0, tr1 = int(train["start"]), int(train["end"])
    mu, sd = _standardize_regime(regime[tr0:tr1])
    regime = (regime - mu) / sd
    del data["num_x"], data["cat_x"]
    gc.collect()

    device = torch.device("cpu")
    net = RegimeFactorNet(regime.shape[1], len(FACTOR_COLS), hidden=16).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(42)
    y_all = split_label_array(data, LABEL_KEY)
    best_ic, best_state, stale = -1e9, None, 0
    epochs, patience = 8, 3
    days = list(range(tr0, tr1))
    print(f"regime_factor params={sum(p.numel() for p in net.parameters())} days={len(days)}")

    def predict_range(start: int, end: int) -> np.ndarray:
        net.eval()
        n_s = factors.shape[1]
        out = np.zeros((end - start, n_s), dtype=np.float32)
        with torch.no_grad():
            for i, t in enumerate(range(start, end)):
                m = np.asarray(data["mask_x"][t], dtype=bool)
                idx = np.flatnonzero(m)
                if idx.size == 0:
                    continue
                r = torch.from_numpy(np.broadcast_to(regime[t], (idx.size, regime.shape[1])).copy())
                f = torch.from_numpy(np.ascontiguousarray(factors[t, idx]))
                out[i, idx] = net(r, f).numpy()
        return out

    for epoch in range(epochs):
        net.train()
        rng.shuffle(days)
        losses: list[float] = []
        for i, t in enumerate(days):
            my = np.asarray(data["mask_y"][t], dtype=bool)
            idx = np.flatnonzero(my)
            if idx.size < 8:
                continue
            if idx.size > 800:
                idx = np.sort(rng.choice(idx, size=800, replace=False))
            r = torch.from_numpy(np.broadcast_to(regime[t], (idx.size, regime.shape[1])).copy())
            f = torch.from_numpy(np.ascontiguousarray(factors[t, idx]))
            yb = torch.from_numpy(np.ascontiguousarray(y_all[t, idx]))
            pred = net(r, f)
            loss = pearson_ic_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            if (i + 1) % 400 == 0:
                print(f"  epoch {epoch + 1} day {i + 1}/{len(days)} loss={float(np.mean(losses[-400:])):.5f}", flush=True)
        pred_v = predict_range(int(valid["start"]), int(valid["end"]))
        ic = float(mean_rank_ic(pred_v, split_label_array(valid, LABEL_KEY), valid["mask_y"]))
        print(f"epoch {epoch + 1}/{epochs} loss={float(np.mean(losses)):.5f} valid_RankIC={ic:.6f}", flush=True)
        if ic > best_ic:
            best_ic, stale = ic, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print("early stop")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    pred_v = predict_range(int(valid["start"]), int(valid["end"]))
    pred_t = predict_range(int(test["start"]), int(test["end"]))
    ic = float(mean_rank_ic(pred_v, split_label_array(valid, LABEL_KEY), valid["mask_y"]))
    print(f"valid mean RankIC={ic:.6f}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "regime_factor_valid.npy", pred_v)
    np.save(OUT / "regime_factor_test.npy", pred_t)
    ckpt = ROOT / "checkpoints" / "task2" / "regime_factor.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "ic": ic}, ckpt)
    save_submission(pred_t, ROOT / "submissions" / "task2_regime_factor.npy")
    (OUT / "regime_factor_summary.json").write_text(json.dumps({"valid_ic": ic}, indent=2), encoding="utf-8")
    print(f"wrote {ckpt}")
    print("VALID_RANKIC", f"{ic:.6f}")
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
