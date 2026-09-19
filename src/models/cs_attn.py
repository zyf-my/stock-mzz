"""Same-day industry self-attention. Stocks look at peers in cat_6, not just industry z-score."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.cs_mlp import CSMLPModel


class CSAttnNet(nn.Module):
    def __init__(
        self,
        d_feat: int,
        hidden: int,
        dropout: float,
        n_heads: int = 2,
        cat_cardinalities: list[int] | None = None,
        cat_embed_dim: int = 8,
    ):
        super().__init__()
        cards = [int(c) for c in (cat_cardinalities or []) if int(c) > 0]
        self.embeds = nn.ModuleList(nn.Embedding(c, cat_embed_dim) for c in cards) if cards else None
        in_dim = d_feat + (len(cards) * cat_embed_dim if cards else 0)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, 1)

    def _encode(self, x: torch.Tensor, cats: torch.Tensor | None) -> torch.Tensor:
        h = x
        if self.embeds is not None:
            if cats is None:
                raise ValueError("CSAttnNet was built with embeddings but cats is None")
            pieces = [h]
            for i, emb in enumerate(self.embeds):
                idx = cats[:, i].clamp(0, emb.num_embeddings - 1)
                pieces.append(emb(idx))
            h = torch.cat(pieces, dim=-1)
        return self.proj(h)

    def forward(
        self,
        x: torch.Tensor,
        cats: torch.Tensor | None = None,
        groups: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self._encode(x, cats)
        if groups is None or groups.numel() == 0:
            return self.fc(h).squeeze(-1)
        _, inv = torch.unique(groups, return_inverse=True)
        n_stocks, hidden = h.shape
        n_groups = int(inv.max().item()) + 1
        counts = torch.bincount(inv, minlength=n_groups)
        max_n = int(counts.max().item())
        padded = h.new_zeros(n_groups, max_n, hidden)
        pad_mask = torch.ones(n_groups, max_n, dtype=torch.bool, device=h.device)
        mixed = h.clone()
        for g in range(n_groups):
            sel = inv == g
            n = int(counts[g].item())
            if n == 0:
                continue
            padded[g, :n] = h[sel]
            pad_mask[g, :n] = False
        attn_out, _ = self.attn(
            padded, padded, padded, key_padding_mask=pad_mask, need_weights=False
        )
        for g in range(n_groups):
            sel = inv == g
            n = int(counts[g].item())
            if n < 2:
                continue
            mixed[sel] = self.norm(h[sel] + attn_out[g, :n])
        return self.fc(mixed).squeeze(-1)


class CSAttnModel(CSMLPModel):
    """Industry grouped attention on the same features as cs_mlp. One architecture change."""

    def __init__(self, cfg: dict, seed: int = 42):
        super().__init__(cfg, seed=seed)
        self.n_heads = int(cfg.get("n_heads", 2))

    def _build_net(self) -> CSAttnNet:
        return CSAttnNet(
            len(self.cols),
            self.hidden_size,
            self.dropout,
            n_heads=self.n_heads,
            cat_cardinalities=self.cat_cardinalities,
            cat_embed_dim=self.cat_embed_dim,
        ).to(self.device)

    def _groups_tensor(self, t: int, idx: np.ndarray) -> torch.Tensor:
        if self.industry is None:
            raise RuntimeError("industry panel missing")
        g = np.ascontiguousarray(self.industry[int(t), idx])
        return torch.from_numpy(g.astype(np.int64, copy=False)).to(self.device)

    def _predict_batch(self, t: int, idx: np.ndarray) -> torch.Tensor:
        if self.net is None:
            raise RuntimeError("CS attention is not fitted")
        return self.net(self._x_tensor(t, idx), self._cats_tensor(t, idx), self._groups_tensor(t, idx))
