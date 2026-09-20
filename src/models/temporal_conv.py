"""Causal multiscale convolution; reuses only the GRU data/training pipeline."""
import torch
from torch import nn
from torch.nn import functional as F

from src.models.gru_ts import GRUModel


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.left = 2 * dilation
        self.depthwise = nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
        self.mix = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        z = self.depthwise(F.pad(x, (self.left, 0)))
        z = self.mix(F.gelu(z))
        # LayerNorm across channels at each time; never across future time steps.
        z = self.norm(z.transpose(1, 2)).transpose(1, 2)
        return x + .3 * F.gelu(z)


class TemporalConvNet(nn.Module):
    def __init__(self, d_feat, width=24, dropout=.1):
        super().__init__()
        self.project = nn.Conv1d(d_feat, width, 1)
        self.blocks = nn.Sequential(*(CausalBlock(width, d) for d in (1, 3, 9)))
        self.head = nn.Sequential(nn.Linear(3*width+2*d_feat, 48), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(48, 1))

    def forward(self, x, cats=None):
        h = self.blocks(F.gelu(self.project(x.transpose(1, 2))))
        summary = torch.cat((h[:, :, -1], h[:, :, -4:].mean(-1), h.mean(-1),
                             x[:, -1], x[:, -1]-x[:, -8:].mean(1)), dim=1)
        return self.head(summary).squeeze(-1)


class TemporalConvModel(GRUModel):
    def _build_net(self):
        if self.cat_indices:
            raise ValueError('TemporalConvModel expects numeric features only')
        return TemporalConvNet(len(self.cols), self.hidden_size, self.head_dropout).to(self.device)
