"""FCM: joint context pooling and per-modality/source channel calibration."""
from __future__ import annotations

import torch
from torch import nn


class FeatureCalibration(nn.Module):
    def __init__(self, modalities: int, sources: int, dim: int):
        super().__init__()
        self.modalities, self.sources, self.dim = modalities, sources, dim
        self.phi = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(modalities * sources)
        ])

    def forward(self, tokens: torch.Tensor, prompts: torch.Tensor, validity: torch.Tensor):
        # tokens [B,M,S,T,d]; prompts [B,M,S,d]; validity [B,M,S].
        eligible = validity[..., None]
        context = ((tokens.mean(3) + prompts) * eligible).sum((1, 2))
        context = context / validity.sum((1, 2)).clamp_min(1)[:, None]
        gamma = torch.stack([torch.sigmoid(phi(context)) for phi in self.phi], dim=1)
        gamma = gamma.view(-1, self.modalities, self.sources, self.dim)
        calibrated = 2 * validity[..., None, None] * gamma[..., None, :] * tokens
        return calibrated, context, gamma

