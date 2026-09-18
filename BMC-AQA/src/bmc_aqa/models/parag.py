"""PARAG: reliability estimation, gated completion, and source annotations."""
from __future__ import annotations

import torch
from torch import nn


class RetrievalConfidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, statistics: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        return self.net(statistics).squeeze(-1) * available


class GatedCompletion(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.seed = nn.Parameter(torch.zeros(1, 1, dim))
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.gate = nn.Sequential(nn.Linear(3 * dim + 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.log_variance = nn.Sequential(
            nn.Linear(3 * dim + 2, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def forward(self, observed: torch.Tensor, retrieved: torch.Tensor,
                retrieval_confidence: torch.Tensor, visible_fraction: torch.Tensor):
        query = retrieved + self.seed
        # The memory is observed context only, as specified in the manuscript.
        for attention, norm in zip(self.attention, self.norms):
            residual, _ = attention(norm(query), observed, observed, need_weights=False)
            query = query + residual
        context = torch.cat((
            query.mean(1), observed.mean(1), retrieved.mean(1),
            retrieval_confidence[:, None], visible_fraction[:, None],
        ), dim=-1)
        alpha = torch.sigmoid(self.gate(context))[:, None]
        nu = self.log_variance(context).squeeze(-1).clamp(-8.0, 8.0)
        generated = alpha * query + (1 - alpha) * retrieved
        return generated, nu, alpha, query


class SourcePrompts(nn.Module):
    def __init__(self, modalities: int, sources: int, dim: int):
        super().__init__()
        self.modality = nn.Parameter(torch.randn(modalities, dim) * 0.02)
        self.source = nn.Parameter(torch.randn(sources, dim) * 0.02)
        self.metadata = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, validity: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        metadata = torch.stack((validity.to(confidence.dtype), confidence), dim=-1)
        # Keep invalid prompt identities for routing and score decoding.
        return self.modality[None, :, None] + self.source[None, None] + self.metadata(metadata)

