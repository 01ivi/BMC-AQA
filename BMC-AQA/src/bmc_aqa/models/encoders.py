"""Modality-specific temporal encoders and mean/std retrieval keys."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.project(tokens)
        length, dim = projected.shape[1:]
        # Construct positions dynamically so the public API has no fixed T limit.
        position = torch.arange(length, device=tokens.device, dtype=torch.float32)[:, None]
        frequency = torch.exp(
            torch.arange(0, dim, 2, device=tokens.device) * (-math.log(10000.0) / dim)
        )
        encoding = projected.new_zeros(length, dim)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[:dim // 2])
        return self.temporal(self.dropout(projected + encoding))


class RetrievalKeyEncoder(nn.Module):
    def __init__(self, input_dims: dict[str, int], key_dim: int):
        super().__init__()
        self.nets = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(2 * size, max(key_dim, min(size, 512))), nn.GELU(),
                nn.Linear(max(key_dim, min(size, 512)), key_dim),
            )
            for name, size in input_dims.items()
        })

    def forward(self, modality: str, tokens: torch.Tensor) -> torch.Tensor:
        statistics = torch.cat((tokens.mean(1), tokens.std(1, unbiased=False)), dim=-1)
        return F.normalize(self.nets[modality](statistics), dim=-1)

