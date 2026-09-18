"""Grade queries decode prompts and fused tokens into a score expectation."""
from __future__ import annotations

import torch
from torch import nn


class GradeScoreHead(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, grades: int, dropout: float):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(grades, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.logit = nn.Linear(dim, 1)
        self.register_buffer("bins", torch.linspace(0, 1, grades))

    def forward(self, fused: torch.Tensor, prompts: torch.Tensor):
        memory = torch.cat((prompts.flatten(1, 2), fused), dim=1)
        decoded = self.decoder(self.queries[None].expand(fused.shape[0], -1, -1), memory)
        logits = self.logit(decoded).squeeze(-1)
        probability = torch.softmax(logits, dim=-1)
        return (probability * self.bins).sum(-1), probability, logits

