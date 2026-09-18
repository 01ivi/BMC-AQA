"""Two-stage MoE: shared provenance experts, then independent modality experts."""
from __future__ import annotations

import torch
from torch import nn


def masked_softmax(logits: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    valid = validity.bool()
    has_candidate = valid.any(dim=-1, keepdim=True)
    masked = logits.masked_fill(~valid, float("-inf"))
    safe = torch.where(has_candidate, masked, torch.zeros_like(masked))
    return torch.softmax(safe, dim=-1) * valid.to(logits.dtype)


class ResidualExpert(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * dim, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.net(tokens)


class HierarchicalMoE(nn.Module):
    def __init__(self, modalities: int, sources: int, dim: int, dropout: float):
        super().__init__()
        # Exactly three source experts, shared across all modalities.
        self.source_experts = nn.ModuleList([ResidualExpert(dim, dropout) for _ in range(sources)])
        self.source_routers = nn.ModuleList([
            nn.Sequential(nn.Linear(sources * (2 * dim + 1) + 1, dim), nn.GELU(), nn.Linear(dim, sources))
            for _ in range(modalities)
        ])
        self.modality_experts = nn.ModuleList([ResidualExpert(dim, dropout) for _ in range(modalities)])
        self.modality_router = nn.Sequential(
            nn.Linear(modalities * (2 * dim + 2), dim), nn.GELU(), nn.Linear(dim, modalities)
        )

    def forward(self, tokens: torch.Tensor, prompts: torch.Tensor, validity: torch.Tensor,
                confidence: torch.Tensor, observed: torch.Tensor):
        expert_tokens = torch.stack([
            expert(tokens[:, :, index]) for index, expert in enumerate(self.source_experts)
        ], dim=2)
        expert_tokens = expert_tokens * validity[..., None, None]
        router_context = torch.cat((
            tokens.mean(3).flatten(2), prompts.flatten(2), confidence, observed[..., None],
        ), dim=-1)
        logits = torch.stack([
            router(router_context[:, index]) for index, router in enumerate(self.source_routers)
        ], dim=1)
        source_weights = masked_softmax(logits, validity)
        selected = (source_weights[..., None, None] * expert_tokens).sum(2)
        modality_tokens = torch.stack([
            expert(selected[:, index]) for index, expert in enumerate(self.modality_experts)
        ], dim=1)
        modality_valid = validity.any(dim=-1)
        modality_tokens = modality_tokens * modality_valid[..., None, None]
        reliability = (source_weights * confidence).sum(-1)
        selected_prompts = (source_weights[..., None] * prompts).sum(2)
        modality_context = torch.cat((
            modality_tokens.mean(2), selected_prompts, reliability[..., None], observed[..., None],
        ), dim=-1).flatten(1)
        modality_weights = masked_softmax(self.modality_router(modality_context), modality_valid)
        fused = (modality_weights[..., None, None] * modality_tokens).sum(1)
        return fused, source_weights, modality_weights

