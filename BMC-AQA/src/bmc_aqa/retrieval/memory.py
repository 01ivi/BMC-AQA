"""Exact training-memory search using observed modality keys only."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

from ..constants import MODALITIES
from ..models.hmoe import masked_softmax


@dataclass
class RetrievalBatch:
    tokens: dict[str, torch.Tensor]
    indices: torch.Tensor
    scores: torch.Tensor
    weights: torch.Tensor
    full_scores: torch.Tensor
    candidate_valid: torch.Tensor
    confidence_features: torch.Tensor
    valid_mask: torch.Tensor


class ModalityMemoryBank:
    def __init__(self, ids: Sequence[str], tokens: dict[str, torch.Tensor],
                 labels: torch.Tensor | None = None):
        self.modalities = MODALITIES
        self.ids = list(map(str, ids))
        self.tokens = {name: tokens[name].detach().cpu().float() for name in self.modalities}
        self.labels = labels.detach().cpu().float() if labels is not None else None
        for name, value in self.tokens.items():
            if value.ndim != 3 or value.shape[0] != len(self.ids):
                raise ValueError(f"Memory {name} must have shape [N,T,d_m] with N=len(ids)")
            if not torch.isfinite(value).all():
                raise ValueError(f"Memory {name} must contain complete, finite training features")
        if self.labels is not None and self.labels.shape != (len(self.ids),):
            raise ValueError("Memory labels must have shape [N]")

    @classmethod
    def from_dataset(cls, dataset):
        if len(dataset) == 0:
            raise ValueError("Cannot build a memory bank from an empty training subset")
        samples = [dataset[index] for index in range(len(dataset))]
        return cls(
            [sample["id"] for sample in samples],
            {name: torch.stack([sample["features"][name] for sample in samples]) for name in MODALITIES},
            torch.stack([sample["label"] for sample in samples]),
        )

    def __len__(self):
        return len(self.ids)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": self.ids, "tokens": self.tokens, "labels": self.labels}, path)

    @classmethod
    def load(cls, path: str | Path):
        return cls(**torch.load(path, map_location="cpu", weights_only=True))

    def retrieve(self, features: dict[str, torch.Tensor], observed: torch.Tensor,
                 key_encoder: Callable, top_k: int = 5, temperature: float = 0.1,
                 exclude_ids: Sequence[str] | None = None, key_batch_size: int = 256):
        if top_k < 1 or temperature <= 0 or key_batch_size < 1:
            raise ValueError("top_k, temperature, and key_batch_size must be positive")
        batch, device = observed.shape[0], observed.device
        if observed.shape != (batch, len(MODALITIES)):
            raise ValueError("Observed mask must have shape [B,3] in RGB/flow/audio order")
        if exclude_ids is not None and len(exclude_ids) != batch:
            raise ValueError("exclude_ids must contain one identifier per query")
        scores = torch.zeros(batch, len(self), device=device)
        for index, name in enumerate(self.modalities):
            rows = observed[:, index].bool()
            if not rows.any() or len(self) == 0:
                continue
            # No query encoding of hidden feature rows, including NaN placeholders.
            query = key_encoder(name, features[name][rows])
            keys = torch.cat([
                key_encoder(name, chunk.to(device))
                for chunk in self.tokens[name].split(key_batch_size)
            ])
            similarity = query @ keys.T
            scores = scores.index_add(0, rows.nonzero().squeeze(1), similarity)
        scores = scores / observed.sum(1).clamp_min(1)[:, None]
        eligible = (observed.sum(1) > 0)[:, None].expand_as(scores).clone()
        if exclude_ids is not None:
            for row, sample_id in enumerate(exclude_ids):
                eligible[row] &= torch.tensor(
                    [value != str(sample_id) for value in self.ids], device=device, dtype=torch.bool
                )
        ranked = scores.masked_fill(~eligible, -1e4)
        selected_scores, indices = ranked.topk(min(top_k, len(self)), dim=-1)
        selected_valid = eligible.gather(1, indices)
        weights = masked_softmax(selected_scores / temperature, selected_valid)
        count = selected_valid.sum(-1)
        available = count > 0
        peak = torch.zeros(batch, device=device)
        gap = torch.zeros_like(peak)
        entropy = torch.zeros_like(peak)
        if indices.shape[1] > 0:
            peak = torch.where(available, selected_scores[:, 0], peak)
            if indices.shape[1] > 1:
                gap = torch.where(count >= 2, selected_scores[:, 0] - selected_scores[:, 1], gap)
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
            entropy = torch.where(count >= 2, entropy / count.clamp_min(2).float().log(), 0)
        retrieved = {
            name: (value[indices.detach().cpu()].to(device) * weights[..., None, None]).sum(1)
            for name, value in self.tokens.items()
        }
        return RetrievalBatch(
            retrieved, indices, selected_scores, weights, scores, eligible,
            torch.stack((peak, gap, entropy), -1), available,
        )

