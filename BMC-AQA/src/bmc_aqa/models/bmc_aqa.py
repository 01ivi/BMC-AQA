"""Paper-facing BMC-AQA model; ground-truth scores are never forward inputs."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ..constants import MODALITIES, SOURCES
from .encoders import RetrievalKeyEncoder, TemporalEncoder
from .parag import GatedCompletion, RetrievalConfidence, SourcePrompts
from .calibration import FeatureCalibration
from .hmoe import HierarchicalMoE
from .score_head import GradeScoreHead

if TYPE_CHECKING:
    from ..retrieval import RetrievalBatch


class BMCAQA(nn.Module):
    def __init__(self, input_dims: dict[str, int], hidden_dim: int = 256,
                 heads: int = 4, encoder_layers: int = 2, generator_layers: int = 2,
                 decoder_layers: int = 2, grades: int = 8, key_dim: int = 128,
                 dropout: float = 0.1, use_fcm: bool = True, use_hmoe: bool = True):
        super().__init__()
        if set(input_dims) != set(MODALITIES):
            raise ValueError("input_dims must define rgb, flow, and audio")
        if hidden_dim < 2 or heads < 1 or hidden_dim % heads or grades < 2 or key_dim < 1:
            raise ValueError("Require hidden_dim >= 2 divisible by heads, grades >= 2, and key_dim >= 1")
        if min(encoder_layers, generator_layers, decoder_layers) < 1:
            raise ValueError("All attention stacks must have at least one layer")
        self.hidden_dim = hidden_dim
        self.use_fcm, self.use_hmoe = use_fcm, use_hmoe
        self.keys = RetrievalKeyEncoder(input_dims, key_dim)
        self.encoders = nn.ModuleDict({
            name: TemporalEncoder(input_dims[name], hidden_dim, heads, encoder_layers, dropout)
            for name in MODALITIES
        })
        self.generators = nn.ModuleDict({
            name: GatedCompletion(hidden_dim, heads, generator_layers, dropout)
            for name in MODALITIES
        })
        self.retrieval_confidence = RetrievalConfidence()
        self.prompts = SourcePrompts(len(MODALITIES), len(SOURCES), hidden_dim)
        self.fcm = FeatureCalibration(len(MODALITIES), len(SOURCES), hidden_dim)
        self.hmoe = HierarchicalMoE(len(MODALITIES), len(SOURCES), hidden_dim, dropout)
        self.score = GradeScoreHead(hidden_dim, heads, decoder_layers, grades, dropout)

    def encode_retrieval_key(self, modality: str, tokens: torch.Tensor):
        return self.keys(modality, tokens)

    def _encode_valid(self, features: dict[str, torch.Tensor], validity: torch.Tensor):
        streams = []
        for index, name in enumerate(MODALITIES):
            reference = features[name]
            stream = reference.new_zeros(*reference.shape[:2], self.hidden_dim)
            rows = validity[:, index].bool()
            if rows.any():
                values = self.encoders[name](reference[rows])
                stream = stream.index_copy(0, rows.nonzero().squeeze(1), values)
            streams.append(stream)
        return torch.stack(streams, dim=1)

    def encode_targets(self, complete_features: dict[str, torch.Tensor]):
        # Deterministic detached targets; no update of encoder parameters here.
        states = {module: module.training for module in self.encoders.modules()}
        try:
            self.encoders.eval()
            with torch.no_grad():
                return torch.stack([
                    self.encoders[name](complete_features[name]) for name in MODALITIES
                ], dim=1).detach()
        finally:
            for module, training in states.items():
                module.training = training

    def forward(self, features: dict[str, torch.Tensor], observed: torch.Tensor,
                retrieval: RetrievalBatch | None = None):
        if observed.ndim != 2 or observed.shape[1] != len(MODALITIES):
            raise ValueError("observed must have shape [B,3] in RGB/flow/audio order")
        if not ((observed == 0) | (observed == 1)).all() or (observed.sum(1) == 0).any():
            raise ValueError("A binary mask retaining at least one modality is required")
        reference_shape = features[MODALITIES[0]].shape[:2]
        if reference_shape[0] != observed.shape[0] or reference_shape[1] == 0:
            raise ValueError("Feature batch must match the mask and contain at least one temporal token")
        if any(features[name].ndim != 3 or features[name].shape[:2] != reference_shape for name in MODALITIES):
            raise ValueError("All feature streams must have matching [B,T,d_m] shapes")
        observed = observed.float()
        encoded = self._encode_valid(features, observed)
        batch = observed.shape[0]
        if retrieval is None:
            available = observed.new_zeros(batch, dtype=torch.bool)
            retrieved = torch.zeros_like(encoded)
            ret_confidence = observed.new_zeros(batch)
        else:
            available = retrieval.valid_mask
            retrieved = self._encode_valid(retrieval.tokens, available[:, None].expand(-1, 3))
            ret_confidence = self.retrieval_confidence(retrieval.confidence_features, available)
        context = encoded.flatten(1, 2)
        completions, log_variances, gates, refined = [], [], [], []
        for index, name in enumerate(MODALITIES):
            generated, nu, alpha, query = self.generators[name](
                context, retrieved[:, index], ret_confidence, observed.mean(-1)
            )
            completions.append(generated)
            log_variances.append(nu)
            gates.append(alpha)
            refined.append(query)
        generated = torch.stack(completions, dim=1)
        nu = torch.stack(log_variances, dim=1)
        missing = 1 - observed
        validity = torch.stack((observed, missing, missing * available[:, None]), dim=2).bool()
        confidence = torch.stack((observed, torch.sigmoid(-nu), ret_confidence[:, None].expand(-1, 3)), dim=2)
        confidence = confidence * validity
        prompts = self.prompts(validity, confidence)
        sources = torch.stack((encoded, generated, retrieved), dim=2)
        if self.use_fcm:
            calibrated, fcm_context, gamma = self.fcm(sources, prompts, validity)
        else:
            calibrated = sources * validity[..., None, None]
            fcm_context, gamma = None, None
        if self.use_hmoe:
            fused, source_weights, modality_weights = self.hmoe(
                calibrated, prompts, validity, confidence, observed
            )
        else:
            # Explicit ablation: average valid sources, then average modalities.
            source_weights = validity.float() / validity.sum(-1, keepdim=True).clamp_min(1)
            modality_weights = observed.new_full((batch, 3), 1 / 3)
            per_modality = (calibrated * source_weights[..., None, None]).sum(2)
            fused = (per_modality * modality_weights[..., None, None]).sum(1)
        prediction, probability, logits = self.score(fused, prompts)
        return {
            "prediction": prediction, "grade_probability": probability, "grade_logits": logits,
            "generated": generated, "log_variance": nu, "retrieved": retrieved,
            "generation_gate": torch.stack(gates, dim=1), "refined_query": torch.stack(refined, dim=1),
            "source_prompts": prompts, "source_validity": validity, "source_confidence": confidence,
            "calibrated_sources": calibrated, "fcm_context": fcm_context, "channel_gates": gamma,
            "source_weights": source_weights, "modality_weights": modality_weights, "fused": fused,
        }

