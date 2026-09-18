"""The three core losses defined in the manuscript."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .retrieval import RetrievalBatch


@dataclass(frozen=True)
class LossConfig:
    lambda_comp: float = 0.5
    lambda_ret: float = 0.02
    eta: float = 0.05
    sigma_y: float = 0.08
    tau_loss: float = 0.1

    def __post_init__(self):
        if min(self.lambda_comp, self.lambda_ret) < 0 or min(self.eta, self.sigma_y, self.tau_loss) <= 0:
            raise ValueError("Loss weights must be nonnegative; eta, sigma_y, and tau_loss must be positive")


def completion_loss(generated: torch.Tensor, targets: torch.Tensor,
                    log_variance: torch.Tensor, observed: torch.Tensor, eta: float):
    error = (generated - targets.detach()).square().mean(dim=(-1, -2))
    nu = log_variance.clamp(-8, 8)
    per_sample = (1 + eta / 2 * torch.exp(-nu)) * error + eta / 2 * nu
    missing = (observed == 0).to(error.dtype)
    counts = missing.sum(0)
    active = counts > 0
    if not active.any():
        return generated.sum() * 0
    # Mean within each masked modality, then mean over active modalities.
    return ((per_sample * missing).sum(0) / counts.clamp_min(1))[active].mean()


def retrieval_loss(retrieval: RetrievalBatch, labels: torch.Tensor,
                   memory_labels: torch.Tensor, sigma_y: float, temperature: float):
    valid = retrieval.candidate_valid
    rows = valid.any(-1)
    if not rows.any():
        return retrieval.full_scores.sum() * 0
    scores, valid = retrieval.full_scores[rows], valid[rows]
    target_logits = -torch.abs(labels[rows, None] - memory_labels.to(labels.device)[None]) / sigma_y
    target = torch.softmax(target_logits.masked_fill(~valid, float("-inf")), dim=-1).detach()
    log_prob = F.log_softmax((scores / temperature).masked_fill(~valid, float("-inf")), dim=-1)
    # Avoid 0 * -inf while normalizing both distributions over the same pool.
    return -(target * log_prob.masked_fill(~valid, 0)).sum(-1).mean()


def core_losses(outputs: dict, labels: torch.Tensor, observed: torch.Tensor,
                targets: torch.Tensor, retrieval: RetrievalBatch,
                memory_labels: torch.Tensor, config: LossConfig):
    task = F.mse_loss(outputs["prediction"], labels)
    comp = completion_loss(outputs["generated"], targets, outputs["log_variance"], observed, config.eta)
    ret = retrieval_loss(retrieval, labels, memory_labels, config.sigma_y, config.tau_loss)
    return {
        "total": task + config.lambda_comp * comp + config.lambda_ret * ret,
        "task": task, "completion": comp, "retrieval": ret,
    }

