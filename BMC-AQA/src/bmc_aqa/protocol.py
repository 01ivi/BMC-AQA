"""Missingness sampling, training holdout, and AQA metrics."""
from __future__ import annotations

from itertools import permutations
import random

import numpy as np
import torch

from .constants import MODALITIES

IMR_SETTINGS = tuple(permutations((0.3, 0.5, 0.7)))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_masks(count: int, rates, generator: torch.Generator | None = None):
    """Independent Bernoulli masks conditioned on at least one observation.

    Sampling stays on CPU for consistent evaluation across device types.
    Conditioning changes the empirical marginal missing rates.
    """
    rates = torch.as_tensor(rates, dtype=torch.float32)
    if rates.shape != (3,) or not torch.isfinite(rates).all() or ((rates < 0) | (rates >= 1)).any():
        raise ValueError("Supply three finite missing rates in [0,1), ordered RGB/flow/audio")
    mask = (torch.rand(count, 3, generator=generator) >= rates).float()
    empty = mask.sum(1) == 0
    while empty.any():
        mask[empty] = (torch.rand(int(empty.sum()), 3, generator=generator) >= rates).float()
        empty = mask.sum(1) == 0
    return mask


def training_holdout(dataset, fraction: float, seed: int):
    if not 0 < fraction < 1 or len(dataset) < 3:
        raise ValueError("Training holdout requires at least three samples and a fraction in (0,1)")
    size = min(len(dataset) - 1, max(1, round(len(dataset) * fraction)))
    ordered = sorted(range(len(dataset)), key=lambda index: dataset.labels[index].score)
    # Select one sample from each score stratum to cover the label range.
    bins = np.array_split(ordered, size)
    rng = np.random.default_rng(seed)
    validation = [int(rng.choice(part)) for part in bins]
    selected = set(validation)
    return [index for index in range(len(dataset)) if index not in selected], validation


def fixed_masks():
    values = [[(number >> index) & 1 for index in range(3)] for number in range(1, 8)]
    return sorted(values, key=lambda row: (-sum(row), tuple(-value for value in row)))


def mask_name(mask):
    return "+".join(name for name, keep in zip(MODALITIES, mask) if keep)


def imr_name(rates):
    return "imr_" + "_".join(f"{name}{float(rate):g}" for name, rate in zip(MODALITIES, rates))


def _ranks(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return np.bincount(inverse, ranks)[inverse] / counts[inverse]


def metrics(predictions, labels):
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    error = float(np.mean((predictions - labels) ** 2))
    if len(labels) < 2 or np.ptp(predictions) == 0 or np.ptp(labels) == 0:
        correlation = None
    else:
        correlation = float(np.corrcoef(_ranks(predictions), _ranks(labels))[0, 1])
    return {"spearman": correlation, "mse": error}


def aggregate_metrics(rows):
    correlations = [row["spearman"] for row in rows if row["spearman"] is not None]
    # Undefined correlations do not remove a task from the arithmetic MSE mean.
    mean_corr = float(np.tanh(np.arctanh(np.clip(correlations, -0.999999, 0.999999)).mean())) if correlations else None
    return {"spearman": mean_corr, "mse": float(np.mean([row["mse"] for row in rows]))}

