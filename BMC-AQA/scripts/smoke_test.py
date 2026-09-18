#!/usr/bin/env python3
"""Run the full core objective on synthetic features, without external data."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmc_aqa import BMCAQA, MODALITIES, ModalityMemoryBank
from bmc_aqa.losses import LossConfig, core_losses


def create_tiny_fs1000(root: Path):
    """Create train/test files using the documented FS1000 directory layout."""
    root = root / "FS1000"
    paths = {"rgb": root / "output_feature_fs1000_new", "flow": root / "i3d_avg_clip8_5s_fs1000",
             "audio": root / "ast_feature_fs1000_new"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    lines = []
    for index in range(10):
        sample_id = f"sample_{index:03d}"
        for name, dim in zip(MODALITIES, (4, 6, 8)):
            np.save(paths[name] / f"{sample_id}.npy", rng.normal(size=(6, dim)).astype("float32"))
        lines.append(f"{sample_id} {40 + index} 30 5 5 5 5 5 1\n")
    (root / "train_fs1000_new.txt").write_text("".join(lines[:8]), encoding="utf-8")
    (root / "val_fs1000_new.txt").write_text("".join(lines[8:]), encoding="utf-8")


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    dims = dict(zip(MODALITIES, (4, 6, 8)))
    features = {name: torch.randn(3, 4, dim) for name, dim in dims.items()}
    observed = torch.eye(3)
    model = BMCAQA(dims, hidden_dim=16, heads=2, encoder_layers=1,
                  generator_layers=1, decoder_layers=1, grades=4, key_dim=8, dropout=0)
    memory = ModalityMemoryBank(["a", "b", "c"], features, torch.tensor([0.2, 0.5, 0.8]))
    evidence = memory.retrieve(features, observed, model.encode_retrieval_key,
                               exclude_ids=memory.ids, top_k=2)
    outputs = model(features, observed, evidence)
    targets = model.encode_targets(features)
    losses = core_losses(outputs, memory.labels, observed, targets, evidence, memory.labels, LossConfig())
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert torch.allclose(outputs["source_weights"].sum(-1), torch.ones(3, 3))
    assert torch.allclose(outputs["modality_weights"].sum(-1), torch.ones(3))
    print("smoke_test_ok: retrieval -> PARAG -> FCM -> HMoE -> grade scoring -> backward")
    print(" | ".join(f"{name}={float(value.detach()):.4f}" for name, value in losses.items()))


if __name__ == "__main__":
    main()

