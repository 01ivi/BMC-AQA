"""Feature-level training and evaluation with a fixed training-only memory."""
from __future__ import annotations

import csv
from pathlib import Path
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import save_json
from .constants import MODALITIES
from .data import AQAFeatureDataset, available_dataset_presets
from .losses import LossConfig, core_losses
from .models import BMCAQA
from .protocol import imr_name, metrics, sample_masks, set_seed, training_holdout
from .retrieval import ModalityMemoryBank

CHECKPOINT_FORMAT = "bmc-aqa-core-v1"


def dataset_from_config(config, root, train):
    data = config["data"]
    paths = dict(available_dataset_presets(root)[data["dataset"]])
    for key in paths:
        if data.get(key):
            paths[key] = str(Path(root) / data[key])
    return AQAFeatureDataset(
        data["dataset"], paths["rgb_path"], paths["audio_path"], paths["flow_path"],
        paths["train_label_path" if train else "test_label_path"],
        data["clip_num"], data["action_type"], train=train, deterministic=not train,
    )


def get_device(request):
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(request)


def move_features(batch, device):
    return {name: value.to(device) for name, value in batch["features"].items()}


def retrieve(model, memory, features, mask, config, exclude_ids=None):
    return memory.retrieve(features, mask, model.encode_retrieval_key,
                           exclude_ids=exclude_ids, **config["retrieval"])


@torch.no_grad()
def evaluate_masks(model, dataset, memory, config, device, masks, name):
    model.eval()
    original = dataset
    while isinstance(original, Subset):
        original = original.dataset
    score_range = original.score_range
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                        num_workers=config["training"]["num_workers"])
    predictions, labels, records = [], [], []
    offset = 0
    for batch in loader:
        features = move_features(batch, device)
        count = len(batch["id"])
        observed = masks[offset:offset + count].to(device)
        evidence = retrieve(model, memory, features, observed, config)
        output = model(features, observed, evidence)["prediction"]
        pred = (output * score_range).cpu().tolist()
        raw = batch["raw_label"].tolist()
        predictions.extend(pred)
        labels.extend(raw)
        records.extend({"setting": name, "id": sample_id, "prediction": score, "label": label,
                        "rgb": int(mask[0]), "flow": int(mask[1]), "audio": int(mask[2])}
                       for sample_id, score, label, mask in zip(batch["id"], pred, raw, observed.cpu().tolist()))
        offset += count
    if offset == 0:
        raise ValueError("Evaluation split is empty")
    return {"setting": name, **metrics(predictions, labels)}, records


def write_metrics(rows, output_dir):
    output = Path(output_dir)
    save_json(rows, output / "metrics.json")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("setting", "spearman", "mse"))
        writer.writeheader()
        writer.writerows(rows)
    table = ["| Setting | Spearman ↑ | MSE ↓ |", "|---|---:|---:|"]
    for row in rows:
        sp = "n/a" if row["spearman"] is None else f"{row['spearman']:.4f}"
        table.append(f"| {row['setting']} | {sp} | {row['mse']:.4f} |")
    (output / "metrics.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def train(config, data_root, output_dir, device):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("last.pt", "best_mse.pt", "memory.pt")):
        raise ValueError("Output directory already contains a run; choose a new --output-dir")
    settings = config["training"]
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["lr"] <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive")
    set_seed(settings["seed"])
    source = dataset_from_config(config, data_root, train=True)
    ids = [entry.sample_id for entry in source.labels]
    if len(ids) != len(set(ids)):
        raise ValueError("Training sample identifiers must be unique")
    train_indices, val_indices = training_holdout(source, settings["val_fraction"], settings["seed"])
    deterministic = source.clone(train=False, deterministic=True)
    fitting, validation = Subset(source, train_indices), Subset(deterministic, val_indices)
    memory = ModalityMemoryBank.from_dataset(Subset(deterministic, train_indices))
    memory.save(output / "memory.pt")
    input_dims = source.infer_input_dims()
    model = BMCAQA(input_dims, **config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    loss_config = LossConfig(**config["loss"])
    loader = DataLoader(fitting, batch_size=settings["batch_size"], shuffle=True,
                        num_workers=settings["num_workers"])
    # Generate validation masks once; evaluation is independent of batch size.
    val_masks = sample_masks(len(validation), settings["imr_rates"],
                             torch.Generator().manual_seed(settings["eval_mask_seed"]))
    manifest = {"training_ids": memory.ids, "validation_ids": [ids[index] for index in val_indices],
                "modality_order": list(MODALITIES), "input_dims": input_dims,
                "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                "device": str(device), "score_range": source.score_range}
    save_json(config, output / "config.json")
    save_json(manifest, output / "split.json")
    history = []
    best_mse, best_sp = float("inf"), float("-inf")
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        totals = dict.fromkeys(("total", "task", "completion", "retrieval"), 0.0)
        sample_count = 0
        for batch in loader:
            features = move_features(batch, device)
            labels = batch["label"].to(device)
            observed = sample_masks(len(labels), settings["imr_rates"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            evidence = retrieve(model, memory, features, observed, config, exclude_ids=batch["id"])
            targets = model.encode_targets(features)
            outputs = model(features, observed, evidence)
            losses = core_losses(outputs, labels, observed, targets, evidence, memory.labels, loss_config)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Nonfinite objective in epoch {epoch}")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            for key, value in losses.items():
                totals[key] += float(value.detach()) * len(labels)
            sample_count += len(labels)
        row, _ = evaluate_masks(model, validation, memory, config, device, val_masks, imr_name(settings["imr_rates"]))
        history.append({"epoch": epoch, "train": {key: value / sample_count for key, value in totals.items()}, "validation": row})
        checkpoint = {"format": CHECKPOINT_FORMAT, "epoch": epoch, "model": model.state_dict(),
                      "config": config, "input_dims": input_dims, "memory_ids": memory.ids,
                      "validation_ids": manifest["validation_ids"], "score_range": source.score_range,
                      "validation": row}
        torch.save(checkpoint, output / "last.pt")
        if row["mse"] < best_mse:
            best_mse = row["mse"]
            torch.save(checkpoint, output / "best_mse.pt")
        if row["spearman"] is not None and row["spearman"] > best_sp:
            best_sp = row["spearman"]
            torch.save(checkpoint, output / "best_spearman.pt")
        sp = "n/a" if row["spearman"] is None else f"{row['spearman']:.4f}"
        print(f"epoch {epoch:03d} | loss {history[-1]['train']['total']:.4f} | validation SP {sp} | MSE {row['mse']:.4f}", flush=True)
        save_json(history, output / "history.json")
    write_metrics([history[-1]["validation"]], output)
    return output / "best_mse.pt"


def load_run(checkpoint_path, memory_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a BMC-AQA core checkpoint; legacy .pt/.pkl weights use a different architecture")
    memory = ModalityMemoryBank.load(memory_path)
    if memory.ids != checkpoint["memory_ids"]:
        raise ValueError("Memory identifiers differ from the training bank saved with this checkpoint")
    if set(memory.ids) & set(checkpoint["validation_ids"]):
        raise ValueError("Validation samples must not occur in the retrieval bank")
    model = BMCAQA(checkpoint["input_dims"], **checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), memory, checkpoint
