#!/usr/bin/env python3
"""Evaluate a BMC-AQA checkpoint using the exact bank saved during training."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmc_aqa.config import save_json
from bmc_aqa.engine import dataset_from_config, evaluate_masks, get_device, load_run, write_metrics
from bmc_aqa.protocol import (
    IMR_SETTINGS, aggregate_metrics, fixed_masks, imr_name, mask_name, sample_masks,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--memory-path", help="Default: memory.pt beside checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--protocol", choices=("imr", "all-imr", "fixed"), default="imr")
    parser.add_argument("--imr-rates", nargs=3, type=float, metavar=("RGB", "FLOW", "AUDIO"))
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.imr_rates is not None and args.protocol != "imr":
        parser.error("--imr-rates is only valid with --protocol imr")
    device = get_device(args.device)
    memory_path = args.memory_path or Path(args.checkpoint).parent / "memory.pt"
    model, memory, checkpoint = load_run(args.checkpoint, memory_path, device)
    config = checkpoint["config"]
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    dataset = dataset_from_config(config, args.data_root, train=False)
    final_ids = {entry.sample_id for entry in dataset.labels}
    if final_ids & (set(memory.ids) | set(checkpoint["validation_ids"])):
        raise ValueError("The official evaluation split overlaps the training/validation identifiers")
    if dataset.infer_input_dims() != checkpoint["input_dims"] or dataset.score_range != checkpoint["score_range"]:
        raise ValueError("Evaluation feature dimensions or score normalization differ from training")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise ValueError("Evaluation output already exists; choose a new --output-dir")
    conditions = []
    if args.protocol == "fixed":
        conditions = [(mask_name(mask), torch.tensor(mask).float().repeat(len(dataset), 1)) for mask in fixed_masks()]
    else:
        rates = IMR_SETTINGS if args.protocol == "all-imr" else [args.imr_rates or config["training"]["imr_rates"]]
        conditions.append(("rgb+flow+audio", torch.ones(len(dataset), 3)))
        for triple in rates:
            generator = torch.Generator().manual_seed(args.mask_seed)
            conditions.append((imr_name(triple), sample_masks(len(dataset), triple, generator)))
    rows, predictions = [], []
    for name, masks in conditions:
        row, records = evaluate_masks(model, dataset, memory, config, device, masks, name)
        rows.append(row)
        predictions.extend(records)
        sp = "n/a" if row["spearman"] is None else f"{row['spearman']:.4f}"
        print(f"{name}: SP={sp}, MSE={row['mse']:.4f}", flush=True)
    if args.protocol == "fixed":
        rows.append({"setting": "average_incomplete", **aggregate_metrics(rows[1:])})
    write_metrics(rows, output)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("setting", "id", "prediction", "label", "rgb", "flow", "audio"))
        writer.writeheader()
        writer.writerows(predictions)
    save_json({"checkpoint": str(Path(args.checkpoint).resolve()), "memory": str(Path(memory_path).resolve()),
               "protocol": args.protocol, "mask_seed": args.mask_seed,
               "modality_order": ["rgb", "flow", "audio"], "score_range": dataset.score_range,
               "checkpoint_epoch": checkpoint["epoch"], "samples": len(dataset)}, output / "evaluation.json")


if __name__ == "__main__":
    main()

