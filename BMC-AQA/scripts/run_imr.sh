#!/usr/bin/env bash
# Train and evaluate one independent BMC-AQA model per paper IMR setting.
set -euo pipefail

config="${1:-configs/fs1000.toml}"
data_root="${2:-data}"
output_root="${3:-runs/fs1000_tes_imr}"
device="${4:-auto}"

for triple in "0.3 0.5 0.7" "0.3 0.7 0.5" "0.5 0.3 0.7" \
              "0.5 0.7 0.3" "0.7 0.3 0.5" "0.7 0.5 0.3"; do
  read -r rgb flow audio <<< "$triple"
  run_dir="${output_root}/rgb${rgb}_flow${flow}_audio${audio}"
  python scripts/train.py --config "$config" --data-root "$data_root" \
    --imr-rates "$rgb" "$flow" "$audio" --device "$device" --output-dir "$run_dir"
  python scripts/evaluate.py --checkpoint "${run_dir}/best_mse.pt" \
    --data-root "$data_root" --device "$device" --protocol imr \
    --mask-seed 2027 --output-dir "${run_dir}/test"
done

