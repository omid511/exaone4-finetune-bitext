#!/bin/bash
# Launch Axolotl Training
# Usage: ./train_axolotl.sh [additional arguments]
# Example: ./train_axolotl.sh --learning_rate 1e-5 --num_epochs 5

# 1. Install DeepSpeed if missing

uv sync

# 2. Set Visible Devices
export CUDA_VISIBLE_DEVICES=0,1

# 3. Launch Axolotl
CONFIG_FILE=${1:-"configs/axolotl_deepspeed.yaml"}
shift # Shift arguments so $@ handles the rest (if any)

echo "Using config: $CONFIG_FILE"

uv run accelerate launch -m axolotl.cli.train "$CONFIG_FILE" "$@"
