#!/bin/bash
export OMP_NUM_THREADS=2

# 1. Install DeepSpeed if missing (Kaggle usually needs this)
uv sync

# 2. Set Visible Devices (ensure we see both T4s)
export CUDA_VISIBLE_DEVICES=0,1

# 3. Launch Accelerate with the DeepSpeed config
# We pass python arguments after the script name
uv run accelerate launch --config_file configs/zero2.yaml scripts/train.py \
    --model_id "LGAI-EXAONE/EXAONE-4.0-1.2B" \
    --output_dir "./exaone-ds-finetune" \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 5e-5