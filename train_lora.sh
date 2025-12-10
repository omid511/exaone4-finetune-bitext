#!/bin/bash
export OMP_NUM_THREADS=2

# 1. Install DeepSpeed if missing (Kaggle usually needs this)
uv sync

# 2. Set Visible Devices (ensure we see both T4s)
export CUDA_VISIBLE_DEVICES=0,1

# 3. Launch Accelerate with the DeepSpeed config
# We pass python arguments after the script name
# Effective Batch Size = 4 (batch) * 4 (grad_accum) * 2 (GPUs) = 32
uv run accelerate launch --config_file configs/zero2.yaml scripts/train_lora.py \
    --model_id "LGAI-EXAONE/EXAONE-4.0-1.2B" \
    --output_dir "./exaone-ds-lora" \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4 \
    --max_seq_length 300 \
    --attn_implementation "eager"
