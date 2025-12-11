#!/bin/bash
export OMP_NUM_THREADS=2

# 1. Install DeepSpeed if missing (Kaggle usually needs this)
uv sync

# 2. Set Visible Devices (ensure we see both T4s)
export CUDA_VISIBLE_DEVICES=0,1

# 3. Launch Accelerate with the DeepSpeed config
# We pass python arguments after the script name

uv run accelerate launch --config_file configs/zero2.yaml scripts/train.py \
    --model_id "Qwen/Qwen3-1.7B-Instruct" \
    --output_dir "./qwen3-finetune" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --max_seq_length 300 \
    --attn_implementation "eager" \
    "$@"