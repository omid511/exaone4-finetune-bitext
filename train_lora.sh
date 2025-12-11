#!/bin/bash
export OMP_NUM_THREADS=2

# 1. Install DeepSpeed if missing (Kaggle usually needs this)
uv sync

# 2. Set Visible Devices (ensure we see both T4s)
export CUDA_VISIBLE_DEVICES=0,1

# 3. Launch Accelerate with the DeepSpeed config
# We pass python arguments after the script name
# Default Configuration:
# - Model: Qwen3 1.7B
# - Repo: omid5/Qwen3-1.7b-cusomer-support-agent
# - QLoRA: Enabled (4-bit)
# - Merge & Push: Enabled
# - Batch Sizes: Train=1 (Efficiency), Eval=4 (Speed)
# - Grad Accum: 16 (Eff. Batch Size = 32)
# - Gradient Checkpointing: Disabled (implied by script update)

uv run accelerate launch --config_file configs/zero2.yaml scripts/train_lora.py \
    --model_id "Qwen/Qwen3-1.7B-Instruct" \
    --output_dir "./qwen3-lora" \
    --hub_model_id "omid5/Qwen3-1.7b-cusomer-support-agent" \
    --push_to_hub \
    --use_bnb \
    --merge_and_push \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --max_seq_length 300 \
    --attn_implementation "eager" \
    "$@"
