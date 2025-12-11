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
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --max_seq_length 300 \
    --max_steps 500 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --fp16 \
    --packing False \
    --dataset_text_field "text" \
    --ddp_find_unused_parameters False \
    --attn_implementation "eager" \
    "$@"