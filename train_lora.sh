export OMP_NUM_THREADS=2

# 1. Install DeepSpeed if missing (Kaggle usually needs this)
uv sync

# 2. Set Visible Devices (ensure we see both T4s)
export CUDA_VISIBLE_DEVICES=0,1

# 3. Launch Accelerate with the DeepSpeed config
# We pass python arguments after the script name
# Default Configuration restored from user request:
# - Model: Qwen3 1.7B
# - Repo: omid5/Qwen3-1.7b-cusomer-support-agent
# - QLoRA: Enabled (4-bit)
# - Merge & Push: Enabled
# - Batch Sizes: Train=1, Eval=4
# - Grad Accum: 16
# - Steps: max=500, eval=50, save=100 (Demo/Quick mode)
# - Technical: FP16, No Packing, No Unused Params Check

uv run accelerate launch --config_file configs/zero2.yaml scripts/train_lora.py \
    --model_id "Qwen/Qwen3-1.7B" \
    --output_dir "./qwen3-lora" \
    --hub_model_id "omid5/Qwen3-1.7b-cusomer-support-agent" \
    --push_to_hub \
    --use_bnb \
    --merge_and_push \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --max_seq_length 300 \
    --num_train_epochs 3 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --disable_tqdm True \
    --fp16 \
    --packing False \
    --dataset_text_field "text" \
    --ddp_find_unused_parameters False \
    --attn_implementation "eager" "$@"
