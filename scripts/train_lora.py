import argparse
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune with DeepSpeed ZeRO and LoRA")
    parser.add_argument("--model_id", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    )
    parser.add_argument("--output_dir", type=str, default="./checkpoints_lora")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    parser.add_argument("--max_seq_length", type=int, default=300)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- 1. Tokenizer Setup ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 2. Dataset Processing ---
    # Load raw dataset
    full_dataset = load_dataset(args.dataset_id, split="train")

    # Split: 90% Train, 10% Validation
    dataset_split = full_dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    # Transformation function
    def apply_chat_style(example):
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    # Map dataset (disable multiprocessing to avoid lock issues in distributed env)
    train_dataset = train_dataset.map(apply_chat_style, num_proc=3)
    val_dataset = val_dataset.map(apply_chat_style, num_proc=3)

    # --- 3. Model Setup ---
    # When using DeepSpeed do not specify device_map.
    # DeepSpeed will manage placing layers on devices.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        attn_implementation=args.attn_implementation,
        use_cache=False,  # Must be False for Gradient Checkpointing
    )

    # Enable Gradient Checkpointing (Industry Standard for saving VRAM)
    model.gradient_checkpointing_enable()

    # --- 4. LoRA Config ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 5. Training Configuration ---
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        # Batch sizes
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Training Duration
        max_steps=1500,
        eval_strategy="steps",
        eval_steps=1500 // 10,
        save_steps=1500 // 10,
        load_best_model_at_end=True,  # Crucial: Loads the best checkpoint when done
        metric_for_best_model="eval_loss",  # Watch the validation loss
        greater_is_better=False,  # Lower loss is better
        save_total_limit=2,
        logging_steps=10,
        # Optimizer & Scheduler
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_ratio=0.05,
        # Technical Settings
        fp16=True,
        packing=False,
        dataset_text_field="text",
        # DeepSpeed specific:
        ddp_find_unused_parameters=False,
    )

    # --- 6. Initialize Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # --- 7. Train ---
    print(">>> Starting DeepSpeed LoRA Training...")
    trainer.train()

    # --- 8. Save Model ---
    # In distributed training, we want to ensure we merge weights properly
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f">>> Training complete. LoRA adapters saved to {args.output_dir}")


if __name__ == "__main__":
    main()
