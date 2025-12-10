import argparse
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune with DeepSpeed ZeRO")
    parser.add_argument("--model_id", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    )
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
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
    # CRITICAL: When using DeepSpeed, do NOT specify device_map.
    # DeepSpeed will manage placing layers on devices.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        attn_implementation="eager",
        use_cache=False,  # Must be False for Gradient Checkpointing
    )

    # Enable Gradient Checkpointing (Industry Standard for saving VRAM)
    model.gradient_checkpointing_enable()

    # --- 4. Training Configuration ---
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        # Batch sizes
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Training Duration
        max_steps=500,  # Shortened for demo, increase for real training
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        # Optimizer & Scheduler
        optim="paged_adamw_8bit",  # Compatible with DeepSpeed
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        # Technical Settings
        fp16=True,
        packing=False,
        dataset_text_field="text",
        # DeepSpeed specific:
        ddp_find_unused_parameters=False,
    )

    # --- 5. Initialize Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # --- 6. Train ---
    print(">>> Starting DeepSpeed Training...")
    trainer.train()

    # --- 7. Save Model ---
    # In distributed training, we want to ensure we merge weights properly
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f">>> Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
