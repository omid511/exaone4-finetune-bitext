import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
)
import torch
import os


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="Qwen/Qwen3-1.7B-Instruct",
        metadata={"help": "The model that you want to train from the Hugging Face hub."},
    )
    dataset_id: str = field(
        default="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    max_seq_length: int = field(
        default=300,
        metadata={"help": "The maximum sequence length for the model."},
    )
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "Attention implementation to use.",
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )


def main():
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        script_args, sft_config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        script_args, sft_config = parser.parse_args_into_dataclasses()

    # --- 1. Tokenizer Setup ---
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 2. Dataset Processing ---
    # Load raw dataset
    full_dataset = load_dataset(script_args.dataset_id, split="train")

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
    train_dataset = train_dataset.map(apply_chat_style, num_proc=1)
    val_dataset = val_dataset.map(apply_chat_style, num_proc=1)

    # --- 3. Model Setup ---
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        dtype=torch.float16,
        attn_implementation=script_args.attn_implementation,
        use_cache=False,  # Must be False for Gradient Checkpointing
    )

    # Enable Gradient Checkpointing (Standard for Full FT)
    model.gradient_checkpointing_enable()

    # --- 4. Initialize Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # --- 5. Train ---
    print(">>> Starting Full Fine-Tuning...")
    trainer.train(resume_from_checkpoint=sft_config.resume_from_checkpoint)

    # --- 6. Save Model ---
    trainer.save_model(sft_config.output_dir)
    tokenizer.save_pretrained(sft_config.output_dir)
    print(f">>> Training complete. Model saved to {sft_config.output_dir}")

    # Push if requested
    if sft_config.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
