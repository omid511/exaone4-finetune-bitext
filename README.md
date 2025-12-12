# 🤖 Qwen3-1.7B Customer Support Agent

Fine-tune **[Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)** on the **[Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)** using **DeepSpeed**, **Accelerate**, or **Axolotl**.

## ✨ Features

- **QLoRA**: 4-bit quantization with Low-Rank Adapters.
- **DoRA**: Support for Weight-Decomposed Low-Rank Adaptation.
- **Full Fine-Tuning**: Script for full parameter training via `train.sh`.
- **Axolotl**: Recipes for both DeepSpeed Zero2 and FSDP.
- **Automated**: Auto-merge and push to Hugging Face Hub (main branch).

## ⚡ Setup

1. **Install `uv`**:
   ```bash
   # Linux / macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Install Dependencies**:
   ```bash
   uv sync
   ```

## 🛠️ Usage

### Option 1: Custom Scripts (Trainer)
The primary training pipelines using `accelerate` + `DeepSpeed`.

**LoRA / QLoRA Training:**
```bash
# Default (QLoRA)
./train_lora.sh

# DoRA
./train_lora.sh --use_dora
```

**Full Fine-Tuning:**
```bash
# Standard Fine-tuning
./train.sh

# Override Arguments
./train.sh --learning_rate 2e-5 --num_train_epochs 3
```

### Option 2: Axolotl
Structured configuration management.

```bash
# Default (DeepSpeed)
./train_axolotl.sh

# FSDP (Full Shard)
./train_axolotl.sh configs/axolotl_fsdp.yaml

# Override Config
./train_axolotl.sh configs/axolotl_deepspeed.yaml --learning_rate 5e-5
```

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `scripts/train_lora.py` | Script for LoRA/QLoRA training |
| `scripts/train.py` | Script for Full Fine-tuning |
| `configs/zero2.yaml` | DeepSpeed configuration |
| `configs/axolotl_*.yaml` | Axolotl recipes (DeepSpeed & FSDP) |
| `train_lora.sh` | Launcher for LoRA training |
| `train.sh` | Launcher for Full Fine-tuning |
| `train_axolotl.sh` | Launcher for Axolotl |
| `MODEL_CARD.md` | Hugging Face Model Card template |

## 📜 License

Apache-2.0
