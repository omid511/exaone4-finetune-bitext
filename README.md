<div align="center">

# 🤖 Qwen3-1.7B Customer Support Agent

**Fine-tune Qwen3-1.7B for customer support using QLoRA, DoRA, or full fine-tuning**

[![Model](https://img.shields.io/badge/🤗%20Model-Qwen3--1.7B-blue)](https://huggingface.co/omid5/Qwen3-1.7b-cusomer-support-agent)
[![Demo](https://img.shields.io/badge/🚀%20Demo-Live%20Chat-green)](https://huggingface.co/spaces/omid5/qwen3-customer-support-demo)
[![Dataset](https://img.shields.io/badge/📊%20Dataset-Bitext-orange)](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Features](#-features) • [Quick Start](#-quick-start) • [Training](#-training) • [Project Structure](#-project-structure) • [Results](#-results)

</div>

---

## ✨ Features

| Feature | Description |
|:--------|:------------|
| **QLoRA** | 4-bit quantization with Low-Rank Adapters — train on 8GB VRAM |
| **DoRA** | Weight-Decomposed Low-Rank Adaptation for improved performance |
| **Full Fine-Tuning** | Complete parameter training via DeepSpeed ZeRO-2 |
| **Axolotl Support** | Pre-configured recipes for DeepSpeed & FSDP |
| **Auto Hub Push** | Automatically merge adapters and push to Hugging Face Hub |

---

## 🚀 Quick Start

### Prerequisites

> 💡 **Note:** Thanks to 4-bit quantization, you can fine-tune on GPUs with as little as **8GB VRAM**!

Install [uv](https://docs.astral.sh/uv/) for fast Python package management:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup

```bash
# Clone and install
git clone https://github.com/omid511/qwen3-1.7b-finetune-customer-support.git
cd qwen3-1.7b-finetune-customer-support
uv sync
```

---

## 🎯 Training

### Option 1: Custom Scripts (Recommended)

Training pipelines using `accelerate` + `DeepSpeed ZeRO-2`.

<details>
<summary><b>LoRA / QLoRA Training</b></summary>

```bash
# QLoRA (default)
./train_lora.sh

# Enable DoRA
./train_lora.sh --use_dora

# Custom parameters
./train_lora.sh --learning_rate 1e-4 --num_train_epochs 5
```

**Key flags:**
- `--use_bnb` — Enable 4-bit quantization (default: enabled)
- `--use_dora` — Use DoRA instead of LoRA
- `--merge_and_push` — Merge adapters and push to Hub

</details>

<details>
<summary><b>Full Fine-Tuning</b></summary>

```bash
# Standard fine-tuning
./train.sh

# Override arguments
./train.sh --learning_rate 2e-5 --num_train_epochs 3
```

</details>

### Option 2: Axolotl

Structured YAML-based configuration for reproducible training.

```bash
# DeepSpeed ZeRO-2 (default)
./train_axolotl.sh

# FSDP (Full Shard)
./train_axolotl.sh configs/axolotl_fsdp.yaml

# Override config values
./train_axolotl.sh configs/axolotl_deepspeed.yaml --learning_rate 5e-5
```

---

## 📂 Project Structure

```
├── scripts/
│   ├── train_lora.py      # QLoRA/DoRA training script
│   └── train.py           # Full fine-tuning script
├── configs/
│   ├── zero2.yaml         # DeepSpeed ZeRO-2 config
│   ├── axolotl_deepspeed.yaml
│   └── axolotl_fsdp.yaml
├── app.py                 # Gradio demo application
├── train_lora.sh          # QLoRA launcher
├── train.sh               # Full FT launcher
└── train_axolotl.sh       # Axolotl launcher
```

---

## 📊 Results

| Metric | Value |
|:-------|------:|
| Validation Loss | 0.5842 |
| Token Accuracy | 81.00% |
| Training Time | ~2.6h |
| Hardware | 2× Tesla T4 |

> [!TIP]
> Check the [Model Card](https://huggingface.co/omid5/Qwen3-1.7b-cusomer-support-agent) for full training details and usage examples.

---

## 🔗 Resources

| Resource | Link |
|:---------|:-----|
| 🤗 Model Card | [omid5/Qwen3-1.7b-cusomer-support-agent](https://huggingface.co/omid5/Qwen3-1.7b-cusomer-support-agent) |
| 🚀 Live Demo | [Hugging Face Spaces](https://huggingface.co/spaces/omid5/qwen3-customer-support-demo) |
| 📊 Dataset | [Bitext Customer Support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) |
| 🏗️ Base Model | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |

---

## 📜 License

Apache-2.0 — see [LICENSE](LICENSE) for details.
