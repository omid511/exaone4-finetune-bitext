---
base_model: Qwen/Qwen3-1.7B
library_name: transformers
model_name: Qwen3-1.7b-cusomer-support-agent
tags:
- generated_from_trainer
- trl
- sft
- deepspeed
- lora
- qlora
- customer-support
- chatbot
- conversational
license: apache-2.0
datasets:
- bitext/Bitext-customer-support-llm-chatbot-training-dataset
language:
- en
pipeline_tag: text-generation
---

<div align="center">

# 🤖 Qwen3-1.7B Customer Support Agent

**A fine-tuned LLM specialized in customer service conversations**

[![Demo](https://img.shields.io/badge/🚀%20Try-Live%20Demo-green)](https://huggingface.co/spaces/omid5/qwen3-customer-support-demo)
[![GitHub](https://img.shields.io/badge/📁%20Code-GitHub-blue)](https://github.com/omid511/qwen3-1.7b-finetune-customer-support)

</div>

---

## Overview

This model is a QLoRA fine-tuned version of [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) trained on the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). It's designed to handle common customer support scenarios including order inquiries, refunds, shipping, and account management.

> 💡 **Tip:** [Try the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/omid5/qwen3-customer-support-demo)

---

## Model Details

| Property | Value |
|:---------|:------|
| **Base Model** | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |
| **Architecture** | Qwen3 (1.7B parameters) |
| **Training Method** | QLoRA (4-bit NF4 quantization + LoRA) |
| **Language** | English |
| **License** | Apache-2.0 |

---

## Quick Start

### Option 1: Load Merged Model (Recommended)

The fully merged model is available on the `merged` branch:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "omid5/Qwen3-1.7b-cusomer-support-agent",
    revision="merged",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "omid5/Qwen3-1.7b-cusomer-support-agent", 
    revision="merged"
)

# Generate response
messages = [{"role": "user", "content": "I want to return my order, what should I do?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option 2: Load LoRA Adapters

For lower memory usage, load adapters separately:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load adapters
model = PeftModel.from_pretrained(base_model, "omid5/Qwen3-1.7b-cusomer-support-agent")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Optional: Merge for faster inference
model = model.merge_and_unload()
```

---

## Training Configuration

### Hyperparameters

| Parameter | Value |
|:----------|------:|
| Epochs | 2 |
| Learning Rate | 2e-4 |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.1 |
| Batch Size (train) | 8 |
| Batch Size (eval) | 16 |
| Gradient Accumulation | 2 |
| Max Sequence Length | 400 |
| Precision | FP16 |

### LoRA Configuration

| Parameter | Value |
|:----------|------:|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

### Quantization

- **Method:** 4-bit NF4 via `bitsandbytes`
- **Double Quantization:** Enabled
- **Compute dtype:** FP16

---

## Results

| Metric | Value |
|:-------|------:|
| **Validation Loss** | 0.5842 |
| **Token Accuracy** | 81.00% |
| **Training Loss** | 0.6846 |
| **Training Runtime** | ~2.6 hours |
| **Samples/Second** | 5.21 |
| **Total Steps** | 1,512 |

### Training Environment

- **Hardware:** 2× Tesla T4 (Kaggle)
- **Framework:** Transformers + TRL + PEFT + DeepSpeed ZeRO-2
- **Tracking:** Weights & Biases

---

## Intended Use

### Recommended Applications

- Customer support chatbots
- Order and shipment inquiries
- Returns and refunds assistance
- Account management help
- FAQ automation

### Limitations

- Trained on English data only
- May generate placeholder tokens from training data (e.g., `{{Customer Name}}`)
- Not suitable for medical, legal, or financial advice
- Responses should be reviewed for production deployment

---

## Citation

```bibtex
@misc{qwen3-customer-support,
  title={Qwen3-1.7B Customer Support Agent},
  author={omid5},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/omid5/Qwen3-1.7b-cusomer-support-agent}
}
```

---

## License

Apache-2.0