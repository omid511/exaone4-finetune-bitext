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
- customer-support
- chatbot
license: apache-2.0
datasets:
- bitext/Bitext-customer-support-llm-chatbot-training-dataset
language:
- en
---

# Model Card for Qwen3-1.7B Customer Support Agent

This model is a fine-tuned version of [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) on the [Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). It is designed to act as a helpful customer support agent.

## Model Details

- **Base Model:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- **Architecture:** Qwen3 (1.7B parameters)
- **Training Method:** QLoRA (4-bit quantization with LoRA adapters)
- **Dataset:** [Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- **Language:** English
- **Code Repository:** [omid511/qwen3-1.7b-finetune-customer-support](https://github.com/omid511/qwen3-1.7b-finetune-customer-support)

## Usage

### Loading the Adapter (Default)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen3-1.7B"
adapter_model_id = "omid5/Qwen3-1.7b-cusomer-support-agent"

# 1. Load Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Load Adapters
model = PeftModel.from_pretrained(base_model, adapter_model_id)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 3. Inference
messages = [
    {"role": "user", "content": "I received a defective item, what should I do?"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Loading the Merged Model

The full merged model is available on the `merged` branch of this repository.

```python
model = AutoModelForCausalLM.from_pretrained(
    "omid5/Qwen3-1.7b-cusomer-support-agent",
    revision="merged",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("omid5/Qwen3-1.7b-cusomer-support-agent", revision="merged")
```

## Training Configuration

The model was trained using `accelerate` and `DeepSpeed` with the following hyperparameters:

- **Epochs:** 2
- **Learning Rate:** 2e-4 (Cosine Schedule)
- **Batch Size:** 8 (per device train) / 16 (per device eval)
- **Gradient Accumulation:** 2
- **Max Sequence Length:** 400
- **LoRA Config:**
    - `r`: 16
    - `alpha`: 32
    - `dropout`: 0.05
    - `target_modules`: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
- **Quantization:** 4-bit (nf4) via `bitsandbytes`

## Training Metrics

| Metric | Value |
| :--- | :--- |
| **Validation Loss** | 0.5842 |
| **Validation Token Acc.** | 81.00% |
| **Training Loss** | 0.6846 |
| **Training Runtime** | 9282s (~2.6h) |
| **Samples/Second** | 5.21 |
| **Total Global Steps** | 1512 |

## Hardware
- **GPUs:** 2x Tesla T4
- **Platform:** Kaggle / Cloud

## License
Apache-2.0