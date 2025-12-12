# app.py
"""Gradio demo for the fine-tuned Qwen3-1.7B model.

A polished chat UI with token streaming for customer support conversations.
Includes post-processing to replace Bitext dataset placeholders with actual values.
"""

import os
import re
from threading import Thread

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextIteratorStreamer

# ------------------- Configuration -------------------
BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
REPO_ID = os.getenv("HF_MODEL_REPO", "omid5/Qwen3-1.7b-cusomer-support-agent")
REVISION = os.getenv("HF_MODEL_REVISION", "merged")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default system prompt with Acme Corp context
DEFAULT_SYSTEM_PROMPT = """You are a helpful customer support agent for Acme Corporation.

Company Information:
- Company Name: Acme Corporation
- Customer Support Phone: 1-800-ACME-HELP (1-800-226-3435)
- Customer Support Email: support@acme.com
- Business Hours: Monday-Friday 9AM-6PM EST
- Website: https://www.acme.com

Policies:
- Refund Policy: Full refund within 30 days of purchase
- Shipping: Free shipping on orders over $50
- Returns: Items must be in original packaging

Be professional, helpful, and concise."""

# Default customer context for demo
DEFAULT_CUSTOMER_CONTEXT = {
    "customer_name": "John Smith",
    "order_number": "ORD-2024-78542",
    "order_date": "December 5, 2024",
    "delivery_date": "December 15, 2024",
    "product": "Wireless Headphones Pro",
    "order_total": "$149.99",
    "tracking_number": "1Z999AA10123456784",
    "delivery_city": "New York",
    "account_email": "john.smith@email.com",
}

# Mapping of Bitext placeholders to context keys
PLACEHOLDER_MAP = {
    "{{Customer Name}}": "customer_name",
    "{{Order Number}}": "order_number",
    "{{Order Date}}": "order_date",
    "{{Delivery Date}}": "delivery_date",
    "{{Product}}": "product",
    "{{Order Total}}": "order_total",
    "{{Tracking Number}}": "tracking_number",
    "{{Delivery City}}": "delivery_city",
    "{{Account Email}}": "account_email",
    "{{Customer Support Phone Number}}": "support_phone",
    "{{Customer Support Email}}": "support_email",
    "{{Company Name}}": "company_name",
    "{{Website}}": "website",
}


def replace_placeholders(text: str, customer_ctx: dict, company_ctx: dict) -> str:
    """Replace Bitext dataset placeholders with actual values."""
    # Merge contexts
    ctx = {
        **customer_ctx,
        "support_phone": company_ctx.get("support_phone", "1-800-ACME-HELP"),
        "support_email": company_ctx.get("support_email", "support@acme.com"),
        "company_name": company_ctx.get("company_name", "Acme Corporation"),
        "website": company_ctx.get("website", "https://www.acme.com"),
    }
    
    result = text
    for placeholder, key in PLACEHOLDER_MAP.items():
        if key in ctx and ctx[key]:
            result = result.replace(placeholder, ctx[key])
    
    # Also handle case variations
    for placeholder, key in PLACEHOLDER_MAP.items():
        if key in ctx and ctx[key]:
            result = result.replace(placeholder.lower(), ctx[key])
            result = result.replace(placeholder.upper(), ctx[key])
    
    return result


# ------------------- Custom CSS -------------------
CUSTOM_CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}

.header-container {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}

.header-container h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
}

.header-container p {
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
    font-size: 1rem;
}

.cpu-badge {
    display: inline-block;
    background: rgba(255, 255, 255, 0.2);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    margin-top: 0.8rem;
}

footer {
    visibility: hidden;
}

.context-fields input {
    font-size: 0.9rem !important;
}
"""


def load_model_and_tokenizer():
    """Load model with fallback: merged model -> LoRA adapters -> base model."""
    print(f"Loading tokenizer from base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    # Attempt 1: Load as merged model
    try:
        print(f"Attempting to load merged model from: {REPO_ID} (revision: {REVISION})")
        config = AutoConfig.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            REPO_ID,
            revision=REVISION,
            config=config,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
        )
        print("Successfully loaded merged model.")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load merged model: {e}")

    # Attempt 2: Load base model with LoRA adapters
    try:
        print(f"Attempting to load LoRA adapters from: {REPO_ID}")
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, REPO_ID)
        model = model.merge_and_unload()
        print("Successfully loaded LoRA adapters.")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load LoRA adapters: {e}")

    # Attempt 3: Fallback to base model
    print(f"Falling back to base model: {BASE_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    return model, tokenizer


# Load model and tokenizer at startup
model, tokenizer = load_model_and_tokenizer()


# ------------------- Streaming Chat -------------------
def generate_response(message: str, history: list[list[str]], system_prompt: str, customer_ctx: dict):
    """Generate a streaming response with system prompt and post-processing."""
    messages = []
    
    # Add system prompt if provided
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    
    # Add conversation history
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Company context from system prompt (simplified extraction)
    company_ctx = {
        "support_phone": "1-800-ACME-HELP (1-800-226-3435)",
        "support_email": "support@acme.com",
        "company_name": "Acme Corporation",
        "website": "https://www.acme.com",
    }

    partial_response = ""
    for new_text in streamer:
        partial_response += new_text
        # Apply post-processing to replace placeholders
        processed = replace_placeholders(partial_response, customer_ctx, company_ctx)
        yield processed

    thread.join()


# ------------------- Event Handlers -------------------
def user_message(message, history):
    """Add user message to history and clear input."""
    if not message.strip():
        return "", history
    return "", history + [[message, None]]


def bot_response(history, system_prompt, name, order_num, order_date, delivery_date, product, total, tracking, city, email):
    """Stream bot response with system prompt and customer context."""
    if not history:
        return history
    
    customer_ctx = {
        "customer_name": name,
        "order_number": order_num,
        "order_date": order_date,
        "delivery_date": delivery_date,
        "product": product,
        "order_total": total,
        "tracking_number": tracking,
        "delivery_city": city,
        "account_email": email,
    }
    
    user_msg = history[-1][0]
    history[-1][1] = ""
    for chunk in generate_response(user_msg, history[:-1], system_prompt, customer_ctx):
        history[-1][1] = chunk
        yield history


# ------------------- Example Prompts -------------------
EXAMPLE_PROMPTS = [
    "What's my order status?",
    "When will my order arrive?",
    "Can I get a refund?",
    "What's my tracking number?",
    "How do I contact support?",
]


# ------------------- Gradio Interface -------------------
cpu_badge = '<div class="cpu-badge">⚡ Running on CPU</div>' if DEVICE == "cpu" else ""

with gr.Blocks(
    title="Qwen3 Customer Support",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:
    # Header
    gr.HTML(
        f"""
        <div class="header-container">
            <h1>🤖 Customer Support Assistant</h1>
            <p>Powered by Qwen3-1.7B fine-tuned on customer support conversations</p>
            {cpu_badge}
        </div>
        """
    )

    with gr.Row():
        # Left column: Chat
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=450,
                show_copy_button=True,
                avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/robot_1f916.png"),
                type="tuples",
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about your order, account, or services...",
                    show_label=False,
                    scale=9,
                    container=False,
                    autofocus=True,
                )
                submit_btn = gr.Button("Send", scale=1, variant="primary", min_width=80)

            clear_btn = gr.ClearButton([msg, chatbot], value="🗑️ Clear Chat", variant="secondary")

            gr.Markdown("**💡 Try asking:**")
            with gr.Row():
                example_btns = []
                for prompt in EXAMPLE_PROMPTS:
                    btn = gr.Button(prompt, size="sm", variant="secondary")
                    example_btns.append((btn, prompt))

        # Right column: Settings
        with gr.Column(scale=1):
            with gr.Accordion("👤 Customer Context", open=True):
                gr.Markdown("*Simulated customer data (replaces placeholders in responses)*")
                ctx_name = gr.Textbox(label="Name", value=DEFAULT_CUSTOMER_CONTEXT["customer_name"])
                ctx_order = gr.Textbox(label="Order #", value=DEFAULT_CUSTOMER_CONTEXT["order_number"])
                ctx_order_date = gr.Textbox(label="Order Date", value=DEFAULT_CUSTOMER_CONTEXT["order_date"])
                ctx_delivery = gr.Textbox(label="Delivery Date", value=DEFAULT_CUSTOMER_CONTEXT["delivery_date"])
                ctx_product = gr.Textbox(label="Product", value=DEFAULT_CUSTOMER_CONTEXT["product"])
                ctx_total = gr.Textbox(label="Order Total", value=DEFAULT_CUSTOMER_CONTEXT["order_total"])
                ctx_tracking = gr.Textbox(label="Tracking #", value=DEFAULT_CUSTOMER_CONTEXT["tracking_number"])
                ctx_city = gr.Textbox(label="City", value=DEFAULT_CUSTOMER_CONTEXT["delivery_city"])
                ctx_email = gr.Textbox(label="Account Email", value=DEFAULT_CUSTOMER_CONTEXT["account_email"])

            with gr.Accordion("⚙️ System Prompt", open=False):
                system_prompt = gr.Textbox(
                    value=DEFAULT_SYSTEM_PROMPT,
                    label="System Prompt",
                    lines=10,
                    max_lines=15,
                )

    # Footer
    gr.Markdown(
        f"""
        ---
        <center>
        <small>
        🔗 <a href="https://huggingface.co/{REPO_ID}" target="_blank">Model Card</a> · 
        📊 <a href="https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset" target="_blank">Bitext Dataset</a> · 
        🏗️ <a href="https://huggingface.co/{BASE_MODEL_ID}" target="_blank">{BASE_MODEL_ID}</a>
        </small>
        </center>
        """
    )

    # All context inputs for event handlers
    context_inputs = [system_prompt, ctx_name, ctx_order, ctx_order_date, ctx_delivery, ctx_product, ctx_total, ctx_tracking, ctx_city, ctx_email]

    # Wire up main events
    msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot] + context_inputs, chatbot
    )
    submit_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_response, [chatbot] + context_inputs, chatbot
    )

    # Wire up example buttons
    for btn, prompt in example_btns:
        btn.click(lambda p=prompt: p, outputs=msg).then(
            user_message, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            bot_response, [chatbot] + context_inputs, chatbot
        )


if __name__ == "__main__":
    demo.queue().launch()
