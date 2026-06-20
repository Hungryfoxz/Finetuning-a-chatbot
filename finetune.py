#!/usr/bin/env python3
"""
Fine-tune Llama-3.2-1B-Instruct on a library FAQ dataset using LoRA.
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login

# ==============================
# Configuration
# ==============================
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "hungryfoxz/LibraryFAQ1000"
OUTPUT_DIR = "./llama32_library_chatbot"

# Optional: login to Hugging Face (uncomment and add your token if needed)
# login("your_hf_token_here") #hf_kHrxPbYIITeecFLZUqKGftGRJcHAsXcod*

# Device setup (MPS for Apple Silicon, else CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# ==============================
# Load and format dataset
# ==============================
def format_example(example):
    """Format a single example into the Llama 3 chat template."""
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['question']}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['answer']}"
        "<|eot_id|>"
    )
    return {"text": text}

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)
dataset = dataset.map(format_example)
print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# ==============================
# Load tokenizer and base model
# ==============================
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
)
model.to(device)

# Set special token IDs
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id

model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id
model.generation_config.bos_token_id = tokenizer.bos_token_id

# ==============================
# LoRA configuration
# ==============================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==============================
# Training arguments
# ==============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    fp16=False,
    bf16=False,
    report_to="none"
)

# ==============================
# Trainer and training
# ==============================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer
)

print("Starting training...")
trainer.train()

# ==============================
# Save the adapter and tokenizer
# ==============================
adapter_path = f"{OUTPUT_DIR}/adapter"
trainer.model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"Adapter saved to {adapter_path}")
