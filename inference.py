#!/usr/bin/env python3
"""
Load the fine-tuned LoRA adapter and run inference on user questions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==============================
# Configuration
# ==============================
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "./llama32_library_chatbot/adapter"   # Path to saved adapter

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"

# ==============================
# Load tokenizer, base model, and adapter
# ==============================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()

# ==============================
# Inference function
# ==============================
def ask_question(question: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
    """Generate an answer for a given question."""
    messages = [{"role": "user", "content": question}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ==============================
# Interactive or single‑question mode
# ==============================
if __name__ == "__main__":
    print("\nLibrary FAQ Chatbot (type 'quit' to exit)\n")
    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        answer = ask_question(user_input)
        print("\nAssistant:", answer, "\n")