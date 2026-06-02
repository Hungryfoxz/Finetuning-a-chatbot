
# 🤖 Library Chatbot: Finetuning Llama Model


An end-to-end machine learning pipeline engineered to fine-tune a **Meta Llama** foundational large language model (LLM). This project specializes the model into an automated **Library Chatbot** capable of handling book recommendations, catalog queries, reservation policies, and academic research assistance.

---

## 🚀 Key Implementation Features

*   **Parameter-Efficient Fine-Tuning (PEFT):** Implements Low-Rank Adaptation (**LoRA**) or QLoRA to fine-tune the model with minimal hardware resources.
*   **Quantization Support:** Leverages 4-bit or 8-bit precision loading configurations (`bitsandbytes`) to drastically optimize GPU memory footprint.
*   **Tailored Library Dataset:** Structured prompt-response pairings designed to align model outputs with standard library domain knowledge.
*   **Fully-Documented Pipeline:** Step-by-step model loading, tokenization, tracking, training loop configuration, and interactive inference within a single notebook.

---

## 🛠️ Machine Learning Stack

*   **Framework Baseline:** PyTorch
*   **LLM Tools:** Hugging Face Ecosystem (`transformers`, `peft`, `accelerate`, `trl`)
*   **Quantization Engine:** `bitsandbytes`
*   **Development Layer:** Jupyter Notebooks (`Finetuning.ipynb`)

---

## 📦 Project Architecture

```bash
├── Finetuning.ipynb     # Main Jupyter notebook containing data prep, training, and testing
├── requirements.txt     # Python dependency environment config file
└── README.md            # Repository documentation layer
```

---

## ⚙️ Quick Start Installation

Follow these steps to configure your environment and begin local training or experimentation:

### 1. Clone the Project
```bash
git clone https://github.com
cd Finetuning-a-chatbot
```

### 2. Setup the Environment
It is highly recommended to run this within a dedicated virtual environment or on a GPU-enabled cloud container (like Google Colab T4/A100, Kaggle, or RunPod).
```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation
A custom dataset was prepared with data from Krishna Kanta Handiqui Library, Gauhati Univeristy, Assam, India.
The dataset is available from the following link --->
```bash
https://huggingface.co/datasets/hungryfoxz/LibraryFAQ
```

### 4. Execution
Launch Jupyter Notebook or your preferred IDE to open and run the pipeline step-by-step:
```bash
jupyter notebook Finetuning.ipynb
```

---

## 📝 Training Pipeline Workflow Summary

1.  **Data Loading:** Ingestion and parsing of customized library operational rules or Q&A pairs.
2.  **Formatting Prompts:** Formatting the text corpus using an explicit instruction format template (e.g., `### Instruction: ... ### Response:`).
3.  **Model Configuration:** Loading the Llama foundational weights under a quantized 4-bit configuration to fit on standard VRAM margins.
4.  **Applying Adapter:** Injecting LoRA matrices onto target modules (like `q_proj` and `v_proj`).
5.  **Execution:** Running the `SFTTrainer` (Supervised Fine-tuning Trainer) wrapper and monitoring loss curves.
6.  **Saving Weights:** Exporting the specialized adapter weights for standalone testing or deployment.

---

## 📜 License

Distributed under the open-source **MIT License**. Check individual base model access tokens for compliance with Meta's community license standards.
