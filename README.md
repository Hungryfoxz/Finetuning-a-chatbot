# 📚 Virtual Library Assistant – Llama 3.2 1B Fine‑tuned for Library FAQs

This repository contains a complete pipeline to **fine‑tune** Meta’s **Llama‑3.2‑1B‑Instruct** on a library FAQ dataset using **LoRA**, and then deploy the fine‑tuned model as an interactive **chatbot** using **Streamlit**.

The project includes:

- `finetune.py` – Fine‑tunes the base model with LoRA on `hungryfoxz/LibraryFAQ1000`
- `inference.py` – Command‑line interface to chat with the fine‑tuned model
- `app.py` – Full‑featured web UI (Streamlit) with adjustable generation parameters
- `requirements.txt` – All Python dependencies

---

## 🖥️ Hardware & Software Requirements

- **Apple Silicon (M1/M2/M3)** – recommended for GPU acceleration via **MPS**  
  *(also works on CPU, but will be slower)*
- **Python 3.10 or higher**
- **At least 8 GB RAM** (16 GB recommended)
- **Hugging Face account** (only needed if you want to download the base model – it’s public, no token required for inference)

---

## 📦 Installation

1. **Clone the repository**
   ```bash
     git clone https://github.com/Hungryfoxz/Finetuning-a-chatbot.git
     cd library-chatbot
   ```
2. **Create a virtual environment**
   ```bash
      python -m venv venv
      source venv/bin/activate        # On Linux/macOS
      # or .\venv\Scripts\activate   # On Windows
   ```
3. **Install dependencies**
   ```bash
     pip install -r requirements.txt
   ```

## 🧠 Training the Model (Optional)
The repository includes a pre‑saved adapter in ```./llama32_library_chatbot/adapter```.
If you want to retrain the model yourself, run:
  ```bash
    python finetune.py
  ```

Training time on a MacBook Air M2: ~10 minutes for 3 epochs.
After training, the adapter and tokenizer are saved to ```./llama32_library_chatbot/adapter.```

## 💬 Running the Chatbot
### Option 1 – Streamlit Web UI (recommended)
Launch the full graphical interface:
  ```bash
    streamlit run app.py
  ```
Then open your browser at ```http://localhost:8501```

**Features:**
- Adjust temperature and max tokens on the fly
- Clear conversation history
- System prompt that keeps the assistant “professional and concise”
- Runs locally – no API calls, no data leaves your machine

### Option 2 – Command‑line interface
For quick testing or headless use:
  ```bash 
    python inference.py
  ```
Type your questions, press Enter, and type ```quit``` to exit.


## ⚙️ Configuration

- **Model** – ```meta-llama/Llama-3.2-1B-Instruct``` (you can change to other Llama 3 models if you have enough memory)
- **Dataset** – ```hungryfoxz/LibraryFAQ1000``` (already formatted for Llama 3 chat template)
- **LoRA** – rank 16, alpha 32, dropout 0.05, applied to all attention and MLP layers
- **Training** – 3 epochs, batch size 1 with gradient accumulation 8 (effective batch 8)

All settings are easily modifiable at the top of each Python file.

## 📁 Project Structure
  ```.
    ├── app.py                 # Streamlit web app
    ├── finetune.py            # LoRA fine‑tuning script
    ├── inference.py           # Command‑line chat interface
    ├── requirements.txt       # Dependencies
    ├── README.md              # This file
    └── llama32_library_chatbot/
        └── adapter/           # Saved LoRA weights + tokenizer (created after training)
  ```

## 🧪 Example Questions

Once the chatbot is running, try asking:

- ***How many books can I borrow?***
- ***How do I get a library card?***
- ***What are the library opening hours?***
- ***How to return a book?***

The assistant answers based on the fine‑tuned dataset. If a question is out of scope, it falls back to general Llama 3 knowledge but remains helpful.

## ⚠️ Troubleshooting

- IProgress not found **warning** – Install ```ipywidgets``` if you want progress bars in Jupyter, but it’s safe to ignore.
- **Out of memory** – Reduce ```per_device_train_batch_size``` to 1 and/or ```max_new_tokens``` in the UI.
- **MPS errors** – If you encounter MPS‑specific issues, set ```DEVICE = "cpu"``` in ```app.py``` or ```inference.py```.
- **Adapter path not found** – Make sure you have run finetune.py at least once, or adjust ```ADAPTER_PATH``` to point to your existing adapter.

🤝 Acknowledgements
- Meta Llama 3 for the base model
- Hugging Face for ```transformers```,``` peft```, ```trl```
- Streamlit for the web UI framework
- Dataset: ```hungryfoxz/LibraryFAQ1000```

## 📄 License
This project is for educational and research purposes.
The base Llama model is subject to Meta’s Acceptable Use Policy.
The fine‑tuned weights (LoRA adapter) inherit the same license.

## 🙋 Contributing
Issues and pull requests are welcome. If you improve the training or the UI, feel free to share!

## Happy building! 📖✨
















