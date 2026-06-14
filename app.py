import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


# =====================================================
# CONFIGURATION
# =====================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Change this to your adapter directory
ADAPTER_PATH = "./llama32_library_chatbot/adapter"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Library Chatbot",
    page_icon="📚",
    layout="wide"
)

st.image(
    "gu_logo.jpeg",
    width=120
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.to(DEVICE)

    model.eval()

    return model, tokenizer


model, tokenizer = load_model()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("⚙️ Settings")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )

    max_tokens = st.slider(
        "Max New Tokens",
        min_value=50,
        max_value=500,
        value=250,
        step=25
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.markdown(
        """
        ### About

        **Model:** Llama-3.2-1B-Instruct (Fine-Tuned)

        **Domain:** Library Information Services

        **Deployment:** Local

        **Framework:** Streamlit

        **Hardware:** Apple Silicon
        """
    )


# =====================================================
# HEADER
# =====================================================

st.title("📚 Virtual Library Assistant")

st.markdown(
    """
    Ask questions related to:

    - Library membership
    - Library services
    - Library policies
    - General library information
    """
)


# =====================================================
# CHAT HISTORY
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =====================================================
# USER INPUT
# =====================================================

query = st.chat_input(
    "Ask a library-related question..."
)


# =====================================================
# GENERATE RESPONSE
# =====================================================

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful and professional library assistant. "
                "Answer user questions accurately, clearly, and concisely."
            )
        },
        {
            "role": "user",
            "content": query
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with st.chat_message("assistant"):

        with st.spinner("Generating response..."):

            with torch.no_grad():

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            full_response = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            assistant_response = full_response

            if query in full_response:
                assistant_response = full_response.split(query)[-1].strip()

            st.markdown(assistant_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response
        }
    )