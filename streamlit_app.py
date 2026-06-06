"""
streamlit_app.py
----------------
Nepali LLM Benchmark Demo

UI Flow:
  1. User selects model (LLaMA or Mistral)
  2. User selects task (Translation / QA / Summarization)
  3. User enters Nepali text
  4. Clicks Compare
  5. Sees Base Model vs Fine-Tuned output side by side
  6. Sees metric improvement

Run:
    pip install streamlit transformers peft torch accelerate bitsandbytes
    streamlit run app/streamlit_app.py
"""

import os
import warnings

# Suppress transformers + torchvision warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", message=".*torchvision.*")

import transformers
transformers.logging.set_verbosity_error()

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Nepali LLM Benchmark",
    page_icon="🇳🇵",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .base-output {
        background-color: #2d1b1b;
        border: 1px solid #ff4444;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .ft-output {
        background-color: #1b2d1b;
        border: 1px solid #44ff44;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-card {
        text-align: center;
        padding: 10px;
    }
    .improvement-badge {
        background-color: #1a4a1a;
        color: #44ff44;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ── Configs ───────────────────────────────────────────────────────────────────

BASE_MODELS = {
    "LLaMA-3.1-8B": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B":   "unsloth/mistral-7b-instruct-v0.3",
}

ADAPTERS = {
    "LLaMA-3.1-8B": {
        "Translation":   "iwasbinod/nepali-llama-translation-qlora",
        "Summarization": "iwasbinod/nepali-llama-summarization-qlora",
        "QA":            "iwasbinod/nepali-llama-qa-qlora",
    },
    "Mistral-7B": {
        "Translation":   "iwasbinod/nepali-mistral-translation-qlora",
        "Summarization": "iwasbinod/nepali-mistral-summarization-qlora",
        "QA":            "iwasbinod/nepali-mistral-qa-qlora",
    },
}

SYSTEM_PROMPTS = {
    "Translation":   "You are a helpful assistant that translates Nepali text to English accurately. Provide only the translation, nothing else.",
    "Summarization": "You are a helpful assistant that summarizes Nepali news articles in one or two sentences. Write the summary in Nepali.",
    "QA":            "You are a helpful assistant that answers questions in Nepali based only on the provided context. Be concise and accurate.",
}

INSTRUCTIONS = {
    "Translation":   "Translate the following Nepali text to English.",
    "Summarization": "Summarize the following Nepali news article in one or two sentences.",
    "QA":            "Answer the following question in Nepali.",
}

EXAMPLES = {
    "Translation": (
        "नेपाल दक्षिण एसियामा अवस्थित एक सुन्दर देश हो। "
        "यहाँ विश्वको सर्वोच्च पर्वत सगरमाथा छ।"
    ),
    "Summarization": (
        "काठमाडौं— नेपाल सरकारले आज नयाँ शिक्षा नीति घोषणा गर्‍यो। "
        "यस नीतिअन्तर्गत सबै विद्यालयमा निःशुल्क शिक्षा प्रदान गरिनेछ। "
        "शिक्षा मन्त्रीले यो नीतिले देशको साक्षरता दर बढाउन मद्दत गर्ने बताए। "
        "यस नीतिको कार्यान्वयन आगामी शैक्षिक सत्रदेखि हुनेछ।"
    ),
    "QA": (
        "नेपालको राजधानी काठमाडौं हो र यो देशको सबैभन्दा ठूलो सहर पनि हो।"
    ),
}

# benchmark results for display
RESULTS = {
    "LLaMA-3.1-8B": {
        "Translation":   {"metric": "BLEU",     "before": 2.09,  "after": 31.48, "pct": 1406},
        "QA":            {"metric": "F1",        "before": 8.78,  "after": 31.68, "pct": 261},
        "Summarization": {"metric": "ROUGE-L",   "before": 7.16,  "after": 32.81, "pct": 358},
    },
    "Mistral-7B": {
        "Translation":   {"metric": "BLEU",     "before": 4.89,  "after": 10.79, "pct": 121},
        "QA":            {"metric": "F1",        "before": 6.97,  "after": 13.36, "pct": 92},
        "Summarization": {"metric": "ROUGE-L",   "before": 0.36,  "after": 31.50, "pct": 8650},
    },
}


# ── Model loader ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_base_model(base_model_name: str):
    """Load base model only — no adapter."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_finetuned_model(base_model_name: str, adapter_repo: str):
    """Load base model + fine-tuned adapter."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_repo)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, task: str, user_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[task]},
        {"role": "user",   "content": f"{INSTRUCTIONS[task]}\n\n{user_input}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=1024,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

    for stop in ["###", "<|", "[INST]", "\n\n\n"]:
        if stop in response:
            response = response[:response.index(stop)]

    return response.strip()


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🇳🇵 Nepali LLM Benchmark Demo")
st.caption(
    "Fine-Tuning and Benchmarking Small Open-Source LLMs "
    "for Low-Resource Nepali NLP Tasks"
)
st.caption(
    "BE Final Year Project — Binod Raj Pant (22070199) | "
    "NAST Dhangadhi | 2026"
)
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    model_choice = st.radio(
        "🤖 Select Model:",
        ["LLaMA-3.1-8B", "Mistral-7B"],
        help="LLaMA-3.1-8B generally performs better on Nepali tasks"
    )

    task = st.radio(
        "📋 Select Task:",
        ["Translation", "Summarization", "QA"],
        captions=[
            "Nepali → English",
            "Article → Short summary",
            "Answer from context",
        ]
    )

    st.divider()

    # show benchmark results
    st.markdown("### 📊 Benchmark Results")
    r = RESULTS[model_choice][task]
    col1, col2 = st.columns(2)
    col1.metric(
        f"Zero-Shot\n{r['metric']}",
        f"{r['before']}",
        delta=None,
    )
    col2.metric(
        f"Fine-Tuned\n{r['metric']}",
        f"{r['after']}",
        delta=f"+{r['pct']}%",
    )

    st.markdown(
        f"<div style='text-align:center'>"
        f"<span class='improvement-badge'>+{r['pct']}% improvement</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("### 🔗 Models on HuggingFace")
    adapter = ADAPTERS[model_choice][task]
    st.markdown(f"[{adapter}](https://huggingface.co/{adapter})")

    st.divider()
    st.markdown("### 📁 GitHub")
    st.markdown("[View Source Code](https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs)")


# ── Main content ──────────────────────────────────────────────────────────────

# task description
task_descriptions = {
    "Translation":   "Enter Nepali text — Base Model vs Fine-Tuned translation to English",
    "Summarization": "Enter a Nepali news article — Base Model vs Fine-Tuned summary",
    "QA":            "Enter a Nepali question — Base Model vs Fine-Tuned answer",
}
st.subheader(f"Task: {task} — {model_choice}")
st.caption(task_descriptions[task])

# example button
col_input, col_example = st.columns([4, 1])
with col_example:
    if st.button("📝 Load Example", use_container_width=True):
        st.session_state["input_text"] = EXAMPLES[task]

with col_input:
    user_input = st.text_area(
        "Enter Nepali text:",
        value=st.session_state.get("input_text", EXAMPLES[task]),
        height=150,
        key="text_input",
        placeholder="Type or paste Nepali text here...",
    )

# compare button
compare_btn = st.button(
    "⚡ Compare: Base Model vs Fine-Tuned",
    type="primary",
    use_container_width=True,
)

# ── Output ────────────────────────────────────────────────────────────────────

if compare_btn and user_input.strip():

    base_model_name = BASE_MODELS[model_choice]
    adapter_repo    = ADAPTERS[model_choice][task]
    r               = RESULTS[model_choice][task]

    col_base, col_ft = st.columns(2)

    # ── LEFT: Base Model ──────────────────────────────────────────────────────
    with col_base:
        st.markdown(f"""
        <div style='text-align:center; padding:8px;
             background:#1a0000; border-radius:8px; margin-bottom:10px;'>
            <h4 style='color:#ff6666; margin:0;'>🔴 Base Model</h4>
            <small style='color:#aaa;'>{model_choice} — No Nepali training</small><br>
            <small style='color:#aaa;'>{r['metric']}: {r['before']}</small>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Loading base model..."):
            try:
                base_model, base_tok = load_base_model(base_model_name)
            except Exception as e:
                st.error(f"Failed to load: {e}")
                st.stop()

        with st.spinner("Generating..."):
            base_out = generate(base_model, base_tok, task, user_input)

        st.markdown(
            f"<div class='base-output'>{base_out}</div>",
            unsafe_allow_html=True
        )
        st.caption(f"Zero-shot output — no fine-tuning on Nepali data")

    # ── RIGHT: Fine-Tuned Model ───────────────────────────────────────────────
    with col_ft:
        st.markdown(f"""
        <div style='text-align:center; padding:8px;
             background:#001a00; border-radius:8px; margin-bottom:10px;'>
            <h4 style='color:#66ff66; margin:0;'>🟢 Fine-Tuned Model</h4>
            <small style='color:#aaa;'>{model_choice} + QLoRA Nepali adapter</small><br>
            <small style='color:#aaa;'>{r['metric']}: {r['after']} (+{r['pct']}%)</small>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Loading fine-tuned model..."):
            try:
                ft_model, ft_tok = load_finetuned_model(
                    base_model_name, adapter_repo
                )
            except Exception as e:
                st.error(f"Failed to load adapter: {e}")
                st.info(
                    f"Make sure adapter exists at: "
                    f"huggingface.co/{adapter_repo}"
                )
                st.stop()

        with st.spinner("Generating..."):
            ft_out = generate(ft_model, ft_tok, task, user_input)

        st.markdown(
            f"<div class='ft-output'>{ft_out}</div>",
            unsafe_allow_html=True
        )
        st.caption(
            f"Fine-tuned with QLoRA on Nepali data — "
            f"adapter: {adapter_repo}"
        )

    # ── Metric improvement bar ────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model",          model_choice)
    m2.metric("Task",           task)
    m3.metric(f"Zero-Shot {r['metric']}", r["before"])
    m4.metric(
        f"Fine-Tuned {r['metric']}",
        r["after"],
        delta=f"+{r['pct']}% improvement"
    )

    st.info(
        f"📊 Fine-tuning improved {r['metric']} score from "
        f"**{r['before']}** to **{r['after']}** — "
        f"a **+{r['pct']}% improvement** on 100 Nepali test examples."
    )

elif compare_btn:
    st.warning("Please enter some Nepali text first.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style='text-align:center; color:gray; font-size:12px;'>
    Nepali LLM Benchmark | BE Final Year Project | NAST Dhangadhi 2026<br>
    LLaMA-3.1-8B + Mistral-7B fine-tuned with QLoRA on Nepali NLP tasks<br>
    <a href='https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs'>
    GitHub</a> |
    <a href='https://huggingface.co/iwasbinod'>HuggingFace</a>
</div>
""", unsafe_allow_html=True)