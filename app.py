"""
app.py — HuggingFace Spaces (Gradio)
--------------------------------------
Nepali LLM Benchmark Demo

Shows Base Model vs Fine-Tuned model side by side.
Base model uses Groq API (instant).
Fine-tuned model loads adapter from HuggingFace.
"""

import os
import torch
import gradio as gr

# ── API Keys from HF Spaces secrets ──────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

# ── Configs ───────────────────────────────────────────────────────────────────

GROQ_MODELS = {
    "LLaMA-3.1-8B": "llama-3.1-8b-instant",
    "Mistral-7B":  "mistral-saba-24b",
    
}

BASE_MODELS = {
    "LLaMA-3.1-8B": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B":   "unsloth/mistral-7b-instruct-v0.3",
}

ADAPTERS = {
    "LLaMA-3.1-8B": {
        "Translation":                         "iwasbinod/nepali-llama-translation-qlora",
        "QA":                                  "iwasbinod/nepali-llama-qa-qlora",
        "Summarization (HuggingFace data)":    "iwasbinod/nepali-llama-summarization-qlora",
        "Summarization (Self-Scraped data)":   "iwasbinod/nepali-llama-summarization-news-scraped-data-qlora",
    },
    "Mistral-7B": {
        "Translation":                         "iwasbinod/nepali-mistral-translation-qlora",
        "QA":                                  "iwasbinod/nepali-mistral-qa-qlora",
        "Summarization (HuggingFace data)":    "iwasbinod/nepali-mistral-summarization-qlora",
        "Summarization (Self-Scraped data)":   "iwasbinod/nepali-mistral-summarization-news_scrap",
    },
}

SYSTEM_PROMPTS = {
    "Translation":                       "You are a helpful assistant that translates Nepali text to English accurately. Provide only the translation, nothing else.",
    "QA":                                "You are a helpful assistant that answers questions in Nepali based only on the provided context. Be concise and accurate.",
    "Summarization (HuggingFace data)":  "You are a helpful assistant that summarizes Nepali news articles in one or two sentences. Write the summary in Nepali.",
    "Summarization (Self-Scraped data)": "You are a helpful assistant that summarizes Nepali news articles in one or two sentences. Write the summary in Nepali.",
}

INSTRUCTIONS = {
    "Translation":                       "Translate the following Nepali text to English.",
    "QA":                                "Answer the following question in Nepali.",
    "Summarization (HuggingFace data)":  "Summarize the following Nepali news article in one or two sentences.",
    "Summarization (Self-Scraped data)": "Summarize the following Nepali news article in one or two sentences.",
}

EXAMPLES = {
    "Translation":                       "नेपाल दक्षिण एसियामा अवस्थित एक सुन्दर देश हो। यहाँ विश्वको सर्वोच्च पर्वत सगरमाथा छ।",
    "QA":                                "नेपालको राजधानी काठमाडौं हो। नेपालको राजधानी के हो?",
    "Summarization (HuggingFace data)":  "काठमाडौं— नेपाल सरकारले आज नयाँ शिक्षा नीति घोषणा गर्‍यो। यस नीतिअन्तर्गत सबै विद्यालयमा निःशुल्क शिक्षा प्रदान गरिनेछ। शिक्षा मन्त्रीले यो नीतिले देशको साक्षरता दर बढाउन मद्दत गर्ने बताए।",
    "Summarization (Self-Scraped data)": "काठमाडौं— नेपाल सरकारले आज नयाँ शिक्षा नीति घोषणा गर्‍यो। यस नीतिअन्तर्गत सबै विद्यालयमा निःशुल्क शिक्षा प्रदान गरिनेछ। शिक्षा मन्त्रीले यो नीतिले देशको साक्षरता दर बढाउन मद्दत गर्ने बताए।",
}

RESULTS = {
    "LLaMA-3.1-8B": {
        "Translation":                       {"metric": "BLEU",    "before": 2.09, "after": 31.48, "pct": 1406},
        "QA":                                {"metric": "F1",      "before": 8.78, "after": 31.68, "pct": 261},
        "Summarization (HuggingFace data)":  {"metric": "ROUGE-L", "before": 7.16, "after": 32.81, "pct": 358},
        "Summarization (Self-Scraped data)": {"metric": "ROUGE-L", "before": 7.16, "after": "TBD", "pct": "TBD"},
    },
    "Mistral-7B": {
        "Translation":                       {"metric": "BLEU",    "before": 4.89, "after": 10.79, "pct": 121},
        "QA":                                {"metric": "F1",      "before": 6.97, "after": 13.36, "pct": 92},
        "Summarization (HuggingFace data)":  {"metric": "ROUGE-L", "before": 0.36, "after": 31.50, "pct": 8650},
        "Summarization (Self-Scraped data)": {"metric": "ROUGE-L", "before": 0.36, "after": "TBD", "pct": "TBD"},
    },
}


# ── Loaded model cache ────────────────────────────────────────────────────────

_model_cache = {}


def load_finetuned(base_model_name: str, adapter_repo: str):
    """Load fine-tuned model. Cached in memory after first load."""
    cache_key = adapter_repo
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading adapter: {adapter_repo}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, token=HF_TOKEN or None
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # device
    if torch.cuda.is_available():
        device = "cuda"
        dtype  = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype  = torch.float16
    else:
        device = "cpu"
        dtype  = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        token=HF_TOKEN or None,
    )
    model = PeftModel.from_pretrained(
        model, adapter_repo,
        torch_dtype=dtype,
        token=HF_TOKEN or None,
    )
    model = model.to(device)
    model.eval()

    _model_cache[cache_key] = (model, tokenizer, device)
    return model, tokenizer, device


def generate_ft(model, tokenizer, task, user_input, device) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[task]},
        {"role": "user",   "content": f"{INSTRUCTIONS[task]}\n\n{user_input}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

    for stop in ["###", "<|", "[INST]", "\n\n\n"]:
        if stop in response:
            response = response[:response.index(stop)]

    return response.strip()


def base_via_groq(model_choice, task, user_input) -> str:
    if not GROQ_API_KEY:
        return " GROQ_API_KEY not set in Space secrets."
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=GROQ_MODELS[model_choice],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user",   "content": f"{INSTRUCTIONS[task]}\n\n{user_input}"},
            ],
            max_tokens=200, temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq error: {e}"


# ── Main inference function ───────────────────────────────────────────────────

def run_comparison(model_choice, task, user_input):
    """
    Called when user clicks Compare button.
    Returns (base_output, finetuned_output, metrics_text)
    """
    if not user_input.strip():
        return "Please enter Nepali text.", "Please enter Nepali text.", ""

    # base model — Groq API
    base_out = base_via_groq(model_choice, task, user_input)

    # fine-tuned model
    adapter_repo    = ADAPTERS[model_choice][task]
    base_model_name = BASE_MODELS[model_choice]

    try:
        ft_model, ft_tok, device = load_finetuned(base_model_name, adapter_repo)
        ft_out = generate_ft(ft_model, ft_tok, task, user_input, device)
    except Exception as e:
        ft_out = f"Error loading adapter: {e}\nAdapter: {adapter_repo}"

    # metrics text
    r = RESULTS[model_choice][task]
    if r["after"] != "TBD":
        metrics = (
            f"  {r['metric']} Score | "
            f"Zero-Shot: {r['before']} → "
            f"Fine-Tuned: {r['after']} | "
            f"+{r['pct']}% improvement"
        )
    else:
        metrics = f"  {r['metric']} Score | Zero-Shot: {r['before']} → Fine-Tuned: Training in progress"

    return base_out, ft_out, metrics


def load_example(task):
    return EXAMPLES.get(task, "")


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Nepali LLM Benchmark",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("""
    # 🇳🇵 Nepali LLM Benchmark Demo
    **Fine-Tuning and Benchmarking Small Open-Source LLMs for Low-Resource Nepali NLP Tasks**
    BE Final Year Project — Binod Raj Pant (22070199) | NAST Dhangadhi | 2026
    """)

    gr.Markdown("""
    ---
    **How it works:**
    -   **Base Model** — original model with no Nepali training (via Groq API, instant)
    -   **Fine-Tuned Model** — trained on Nepali data using QLoRA (~3 min first load, cached after)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Radio(
                choices=["LLaMA-3.1-8B", "Mistral-7B"],
                value="LLaMA-3.1-8B",
                label="  Select Model",
            )
            task = gr.Radio(
                choices=list(ADAPTERS["LLaMA-3.1-8B"].keys()),
                value="Translation",
                label="  Select Task",
            )
            gr.Markdown("""
            **Task descriptions:**
            - Translation → Nepali to English
            - QA → Answer Nepali question
            - Summarization (HF data) → trained on 3000 examples
            - Summarization (Scraped) → trained on YOUR 383 scraped articles
            """)

        with gr.Column(scale=3):
            user_input = gr.Textbox(
                label="Enter Nepali Text",
                placeholder="Type or paste Nepali text here...",
                lines=5,
            )

            with gr.Row():
                example_btn = gr.Button("  Load Example", variant="secondary")
                run_btn     = gr.Button("⚡ Compare: Base vs Fine-Tuned", variant="primary")

            metrics_out = gr.Textbox(
                label="  Benchmark Results",
                interactive=False,
                lines=1,
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("###   Base Model (Zero-Shot)")
                    gr.Markdown("*No Nepali training — original model*")
                    base_out = gr.Textbox(
                        label="Base Model Output",
                        lines=5,
                        interactive=False,
                    )

                with gr.Column():
                    gr.Markdown("###   Fine-Tuned Model (QLoRA)")
                    gr.Markdown("*Trained on Nepali data — your adapter*")
                    ft_out = gr.Textbox(
                        label="Fine-Tuned Output",
                        lines=5,
                        interactive=False,
                    )

    # button actions
    run_btn.click(
        fn=run_comparison,
        inputs=[model_choice, task, user_input],
        outputs=[base_out, ft_out, metrics_out],
    )

    example_btn.click(
        fn=load_example,
        inputs=[task],
        outputs=[user_input],
    )

    # update task choices when model changes
    def update_tasks(model):
        tasks = list(ADAPTERS[model].keys())
        return gr.Radio(choices=tasks, value=tasks[0])

    model_choice.change(
        fn=update_tasks,
        inputs=[model_choice],
        outputs=[task],
    )

    gr.Markdown("""
    ---
     [GitHub](https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs) |
    🤗 [HuggingFace Models](https://huggingface.co/iwasbinod)
    """)


if __name__ == "__main__":
    demo.launch()