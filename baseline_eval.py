"""
baseline_eval.py
----------------
Kaggle notebook — Zero-shot baseline evaluation.

Runs LLaMA-3-8B and Mistral-7B on Nepali test data
WITHOUT any fine-tuning. Records scores as "before" numbers.

Run this on Kaggle with GPU T4 enabled.
Expected time: ~45 minutes total for both models.
"""

# 
# CELL 1 — Install packages
# 

# !pip install -q unsloth transformers datasets
# !pip install -q evaluate sacrebleu rouge-score

# 
# CELL 2 — Pull your repo from GitHub
# 

# !git clone https://github.com/YOUR_USERNAME/nepali-llm-benchmark
# %cd nepali-llm-benchmark

# 
# CELL 3 — HuggingFace login
# 

# from kaggle_secrets import UserSecretsClient
# secrets = UserSecretsClient()
# hf_token = secrets.get_secret("HF_TOKEN")
#
# from huggingface_hub import login
# login(token=hf_token)
# print("HuggingFace login successful")

# 
# CELL 4 — Imports
# 

import json
import sys
import time
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# add repo root to path so we can import our metrics
sys.path.append(".")
from evaluation.metrics import compute_metrics

# output dir for results
Path("results").mkdir(exist_ok=True)

print(f"GPU available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name      : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# 
# CELL 5 — Config
# 

# how many test examples to evaluate per task
# 100 is fast (~15 min per model) and statistically representative
EVAL_SAMPLES = 100

MODELS = {
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

TASKS = {
    "translation":   {
        "test_path":   "outputs/formatted/translation/test.jsonl",
        "input_field": "source",
        "ref_field":   "target",
        "metric":      "translation",
        "prompt_fn":   lambda src: (
            f"### Instruction:\nTranslate the following Nepali text to English."
            f"\n\n### Input:\n{src}\n\n### Response:\n"
        ),
    },
    "qa": {
        "test_path":   "outputs/formatted/qa/test.jsonl",
        "input_field": "question",
        "ref_field":   "answer",
        "metric":      "qa",
        "prompt_fn":   lambda q: (
            f"### Instruction:\nAnswer the following question in Nepali."
            f"\n\n### Question:\n{q}\n\n### Response:\n"
        ),
    },
    "summarization": {
        "test_path":   "outputs/formatted/summarization/test.jsonl",
        "input_field": "article",
        "ref_field":   "summary",
        "metric":      "summarization",
        "prompt_fn":   lambda art: (
            f"### Instruction:\nSummarize the following Nepali news article "
            f"in one or two sentences.\n\n### Article:\n{art}\n\n### Response:\n"
        ),
    },
}

print("Config loaded")
print(f"Eval samples per task: {EVAL_SAMPLES}")


# 
# CELL 6 — Load test data
# 

def load_test_data(path: str, n: int) -> list[dict]:
    """Load n examples from a JSONL file."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    # shuffle with fixed seed for reproducibility
    import random
    random.seed(42)
    random.shuffle(data)
    return data[:n]


test_data = {}
for task, cfg in TASKS.items():
    data = load_test_data(cfg["test_path"], EVAL_SAMPLES)
    test_data[task] = data
    print(f"  {task:<16} : {len(data)} test examples loaded")


# 
# CELL 7 — Model loader
# 

def load_model(model_name: str):
    """
    Load model in 4-bit quantization — fits on Kaggle T4 (16GB).
    Uses NF4 quantization same as QLoRA training will use.
    """
    print(f"\nLoading {model_name}...")
    print("  This takes 3-5 minutes — downloading ~14-16GB")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # important for generation

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()

    print(f"  Model loaded — {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
    return model, tokenizer


# 
# CELL 8 — Inference function
# 

def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """
    Generate model output for one prompt.
    Returns only the generated text (not the prompt).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,         # greedy — deterministic, reproducible
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # stop at ### if model continues into next section
    for stop in ["###", "\n\n\n"]:
        if stop in text:
            text = text[:text.index(stop)]

    return text.strip()


# 
# CELL 9 — Evaluate one model on all tasks
# 

def evaluate_model(model_key: str, model_name: str) -> dict:
    """
    Run zero-shot evaluation of one model on all three tasks.
    Returns results dict.
    """
    print(f"\n{'═'*55}")
    print(f"  Evaluating: {model_key.upper()}  ({model_name})")
    print(f"{'═'*55}")

    model, tokenizer = load_model(model_name)

    results = {
        "model":      model_name,
        "model_key":  model_key,
        "eval_type":  "zero_shot",
        "n_samples":  EVAL_SAMPLES,
        "tasks":      {},
        "evaluated_at": datetime.now().isoformat(),
    }

    for task, cfg in TASKS.items():
        print(f"\n  Task: {task}")
        data   = test_data[task]
        prompt_fn    = cfg["prompt_fn"]
        input_field  = cfg["input_field"]
        ref_field    = cfg["ref_field"]

        predictions = []
        references  = []
        errors      = 0

        for ex in tqdm(data, desc=f"    {task}"):
            try:
                input_text = ex.get(input_field, "")
                reference  = ex.get(ref_field,   "")

                if not input_text or not reference:
                    continue

                prompt     = prompt_fn(input_text)
                prediction = generate(model, tokenizer, prompt)

                predictions.append(prediction)
                references.append(reference)

            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"    Error on example: {e}")
                continue

        # compute metrics
        if predictions:
            scores = compute_metrics(cfg["metric"], predictions, references)
        else:
            scores = {"error": "no predictions generated"}

        results["tasks"][task] = {
            "scores":      scores,
            "n_evaluated": len(predictions),
            "n_errors":    errors,
        }

        print(f"    Scores : {scores}")
        print(f"    Evaluated {len(predictions)}/{len(data)} examples")

        # show 3 examples so you can sanity check outputs
        print(f"\n    Sample predictions:")
        for i, (pred, ref) in enumerate(zip(predictions[:3], references[:3])):
            print(f"    [{i+1}] Ref  : {ref[:80]}")
            print(f"         Pred : {pred[:80]}")
            print()

    # save after each model — don't lose results if session dies
    out_path = Path(f"results/baseline_{model_key}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved → {out_path}")

    # free GPU memory before loading next model
    del model
    torch.cuda.empty_cache()
    print("  GPU memory cleared")

    return results


# 
# CELL 10 — Run evaluation for both models
# 

all_results = {}

for model_key, model_name in MODELS.items():
    result = evaluate_model(model_key, model_name)
    all_results[model_key] = result


# CELL 11 — Print comparison table

print("\n" + "═"*65)
print("  BASELINE RESULTS — ZERO SHOT")
print("  (These are your BEFORE numbers)")
print("═"*65)

print(f"\n  {'Task':<16} {'Metric':<12} {'LLaMA':>10} {'Mistral':>10}")
print("  " + "─"*50)

metric_map = {
    "translation":   "bleu",
    "qa":            "f1",
    "summarization": "rouge_l",
}

for task, metric in metric_map.items():
    llama_score   = all_results.get("llama",   {}).get("tasks", {}).get(task, {}).get("scores", {}).get(metric, "N/A")
    mistral_score = all_results.get("mistral", {}).get("tasks", {}).get(task, {}).get("scores", {}).get(metric, "N/A")
    print(f"  {task:<16} {metric:<12} {str(llama_score):>10} {str(mistral_score):>10}")

print("\n  Save these numbers — you need them for your results chapter")
print("  Files: results/baseline_llama.json")
print("         results/baseline_mistral.json\n")

# save combined
with open("results/baseline_all.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print("  Combined → results/baseline_all.json")


