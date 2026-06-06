"""
qlora_trainer.py
----------------
QLoRA fine-tuning for Nepali NLP tasks using Unsloth.

FIXES IN THIS VERSION:
  - Uses UnslothTrainer instead of SFTTrainer (fixes AttributeError mean)
  - Removed warmup_ratio (deprecated) → uses warmup_steps instead
  - Removed logging_dir (deprecated)
  - Compatible with Unsloth 2026.x + Transformers 5.x

HOW TO USE ON KAGGLE:
  Cell 1: !pip install -q unsloth transformers datasets trl
  Cell 2: !git clone https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs
          %cd Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs
  Cell 3: from kaggle_secrets import UserSecretsClient
          from huggingface_hub import login
          login(token=UserSecretsClient().get_secret("HF_TOKEN"))
  Cell 4: MODEL = "llama"        # or "mistral"
          TASK  = "translation"  # or "qa" or "summarization"
          exec(open("qlora_trainer.py").read())

6 RUNS:
  MODEL="llama",   TASK="translation"
  MODEL="llama",   TASK="qa"
  MODEL="llama",   TASK="summarization"
  MODEL="mistral", TASK="translation"
  MODEL="mistral", TASK="qa"
  MODEL="mistral", TASK="summarization"
"""

import os
import sys
import json
import torch
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.append(".")
from evaluation.metrics import compute_metrics

Path("results").mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

if "MODEL" not in dir(): MODEL = "llama"
if "TASK"  not in dir(): TASK  = "translation"

print(f"\n{'═'*55}")
print(f"  QLoRA Fine-Tuning")
print(f"  Model : {MODEL}")
print(f"  Task  : {TASK}")
print(f"{'═'*55}\n")

MODEL_CONFIGS = {
    "llama": {
        "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "hf_org":     "iwasbinod",
    },
    "mistral": {
        "model_name": "unsloth/mistral-7b-instruct-v0.3",
        "hf_org":     "iwasbinod",
    },
}

TRAINING_CONFIG = {
    "translation": {
        "num_epochs":    3,
        "batch_size":    4,
        "grad_accum":    4,
        "lr":            2e-4,
        "max_seq_len":   256,
        "warmup_steps":  20,
    },
    "qa": {
        "num_epochs":    3,
        "batch_size":    2,
        "grad_accum":    8,
        "lr":            2e-4,
        "max_seq_len":   512,
        "warmup_steps":  20,
    },
    "summarization": {
        "num_epochs":    3,
        "batch_size":    2,
        "grad_accum":    8,
        "lr":            2e-4,
        "max_seq_len":   512,
        "warmup_steps":  20,
    },
}

model_cfg = MODEL_CONFIGS[MODEL]
train_cfg = TRAINING_CONFIG[TASK]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

print("STEP 1 — Loading base model...")

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_cfg["model_name"],
    max_seq_length=train_cfg["max_seq_len"],
    load_in_4bit=True,
    dtype=None,
)

print(f"  ✓ Base model loaded")
print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ADD LORA ADAPTERS
# ═══════════════════════════════════════════════════════════════════════════════

print("\nSTEP 2 — Adding LoRA adapters...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,           # 0 dropout — required for Unsloth fast path
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  ✓ LoRA adapters added")
print(f"  Trainable : {trainable/1e6:.1f}M / {total/1e9:.1f}B ({100*trainable/total:.2f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — LOAD DATASET
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nSTEP 3 — Loading {TASK} training data...")

from datasets import Dataset

TASK_PATHS = {
    "translation":   "outputs/formatted/translation/train.jsonl",
    "qa":            "outputs/formatted/qa/train.jsonl",
    "summarization": "outputs/formatted/summarization/train.jsonl",
}

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

raw_data = load_jsonl(TASK_PATHS[TASK])
print(f"  Loaded {len(raw_data)} examples")

# apply chat template — converts messages → model specific tokens
def format_sample(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_sample, desc="Applying chat template")

print(f"  ✓ Formatted")
print(f"  Sample: {dataset[0]['text'][:150].replace(chr(10),' ')}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nSTEP 4 — Training...")
print(f"  Epochs      : {train_cfg['num_epochs']}")
print(f"  Batch size  : {train_cfg['batch_size']} × {train_cfg['grad_accum']} = {train_cfg['batch_size']*train_cfg['grad_accum']} effective")
print(f"  LR          : {train_cfg['lr']}")
print(f"  Max seq len : {train_cfg['max_seq_len']}\n")

from unsloth import UnslothTrainer, UnslothTrainingArguments

output_dir = f"outputs/trained/{MODEL}_{TASK}"
Path(output_dir).mkdir(parents=True, exist_ok=True)

training_args = UnslothTrainingArguments(
    output_dir=output_dir,

    # epochs + batch
    num_train_epochs=train_cfg["num_epochs"],
    per_device_train_batch_size=train_cfg["batch_size"],
    gradient_accumulation_steps=train_cfg["grad_accum"],

    # learning rate
    learning_rate=train_cfg["lr"],
    warmup_steps=train_cfg["warmup_steps"],  # fixed steps, not ratio
    lr_scheduler_type="cosine",

    # precision — T4 uses fp16
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),

    # memory
    optim="adamw_8bit",

    # logging
    logging_steps=10,

    # saving
    save_strategy="epoch",
    save_total_limit=1,

    # misc
    seed=42,
    report_to="none",
    do_eval=False,
)

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=train_cfg["max_seq_len"],
    args=training_args,
    dataset_num_proc=2,
)

train_result = trainer.train()

print(f"\n  ✓ Training complete")
print(f"  Final loss : {train_result.training_loss:.4f}")
print(f"  Steps      : {train_result.global_step}")
print(f"  Time       : {train_result.metrics.get('train_runtime',0)/60:.1f} min")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nSTEP 5 — Evaluating on test set...")

FastLanguageModel.for_inference(model)

TEST_PATHS = {
    "translation":   "outputs/formatted/translation/test.jsonl",
    "qa":            "outputs/formatted/qa/test.jsonl",
    "summarization": "outputs/formatted/summarization/test.jsonl",
}

REF_FIELDS = {
    "translation":   "target",
    "qa":            "answer",
    "summarization": "summary",
}

SYSTEM_PROMPTS = {
    "translation":   "You are a helpful assistant that translates Nepali text to English accurately. Provide only the translation, nothing else.",
    "qa":            "You are a helpful assistant that answers questions in Nepali based only on the provided context. Be concise and accurate.",
    "summarization": "You are a helpful assistant that summarizes Nepali news articles in one or two sentences. Write the summary in Nepali.",
}

def build_user_message(task, ex):
    if task == "translation":
        return f"Translate the following Nepali text to English.\n\nNepali:\n{ex['source']}"
    elif task == "qa":
        ctx = ex.get("context","")
        q   = ex.get("question","")
        if ctx:
            return f"Read the following context carefully and answer the question.\n\nContext:\n{ctx}\n\nQuestion:\n{q}"
        return f"Answer the following question in Nepali.\n\nQuestion:\n{q}"
    elif task == "summarization":
        return f"Summarize the following Nepali news article in one or two sentences.\n\nArticle:\n{ex['article']}"

test_data = load_jsonl(TEST_PATHS[TASK])
random.seed(42)
random.shuffle(test_data)
test_data = test_data[:100]

predictions, references = [], []

for ex in tqdm(test_data, desc=f"  {TASK}"):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[TASK]},
            {"role": "user",   "content": build_user_message(TASK, ex)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=train_cfg["max_seq_len"]
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

        for stop in ["###", "<|", "[INST]", "\n\n\n"]:
            if stop in response:
                response = response[:response.index(stop)]

        ref = ex.get(REF_FIELDS[TASK],"").strip()
        if response.strip() and ref:
            predictions.append(response.strip())
            references.append(ref)

    except Exception as e:
        continue

scores = compute_metrics(TASK, predictions, references)
print(f"\n  ✓ Fine-tuned scores : {scores}")
print(f"  Evaluated {len(predictions)}/100 examples")

print(f"\n  Sample predictions:")
for i, (p, r) in enumerate(zip(predictions[:3], references[:3])):
    print(f"  [{i+1}] Ref  : {r[:70]}")
    print(f"       Pred : {p[:70]}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SAVE TO HUGGINGFACE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nSTEP 6 — Saving adapter to HuggingFace Hub...")

hf_repo = f"{model_cfg['hf_org']}/nepali-{MODEL}-{TASK}-qlora"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

try:
    model.push_to_hub(hf_repo)
    tokenizer.push_to_hub(hf_repo)
    print(f"  ✓ Uploaded → huggingface.co/{hf_repo}")
except Exception as e:
    print(f"  Upload failed: {e}")
    print(f"  Adapter saved locally → {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SAVE RESULTS + COMPARE WITH BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nSTEP 7 — Saving results...")

results = {
    "model":         model_cfg["model_name"],
    "model_key":     MODEL,
    "task":          TASK,
    "eval_type":     "fine_tuned",
    "scores":        scores,
    "n_evaluated":   len(predictions),
    "training_loss": train_result.training_loss,
    "train_steps":   train_result.global_step,
    "hf_adapter":    hf_repo,
    "trained_at":    datetime.now().isoformat(),
}

result_path = f"results/finetuned_{MODEL}_{TASK}.json"
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"  ✓ Results → {result_path}")

# compare with baseline
METRIC_MAP = {"translation":"bleu", "qa":"f1", "summarization":"rouge_l"}
metric = METRIC_MAP[TASK]

baseline_path = f"results/baseline_{MODEL}.json"
if Path(baseline_path).exists():
    with open(baseline_path) as f:
        baseline = json.load(f)
    before = baseline.get("tasks",{}).get(TASK,{}).get("scores",{}).get(metric,"N/A")
    after  = scores.get(metric, "N/A")

    print(f"\n{'─'*50}")
    print(f"  RESULT — {MODEL.upper()} + {TASK.upper()}")
    print(f"{'─'*50}")
    print(f"  Metric             : {metric}")
    print(f"  Before (zero-shot) : {before}")
    print(f"  After  (fine-tuned): {after}")
    if isinstance(before,(int,float)) and isinstance(after,(int,float)) and before > 0:
        print(f"  Improvement        : +{after-before:.2f} ({(after-before)/before*100:.0f}%)")

print(f"\n{'═'*55}")
print(f"  DONE: {MODEL.upper()} + {TASK.upper()}")
print(f"  Adapter : huggingface.co/{hf_repo}")
print(f"  Results : {result_path}")
print(f"{'═'*55}")

# print(f"""
#   Push results to GitHub:
#   import os
#   from kaggle_secrets import UserSecretsClient
#   GITHUB_TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
#   os.system('git add results/ outputs/trained/')
#   os.system('git commit -m "results: finetuned {MODEL} {TASK}"')
#   os.system(f'git push https://Just-Binod:{{GITHUB_TOKEN}}@github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs.git main')
# """)
