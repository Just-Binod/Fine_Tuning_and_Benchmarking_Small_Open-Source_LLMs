

#
"""
qlora_trainer.py  (v2)
----------------------
QLoRA fine-tuning for Nepali NLP tasks using Unsloth.

CHANGES FROM v1:
  [BUG FIX] predictions.append / references.append were commented out →
             benchmark scores were computed on empty lists. Now fixed.
  [FIX]     max_seq_len for summarization: 512 → 768
             Nepali articles were truncated mid-sentence during training,
             causing broken/incomplete output at inference.
  [FIX]     lora_dropout: 0 → 0.05
             Zero dropout caused overfitting on small Nepali datasets.
  [FIX]     temperature in eval: 0.65 → 0.3-0.4
             High temperature caused mixed-language and hallucinated output.
  [FIX]     Removed min_new_tokens from eval
             Was forcing generation past natural EOS → garbage tokens.
  [NEW]     HF_REPO_SUFFIX support — set "-v2" to push to new repo,
             keeping old adapter safe as backup on HuggingFace.
  [NEW]     HYPERPARAMS_OVERRIDE dict — patch config from notebook cell
             without editing this file.
  [NEW]     Validation split (90/10) + load_best_model_at_end=True
             Saves best checkpoint, not last. Prevents catastrophic forgetting.
  [NEW]     DandaStoppingCriteria for summarization eval
             Stops after 2 Nepali dandas (।) — clean 1-2 sentence summaries.
  [NEW]     clean_output() + is_output_valid() — task-aware cleaning.
  [NEW]     Score comparison vs previous fine-tuned at end of run.
  Compatible with Unsloth 2026.x + Transformers 5.x

HOW TO USE ON KAGGLE:
  Cell 1: !pip install -q unsloth transformers datasets trl
  Cell 2: !git clone https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs
          %cd Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs
  Cell 3: from kaggle_secrets import UserSecretsClient
          from huggingface_hub import login
          login(token=UserSecretsClient().get_secret("HF_TOKEN"))

  Original 6 runs (unchanged):
    MODEL="llama",   TASK="translation"
    MODEL="llama",   TASK="qa"
    MODEL="llama",   TASK="summarization"
    MODEL="mistral", TASK="translation"
    MODEL="mistral", TASK="qa"
    MODEL="mistral", TASK="summarization"
    exec(open("qlora_trainer.py").read())

  Retraining runs (v2 — keeps old adapter safe):
    MODEL="llama";   TASK="summarization"; HF_REPO_SUFFIX="-v2"
    MODEL="mistral"; TASK="translation";   HF_REPO_SUFFIX="-v2"
    exec(open("qlora_trainer.py").read())
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import re
import json
import gc
import torch
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.append(".")
from evaluation.metrics import compute_metrics

Path("results").mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

if "MODEL"          not in dir(): MODEL          = "llama"
if "TASK"           not in dir(): TASK           = "translation"
if "HF_REPO_SUFFIX" not in dir(): HF_REPO_SUFFIX = ""  # set "-v2" for retraining

print(f"\n{'═'*55}")
print(f"  QLoRA Fine-Tuning  {'(RETRAINING v2)' if HF_REPO_SUFFIX else '(v2)'}")
print(f"  Model  : {MODEL}")
print(f"  Task   : {TASK}")
if HF_REPO_SUFFIX:
    print(f"  Suffix : '{HF_REPO_SUFFIX}' ← new repo, old adapter stays safe")
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

# ── Training config (v2 values) ─────────────────────────────────
TRAINING_CONFIG = {
    "translation": {
        "num_epochs":   3,
        "batch_size":   2,
        "grad_accum":   8,
        "lr":           8e-5,
        "max_seq_len":  512,
        "warmup_steps": 40,
    },
    "qa": {
        "num_epochs":   3,
        "batch_size":   2,
        "grad_accum":   8,
        "lr":           8e-5,
        "max_seq_len":  512,
        "warmup_steps": 40,
    },
    "summarization": {
        "num_epochs":   4,       # v2: was 3
        "batch_size":   1,       # v2: was 2 — reduced to fit 768 seq_len
        "grad_accum":   16,      # v2: was 8 — keeps effective batch=16
        "lr":           1.2e-4,  # v2: was 8e-5
        "max_seq_len":  768,     # v2: was 512 — KEY FIX for Nepali truncation
        "warmup_steps": 60,      # v2: was 40
    },
}

# LoRA rank
LORA_RANK = 24 if TASK == "translation" else 16

model_cfg = MODEL_CONFIGS[MODEL]
train_cfg  = TRAINING_CONFIG[TASK]

# ── Task-specific overrides (same as v1 for translation/qa) ─────
if TASK == "translation":
    train_cfg["num_epochs"]   = 5
    train_cfg["lr"]           = 1.5e-4
    train_cfg["max_seq_len"]  = 512
    train_cfg["batch_size"]   = 2
    train_cfg["grad_accum"]   = 8
    train_cfg["warmup_steps"] = 60    # v2: was 40

# ── Apply external override if set from notebook ─────────────────
if "HYPERPARAMS_OVERRIDE" in dir() and isinstance(HYPERPARAMS_OVERRIDE, dict):
    train_cfg.update(HYPERPARAMS_OVERRIDE)
    print(f"  Hyperparams overridden:")
    for k, v in HYPERPARAMS_OVERRIDE.items():
        print(f"    {k} = {v}")
    print()

print(f"  Training config:")
for k, v in train_cfg.items():
    print(f"    {k:15s} = {v}")
print(f"    {'lora_rank':15s} = {LORA_RANK}")
print()


# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD MODEL
# ═══════════════════════════════════════════════════════════════

print("STEP 1 — Loading base model...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = model_cfg["model_name"],
    max_seq_length = train_cfg["max_seq_len"],
    load_in_4bit   = True,
    dtype          = None,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"  ✓ Loaded: {model_cfg['model_name']}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — ADD LORA ADAPTERS
# ═══════════════════════════════════════════════════════════════

print("\nSTEP 2 — Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    lora_alpha     = LORA_RANK * 2,
    lora_dropout   = 0.05,    # v2: was 0 — prevents overfitting
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias           = "none",
    use_rslora     = True,
    use_gradient_checkpointing = "unsloth",
    random_state   = 42,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  ✓ LoRA added (r={LORA_RANK}, dropout=0.05)")
print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e9:.1f}B ({100*trainable/total:.2f}%)")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — LOAD DATASET
# ═══════════════════════════════════════════════════════════════

print(f"\nSTEP 3 — Loading {TASK} data...")
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

# Shuffle with fixed seed
random.seed(42)
random.shuffle(raw_data)

# v2: 90/10 train/val split — enables eval during training
val_size   = max(50, int(len(raw_data) * 0.10))
val_data   = raw_data[:val_size]
train_data = raw_data[val_size:]
print(f"  Train: {len(train_data)}  |  Val: {len(val_data)}")

def format_sample(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_dataset = Dataset.from_list(train_data)
train_dataset = train_dataset.map(format_sample, desc="Formatting train")

val_dataset = Dataset.from_list(val_data)
val_dataset = val_dataset.map(format_sample, desc="Formatting val")

print(f"  ✓ Formatted")
print(f"  Sample: {train_dataset[0]['text'][:120].replace(chr(10),' ')}")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — TRAIN
# ═══════════════════════════════════════════════════════════════

print(f"\nSTEP 4 — Training...")
print(f"  Epochs     : {train_cfg['num_epochs']}")
print(f"  Eff. batch : {train_cfg['batch_size']} × {train_cfg['grad_accum']} = {train_cfg['batch_size']*train_cfg['grad_accum']}")
print(f"  LR         : {train_cfg['lr']}")
print(f"  Max seq len: {train_cfg['max_seq_len']}")

from unsloth import UnslothTrainer, UnslothTrainingArguments

output_dir = f"outputs/trained/{MODEL}_{TASK}{HF_REPO_SUFFIX}"
Path(output_dir).mkdir(parents=True, exist_ok=True)

training_args = UnslothTrainingArguments(
    output_dir                  = output_dir,
    num_train_epochs            = train_cfg["num_epochs"],
    per_device_train_batch_size = train_cfg["batch_size"],
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = train_cfg["grad_accum"],
    learning_rate               = train_cfg["lr"],
    warmup_steps                = train_cfg["warmup_steps"],
    lr_scheduler_type           = "cosine",
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    optim                       = "adamw_8bit",
    logging_steps               = 10,
    # v2: eval + save best checkpoint
    eval_strategy               = "steps",
    eval_steps                  = 50,
    save_strategy               = "steps",
    save_steps                  = 50,
    save_total_limit            = 2,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    seed                        = 42,
    report_to                   = "none",
    do_eval                     = True,
)

trainer = UnslothTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_dataset,
    eval_dataset       = val_dataset,    # v2: added
    dataset_text_field = "text",
    max_seq_length     = train_cfg["max_seq_len"],
    args               = training_args,
    dataset_num_proc   = 2,
)

print("\n  Watch the logs:")
print("  train_loss + eval_loss both decreasing = good")
print("  eval_loss stable, train_loss drops = acceptable")
print("   eval_loss rising = overfitting (best ckpt still saved)\n")

train_result = trainer.train()

print(f"\n  ✓ Training complete")
print(f"  Final train loss: {train_result.training_loss:.4f}")
print(f"  Steps: {train_result.global_step}")
print(f"  Time : {train_result.metrics.get('train_runtime',0)/60:.1f} min")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — EVALUATE
# ═══════════════════════════════════════════════════════════════

print(f"\nSTEP 5 — Evaluating on test set...")

del trainer
gc.collect()
torch.cuda.empty_cache()

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
    "translation": (
        "You are a professional Nepali-English translator.\n"
        "Translate the given Nepali text into natural, fluent, and accurate English.\n"
        "Preserve the original meaning exactly. Use natural English expressions.\n"
        "Do not add any extra information, explanations, or notes. "
        "Output only the translation."
    ),
    "qa": (
        "You are an accurate and concise assistant.\n"
        "Answer the question in Nepali based only on the provided context.\n"
        "If the answer is not in the context, say "
        "\"माफ गर्नुहोस्, दिइएको सन्दर्भमा यो प्रश्नको जवाफ उपलब्ध छैन।\"\n"
        "Do not make up information."
    ),
    "summarization": (
        "You are an expert Nepali news summarizer.\n"
        "Summarize the given Nepali news article in 1 to 2 clear and concise "
        "sentences in Nepali.\n"
        "Capture the main points and key information. "
        "Do not add your own opinions."
    ),
}


def build_user_message(task, ex):
    if task == "translation":
        return (
            f"Translate the following Nepali text to natural, fluent English.\n\n"
            f"Nepali:\n{ex['source']}\n\nEnglish:"
        )
    elif task == "qa":
        ctx = ex.get("context", "")
        q   = ex.get("question", "")
        if ctx:
            return (
                f"Read the following context carefully and answer the question.\n\n"
                f"Context:\n{ctx}\n\nQuestion:\n{q}"
            )
        return f"Answer the following question in Nepali.\n\nQuestion:\n{q}"
    else:  # summarization
        return (
            f"Summarize the following Nepali news article in one or two sentences."
            f"\n\nArticle:\n{ex['article']}"
        )


# ── v2: DandaStoppingCriteria for summarization ─────────────────
from transformers import StoppingCriteria, StoppingCriteriaList

class DandaStoppingCriteria(StoppingCriteria):
    """Stop after max_sentences Nepali dandas (।)."""
    def __init__(self, tokenizer, max_sentences=2):
        self.danda_ids = set()
        for i in range(len(tokenizer)):
            try:
                if "।" in tokenizer.decode([i], skip_special_tokens=True):
                    self.danda_ids.add(i)
            except Exception:
                continue
        self.count = 0
        self.max_sentences = max_sentences

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0, -1].item() in self.danda_ids:
            self.count += 1
        return self.count >= self.max_sentences


# ── v2: clean_output ────────────────────────────────────────────
def clean_output(text: str, task: str) -> str:
    if not text:
        return ""
    stops = [
        "###", "<|end_of_text|>", "<|eot_id|>", "<|assistant|>",
        "English:", "Or, in English", "Article:", "Summarize", "[INST]", "<|",
    ]
    for stop in stops:
        if stop in text:
            text = text.split(stop)[0].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text  = "\n".join(lines)
    if task == "translation":
        text = re.sub(r'[\u0900-\u097F]+', '', text)
        text = " ".join(text.split())
    return text.strip()


# ── v2: is_output_valid ─────────────────────────────────────────
def is_output_valid(text: str, task: str) -> bool:
    if not text or len(text) < 5:
        return False
    if task == "summarization":
        return any('\u0900' <= c <= '\u097F' for c in text)
    if task == "translation":
        return any('a' <= c.lower() <= 'z' for c in text)
    return True


# ── Load test data ───────────────────────────────────────────────
test_data = load_jsonl(TEST_PATHS[TASK])
random.seed(42)
random.shuffle(test_data)
test_data = test_data[:100]

# ── v2: task-specific generation params ─────────────────────────
if TASK == "translation":
    GEN_KWARGS = dict(
        max_new_tokens    = 150,
        temperature       = 0.3,    # v2: was 0.65
        top_p             = 0.92,
        do_sample         = True,
        repetition_penalty= 1.1,    # v2: was 1.08
    )
    stopping_criteria = None

elif TASK == "qa":
    GEN_KWARGS = dict(
        max_new_tokens    = 120,
        temperature       = 0.4,    # v2: was 0.65
        top_p             = 0.92,
        do_sample         = True,
        repetition_penalty= 1.1,
    )
    stopping_criteria = None

else:  # summarization
    GEN_KWARGS = dict(
        max_new_tokens    = 150,    # v2: was 256
        temperature       = 0.35,   # v2: was 0.65
        top_p             = 0.92,
        do_sample         = True,
        repetition_penalty= 1.1,
    )
    # v2: stop at 2 natural Nepali sentence endings
    danda_stopper     = DandaStoppingCriteria(tokenizer, max_sentences=2)
    stopping_criteria = StoppingCriteriaList([danda_stopper])


# ── Evaluation loop ──────────────────────────────────────────────
# v2: FIXED — predictions/references now correctly appended
predictions, references = [], []

for idx, ex in enumerate(tqdm(test_data, desc=f"  Evaluating {TASK}")):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[TASK]},
            {"role": "user",   "content": build_user_message(TASK, ex)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = train_cfg["max_seq_len"],
        ).to(model.device)

        gen_kwargs = {
            **GEN_KWARGS,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria
            danda_stopper.count = 0   # reset per example

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        response   = clean_output(response, TASK)

        # v2: FIXED — this was commented out in v1
        ref = ex.get(REF_FIELDS[TASK], "").strip()
        if is_output_valid(response, TASK) and ref:
            predictions.append(response)
            references.append(ref)

    except Exception as e:
        if idx < 5:
            print(f"  ⚠ Example {idx} failed: {e}")
        continue

    if (idx + 1) % 20 == 0:
        torch.cuda.empty_cache()


# ── Compute + print scores ───────────────────────────────────────
scores = compute_metrics(TASK, predictions, references)

print(f"\n  ✓ Fine-tuned scores : {scores}")
print(f"  Evaluated {len(predictions)}/100 examples")

if len(predictions) < 50:
    print(f"  ⚠ Only {len(predictions)} valid predictions — check outputs above")

print(f"\n  Sample predictions:")
for i, (p, r) in enumerate(zip(predictions[:3], references[:3])):
    print(f"  [{i+1}] Ref  : {r[:80]}")
    print(f"       Pred : {p[:80]}")
    print()


# ═══════════════════════════════════════════════════════════════
# STEP 6 — SAVE TO HUGGINGFACE
# ═══════════════════════════════════════════════════════════════

print(f"\nSTEP 6 — Pushing to HuggingFace...")

hf_repo = f"{model_cfg['hf_org']}/nepali-{MODEL}-{TASK}-qlora{HF_REPO_SUFFIX}"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

if HF_REPO_SUFFIX:
    old_repo = f"{model_cfg['hf_org']}/nepali-{MODEL}-{TASK}-qlora"
    print(f"  New repo : huggingface.co/{hf_repo}")
    print(f"  Old repo : huggingface.co/{old_repo} ← UNTOUCHED")
else:
    print(f"  Repo: huggingface.co/{hf_repo}")

try:
    model.push_to_hub(hf_repo)
    tokenizer.push_to_hub(hf_repo)
    print(f"  ✓ Uploaded → huggingface.co/{hf_repo}")
except Exception as e:
    print(f"  ✗ Upload failed: {e}")
    print(f"  Saved locally → {output_dir}")


# ═══════════════════════════════════════════════════════════════
# STEP 7 — SAVE RESULTS + COMPARE
# ═══════════════════════════════════════════════════════════════

print(f"\nSTEP 7 — Saving results...")

results = {
    "model":         model_cfg["model_name"],
    "model_key":     MODEL,
    "task":          TASK,
    "eval_type":     f"fine_tuned{HF_REPO_SUFFIX}",
    "scores":        scores,
    "n_evaluated":   len(predictions),
    "training_loss": train_result.training_loss,
    "train_steps":   train_result.global_step,
    "hf_adapter":    hf_repo,
    "hyperparams":   train_cfg,
    "lora_rank":     LORA_RANK,
    "trained_at":    datetime.now().isoformat(),
}

result_path = f"results/finetuned_{MODEL}_{TASK}{HF_REPO_SUFFIX}.json"
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"  ✓ Results → {result_path}")

METRIC_MAP = {"translation": "bleu", "qa": "f1", "summarization": "rouge_l"}
metric     = METRIC_MAP[TASK]

# Compare vs baseline
baseline_path = f"results/baseline_{MODEL}.json"
if Path(baseline_path).exists():
    with open(baseline_path) as f:
        baseline = json.load(f)
    before = baseline.get("tasks", {}).get(TASK, {}).get("scores", {}).get(metric, "N/A")
    after  = scores.get(metric, "N/A")
    print(f"\n{'─'*55}")
    print(f"  RESULT — {MODEL.upper()} + {TASK.upper()}")
    print(f"{'─'*55}")
    print(f"  Metric             : {metric}")
    print(f"  Before (zero-shot) : {before}")
    print(f"  After  (fine-tuned): {after}")
    if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before > 0:
        pct = (after - before) / before * 100
        sym = "✅" if after > before else ""
        print(f"  Improvement        : {sym} {after-before:+.2f} ({pct:+.0f}%)")

# Compare vs previous fine-tuned (only for retraining)
if HF_REPO_SUFFIX:
    prev_path = f"results/finetuned_{MODEL}_{TASK}.json"
    if Path(prev_path).exists():
        with open(prev_path) as f:
            prev = json.load(f)
        prev_score = prev.get("scores", {}).get(metric, "N/A")
        new_score  = scores.get(metric, "N/A")
        print(f"\n  VS PREVIOUS FINE-TUNED:")
        print(f"  Old adapter score : {prev_score}")
        print(f"  New v2 score      : {new_score}")
        if isinstance(prev_score, (int,float)) and isinstance(new_score, (int,float)):
            diff = new_score - prev_score
            if diff > 0:
                print(f"  v2 is BETTER by {diff:+.2f} → update demo notebook:")
                print(f'     "{TASK}": "{hf_repo}"')
            else:
                print(f"   v2 is WORSE by {diff:.2f} → keep old adapter, do NOT update demo")

print(f"\n{'═'*55}")
print(f"  DONE: {MODEL.upper()} + {TASK.upper()}{HF_REPO_SUFFIX.upper()}")
print(f"  Adapter : huggingface.co/{hf_repo}")
print(f"  Results : {result_path}")
print(f"{'═'*55}")


#






# """
# qlora_trainer.py
# ----------------
# QLoRA fine-tuning for Nepali NLP tasks using Unsloth.

# VERSION NOTES:
#   - Uses UnslothTrainer instead of SFTTrainer (fixes AttributeError mean)
#   - Removed warmup_ratio (deprecated) -> uses warmup_steps instead
#   - Removed logging_dir (deprecated)
#   - Compatible with Unsloth 2026.x + Transformers 5.x
#   - Translation task uses larger LoRA rank (32) + longer context (512) for
#     better quality, with batch_size/grad_accum rebalanced to avoid OOM
#   - expandable_segments + cache clearing added to prevent CUDA OOM /
#     fragmentation on ~15GB GPUs (T4-class)

# HOW TO USE ON KAGGLE:
#   Cell 1: !pip install -q unsloth transformers datasets trl
#   Cell 2: !git clone https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs
#           %cd Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs
#   Cell 3: from kaggle_secrets import UserSecretsClient
#           from huggingface_hub import login
#           login(token=UserSecretsClient().get_secret("HF_TOKEN"))
#   Cell 4: MODEL = "llama"        # or "mistral"
#           TASK  = "translation"  # or "qa" or "summarization"
#           exec(open("qlora_trainer.py").read())

# 6 RUNS:
#   MODEL="llama",   TASK="translation"
#   MODEL="llama",   TASK="qa"
#   MODEL="llama",   TASK="summarization"
#   MODEL="mistral", TASK="translation"
#   MODEL="mistral", TASK="qa"
#   MODEL="mistral", TASK="summarization"
# """

# import os
# # Must be set before torch initializes CUDA — reduces OOM from fragmentation
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# import sys
# import json
# import gc
# import torch
# import random
# from pathlib import Path
# from datetime import datetime
# from tqdm import tqdm

# sys.path.append(".")
# from evaluation.metrics import compute_metrics

# Path("results").mkdir(exist_ok=True)


# # 
# # CONFIG
# # 

# if "MODEL" not in dir(): MODEL = "llama"
# if "TASK"  not in dir(): TASK  = "translation"

# print(f"\n{'═'*55}")
# print(f"  QLoRA Fine-Tuning")
# print(f"  Model : {MODEL}")
# print(f"  Task  : {TASK}")
# print(f"{'═'*55}\n")

# MODEL_CONFIGS = {
#     "llama": {
#         "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",
#         "hf_org":     "iwasbinod",
#     },
#     "mistral": {
#         "model_name": "unsloth/mistral-7b-instruct-v0.3",
#         "hf_org":     "iwasbinod",
#     },
# }

# # TRAINING_CONFIG = {
# #     "translation": {
# #         "num_epochs":    3,
# #         "batch_size":    4,
# #         "grad_accum":    4,
# #         "lr":            2e-4,
# #         "max_seq_len":   256,
# #         "warmup_steps":  20,
# #     },
# #     "qa": {
# #         "num_epochs":    3,
# #         "batch_size":    2,
# #         "grad_accum":    8,
# #         "lr":            2e-4,
# #         "max_seq_len":   512,
# #         "warmup_steps":  20,
# #     },
# #     "summarization": {
# #         "num_epochs":    3,
# #         "batch_size":    2,
# #         "grad_accum":    8,
# #         "lr":            2e-4,
# #         "max_seq_len":   512,
# #         "warmup_steps":  20,
# #     },
# # }

# TRAINING_CONFIG = {
#     "translation": {
#         "num_epochs":    3,
#         "batch_size":    2,
#         "grad_accum":    8,
#         "lr":            8e-5,
#         "max_seq_len":   512,
#         "warmup_steps":  40,
#     },
#     "qa": {
#         "num_epochs":    3,
#         "batch_size":    2,
#         "grad_accum":    8,
#         "lr":            8e-5,
#         "max_seq_len":   512,
#         "warmup_steps":  40,
#     },
#     "summarization": {
#         "num_epochs":    3,
#         "batch_size":    2,
#         "grad_accum":    8,
#         "lr":            8e-5,
#         "max_seq_len":   512,
#         "warmup_steps":  40,
#     },
# }
# # LoRA
# LORA_RANK = 24 if TASK == "translation" else 16

# model_cfg = MODEL_CONFIGS[MODEL]
# train_cfg = TRAINING_CONFIG[TASK]

# # Task-specific overrides.
# # Translation gets a bigger LoRA rank + longer context for better quality.
# # Both increase activation/gradient memory, so batch_size is cut and
# # grad_accum raised to keep the same effective batch size (16) without OOM.
# if TASK == "translation":
#     train_cfg["num_epochs"]  = 5
#     train_cfg["lr"]          = 1.5e-4
#     train_cfg["max_seq_len"] = 512
#     train_cfg["batch_size"]  = 2
#     train_cfg["grad_accum"]  = 8

# # LORA_RANK = 32 if TASK == "translation" else 16

# # 
# # STEP 1 — LOAD MODEL
# # 

# print("STEP 1 — Loading base model...")

# from unsloth import FastLanguageModel

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=model_cfg["model_name"],
#     max_seq_length=train_cfg["max_seq_len"],
#     load_in_4bit=True,
#     dtype=None,
# )

# print(f"  ✓ Base model loaded")
# print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")


# # 
# # STEP 2 — ADD LORA ADAPTERS
# # 

# print("\nSTEP 2 — Adding LoRA adapters...")

# model = FastLanguageModel.get_peft_model(
#     model,
#     r=LORA_RANK,
#     lora_alpha=LORA_RANK * 2,
#     lora_dropout=0,           # 0 dropout — required for Unsloth fast path
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ],
#     bias="none",
#     use_rslora=True,
#     use_gradient_checkpointing="unsloth",
# )

# trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# total     = sum(p.numel() for p in model.parameters())
# print(f"  ✓ LoRA adapters added (r={LORA_RANK})")
# print(f"  Trainable : {trainable/1e6:.1f}M / {total/1e9:.1f}B ({100*trainable/total:.2f}%)")


# # 
# # STEP 3 — LOAD DATASET
# # 

# print(f"\nSTEP 3 — Loading {TASK} training data...")

# from datasets import Dataset

# TASK_PATHS = {
#     "translation":   "outputs/formatted/translation/train.jsonl",
#     "qa":            "outputs/formatted/qa/train.jsonl",
#     "summarization": "outputs/formatted/summarization/train.jsonl",
# }

# def load_jsonl(path):
#     with open(path, encoding="utf-8") as f:
#         return [json.loads(l) for l in f if l.strip()]

# raw_data = load_jsonl(TASK_PATHS[TASK])
# print(f"  Loaded {len(raw_data)} examples")

# # apply chat template — converts messages → model specific tokens
# def format_sample(example):
#     text = tokenizer.apply_chat_template(
#         example["messages"],
#         tokenize=False,
#         add_generation_prompt=False,
#     )
#     return {"text": text}

# dataset = Dataset.from_list(raw_data)
# dataset = dataset.map(format_sample, desc="Applying chat template")

# print(f"  ✓ Formatted")
# print(f"  Sample: {dataset[0]['text'][:150].replace(chr(10),' ')}")


# # 
# # STEP 4 — TRAIN
# # 

# print(f"\nSTEP 4 — Training...")
# print(f"  Epochs      : {train_cfg['num_epochs']}")
# print(f"  Batch size  : {train_cfg['batch_size']} × {train_cfg['grad_accum']} = {train_cfg['batch_size']*train_cfg['grad_accum']} effective")
# print(f"  LR          : {train_cfg['lr']}")
# print(f"  Max seq len : {train_cfg['max_seq_len']}\n")

# from unsloth import UnslothTrainer, UnslothTrainingArguments

# output_dir = f"outputs/trained/{MODEL}_{TASK}"
# Path(output_dir).mkdir(parents=True, exist_ok=True)

# training_args = UnslothTrainingArguments(
#     output_dir=output_dir,

#     # epochs + batch
#     num_train_epochs=train_cfg["num_epochs"],
#     per_device_train_batch_size=train_cfg["batch_size"],
#     gradient_accumulation_steps=train_cfg["grad_accum"],

#     # learning rate
#     learning_rate=train_cfg["lr"],
#     warmup_steps=train_cfg["warmup_steps"],  # fixed steps, not ratio
#     lr_scheduler_type="cosine",

#     # precision — T4 uses fp16
#     fp16=not torch.cuda.is_bf16_supported(),
#     bf16=torch.cuda.is_bf16_supported(),

#     # memory
#     optim="adamw_8bit",

#     # logging
#     logging_steps=10,

#     # saving
#     save_strategy="epoch",
#     save_total_limit=1,

#     # misc
#     seed=42,
#     report_to="none",
#     do_eval=False,
# )

# trainer = UnslothTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     max_seq_length=train_cfg["max_seq_len"],
#     args=training_args,
#     dataset_num_proc=2,
# )

# train_result = trainer.train()

# print(f"\n  ✓ Training complete")
# print(f"  Final loss : {train_result.training_loss:.4f}")
# print(f"  Steps      : {train_result.global_step}")
# print(f"  Time       : {train_result.metrics.get('train_runtime',0)/60:.1f} min")


# # 
# # STEP 5 — EVALUATE
# # 

# print(f"\nSTEP 5 — Evaluating on test set...")

# # Free memory left over from training (optimizer states, gradients, cached
# # activations) before switching to inference mode — prevents OOM/fragmentation
# # carrying over into the generation loop below.
# del trainer
# gc.collect()
# torch.cuda.empty_cache()

# FastLanguageModel.for_inference(model)

# TEST_PATHS = {
#     "translation":   "outputs/formatted/translation/test.jsonl",
#     "qa":            "outputs/formatted/qa/test.jsonl",
#     "summarization": "outputs/formatted/summarization/test.jsonl",
# }

# REF_FIELDS = {
#     "translation":   "target",
#     "qa":            "answer",
#     "summarization": "summary",
# }

# SYSTEM_PROMPTS = {
#     "translation": """You are a professional Nepali-English translator.
# Translate the given Nepali text into natural, fluent, and accurate English.
# Preserve the original meaning exactly. Use natural English expressions.
# Do not add any extra information, explanations, or notes. Output only the translation.""",

#     "qa": """You are an accurate and concise assistant.
# Answer the question in Nepali based **only** on the provided context.
# If the answer is not in the context, say "माफ गर्नुहोस्, दिइएको सन्दर्भमा यो प्रश्नको जवाफ उपलब्ध छैन।"
# Do not make up information.""",

#     "summarization": """You are an expert Nepali news summarizer.
# Summarize the given Nepali news article in **1 to 2 clear and concise sentences** in Nepali.
# Capture the main points and key information. Do not add your own opinions.""",
# }

# def build_user_message(task, ex):
#     if task == "translation":
#         return f"""Translate the following Nepali text to natural, fluent English.

# Nepali:
# {ex['source']}

# English:"""
#     elif task == "qa":
#         ctx = ex.get("context", "")
#         q   = ex.get("question", "")
#         if ctx:
#             return f"Read the following context carefully and answer the question.\n\nContext:\n{ctx}\n\nQuestion:\n{q}"
#         return f"Answer the following question in Nepali.\n\nQuestion:\n{q}"
#     elif task == "summarization":
#         return f"Summarize the following Nepali news article in one or two sentences.\n\nArticle:\n{ex['article']}"


# test_data = load_jsonl(TEST_PATHS[TASK])
# random.seed(42)
# random.shuffle(test_data)
# test_data = test_data[:100]

# predictions, references = [], []

# for idx, ex in enumerate(tqdm(test_data, desc=f"  {TASK}")):
#     try:
#         messages = [
#             {"role": "system", "content": SYSTEM_PROMPTS[TASK]},
#             {"role": "user",   "content": build_user_message(TASK, ex)},
#         ]
#         prompt = tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )


#         inputs = tokenizer(
#             prompt, return_tensors="pt",
#             truncation=True, max_length=512
#         ).to(model.device)

#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=256,
#                 min_new_tokens=20,
#                 temperature=0.65,
#                 top_p=0.92,
#                 do_sample=True,
#                 repetition_penalty=1.08,
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#             )

#         new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
#         response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

#         # Strong cleaning
#         stops = ["###", "<|end_of_text|>", "<|eot_id|>", "<|assistant|>", "English:", "Or, in English"]
#         for stop in stops:
#             if stop in response:
#                 response = response.split(stop)[0].strip()

#         response = response.replace("English:", "").strip()

#         # inputs = tokenizer(
#         #     prompt, return_tensors="pt",
#         #     truncation=True, max_length=train_cfg["max_seq_len"]
#         # ).to(model.device)

#         # with torch.no_grad():
#         #     outputs = model.generate(
#         #         **inputs,
#         #         #max_new_tokens=256,
#         #         #############
#         #         max_new_tokens=128,

#         #         ############
#         #         temperature=0.3,
#         #         top_p=0.9,
#         #         do_sample=True,
#         #         repetition_penalty=1.1,
#         #         pad_token_id=tokenizer.eos_token_id,
#         #     )

#         # new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
#         # response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

#         # for stop in ["###", "<|", "[INST]", "\n\n\n"]:
#         #     if stop in response:
#         #         response = response[:response.index(stop)]

#         # ref = ex.get(REF_FIELDS[TASK], "").strip()
#         # if response.strip() and ref:
#         #     predictions.append(response.strip())
#         #     references.append(ref)

#     except Exception:
#         continue

#     # Periodically clear cache to prevent fragmentation buildup over many
#     # generate() calls.
#     if (idx + 1) % 20 == 0:
#         torch.cuda.empty_cache()

# scores = compute_metrics(TASK, predictions, references)
# print(f"\n  ✓ Fine-tuned scores : {scores}")
# print(f"  Evaluated {len(predictions)}/100 examples")

# print(f"\n  Sample predictions:")
# for i, (p, r) in enumerate(zip(predictions[:3], references[:3])):
#     print(f"  [{i+1}] Ref  : {r[:70]}")
#     print(f"       Pred : {p[:70]}")
#     print()


# # 
# # STEP 6 — SAVE TO HUGGINGFACE
# # 

# print(f"\nSTEP 6 — Saving adapter to HuggingFace Hub...")

# hf_repo = f"{model_cfg['hf_org']}/nepali-{MODEL}-{TASK}-qlora"

# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# try:
#     model.push_to_hub(hf_repo)
#     tokenizer.push_to_hub(hf_repo)
#     print(f"   Uploaded → huggingface.co/{hf_repo}")
# except Exception as e:
#     print(f"  Upload failed: {e}")
#     print(f"  Adapter saved locally → {output_dir}")


# # 
# # STEP 7 — SAVE RESULTS + COMPARE WITH BASELINE
# # 

# print(f"\nSTEP 7 — Saving results...")

# results = {
#     "model":         model_cfg["model_name"],
#     "model_key":     MODEL,
#     "task":          TASK,
#     "eval_type":     "fine_tuned",
#     "scores":        scores,
#     "n_evaluated":   len(predictions),
#     "training_loss": train_result.training_loss,
#     "train_steps":   train_result.global_step,
#     "hf_adapter":    hf_repo,
#     "trained_at":    datetime.now().isoformat(),
# }

# result_path = f"results/finetuned_{MODEL}_{TASK}.json"
# with open(result_path, "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=2, ensure_ascii=False)

# print(f"  ✓ Results → {result_path}")

# # compare with baseline
# METRIC_MAP = {"translation": "bleu", "qa": "f1", "summarization": "rouge_l"}
# metric = METRIC_MAP[TASK]

# baseline_path = f"results/baseline_{MODEL}.json"
# if Path(baseline_path).exists():
#     with open(baseline_path) as f:
#         baseline = json.load(f)
#     before = baseline.get("tasks", {}).get(TASK, {}).get("scores", {}).get(metric, "N/A")
#     after  = scores.get(metric, "N/A")

#     print(f"\n{'─'*50}")
#     print(f"  RESULT — {MODEL.upper()} + {TASK.upper()}")
#     print(f"{'─'*50}")
#     print(f"  Metric             : {metric}")
#     print(f"  Before (zero-shot) : {before}")
#     print(f"  After  (fine-tuned): {after}")
#     if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before > 0:
#         print(f"  Improvement        : +{after-before:.2f} ({(after-before)/before*100:.0f}%)")

# print(f"\n{'═'*55}")
# print(f"  DONE: {MODEL.upper()} + {TASK.upper()}")
# print(f"  Adapter : huggingface.co/{hf_repo}")
# print(f"  Results : {result_path}")
# print(f"{'═'*55}")