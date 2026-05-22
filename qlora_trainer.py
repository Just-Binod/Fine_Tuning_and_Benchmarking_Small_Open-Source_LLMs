"""
qlora_trainer.py
----------------
QLoRA fine-tuning for Nepali NLP tasks using Unsloth.

WHAT THIS DOES:
  Fine-tunes LLaMA-3.1-8B or Mistral-7B on one Nepali task at a time.
  Uses QLoRA — trains only small adapter layers, not the whole model.
  Saves adapter weights to HuggingFace Hub after training.

WHY UNSLOTH:
  2x faster training than plain HuggingFace
  60% less GPU memory — critical for free Kaggle T4
  Same results as standard QLoRA

HOW TO USE ON KAGGLE:
  Run 6 separate notebooks — one per combination:
    Notebook 1: MODEL="llama"   TASK="translation"
    Notebook 2: MODEL="llama"   TASK="qa"
    Notebook 3: MODEL="llama"   TASK="summarization"
    Notebook 4: MODEL="mistral" TASK="translation"
    Notebook 5: MODEL="mistral" TASK="qa"
    Notebook 6: MODEL="mistral" TASK="summarization"

  Each notebook:
    Cell 1: !pip install unsloth transformers datasets tqdm
    Cell 2: !git clone https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs
            %cd Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs
    Cell 3: login to HuggingFace
    Cell 4: MODEL = "llama"        # or "mistral"
            TASK  = "translation"  # or "qa" or "summarization"
            exec(open("qlora_trainer.py").read())

EXPECTED TIME PER RUN:
  Translation   → ~2-3 hours on T4
  QA            → ~1.5-2 hours on T4
  Summarization → ~2-3 hours on T4

OUTPUT:
  Adapter saved to HuggingFace Hub:
    yourusername/nepali-llama-translation-qlora
    yourusername/nepali-llama-qa-qlora
    etc.
  Local checkpoint: outputs/trained/<model>_<task>/
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# ── make sure we can import our metrics ──────────────────────────────────────
sys.path.append(".")
from evaluation.metrics import compute_metrics


# 
# CONFIG — CHANGE THESE TWO LINES PER RUN
# 

# Set these before running:
# MODEL = "llama"    or  "mistral"
# TASK  = "translation"  or  "qa"  or  "summarization"

# If not set externally, defaults to llama + translation
if "MODEL" not in dir():
    MODEL = "llama"
if "TASK" not in dir():
    TASK = "translation"

print(f"\n{'═'*55}")
print(f"  QLoRA Fine-Tuning")
print(f"  Model : {MODEL}")
print(f"  Task  : {TASK}")
print(f"{'═'*55}\n")


# 
# MODEL CONFIGS
# 

MODEL_CONFIGS = {
    "llama": {
        "model_name":  "unsloth/Meta-Llama-3.1-8B-Instruct",
        "hf_org":      "Just-Binod",   # ← your HuggingFace username
        "max_seq_len": 2048,
    },
    "mistral": {
        "model_name":  "unsloth/mistral-7b-instruct-v0.3",
        "hf_org":      "Just-Binod",
        "max_seq_len": 2048,
    },
}

# ── QLoRA hyperparameters ────────────────────────────────────────────────────
# These are well-tested defaults for 7-8B models on low-resource tasks
# r=16 means we train 16-dimensional adapter matrices
# larger r = more capacity but more memory

LORA_CONFIG = {
    "r":              16,      # LoRA rank — 16 is standard for task adaptation
    "lora_alpha":     32,      # scaling factor — usually 2x rank
    "lora_dropout":   0.05,    # small dropout to prevent overfitting
    "target_modules": [        # which layers to add LoRA adapters to
        "q_proj",              # query projection — attention
        "k_proj",              # key projection — attention
        "v_proj",              # value projection — attention
        "o_proj",              # output projection — attention
        "gate_proj",           # feedforward network
        "up_proj",             # feedforward network
        "down_proj",           # feedforward network
    ],
    "bias":           "none",
    "use_rslora":     True,    # rank-stabilized LoRA — better than standard
}

# ── Training hyperparameters ──────────────────────────────────────────────────
TRAINING_CONFIG = {
    "translation": {
        "num_epochs":   3,
        "batch_size":   4,       # per device batch size
        "grad_accum":   4,       # effective batch = 4*4 = 16
        "lr":           2e-4,
        "max_seq_len":  256,     # translation sentences are short
        "warmup_ratio": 0.1,
    },
    "qa": {
        "num_epochs":   3,
        "batch_size":   2,       # smaller batch — QA examples are longer
        "grad_accum":   8,       # effective batch = 2*8 = 16
        "lr":           2e-4,
        "max_seq_len":  512,
        "warmup_ratio": 0.1,
    },
    "summarization": {
        "num_epochs":   3,
        "batch_size":   2,
        "grad_accum":   8,
        "lr":           2e-4,
        "max_seq_len":  512,
        "warmup_ratio": 0.1,
    },
}


# 
# STEP 1 — LOAD MODEL WITH UNSLOTH
# 

print("STEP 1 — Loading base model...")
print("  Installing unsloth if needed...")
os.system("pip install -q unsloth")

from unsloth import FastLanguageModel

model_cfg  = MODEL_CONFIGS[MODEL]
train_cfg  = TRAINING_CONFIG[TASK]
model_name = model_cfg["model_name"]

print(f"  Loading {model_name}...")
print(f"  This takes 3-5 minutes on first run...")

# Unsloth loads model in 4-bit NF4 quantization automatically
# much simpler than manual BitsAndBytesConfig
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=train_cfg["max_seq_len"],
    load_in_4bit=True,          # 4-bit quantization — fits on T4
    dtype=None,                 # auto-detect: bfloat16 on newer GPUs
)

print(f"  ✓ Base model loaded")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")


# 
# STEP 2 — ADD LORA ADAPTERS
# 

print("\nSTEP 2 — Adding LoRA adapters...")

# this adds small trainable matrices to the frozen base model
# instead of training 8B parameters, we train ~20-50M adapter parameters
# that's 99% less memory and compute
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_CONFIG["r"],
    lora_alpha=LORA_CONFIG["lora_alpha"],
    lora_dropout=LORA_CONFIG["lora_dropout"],
    target_modules=LORA_CONFIG["target_modules"],
    bias=LORA_CONFIG["bias"],
    use_rslora=LORA_CONFIG["use_rslora"],
    use_gradient_checkpointing="unsloth",  # saves memory during training
)

# count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  ✓ LoRA adapters added")
print(f"  Trainable params : {trainable/1e6:.1f}M / {total/1e9:.1f}B")
print(f"  Training ratio   : {100*trainable/total:.2f}%")


# 
# STEP 3 — LOAD AND PREPARE DATASET
# 

print(f"\nSTEP 3 — Loading {TASK} training data...")

import json
from datasets import Dataset

def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

TASK_DATA_PATHS = {
    "translation":   "outputs/formatted/translation/train.jsonl",
    "qa":            "outputs/formatted/qa/train.jsonl",
    "summarization": "outputs/formatted/summarization/train.jsonl",
}

train_path = TASK_DATA_PATHS[TASK]
if not Path(train_path).exists():
    raise FileNotFoundError(
        f"\nTraining data not found: {train_path}\n"
        f"Run on Mac first: python data/formatter.py\n"
        f"Then push to GitHub and pull here.\n"
    )

raw_data = load_jsonl(train_path)
print(f"  Loaded {len(raw_data)} training examples")

# apply chat template — converts messages list to model-specific format
# LLaMA-3  → <|begin_of_text|><|start_header_id|>...
# Mistral  → <s>[INST]...[/INST]...
# Unsloth handles this automatically via tokenizer.apply_chat_template

def format_for_training(example):
    """
    Convert messages list to training text using model's chat template.
    The template adds the correct special tokens for each model.
    """
    messages = example["messages"]

    # apply_chat_template formats messages into a single string
    # with the correct special tokens for this specific model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # False for training — include full response
    )
    return {"text": text}

# convert to HuggingFace Dataset
dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_for_training, desc="Applying chat template")

print(f"  ✓ Dataset formatted")
print(f"  Sample (first 200 chars):")
print(f"  {dataset[0]['text'][:200].replace(chr(10), ' ')}")


# 
# STEP 4 — TRAIN
# 

print(f"\nSTEP 4 — Starting QLoRA training...")
print(f"  Epochs     : {train_cfg['num_epochs']}")
print(f"  Batch size : {train_cfg['batch_size']} × {train_cfg['grad_accum']} accum = {train_cfg['batch_size']*train_cfg['grad_accum']} effective")
print(f"  LR         : {train_cfg['lr']}")
print(f"  Max seq len: {train_cfg['max_seq_len']}")

from trl import SFTTrainer
from transformers import TrainingArguments

output_dir = f"outputs/trained/{MODEL}_{TASK}"
Path(output_dir).mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,

    # epochs and batch
    num_train_epochs=train_cfg["num_epochs"],
    per_device_train_batch_size=train_cfg["batch_size"],
    gradient_accumulation_steps=train_cfg["grad_accum"],

    # learning rate with warmup + cosine decay
    learning_rate=train_cfg["lr"],
    warmup_ratio=train_cfg["warmup_ratio"],
    lr_scheduler_type="cosine",

    # memory optimizations
    fp16=not torch.cuda.is_bf16_supported(),   # fp16 on T4
    bf16=torch.cuda.is_bf16_supported(),        # bf16 on A100
    optim="adamw_8bit",                         # 8-bit optimizer saves memory
    gradient_checkpointing=True,

    # logging — watch loss go down
    logging_steps=10,
    logging_dir=f"{output_dir}/logs",

    # saving
    save_strategy="epoch",
    save_total_limit=1,          # only keep last checkpoint — saves disk space

    # evaluation
    do_eval=False,               # skip eval during training — saves time

    # reproducibility
    seed=42,

    # reporting
    report_to="none",            # disable wandb/tensorboard
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",   # the field we created in format_for_training
    max_seq_length=train_cfg["max_seq_len"],
    args=training_args,
    dataset_num_proc=2,
)

print(f"\n  Training started — watch loss decrease...")
print(f"  Steps per epoch: {len(dataset) // (train_cfg['batch_size'] * train_cfg['grad_accum'])}")
print(f"  Total steps    : {len(dataset) * train_cfg['num_epochs'] // (train_cfg['batch_size'] * train_cfg['grad_accum'])}\n")

train_result = trainer.train()

print(f"\n  ✓ Training complete")
print(f"  Final loss     : {train_result.training_loss:.4f}")
print(f"  Total steps    : {train_result.global_step}")
print(f"  Time taken     : {train_result.metrics.get('train_runtime', 0)/60:.1f} minutes")


# 
# STEP 5 — QUICK EVAL ON TEST SET
# 

print(f"\nSTEP 5 — Quick evaluation on test set...")

# switch model to inference mode — faster generation
FastLanguageModel.for_inference(model)

TASK_TEST_PATHS = {
    "translation":   "outputs/formatted/translation/test.jsonl",
    "qa":            "outputs/formatted/qa/test.jsonl",
    "summarization": "outputs/formatted/summarization/test.jsonl",
}

TASK_METRICS = {
    "translation":   "translation",
    "qa":            "qa",
    "summarization": "summarization",
}

REF_FIELDS = {
    "translation":   "target",
    "qa":            "answer",
    "summarization": "summary",
}

# evaluate on 100 examples (same as baseline)
test_data = load_jsonl(TASK_TEST_PATHS[TASK])
import random
random.seed(42)
random.shuffle(test_data)
test_data = test_data[:100]

SYSTEM_PROMPTS = {
    "translation": "You are a helpful assistant that translates Nepali text to English accurately. Provide only the translation, nothing else.",
    "qa":          "You are a helpful assistant that answers questions in Nepali based only on the provided context. Be concise and accurate.",
    "summarization": "You are a helpful assistant that summarizes Nepali news articles in one or two sentences. Write the summary in Nepali.",
}

def build_user_message(task, example):
    if task == "translation":
        return f"Translate the following Nepali text to English.\n\nNepali:\n{example['source']}"
    elif task == "qa":
        context  = example.get("context",  "")
        question = example.get("question", "")
        if context:
            return f"Read the following context carefully and answer the question based only on the information provided.\n\nContext:\n{context}\n\nQuestion:\n{question}"
        return f"Answer the following question in Nepali.\n\nQuestion:\n{question}"
    elif task == "summarization":
        return f"Summarize the following Nepali news article in one or two sentences.\n\nArticle:\n{example['article']}"

predictions = []
references  = []

from tqdm import tqdm

for ex in tqdm(test_data, desc=f"  Evaluating {TASK}"):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[TASK]},
            {"role": "user",   "content": build_user_message(TASK, ex)},
        ]

        # apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=train_cfg["max_seq_len"],
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

        for stop in ["###", "<|", "[INST]", "\n\n\n"]:
            if stop in response:
                response = response[:response.index(stop)]

        ref = ex.get(REF_FIELDS[TASK], "").strip()
        if response.strip() and ref:
            predictions.append(response.strip())
            references.append(ref)

    except Exception as e:
        continue

scores = compute_metrics(TASK_METRICS[TASK], predictions, references)
print(f"\n  ✓ Fine-tuned scores: {scores}")
print(f"  Evaluated {len(predictions)}/100 examples")

# show 3 samples
print(f"\n  Sample predictions:")
for i, (pred, ref) in enumerate(zip(predictions[:3], references[:3])):
    print(f"  [{i+1}] Ref  : {ref[:70]}")
    print(f"       Pred : {pred[:70]}")
    print()


# 
# STEP 6 — SAVE ADAPTER TO HUGGINGFACE HUB
# 

print(f"\nSTEP 6 — Saving adapter to HuggingFace Hub...")

hf_repo_name = f"{model_cfg['hf_org']}/nepali-{MODEL}-{TASK}-qlora"
print(f"  Uploading to: {hf_repo_name}")

# save adapter weights + tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# push to HuggingFace Hub
try:
    model.push_to_hub(hf_repo_name)
    tokenizer.push_to_hub(hf_repo_name)
    print(f"  ✓ Adapter uploaded → huggingface.co/{hf_repo_name}")
except Exception as e:
    print(f"  Upload failed: {e}")
    print(f"  Adapter saved locally → {output_dir}")
    print(f"  You can upload manually later")


# 
# STEP 7 — SAVE RESULTS
# 

print(f"\nSTEP 7 — Saving results...")

Path("results").mkdir(exist_ok=True)

results = {
    "model":          model_name,
    "model_key":      MODEL,
    "task":           TASK,
    "eval_type":      "fine_tuned",
    "scores":         scores,
    "n_evaluated":    len(predictions),
    "training_loss":  train_result.training_loss,
    "train_steps":    train_result.global_step,
    "hf_adapter":     hf_repo_name,
    "lora_config":    LORA_CONFIG,
    "train_config":   train_cfg,
    "trained_at":     datetime.now().isoformat(),
}

result_path = f"results/finetuned_{MODEL}_{TASK}.json"
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"  ✓ Results saved → {result_path}")

# compare with baseline
baseline_path = f"results/baseline_{MODEL}.json"
if Path(baseline_path).exists():
    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_scores = (
        baseline.get("tasks", {})
        .get(TASK, {})
        .get("scores", {})
    )

    print(f"\n{'─'*50}")
    print(f"  COMPARISON — {MODEL.upper()} {TASK.upper()}")
    print(f"{'─'*50}")

    metric_map = {
        "translation":   "bleu",
        "qa":            "f1",
        "summarization": "rouge_l",
    }
    metric = metric_map[TASK]

    before = baseline_scores.get(metric, "N/A")
    after  = scores.get(metric, "N/A")

    print(f"  Before (zero-shot) : {before}")
    print(f"  After  (fine-tuned): {after}")

    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        improvement = after - before
        pct         = (improvement / max(before, 0.01)) * 100
        print(f"  Improvement        : +{improvement:.2f} ({pct:.0f}%)")

print(f"\n{'─'*55}")
print(f"  RUN COMPLETE: {MODEL.upper()} + {TASK.upper()}")
print(f"  Adapter : huggingface.co/{hf_repo_name}")
print(f"  Results : {result_path}")
print(f"{'-'*55}\n")

print("  Push results to GitHub:")
print("  git add results/ && git commit -m 'results: finetuned_{MODEL}_{TASK}'")
print("  git push ...")