"""
formatter.py
------------
Converts raw JSONL files into messages format for QLoRA fine-tuning.

WHY MESSAGES FORMAT:
  Your models are LLaMA-3-8B-Instruct and Mistral-7B-Instruct.
  Both are "Instruct" models — trained with specific chat templates.
  Messages format lets Unsloth apply the correct template automatically:
    - LLaMA-3  → <|begin_of_text|><|start_header_id|>user<|end_header_id|>...
    - Mistral  → <s>[INST]...[/INST]...
  You write messages once, Unsloth handles both models correctly.

OUTPUT FORMAT per example:
  {
    "messages": [
      {"role": "system",    "content": "You are a helpful Nepali assistant."},
      {"role": "user",      "content": "Translate: नमस्ते"},
      {"role": "assistant", "content": "Hello"}   ← empty string for test
    ],
    "source":   "नमस्ते",     ← kept for evaluation
    "target":   "Hello",      ← kept for evaluation
    "task":     "translation",
    "lang":     "nepali"
  }

Run:
    python data/formatter.py --debug
    python data/formatter.py
"""

import json
import re
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict


# ── Config ────────────────────────────────────────────────────────────────────

MAX_WORDS = 400    # max words per example — keeps within GPU memory limits

TASKS = {
    "translation":   Path("outputs/flores"),
    "qa":            Path("outputs/xquad"),
    "summarization": Path("outputs/news"),
}

OUTPUT_DIR = Path("outputs/formatted")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# system prompt per task — tells the model its role
SYSTEM_PROMPTS = {
    "translation": (
        "You are a helpful assistant that translates Nepali text to English accurately. "
        "Provide only the translation, nothing else."
    ),
    "qa": (
        "You are a helpful assistant that answers questions in Nepali "
        "based only on the provided context. Be concise and accurate."
    ),
    "summarization": (
        "You are a helpful assistant that summarizes Nepali news articles "
        "in one or two sentences. Write the summary in Nepali."
    ),
}

DEVANAGARI = re.compile(r'[\u0900-\u097F]')


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"\n  File not found: {path}"
            f"\n  Run first:\n"
            f"\n    python data/load_flores.py"
            f"\n    python data/load_xquad.py"
            f"\n    python data/load_news.py\n"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def has_devanagari(text: str, min_chars: int = 5) -> bool:
    return len(DEVANAGARI.findall(text)) >= min_chars


def truncate(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate from messages list."""
    total_text = " ".join(m["content"] for m in messages)
    return len(total_text.split()) * 2


# ── Messages formatter — one function per task ────────────────────────────────

def format_translation(example: dict, for_training: bool) -> dict | None:
    """Nepali → English translation."""
    nepali  = str(example.get("source", "") or "").strip()
    english = str(example.get("target", "") or "").strip()

    if not nepali or not has_devanagari(nepali):
        return None
    if for_training and not english:
        return None

    user_content = f"Translate the following Nepali text to English.\n\nNepali:\n{nepali}"

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPTS["translation"]},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": english if for_training else ""},
        ],
        # keep raw fields for evaluation
        "source": nepali,
        "target": english,
        "task":   "translation",
        "lang":   "nepali",
    }


def format_qa(example: dict, for_training: bool) -> dict | None:
    """Nepali question answering."""
    context  = str(example.get("context",  "") or "").strip()
    question = str(example.get("question", "") or "").strip()
    answer   = str(example.get("answer",   "") or "").strip()

    if not question:
        return None
    if for_training and not answer:
        return None

    # truncate long contexts
    context = truncate(context, max_words=150)

    if context and has_devanagari(context):
        user_content = (
            f"Read the following context carefully and answer the question "
            f"based only on the information provided.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )
    else:
        user_content = f"Answer the following question in Nepali.\n\nQuestion:\n{question}"

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPTS["qa"]},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": answer if for_training else ""},
        ],
        "context":  context,
        "question": question,
        "answer":   answer,
        "task":     "qa",
        "lang":     "nepali",
    }


def format_summarization(example: dict, for_training: bool) -> dict | None:
    """Nepali news summarization."""
    article = str(example.get("article", "") or "").strip()
    summary = str(example.get("summary", "") or "").strip()

    if not article or not has_devanagari(article):
        return None
    if for_training and not summary:
        return None

    article = truncate(article, max_words=350)

    user_content = (
        f"Summarize the following Nepali news article "
        f"in one or two sentences.\n\nArticle:\n{article}"
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPTS["summarization"]},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": summary if for_training else ""},
        ],
        "article": article,
        "summary": summary,
        "task":    "summarization",
        "lang":    "nepali",
    }


# map task name → formatter function
FORMATTERS = {
    "translation":   format_translation,
    "qa":            format_qa,
    "summarization": format_summarization,
}


# ── Process one task ──────────────────────────────────────────────────────────

def process_task(task: str, data_dir: Path, debug: bool) -> dict:
    print(f"\n── {task.upper()} {'─'*(38-len(task))}")

    # load
    train_raw = load_jsonl(data_dir / "train.jsonl")
    test_raw  = load_jsonl(data_dir / "test.jsonl")

    if debug:
        train_raw = train_raw[:10]
        test_raw  = test_raw[:5]
        print(f"  [DEBUG] {len(train_raw)} train / {len(test_raw)} test")

    print(f"  Loaded   : {len(train_raw)} train / {len(test_raw)} test")

    formatter = FORMATTERS[task]

    # format
    train_out, test_out = [], []
    skipped = 0

    # for ex in train_raw:
    #     result = formatter(ex, for_training=True)
    #     if result:
    #         train_out.append(result)
    #     else:
    #         skipped += 1



    # 
    MAX_QA_TOKENS = 512

    for ex in train_raw:
        result = formatter(ex, for_training=True)
        if result:
            # for QA specifically, skip if still too long
            if task == "qa":
                tok_len = estimate_tokens(result["messages"])
                if tok_len > MAX_QA_TOKENS:
                    skipped += 1
                    continue
            train_out.append(result)
        else:
            skipped += 1

    for ex in test_raw:
        result = formatter(ex, for_training=False)
        if result:
            if task == "qa":
                tok_len = estimate_tokens(result["messages"])
                if tok_len > MAX_QA_TOKENS:
                    skipped += 1
                    continue
            test_out.append(result)
        else:
            skipped += 1

# updated code 



    # 

    for ex in test_raw:
        result = formatter(ex, for_training=False)
        if result:
            test_out.append(result)
        else:
            skipped += 1

    if skipped:
        print(f"  Skipped  : {skipped} invalid examples")

    print(f"  Clean    : {len(train_out)} train / {len(test_out)} test")

    # token stats
    token_lengths = [estimate_tokens(ex["messages"]) for ex in train_out]
    avg_tok = round(sum(token_lengths) / max(len(token_lengths), 1), 1)
    p95_tok = sorted(token_lengths)[int(len(token_lengths) * 0.95)] if token_lengths else 0
    max_tok = max(token_lengths) if token_lengths else 0
    recommended = min(p95_tok + 64, 2048)

    print(f"  Tokens   : avg={avg_tok}  p95={p95_tok}  max={max_tok}")
    print(f"  Use max_seq_length={recommended} in trainer")

    # save
    task_out = OUTPUT_DIR / task
    task_out.mkdir(exist_ok=True)

    # JSONL for inspection
    for split_name, split_data in [("train", train_out), ("test", test_out)]:
        path = task_out / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # HuggingFace Dataset arrow format — fastest to load on Kaggle
    ds = DatasetDict({
        "train": Dataset.from_list(train_out),
        "test":  Dataset.from_list(test_out),
    })
    ds.save_to_disk(str(task_out / "hf_dataset"))

    print(f"  Saved    : {task_out}/")

    # show sample — verify messages structure looks correct
    sample = train_out[0]
    print(f"\n  Messages structure:")
    for msg in sample["messages"]:
        role    = msg["role"]
        content = msg["content"][:80].replace("\n", " ")
        print(f"    [{role}]: {content}")
    print()

    return {
        "train":           len(train_out),
        "test":            len(test_out),
        "avg_tokens":      avg_tok,
        "p95_tokens":      p95_tok,
        "max_tokens":      max_tok,
        "max_seq_length":  recommended,
        "skipped":         skipped,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print("\n" + "═"*50)
    print("  Formatter — Messages Format for QLoRA")
    print("  Works with LLaMA-3 and Mistral automatically")
    print("═"*50)

    all_stats = {}
    for task, data_dir in TASKS.items():
        all_stats[task] = process_task(task, data_dir, args.debug)

    # save stats
    with open(OUTPUT_DIR / "format_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    # summary table
    print("\n" + "═"*60)
    print("  SUMMARY")
    print("═"*60)
    print(f"  {'Task':<16} {'Train':>6} {'Test':>6} {'AvgTok':>8} {'MaxSeqLen':>10}")
    print("  " + "─"*50)
    for task, s in all_stats.items():
        print(f"  {task:<16} {s['train']:>6} {s['test']:>6} "
              f"{s['avg_tokens']:>8} {s['max_seq_length']:>10}")

    print(f"\n  Format  : messages (works with LLaMA-3 + Mistral)")
    print(f"  Saved to: {OUTPUT_DIR}/")
    # print(f"\n  ✓ Ready for Kaggle training\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="10 examples only")
    main(parser.parse_args())