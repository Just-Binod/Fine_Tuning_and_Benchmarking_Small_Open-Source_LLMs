"""
formatter.py
------------
Converts raw JSONL files into model-ready format for QLoRA fine-tuning.

What this does:
  - Loads train/test JSONL from outputs/
  - Verifies Devanagari text is present
  - Checks token lengths (important for Colab memory limits)
  - Saves combined dataset ready to pass to trainer

Run:
    python data/formatter.py --debug
    python data/formatter.py
"""

import json
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict


# ── Config ────────────────────────────────────────────────────────────────────

MAX_TOKENS_ESTIMATE = 512   # rough word-based estimate before tokenization
                             # real tokenizer check happens in trainer

TASKS = {
    "translation":   Path("outputs/flores"),
    "qa":            Path("outputs/xquad"),
    "summarization": Path("outputs/news"),
}

OUTPUT_DIR = Path("outputs/formatted")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"\n  File not found: {path}"
            f"\n  Run the data loader first:\n"
            f"\n  python data/load_flores.py"
            f"\n  python data/load_xquad.py"
            f"\n  python data/load_news.py\n"
        )
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate without loading a tokenizer.
    Nepali Devanagari tokenizes at ~3-4 chars per token on average.
    English tokenizes at ~4-5 chars per token.
    We use word count * 2 as a safe upper bound.
    """
    return len(text.split()) * 2


def validate_and_filter(examples: list[dict], task: str, max_tokens: int) -> tuple[list, dict]:
    """
    Validates examples and filters out bad ones.
    Returns (clean_examples, stats_dict).
    """
    clean     = []
    skipped   = {"empty": 0, "too_long": 0, "no_devanagari": 0}

    import re
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')

    for ex in examples:
        text = ex.get("text", "")

        # skip empty
        if not text.strip():
            skipped["empty"] += 1
            continue

        # skip if no Devanagari at all
        if len(devanagari_pattern.findall(text)) < 5:
            skipped["no_devanagari"] += 1
            continue

        # skip if too long for Colab
        if estimate_tokens(text) > max_tokens:
            skipped["too_long"] += 1
            continue

        clean.append(ex)

    return clean, skipped


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print("\n" + "═"*50)
    print("  Formatter — Preparing Data for QLoRA")
    print("═"*50 + "\n")

    all_stats = {}

    for task, data_dir in TASKS.items():
        print(f"\n── {task.upper()} ──────────────────────────────────────")

        # load
        train_path = data_dir / "train.jsonl"
        test_path  = data_dir / "test.jsonl"

        train_raw = load_jsonl(train_path)
        test_raw  = load_jsonl(test_path)

        if args.debug:
            train_raw = train_raw[:10]
            test_raw  = test_raw[:5]
            print(f"  [DEBUG] {len(train_raw)} train / {len(test_raw)} test")

        print(f"  Loaded  : {len(train_raw)} train / {len(test_raw)} test")

        # validate
        train_clean, train_skipped = validate_and_filter(
            train_raw, task, MAX_TOKENS_ESTIMATE
        )
        test_clean, test_skipped = validate_and_filter(
            test_raw, task, MAX_TOKENS_ESTIMATE
        )

        if train_skipped["too_long"] or train_skipped["empty"]:
            print(f"  Skipped : {train_skipped}")

        print(f"  Clean   : {len(train_clean)} train / {len(test_clean)} test")

        # token length stats — important for setting max_seq_length in trainer
        train_lengths = [estimate_tokens(ex["text"]) for ex in train_clean]
        avg_len = round(sum(train_lengths) / max(len(train_lengths), 1), 1)
        max_len = max(train_lengths) if train_lengths else 0
        p95_len = sorted(train_lengths)[int(len(train_lengths) * 0.95)] if train_lengths else 0

        print(f"  Token estimates → avg: {avg_len}  p95: {p95_len}  max: {max_len}")
        print(f"  Recommended max_seq_length for trainer: {min(p95_len + 64, 2048)}")

        # save as HuggingFace Dataset (makes loading in trainer trivial)
        task_out = OUTPUT_DIR / task
        task_out.mkdir(exist_ok=True)

        # save JSONL (for inspection)
        for split_name, split_data in [("train", train_clean), ("test", test_clean)]:
            jsonl_path = task_out / f"{split_name}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for ex in split_data:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        # save as HuggingFace Dataset arrow format (faster to load on Kaggle)
        ds = DatasetDict({
            "train": Dataset.from_list(train_clean),
            "test":  Dataset.from_list(test_clean),
        })
        ds.save_to_disk(str(task_out / "hf_dataset"))

        print(f"  Saved   : {task_out}/")

        # show one sample
        print(f"\n  Sample ({task}):")
        sample = train_clean[0]["text"]
        print("  " + sample[:300].replace("\n", "\n  "))
        print()

        all_stats[task] = {
            "train":      len(train_clean),
            "test":       len(test_clean),
            "avg_tokens": avg_len,
            "p95_tokens": p95_len,
            "max_tokens": max_len,
            "recommended_max_seq_length": min(p95_len + 64, 2048),
            "skipped":    train_skipped,
        }

    # save combined stats
    stats_path = OUTPUT_DIR / "format_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # print summary table
    print("\n" + "═"*50)
    print("  SUMMARY")
    print("═"*50)
    print(f"  {'Task':<16} {'Train':>6} {'Test':>6} {'AvgTok':>8} {'MaxSeqLen':>10}")
    print("  " + "─"*46)
    for task, s in all_stats.items():
        print(f"  {task:<16} {s['train']:>6} {s['test']:>6} "
              f"{s['avg_tokens']:>8} {s['recommended_max_seq_length']:>10}")

    print(f"\n  Saved to: {OUTPUT_DIR}/")
    print("  Format stats: format_stats.json\n")
    print("  ✓ Ready for baseline evaluation and QLoRA training\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="10 examples only")
    main(parser.parse_args())