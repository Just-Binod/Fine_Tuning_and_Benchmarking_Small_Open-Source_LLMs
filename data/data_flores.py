"""
load_flores.py
--------------
Downloads Nepali-English parallel translation data.

Source: Helsinki-NLP/opus-100 config 'en-ne'
  (English-Nepali — Nepali is the 'ne' side)

Run:
    python data/load_flores.py --debug
    python data/load_flores.py
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

OUTPUT_DIR  = Path("outputs/flores")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SAMPLE = 2000
RANDOM_SEED  = 42

PROMPT_TEMPLATE = """\
### Instruction:
Translate the following Nepali text to English.

### Input:
{nepali}

### Response:
{english}"""


def download():
    print("Downloading Helsinki-NLP/opus-100 (en-ne)...")

    # config is 'en-ne' — English on left, Nepali on right
    # each row: {"translation": {"en": "...", "ne": "..."}}
    ds = load_dataset("Helsinki-NLP/opus-100", "en-ne")

    print(f"  Splits     : {list(ds.keys())}")
    print(f"  Train size : {len(ds['train']):,}")

    def extract(example):
        t = example["translation"]
        return {
            "nepali":  t["ne"].strip(),
            "english": t["en"].strip(),
        }

    train_raw = [extract(ex) for ex in ds["train"]]

    # use validation as test set
    test_key  = "validation" if "validation" in ds else "test"
    test_raw  = [extract(ex) for ex in ds[test_key]]

    # filter empty
    train_raw = [p for p in train_raw if p["nepali"] and p["english"]]
    test_raw  = [p for p in test_raw  if p["nepali"] and p["english"]]

    # sample train down to 2000
    random.seed(RANDOM_SEED)
    random.shuffle(train_raw)
    train_raw = train_raw[:TRAIN_SAMPLE]

    print(f"  Using      : {len(train_raw)} train / {len(test_raw)} test")
    return train_raw, test_raw


def format_example(pair, for_training):
    return {
        "text": PROMPT_TEMPLATE.format(
            nepali=pair["nepali"],
            english=pair["english"] if for_training else "",
        ),
        "source": pair["nepali"],
        "target": pair["english"],
        "task":   "translation",
        "lang":   "ne-en",
    }


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} → {path}")


def main(args):
    print("\n" + "═"*50)
    print("  Nepali → English  |  opus-100 en-ne")
    print("═"*50 + "\n")

    train_raw, test_raw = download()

    if args.debug:
        train_raw = train_raw[:10]
        test_raw  = test_raw[:10]
        print("[DEBUG] 10 examples only\n")

    print("\nFormatting...")
    train_out = [format_example(p, True)  for p in tqdm(train_raw, desc="  train")]
    test_out  = [format_example(p, False) for p in tqdm(test_raw,  desc="  test ")]

    print()
    save_jsonl(train_out, OUTPUT_DIR / "train.jsonl")
    save_jsonl(test_out,  OUTPUT_DIR / "test.jsonl")

    stats = {
        "train_size":       len(train_out),
        "test_size":        len(test_out),
        "avg_nepali_words": round(
            sum(len(e["source"].split()) for e in train_out) / max(len(train_out), 1), 1
        ),
        "avg_english_words": round(
            sum(len(e["target"].split()) for e in train_out) / max(len(train_out), 1), 1
        ),
        "source": "Helsinki-NLP/opus-100 en-ne",
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n── Sample — verify Devanagari is readable ───────────────")
    print(train_out[0]["text"])
    print("─"*50)
    print(f"\n✓  Train            : {stats['train_size']}")
    print(f"   Test             : {stats['test_size']}")
    print(f"   Avg Nepali words : {stats['avg_nepali_words']}")
    print(f"   Avg Eng words    : {stats['avg_english_words']}")
    print(f"   Saved to         : {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="10 examples only")
    main(parser.parse_args())