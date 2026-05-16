"""
load_xquad.py
-------------
Downloads Nepali Question Answering data.

Sources:
    1. Chhabi/Nepali-Health-QA
       columns: Translated_Context, Translated_Response
    2. Yunika/Nepali-QA (skipped — broken generator on HF)

Run:
    python data/load_xquad.py --debug
    python data/load_xquad.py
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

OUTPUT_DIR  = Path("outputs/xquad")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_RATIO  = 0.15
RANDOM_SEED = 42

PROMPT_TEMPLATE = """\
### Instruction:
Read the context and answer the question based only on the information provided.

### Context:
{context}

### Question:
{question}

### Response:
{answer}"""

PROMPT_SIMPLE = """\
### Instruction:
Answer the following question in Nepali.

### Question:
{question}

### Response:
{answer}"""


def load_health_qa():
    """
    Chhabi/Nepali-Health-QA
    Actual columns: Translated_Context, Translated_Response
    """
    print("  Loading Chhabi/Nepali-Health-QA...")
    ds    = load_dataset("Chhabi/Nepali-Health-QA")
    split = list(ds.keys())[0]
    data  = list(ds[split])
    print(f"    Rows    : {len(data)}")
    print(f"    Columns : {list(data[0].keys())}")

    pairs = []
    for ex in data:
        context  = str(ex.get("Translated_Context",  "") or "").strip()
        response = str(ex.get("Translated_Response", "") or "").strip()

        if not response:
            continue

        # context is the passage, response is the answer
        # we split context into question + passage if possible
        # otherwise treat context as question
        pairs.append({
            "context":  context,
            "question": "",       # no separate question field in this dataset
            "answer":   response,
            "source":   "Chhabi/Nepali-Health-QA",
        })

    print(f"    Valid   : {len(pairs)}")
    return pairs


def load_rxnach_health():
    """
    rxnach/nepali-health-forum-corpus-questions-and-answers (Kaggle mirror on HF)
    Fallback source with 2500+ Q&A pairs.
    """
    try:
        print("  Loading sagearbor/nepali-qa (fallback)...")
        ds = load_dataset("sagearbor/nepali-qa")
        split = list(ds.keys())[0]
        data  = list(ds[split])
        print(f"    Columns: {list(data[0].keys())}")

        pairs = []
        for ex in data:
            cols = list(ex.keys())
            q_col = next((c for c in cols if "question" in c.lower() or "q" == c.lower()), None)
            a_col = next((c for c in cols if "answer"   in c.lower() or "a" == c.lower()), None)
            if not q_col or not a_col:
                break
            q = str(ex[q_col]).strip()
            a = str(ex[a_col]).strip()
            if q and a:
                pairs.append({"context": "", "question": q, "answer": a, "source": "sagearbor/nepali-qa"})
        return pairs
    except Exception as e:
        print(f"    Failed: {e}")
        return []


def format_example(pair: dict, for_training: bool) -> dict:
    context  = pair.get("context",  "").strip()
    question = pair.get("question", "").strip()
    answer   = pair.get("answer",   "").strip()

    # truncate long context
    words = context.split()
    if len(words) > 300:
        context = " ".join(words[:300]) + "..."

    # if we have both context and question → use full template
    if context and question:
        text = PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            answer=answer if for_training else "",
        )
    # if only context (which acts as question/answer pair)
    elif context:
        text = PROMPT_SIMPLE.format(
            question=context,
            answer=answer if for_training else "",
        )
    else:
        text = PROMPT_SIMPLE.format(
            question=question,
            answer=answer if for_training else "",
        )

    return {
        "text":     text,
        "context":  context,
        "question": question or context,
        "answer":   answer,
        "source":   pair.get("source", ""),
        "task":     "qa",
        "lang":     "nepali",
    }


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} → {path}")


def main(args):
    print("\n" + "═"*50)
    print("  Nepali QA Dataset")
    print("═"*50 + "\n")

    print("Loading datasets...")
    all_data = []
    all_data += load_health_qa()

    if len(all_data) < 100:
        all_data += load_rxnach_health()

    if not all_data:
        raise RuntimeError("All QA sources failed.")

    print(f"\nTotal: {len(all_data)} examples")

    # shuffle + split
    random.seed(RANDOM_SEED)
    random.shuffle(all_data)
    split     = int(len(all_data) * (1 - TEST_RATIO))
    train_raw = all_data[:split]
    test_raw  = all_data[split:]
    print(f"Split : {len(train_raw)} train / {len(test_raw)} test")

    if args.debug:
        train_raw = train_raw[:10]
        test_raw  = test_raw[:10]
        print("[DEBUG] 10 examples only\n")

    # format
    print("\nFormatting...")
    train_out = [format_example(ex, True)  for ex in tqdm(train_raw, desc="  train")]
    test_out  = [format_example(ex, False) for ex in tqdm(test_raw,  desc="  test ")]

    # save
    print()
    save_jsonl(train_out, OUTPUT_DIR / "train.jsonl")
    save_jsonl(test_out,  OUTPUT_DIR / "test.jsonl")

    stats = {
        "train_size":       len(train_out),
        "test_size":        len(test_out),
        "avg_answer_words": round(
            sum(len(e["answer"].split()) for e in train_out) / max(len(train_out), 1), 1
        ),
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n── Sample ───────────────────────────────────────────────")
    print(train_out[0]["text"][:500])
    print("─"*50)
    print(f"\n✓  Train : {stats['train_size']}")
    print(f"   Test  : {stats['test_size']}")
    print(f"   Saved → {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())