"""
load_xquad.py
-------------
Downloads proper Nepali QA data.

CONFIRMED WORKING:
  1. dineshkarki/textbooks-qa-nepali → 5201 rows (conversations format)
     NOTE: context has OCR noise — we skip noisy context, use question+answer only
  2. Yunika/Nepali-QA → 266 SQuAD-format QA pairs (JSON)
  3. Chhabi/Nepali-Health-QA → 1570 rows (fallback)

Run:
    python data/load_xquad.py --debug
    python data/load_xquad.py
"""

import json
import re
import random
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR  = Path("outputs/xquad")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_RATIO  = 0.15
RANDOM_SEED = 42
MAX_TRAIN   = 2000

DEVANAGARI = re.compile(r'[\u0900-\u097F]')

PROMPT_WITH_CONTEXT = """\
### Instruction:
Read the following context carefully and answer the question \
based only on the information provided.

### Context:
{context}

### Question:
{question}

### Response:
{answer}"""

PROMPT_WITHOUT_CONTEXT = """\
### Instruction:
Answer the following question in Nepali.

### Question:
{question}

### Response:
{answer}"""


def is_clean_nepali(text: str, min_ratio: float = 0.4) -> bool:
    """
    Returns True if text has enough Devanagari characters.
    Filters out OCR noise like 'eit VA a zs oe"'
    """
    if not text or len(text) < 10:
        return False
    deva_count  = len(DEVANAGARI.findall(text))
    total_chars = len(text.replace(" ", ""))
    return (deva_count / max(total_chars, 1)) >= min_ratio


def truncate_answer(text: str, max_words: int = 40) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


# ── Source 1: dineshkarki/textbooks-qa-nepali ─────────────────────────────────

def load_textbooks_qa():
    """
    5201 rows. Conversations format:
    [{"from": "human", "value": "question"}, {"from": "gpt", "value": "answer"}]

    Context is noisy (OCR from scanned PDFs) — skip context, use Q+A only.
    Questions and answers are clean Nepali.
    """
    from datasets import load_dataset
    print("  Loading dineshkarki/textbooks-qa-nepali...")
    try:
        ds   = load_dataset("dineshkarki/textbooks-qa-nepali")
        data = list(ds[list(ds.keys())[0]])

        pairs = []
        for ex in data:
            convs = ex.get("conversations", [])
            if not isinstance(convs, list):
                continue

            question = ""
            answer   = ""
            for turn in convs:
                if not isinstance(turn, dict):
                    continue
                role  = turn.get("from", "").lower()
                value = str(turn.get("value", "")).strip()
                if role in ["human", "user"] and not question:
                    question = value
                elif role in ["gpt", "assistant"] and not answer:
                    answer = value

            if not question or not answer:
                continue

            # skip if question is not clean Nepali
            if not is_clean_nepali(question, min_ratio=0.3):
                continue

            # skip if answer is not clean Nepali
            if not is_clean_nepali(answer, min_ratio=0.2):
                continue

            # try to use context_text if it's clean
            context = str(ex.get("context_text", "") or "").strip()
            if not is_clean_nepali(context, min_ratio=0.5):
                context = ""   # skip noisy OCR context

            pairs.append({
                "context":  context[:400] if context else "",
                "question": question,
                "answer":   truncate_answer(answer),
                "source":   "textbooks-qa-nepali",
            })

        print(f"    Valid   : {len(pairs)}")
        return pairs

    except Exception as e:
        print(f"    Failed: {e}")
        return []


# ── Source 2: Yunika/Nepali-QA — SQuAD JSON format ───────────────────────────

def load_yunika_json():
    """
    Yunika/Nepali-QA — 266 extractive QA pairs.
    SQuAD format: {"data": [{"paragraphs": [{"context":"...", "qas":[...]}]}]}
    """
    print("  Loading Yunika/Nepali-QA (SQuAD JSON)...")
    try:
        url  = "https://huggingface.co/datasets/Yunika/Nepali-QA/resolve/main/nepali-qa.json"
        resp = requests.get(url, timeout=30)
        resp.encoding = "utf-8"
        raw  = resp.json()

        pairs = []

        # SQuAD format traversal
        for article in raw.get("data", []):
            for para in article.get("paragraphs", []):
                context = str(para.get("context", "")).strip()

                for qa in para.get("qas", []):
                    question = str(qa.get("question", "")).strip()
                    answers  = qa.get("answers", [])

                    # answers is list of {"text": "...", "answer_start": N}
                    if not answers:
                        continue

                    answer = ""
                    for ans in answers:
                        if isinstance(ans, dict):
                            t = str(ans.get("text", "")).strip()
                            if t:
                                answer = t
                                break
                        elif isinstance(ans, str):
                            answer = ans.strip()
                            break

                    if question and answer:
                        pairs.append({
                            "context":  context,
                            "question": question,
                            "answer":   answer,
                            "source":   "Yunika-Nepali-QA",
                        })

        print(f"    Valid   : {len(pairs)}")
        return pairs

    except Exception as e:
        print(f"    Failed: {e}")
        return []


# ── Source 3: Chhabi health QA fixed ─────────────────────────────────────────

def load_health_qa_fixed():
    """Chhabi/Nepali-Health-QA with short answers."""
    from datasets import load_dataset
    print("  Loading Chhabi/Nepali-Health-QA (fixed)...")
    try:
        ds   = load_dataset("Chhabi/Nepali-Health-QA")
        data = list(ds[list(ds.keys())[0]])

        pairs = []
        for ex in data:
            context  = str(ex.get("Translated_Context",  "") or "").strip()
            response = str(ex.get("Translated_Response", "") or "").strip()

            if not context or not response:
                continue

            # extract question from first sentence
            for sep in ["?", "।", "."]:
                if sep in context:
                    question = context.split(sep)[0].strip() + sep
                    break
            else:
                question = context[:80] + "?"

            if len(question.split()) < 4:
                continue

            # short answer
            sentences = [s.strip() for s in response.split("।") if s.strip()]
            answer    = "।".join(sentences[:2]) or response[:150]
            answer    = truncate_answer(answer)

            pairs.append({
                "context":  context[:400],
                "question": question,
                "answer":   answer,
                "source":   "health-qa-fixed",
            })

        print(f"    Valid   : {len(pairs)}")
        return pairs

    except Exception as e:
        print(f"    Failed: {e}")
        return []


# ── Format ────────────────────────────────────────────────────────────────────

def format_example(pair: dict, for_training: bool) -> dict:
    context  = pair.get("context",  "").strip()
    question = pair.get("question", "").strip()
    answer   = pair.get("answer",   "").strip()

    words = context.split()
    if len(words) > 200:
        context = " ".join(words[:200]) + "..."

    if context:
        text = PROMPT_WITH_CONTEXT.format(
            context=context,
            question=question,
            answer=answer if for_training else "",
        )
    else:
        text = PROMPT_WITHOUT_CONTEXT.format(
            question=question,
            answer=answer if for_training else "",
        )

    return {
        "text":     text,
        "context":  context,
        "question": question,
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print("\n" + "═"*50)
    print("  Nepali QA — Clean Data with Real Questions")
    print("═"*50 + "\n")

    all_pairs = []
    for loader in [load_textbooks_qa, load_yunika_json, load_health_qa_fixed]:
        pairs = loader()
        all_pairs.extend(pairs)
        print(f"  Running total: {len(all_pairs)}")

    if not all_pairs:
        raise RuntimeError("All sources failed")

    print(f"\nTotal: {len(all_pairs)} pairs")

    from collections import Counter
    sources = Counter(p["source"] for p in all_pairs)
    print(f"Sources: {dict(sources)}")

    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)
    all_pairs = all_pairs[:MAX_TRAIN + 500]

    split     = int(len(all_pairs) * (1 - TEST_RATIO))
    train_raw = all_pairs[:split]
    test_raw  = all_pairs[split:]

    if args.debug:
        train_raw = train_raw[:10]
        test_raw  = test_raw[:10]
        print("[DEBUG] 10 examples only\n")

    print(f"\nSplit: {len(train_raw)} train / {len(test_raw)} test")

    print("\nFormatting...")
    train_out = [format_example(ex, True)  for ex in tqdm(train_raw, desc="  train")]
    test_out  = [format_example(ex, False) for ex in tqdm(test_raw,  desc="  test ")]

    print()
    save_jsonl(train_out, OUTPUT_DIR / "train.jsonl")
    save_jsonl(test_out,  OUTPUT_DIR / "test.jsonl")

    stats = {
        "train_size":         len(train_out),
        "test_size":          len(test_out),
        "sources":            dict(sources),
        "avg_question_words": round(
            sum(len(e["question"].split()) for e in train_out) / max(len(train_out),1), 1
        ),
        "avg_answer_words":   round(
            sum(len(e["answer"].split()) for e in train_out) / max(len(train_out),1), 1
        ),
        "with_context":       sum(1 for e in train_out if e["context"]),
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # show 3 samples to verify quality
    print("\n── Samples (verify quality) ─────────────────────────────")
    for i, ex in enumerate(train_out[:3]):
        print(f"\n[{i+1}] Source: {ex['source']}")
        print(f"  Q: {ex['question'][:80]}")
        print(f"  A: {ex['answer'][:80]}")
    print("─"*50)

    print(f"\n✓  Train              : {stats['train_size']}")
    print(f"   Test               : {stats['test_size']}")
    print(f"   Sources            : {dict(sources)}")
    print(f"   Avg question words : {stats['avg_question_words']}")
    print(f"   Avg answer words   : {stats['avg_answer_words']}")
    print(f"   With context       : {stats['with_context']}/{len(train_out)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())