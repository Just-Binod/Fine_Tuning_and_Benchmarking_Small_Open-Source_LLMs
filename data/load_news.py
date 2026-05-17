"""
load_news.py
------------
Downloads Nepali news summarization data from HuggingFace.
NO scraping needed — clean datasets already exist.

Why not scrape?
  OnlineKhabar/Setopati/Ratopati sitemaps return 404 or block bots.
  These HuggingFace datasets were already scraped from those same sites
  by researchers — cleaner, larger, and immediately available.

Sources (tried in order):
  1. sanjeev-bhandari01/nepali-summarization-dataset  (HF, clean)
  2. sanjeev-bhandari01/XLSum-nepali                  (HF, XL-Sum format)
  3. Someman/news_nepali                               (HF, used by GenzNepal)
  4. Suyogyart/np20ng                                 (HF, 200k+ news docs)

Task: article → headline (headline generation = summarization)

Run:
    python data/load_news.py --debug
    python data/load_news.py
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

OUTPUT_DIR  = Path("outputs/news")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SAMPLE = 3000   # enough for fine-tuning, keeps Colab fast
TEST_SAMPLE  = 500
RANDOM_SEED  = 42

PROMPT_TEMPLATE = """\
### Instruction:
Summarize the following Nepali news article in one or two sentences.

### Article:
{article}

### Response:
{summary}"""


# ── Loaders — one per source ──────────────────────────────────────────────────

def load_sanjeev_summarization():
    """sanjeev-bhandari01/nepali-summarization-dataset"""
    print("  Trying sanjeev-bhandari01/nepali-summarization-dataset...")
    ds    = load_dataset("sanjeev-bhandari01/nepali-summarization-dataset")
    split = list(ds.keys())[0]
    data  = list(ds[split])
    cols  = list(data[0].keys())
    print(f"    Rows: {len(data)}, Columns: {cols}")

    pairs = []
    for ex in data:
        # find article and summary columns
        article = (
            ex.get("article") or ex.get("text") or
            ex.get("body")    or ex.get("content") or ""
        )
        summary = (
            ex.get("summary")  or ex.get("headline") or
            ex.get("title")    or ex.get("abstract") or ""
        )
        article = str(article).strip()
        summary = str(summary).strip()
        if article and summary and len(article.split()) > 30:
            pairs.append({"article": article, "summary": summary,
                          "source": "sanjeev-bhandari01/nepali-summarization-dataset"})

    print(f"    Valid pairs: {len(pairs)}")
    return pairs


def load_xlsum_nepali():
    """sanjeev-bhandari01/XLSum-nepali"""
    print("  Trying sanjeev-bhandari01/XLSum-nepali...")
    ds   = load_dataset("sanjeev-bhandari01/XLSum-nepali")
    cols = ds[list(ds.keys())[0]].column_names
    print(f"    Columns: {cols}")

    pairs = []
    for split in ds.keys():
        for ex in ds[split]:
            article = str(ex.get("text", "") or "").strip()
            summary = str(ex.get("summary", "") or ex.get("title", "") or "").strip()
            if article and summary and len(article.split()) > 30:
                pairs.append({"article": article, "summary": summary,
                              "source": "XLSum-nepali"})

    print(f"    Valid pairs: {len(pairs)}")
    return pairs


def load_someman_news():
    """Someman/news_nepali — used in published Nepali summarization models"""
    print("  Trying Someman/news_nepali...")
    ds   = load_dataset("Someman/news_nepali")
    cols = ds[list(ds.keys())[0]].column_names
    print(f"    Columns: {cols}")

    pairs = []
    for split in ds.keys():
        for ex in ds[split]:
            # inspect all columns — print first example to debug
            article = (
                ex.get("article") or ex.get("text") or
                ex.get("body")    or ex.get("content") or
                ex.get("news")    or ""
            )
            summary = (
                ex.get("summary")  or ex.get("headline") or
                ex.get("title")    or ex.get("abstract") or
                ex.get("summary_text") or ""
            )
            article = str(article).strip()
            summary = str(summary).strip()
            if article and summary and len(article.split()) > 30:
                pairs.append({"article": article, "summary": summary,
                              "source": "Someman/news_nepali"})

    print(f"    Valid pairs: {len(pairs)}")
    return pairs


def load_np20ng():
    """
    Suyogyart/np20ng — 200k+ Nepali news documents with headings.
    Uses heading as summary, document text as article.
    """
    print("  Trying Suyogyart/np20ng...")
    ds   = load_dataset("Suyogyart/np20ng")
    cols = ds[list(ds.keys())[0]].column_names
    print(f"    Columns: {cols}")

    pairs = []
    split = list(ds.keys())[0]
    for ex in ds[split]:
        # np20ng has 'text' and category labels
        # text field usually starts with heading then body
        text = str(ex.get("text", "") or "").strip()
        if not text or len(text.split()) < 50:
            continue

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 2:
            continue

        # first line = headline, rest = article body
        summary = lines[0]
        article = " ".join(lines[1:])

        if len(article.split()) > 30 and len(summary.split()) > 2:
            pairs.append({"article": article, "summary": summary,
                          "source": "Suyogyart/np20ng"})

    print(f"    Valid pairs: {len(pairs)}")
    return pairs


# ── Download — try sources in order ──────────────────────────────────────────

def download() -> list[dict]:
    """Try all sources, return first successful one with enough data."""

    sources = [
        load_sanjeev_summarization,
        load_xlsum_nepali,
        load_someman_news,
        load_np20ng,
    ]

    for loader in sources:
        try:
            pairs = loader()
            if len(pairs) >= 100:
                return pairs
            print(f"    Too few pairs ({len(pairs)}), trying next source...")
        except Exception as e:
            print(f"    Failed: {e}")

    raise RuntimeError(
        "All summarization sources failed.\n"
        "Run: pip install --upgrade datasets\n"
    )


# ── Format ────────────────────────────────────────────────────────────────────

def format_example(pair: dict, for_training: bool) -> dict:
    article = pair["article"]
    summary = pair["summary"]

    # truncate very long articles
    words = article.split()
    if len(words) > 500:
        article = " ".join(words[:500]) + "..."

    return {
        "text":    PROMPT_TEMPLATE.format(
                       article=article,
                       summary=summary if for_training else ""
                   ),
        "article": article,
        "summary": summary,
        "source":  pair.get("source", ""),
        "task":    "summarization",
        "lang":    "nepali",
    }


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print("\n" + "═"*50)
    print("  Nepali Summarization Data")
    print("═"*50 + "\n")

    print("Loading datasets...")
    all_pairs = download()
    print(f"\nTotal valid pairs: {len(all_pairs)}")

    # sample down to manageable size
    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)

    train_raw = all_pairs[:TRAIN_SAMPLE]
    test_raw  = all_pairs[TRAIN_SAMPLE:TRAIN_SAMPLE + TEST_SAMPLE]

    if args.debug:
        train_raw = train_raw[:10]
        test_raw  = test_raw[:10]
        print("[DEBUG] 10 examples only\n")

    print(f"Using: {len(train_raw)} train / {len(test_raw)} test")

    # format
    print("\nFormatting...")
    train_out = [format_example(p, True)  for p in tqdm(train_raw, desc="  train")]
    test_out  = [format_example(p, False) for p in tqdm(test_raw,  desc="  test ")]

    # save
    print()
    save_jsonl(train_out, OUTPUT_DIR / "train.jsonl")
    save_jsonl(test_out,  OUTPUT_DIR / "test.jsonl")

    stats = {
        "train_size": len(train_out),
        "test_size":  len(test_out),
        "source":     all_pairs[0].get("source", "unknown") if all_pairs else "none",
        "avg_article_words": round(
            sum(len(e["article"].split()) for e in train_out) / max(len(train_out), 1), 1
        ),
        "avg_summary_words": round(
            sum(len(e["summary"].split()) for e in train_out) / max(len(train_out), 1), 1
        ),
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n── Sample — verify Devanagari is readable ───────────────")
    print(train_out[0]["text"][:400])
    print("─"*50)
    print(f"\n✓  Train               : {stats['train_size']}")
    print(f"   Test                : {stats['test_size']}")
    print(f"   Source              : {stats['source']}")
    print(f"   Avg article words   : {stats['avg_article_words']}")
    print(f"   Avg summary words   : {stats['avg_summary_words']}")
    print(f"   Saved to            : {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="10 examples only")
    main(parser.parse_args())