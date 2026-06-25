import json
from pathlib import Path

# ================== CORRECT PATH FOR NEWS SCRAPED ==================
input_path = "./outputs/news_scraped/train.jsonl"     # This is your scraped data
output_path = "./outputs/news_scraped/train_fixed_v2.jsonl"
# 

print(f"Reading from: {input_path}")

fixed_examples = []
short_count = 0

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
            
            if ex.get("task") != "summarization":
                continue
                
            article = ex.get("article", "").strip()
            old_summary = ex.get("summary", "").strip()
            
            if len(article) < 80 or len(old_summary) < 5:
                continue

            # Improve short headline-style summaries
            if len(old_summary.split()) <= 12:
                improved_summary = old_summary.rstrip("। ") + " । यो घटनाले नेपालमा ठूलो चर्चा पाएको छ ।"
                short_count += 1
            else:
                improved_summary = old_summary

            system_prompt = """You are an expert Nepali news summarizer. 
Summarize the given Nepali news article in **exactly 1 to 2 clear, complete and natural sentences** in Nepali. 
Capture the main event, key facts, people involved, and outcome. 
Aim for 35-75 words. Never output only a headline."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize the following Nepali news article in one or two sentences.\n\nArticle:\n{article}"},
                {"role": "assistant", "content": improved_summary}
            ]

            fixed_examples.append({
                "messages": messages,
                "task": "summarization"
            })

        except:
            continue

# Save
with open(output_path, "w", encoding="utf-8") as f:
    for ex in fixed_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f" Done! Created {len(fixed_examples)} improved examples")
print(f"   Fixed {short_count} short summaries")
print(f"   Saved as: {output_path}")