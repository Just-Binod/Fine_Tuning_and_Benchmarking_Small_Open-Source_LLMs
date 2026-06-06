"""
scrape_categories.py
--------------------
Scrapes Nepali news from category/listing pages directly.
Gets more articles than RSS (which only shows recent 20-50).

Category pages list many articles → follow each link → extract body.

Target: 500-1000 articles across all 3 sites.

Run:
    python data/scrape_categories.py --debug
    python data/scrape_categories.py
"""

import json, re, time, random, logging, argparse
import unicodedata
from pathlib import Path
from datetime import datetime
from collections import Counter

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

OUTPUT_DIR = Path("outputs/news_scraped")
RAW_DIR    = OUTPUT_DIR / "raw"
CLEAN_DIR  = OUTPUT_DIR / "cleaned"
LOG_DIR    = Path("logs")

for d in [OUTPUT_DIR, RAW_DIR, CLEAN_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "category_scraper.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

DELAY_SEC = 2.0
MIN_WORDS = 60

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ne,en;q=0.9",
}

PROMPT_TEMPLATE = """\
### Instruction:
Summarize the following Nepali news article in one or two sentences.

### Article:
{article}

### Response:
{summary}"""

# Category pages — multiple pages of article listings
SITES = {
    "onlinekhabar": {
        "category_urls": [
            "https://www.onlinekhabar.com/content/news",
            "https://www.onlinekhabar.com/content/news?page=2",
            "https://www.onlinekhabar.com/content/news?page=3",
            "https://www.onlinekhabar.com/content/politics",
            "https://www.onlinekhabar.com/content/economics",
        ],
        # CSS selectors to find article links on listing page
        "link_selectors": [
            "h2.ok18-single-post-title a",
            "h3.ok18-single-post-title a",
            "a.ok18-post-title",
            ".post-title a",
            "article h2 a",
            "article h3 a",
        ],
        # CSS selectors to find article body
        "body_selectors": [
            "div.ok18-single-post-content-main",
            "div.post-content",
            "article .content",
            "article",
        ],
        # filter to only keep article URLs
        "url_filter": lambda u: "/content/" in u or re.search(r'/\d{4}/\d{2}/', u),
    },
    "setopati": {
        "category_urls": [
            "https://www.setopati.com/national",
            "https://www.setopati.com/politics",
            "https://www.setopati.com/economy",
            "https://www.setopati.com/society",
        ],
        "link_selectors": [
            "h2.news-title a",
            "h3.news-title a",
            ".article-title a",
            "article h2 a",
            "article h3 a",
            ".post-title a",
        ],
        "body_selectors": [
            "div.article-body",
            "div.content-area",
            "article .body",
            "article",
        ],
        "url_filter": lambda u: "setopati.com" in u and len(u) > 35,
    },
    "ratopati": {
        "category_urls": [
            "https://ratopati.com/category/news",
            "https://ratopati.com/category/politics",
            "https://ratopati.com/category/business",
        ],
        "link_selectors": [
            "h2.entry-title a",
            "h3.entry-title a",
            ".post-title a",
            "article h2 a",
            "article h3 a",
        ],
        "body_selectors": [
            "div.news-content",
            "div.single-content",
            "div.post-content",
            "article",
        ],
        "url_filter": lambda u: "ratopati.com" in u and len(u) > 25,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch(session, url, timeout=20):
    try:
        time.sleep(DELAY_SEC + random.uniform(0, 0.8))
        r = session.get(url, timeout=timeout)
        r.encoding = "utf-8"
        return r if r.status_code == 200 else None
    except Exception as e:
        log.warning(f"Fetch failed: {url} | {e}")
        return None


def is_nepali(text, min_ratio=0.25):
    deva  = len(re.findall(r'[\u0900-\u097F]', text))
    total = len(text.replace(' ', ''))
    return total > 0 and deva / total >= min_ratio


def preprocess(raw: str) -> str:
    """7-step Nepali text preprocessing pipeline."""
    text = raw.encode("utf-8", errors="ignore").decode("utf-8")
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\u00ad]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'&[a-zA-Z]+;|&#\d+;', ' ', text)
    text = unicodedata.normalize('NFC', text)
    lines = [
        l for l in text.split('\n')
        if l.strip() and (
            len(re.findall(r'[\u0900-\u097F]', l)) /
            max(len(l.replace(' ','')), 1) >= 0.2
            or len(l.strip()) < 5
        )
    ]
    text = ' '.join(lines)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_links(soup, selectors, url_filter, base_url) -> list[str]:
    """Find article links on a category/listing page."""
    links = set()

    for sel in selectors:
        try:
            for a in soup.select(sel):
                href = a.get("href", "")
                if not href:
                    continue
                # make absolute URL
                if href.startswith("/"):
                    href = base_url + href
                if url_filter(href):
                    links.add(href)
        except Exception:
            continue

    # fallback: find all links that look like articles
    if not links:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/"):
                href = base_url.rstrip("/") + href
            if url_filter(href) and len(a.get_text(strip=True)) > 10:
                links.add(href)

    return list(links)


def extract_body(soup, selectors) -> str:
    """Extract article body text."""
    for sel in selectors:
        try:
            el = soup.select_one(sel)
            if el:
                for noise in el.find_all(
                    ["script","style","aside","figure","nav","footer"]
                ):
                    noise.decompose()
                paras = [p.get_text(strip=True) for p in el.find_all("p")]
                text  = " ".join(p for p in paras if p)
                if is_nepali(text) and len(text.split()) >= MIN_WORDS:
                    return text
        except Exception:
            continue

    # fallback: score divs by Devanagari density
    best_score, best_text = 0, ""
    for div in soup.find_all("div"):
        try:
            text  = div.get_text(" ", strip=True)
            words = text.split()
            if len(words) < MIN_WORDS:
                continue
            deva  = len(re.findall(r'[\u0900-\u097F]', text))
            score = deva / max(len(text), 1) * len(words)
            if score > best_score:
                best_score = score
                best_text  = text
        except Exception:
            continue

    return best_text


def extract_headline(soup) -> str:
    """Extract article headline."""
    for tag in ["h1", "h2"]:
        el = soup.find(tag)
        if el:
            text = el.get_text(strip=True)
            if is_nepali(text, min_ratio=0.1):
                return text
    return ""


# ── Core scraper ──────────────────────────────────────────────────────────────

def scrape_site(name, config, session, max_articles, debug) -> list[dict]:
    log.info(f"\n{'─'*50}")
    log.info(f"  Scraping: {name}")

    base_url = "https://" + name.replace("onlinekhabar", "www.onlinekhabar.com")\
                                 .replace("setopati", "www.setopati.com")\
                                 .replace("ratopati", "ratopati.com")

    # Step 1: get article URLs from category pages
    all_article_urls = set()
    cat_urls = config["category_urls"][:2] if debug else config["category_urls"]

    for cat_url in cat_urls:
        log.info(f"  Fetching category: {cat_url}")
        resp = fetch(session, cat_url, timeout=15)
        if resp is None:
            continue
        soup  = BeautifulSoup(resp.text, "html.parser")
        links = extract_links(
            soup,
            config["link_selectors"],
            config["url_filter"],
            base_url,
        )
        log.info(f"    Found {len(links)} article links")
        all_article_urls.update(links)

    log.info(f"  Total unique article URLs: {len(all_article_urls)}")

    if not all_article_urls:
        log.error(f"  [{name}] No article URLs found")
        return []

    urls = list(all_article_urls)[:5 if debug else max_articles]
    random.shuffle(urls)

    # Step 2: scrape each article
    articles = []
    skipped  = 0

    for url in tqdm(urls, desc=f"  {name}"):
        resp = fetch(session, url)
        if resp is None:
            skipped += 1
            continue

        soup     = BeautifulSoup(resp.text, "html.parser")
        headline = extract_headline(soup)
        body     = extract_body(soup, config["body_selectors"])

        if not headline or not body:
            skipped += 1
            continue

        # save raw
        headline_raw = headline
        body_raw     = body

        # preprocess
        headline = preprocess(headline)
        body     = preprocess(body)

        if not is_nepali(headline) or not is_nepali(body):
            skipped += 1
            continue
        if len(body.split()) < MIN_WORDS:
            skipped += 1
            continue

        articles.append({
            "headline":     headline,
            "body":         body,
            "headline_raw": headline_raw,
            "body_raw":     body_raw,
            "url":          url,
            "source":       name,
            "scraped":      datetime.now().isoformat(),
        })

    log.info(f"  [{name}] {len(articles)} kept / {skipped} skipped")

    # save raw
    raw_path = RAW_DIR / f"{name}_cat.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps({
                "headline": a["headline_raw"],
                "body":     a["body_raw"],
                "url":      a["url"],
                "source":   a["source"],
            }, ensure_ascii=False) + "\n")

    # save cleaned
    clean_path = CLEAN_DIR / f"{name}_cat.jsonl"
    with open(clean_path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps({
                "headline": a["headline"],
                "body":     a["body"],
                "url":      a["url"],
                "source":   a["source"],
            }, ensure_ascii=False) + "\n")

    log.info(f"  Raw   → {raw_path}")
    log.info(f"  Clean → {clean_path}")
    return articles


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    log.info("\n" + "═"*55)
    log.info("  Category Page Scraper")
    log.info("  Target: 500-1000 Nepali news articles")
    log.info("═"*55)

    session = requests.Session()
    session.headers.update(HEADERS)

    sites = {args.site: SITES[args.site]} if args.site else SITES
    all_articles = []

    for name, config in sites.items():
        arts = scrape_site(
            name, config, session,
            max_articles=400,
            debug=args.debug,
        )
        all_articles.extend(arts)

    if not all_articles:
        log.error("No articles scraped. Check logs/category_scraper.log")
        return

    # combine with RSS scrape if exists
    rss_dir = RAW_DIR
    for rss_file in rss_dir.glob("*_raw.jsonl"):
        try:
            with open(rss_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # check not duplicate
                        existing_urls = {a["url"] for a in all_articles}
                        if item.get("url") not in existing_urls:
                            all_articles.append({
                                "headline":     item.get("headline",""),
                                "body":         item.get("body",""),
                                "headline_raw": item.get("headline",""),
                                "body_raw":     item.get("body",""),
                                "url":          item.get("url",""),
                                "source":       item.get("source",""),
                                "scraped":      "",
                            })
            log.info(f"Merged RSS data from {rss_file.name}")
        except Exception:
            pass

    log.info(f"\nTotal articles (category + RSS): {len(all_articles)}")

    # format for training
    random.seed(42)
    random.shuffle(all_articles)
    split     = int(len(all_articles) * 0.85)
    train_raw = all_articles[:split]
    test_raw  = all_articles[split:]

    def fmt(a, train):
        body = " ".join(a["body"].split()[:400])
        return {
            "text":    PROMPT_TEMPLATE.format(
                           article=body,
                           summary=a["headline"] if train else ""
                       ),
            "article": body,
            "summary": a["headline"],
            "source":  a["source"],
            "url":     a["url"],
            "task":    "summarization",
            "lang":    "nepali",
        }

    train_out = [fmt(a, True)  for a in train_raw]
    test_out  = [fmt(a, False) for a in test_raw]

    def save(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log.info(f"Saved {len(data):,} → {path}")

    save(train_out, OUTPUT_DIR / "train.jsonl")
    save(test_out,  OUTPUT_DIR / "test.jsonl")

    sources = dict(Counter(a["source"] for a in all_articles))
    stats   = {
        "total":      len(all_articles),
        "train":      len(train_out),
        "test":       len(test_out),
        "sources":    sources,
        "scraped_at": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # show sample
    if all_articles:
        s = all_articles[0]
        log.info(f"\n── Sample ───────────────────────────────────")
        log.info(f"  Headline : {s['headline'][:70]}")
        log.info(f"  Body     : {s['body'][:100]}")

    log.info(f"\n✓  Total  : {stats['total']}")
    log.info(f"   Train  : {stats['train']}")
    log.info(f"   Test   : {stats['test']}")
    log.info(f"   Sources: {sources}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--site", choices=list(SITES.keys()), default=None)
    main(parser.parse_args())