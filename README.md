# Yamaha JP FAQ Crawler

Python crawler that harvests Q/A entries from https://faq.yamaha.com/jp/s/ using Playwright for dynamic page rendering, lxml for parsing, and pandas for data assembly/export. Collected data is written to JSON.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install
```

## Run

```bash
python main.py --limit 50
python main.py --limit 10 --no-headless  # launch with a visible browser window
```

The crawler respects the site's robots.txt directives by skipping disallowed URL patterns such as `*?nocache=*` and image-only endpoints. Results are saved exclusively as partitioned JSON files inside `data/faq_articles_by_topic/` (or the directory specified via `--output`), with filenames following `製品タイプ__製品名+タイトル.json` (product type + product name + title). Each JSON contains plain-text answers (no HTML markup).
