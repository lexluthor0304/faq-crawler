import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.request import Request, urlopen

import pandas as pd
from lxml import etree, html
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
DISALLOWED_PATTERNS = (
    "/secur/forgotpassword.jsp",
    "?nocache=",
    "/servlet/rtaImage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl Yamaha JP FAQ articles into JSON.")
    parser.add_argument(
        "--sitemap-index",
        default="https://faq.yamaha.com/jp/s/sitemap.xml",
        help="Root sitemap index to start from.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of articles to fetch (default: 20).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between article fetches to be polite.",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=2000,
        help="Extra wait in milliseconds after the page reports ready (default: 2000).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60000,
        help="Per-page timeout in milliseconds for Playwright waits (default: 60000).",
    )
    parser.add_argument(
        "--output",
        default="data/faq_articles_by_topic",
        help="Directory (or legacy JSON path) where partitioned JSON files will be saved.",
    )
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        default=True,
        help="Run the browser in headless mode (default: enabled).",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Run the browser with a visible window.",
    )
    return parser.parse_args()


def fetch_xml(url: str) -> etree._Element:
    req = Request(url, headers={"User-Agent": "faq-crawler/1.0"})
    with urlopen(req) as resp:
        data = resp.read()
    return etree.fromstring(data)


def iter_article_urls(url: str) -> Iterable[Tuple[str, Optional[str]]]:
    root = fetch_xml(url)
    tag = etree.QName(root).localname
    if tag == "sitemapindex":
        for loc in root.xpath("//sm:sitemap/sm:loc/text()", namespaces=SITEMAP_NS):
            loc = loc.strip()
            if "topicarticle" not in loc:
                continue
            yield from iter_article_urls(loc)
    else:
        for url_node in root.xpath("//sm:url", namespaces=SITEMAP_NS):
            loc = url_node.xpath("./sm:loc/text()", namespaces=SITEMAP_NS)
            lastmod = url_node.xpath("./sm:lastmod/text()", namespaces=SITEMAP_NS)
            if not loc:
                continue
            yield loc[0].strip(), (lastmod[0].strip() if lastmod else None)


def is_allowed(url: str) -> bool:
    return not any(pattern in url for pattern in DISALLOWED_PATTERNS)


def extract_field(root: html.HtmlElement, label: str) -> str:
    node = root.xpath(
        "//span[contains(@class,'test-id__field-label') and normalize-space(text())=$label]/../../div[contains(@class,'slds-form-element__control')]//*[contains(@class,'test-id__field-value')]",
        label=label,
    )
    if not node:
        return ""
    text = node[0].text_content().replace("\u00a0", " ")
    return text.strip()


def extract_breadcrumbs(root: html.HtmlElement) -> Dict[str, List[str]]:
    groups = root.xpath("//div[contains(@class,'cFAQBreadcrumbList')]/div[contains(@class,'breadcrumbListText')]")
    trail_primary: List[str] = []
    topic_groups: List[str] = []
    if groups:
        primary_links = groups[0].xpath(".//a/text()")
        trail_primary = [txt.strip() for txt in primary_links if txt.strip()]
        for block in groups[1:]:
            for paragraph in block.xpath(".//p"):
                labels = [a.text_content().strip() for a in paragraph.xpath(".//a")]  # type: ignore[arg-type]
                if labels:
                    topic_groups.append(" > ".join(labels))
    return {"site_path": trail_primary, "topic_paths": topic_groups}


def scrape_article(page, url: str, wait_ms: int, timeout: int) -> Dict[str, object]:
    page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    page.wait_for_selector("span.test-id__field-label", timeout=timeout)
    if wait_ms:
        page.wait_for_timeout(wait_ms)
    content = page.content()
    root = html.fromstring(content)
    title = extract_field(root, "タイトル") or root.xpath("//title/text()")[:1]
    if isinstance(title, list):
        title = title[0] if title else ""
    answer_text = extract_field(root, "アンサー")
    article_id = extract_field(root, "URL 名")
    breadcrumbs = extract_breadcrumbs(root)
    return {
        "url": url,
        "article_id": article_id,
        "title": title,
        "answer_text": answer_text,
        "site_path": breadcrumbs["site_path"],
        "topic_paths": breadcrumbs["topic_paths"],
    }


FILENAME_SANITIZE_RE = re.compile(r"[\s/\\:*?\"<>|]+")


def sanitize_filename_component(value: str, fallback: str) -> str:
    if not value:
        return fallback
    sanitized = FILENAME_SANITIZE_RE.sub("_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or fallback


def build_article_filename(
    product_type: str,
    product_name: str,
    title: str,
    article_id: str,
    existing: set[str],
) -> str:
    type_part = sanitize_filename_component(product_type, "Uncategorized")
    name_part = sanitize_filename_component(product_name, "General")
    title_part = sanitize_filename_component(title, article_id or "Article")
    base = f"{type_part}__{name_part}+{title_part}"
    filename = f"{base}.json"
    if filename in existing:
        suffix = sanitize_filename_component(article_id, "dup") if article_id else "dup"
        filename = f"{base}_{suffix}.json"
        counter = 2
        while filename in existing:
            filename = f"{base}_{suffix}_{counter}.json"
            counter += 1
    return filename


def derive_product(record: Dict[str, object]) -> Tuple[str, str]:
    for raw_path in record.get("topic_paths") or []:
        parts = [segment.strip() for segment in str(raw_path).split(">") if segment.strip()]
        if parts:
            product_type = parts[0]
            product_name = parts[1] if len(parts) > 1 else "General"
            return product_type, product_name
    return "Uncategorized", "General"


def resolve_partition_root(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.suffix.lower() == ".json":
        base_dir = path.parent
        base_dir.mkdir(parents=True, exist_ok=True)
        partition_root = base_dir / f"{path.stem}_by_topic"
    else:
        partition_root = path
    partition_root.mkdir(parents=True, exist_ok=True)
    return partition_root


def write_partitioned_json(
    records: List[Dict[str, object]],
    partition_root: Path,
    existing_files: Optional[Set[str]] = None,
) -> List[Dict[str, object]]:
    written_records: List[Dict[str, object]] = []
    session_existing: set[str] = set()
    for record in records:
        product_type, product_name = derive_product(record)
        item = dict(record)
        item["product_type"] = product_type
        item["product_name"] = product_name
        filename = build_article_filename(
            product_type,
            product_name,
            item.get("title", ""),
            item.get("article_id", ""),
            session_existing,
        )
        if existing_files is not None and filename in existing_files:
            continue
        output_path = partition_root / filename
        frame = pd.DataFrame([item])
        data = frame.to_dict(orient="records")[0]
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        session_existing.add(filename)
        if existing_files is not None:
            existing_files.add(filename)
        written_records.append(record)
    return written_records


def main() -> int:
    args = parse_args()
    partition_root = resolve_partition_root(args.output)
    state_path = partition_root / ".crawl_state.json"
    processed_urls: Set[str] = set()
    processed_article_ids: Set[str] = set()
    last_processed_url: Optional[str] = None
    last_processed_timestamp: Optional[str] = None

    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as handle:
                state_data = json.load(handle)
            processed_urls = set(state_data.get("processed_urls", []))
            processed_article_ids = set(state_data.get("processed_article_ids", []))
            last_processed_url = state_data.get("last_processed_url")
            last_processed_timestamp = state_data.get("last_processed_timestamp")
            state_output = state_data.get("output_dir")
            if state_output and Path(state_output) != partition_root:
                print(
                    f"Loaded state from {state_path} for output {state_output}",
                    file=sys.stderr,
                )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to load state file {state_path}: {exc}", file=sys.stderr)
            processed_urls = set()
            processed_article_ids = set()
            last_processed_url = None
            last_processed_timestamp = None

    existing_files: Set[str] = {path.name for path in partition_root.glob("*.json")}

    batch: List[Dict[str, object]] = []
    total_collected = 0
    total_written = 0

    def flush_batch() -> None:
        nonlocal total_written, last_processed_url, last_processed_timestamp
        if not batch:
            return
        written_records = write_partitioned_json(batch, partition_root, existing_files)
        if not written_records:
            batch.clear()
            return
        total_written += len(written_records)
        now = datetime.now(timezone.utc).isoformat()
        for record in written_records:
            url = str(record.get("url") or "")
            if url:
                processed_urls.add(url)
                last_processed_url = url
            article_id = str(record.get("article_id") or "")
            if article_id:
                processed_article_ids.add(article_id)
            last_processed_timestamp = now
        state_payload = {
            "processed_urls": sorted(processed_urls),
            "processed_article_ids": sorted(processed_article_ids),
            "last_processed_url": last_processed_url,
            "last_processed_timestamp": last_processed_timestamp,
            "output_dir": str(partition_root),
        }
        try:
            with state_path.open("w", encoding="utf-8") as handle:
                json.dump(state_payload, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"Failed to write state file {state_path}: {exc}", file=sys.stderr)
        batch.clear()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        page = browser.new_page()
        try:
            stop = False
            for article_url, lastmod in iter_article_urls(args.sitemap_index):
                if args.limit and total_written >= args.limit:
                    break
                if args.limit and batch and total_written + len(batch) >= args.limit:
                    flush_batch()
                    if args.limit and total_written >= args.limit:
                        break
                if article_url in processed_urls:
                    continue
                if not is_allowed(article_url):
                    continue
                try:
                    record = scrape_article(page, article_url, args.wait, args.timeout)
                    record["lastmod"] = lastmod
                    article_id = str(record.get("article_id") or "")
                    if article_url in processed_urls or (
                        article_id and article_id in processed_article_ids
                    ):
                        continue
                    batch.append(record)
                    total_collected += 1
                    print(f"[{total_collected}] {record['title']}")
                    if args.limit and total_written + len(batch) >= args.limit:
                        flush_batch()
                        if args.limit and total_written >= args.limit:
                            stop = True
                            break
                    elif len(batch) >= 10:
                        flush_batch()
                except PlaywrightTimeoutError as exc:
                    print(f"Timeout loading {article_url}: {exc}", file=sys.stderr)
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to process {article_url}: {exc}", file=sys.stderr)
                if stop:
                    break
                time.sleep(args.delay)
        finally:
            browser.close()

    if batch:
        flush_batch()

    if not total_collected:
        print("No articles collected", file=sys.stderr)
        return 1

    print(
        f"Partitioned {total_collected} articles into {total_written} files under {partition_root}"
    )
    print(f"State saved to {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
