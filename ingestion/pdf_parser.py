"""
ingestion/pdf_parser.py
------------------------
Takes a PDF file path + its row from papers_enriched.csv and returns
a structured dict with paper-level metadata and cleaned text per page.

Used by: ingestion/embedder.py → stores/ → api/main.py
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean raw PDF text:
    1. Fix soft hyphens and line-break hyphenation (knowl-\nedge → knowledge)
    2. Normalize unicode characters
    3. Collapse whitespace
    4. Strip non-printable characters
    """
    if not text:
        return ""

    # Fix hyphenated line breaks — very common in academic PDFs
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Normalize unicode (é → e, etc.)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Replace newlines inside paragraphs with space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines into 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", "", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata(
    pdf_path: str,
    csv_row: Optional[pd.Series] = None,
) -> dict:
    """
    Extract paper-level metadata.
    Priority order:
        1. papers_enriched.csv row  (most reliable for arXiv papers)
        2. PDF built-in metadata    (often empty or wrong)
        3. Filename fallback        (paper_id only)
    """
    paper_id = Path(pdf_path).stem   # e.g. "2604.15309v1"
    meta = {
        "paper_id": paper_id,
        "title":    None,
        "authors":  [],
        "year":     None,
        "venue":    None,
        "doi":      None,
        "topics":   [],
        "abstract": None,
        "pdf_path": str(pdf_path),
    }

    # --- Source 1: CSV row (highest priority) ---
    if csv_row is not None:
        meta["paper_id"] = str(csv_row.get("paper_id", paper_id))
        meta["title"]    = csv_row.get("title")    or None
        meta["year"]     = csv_row.get("year")     or None
        meta["venue"]    = csv_row.get("venue")    or None
        meta["abstract"] = csv_row.get("abstract") or None
        meta["pdf_path"] = csv_row.get("pdf_path") or str(pdf_path)

        # Authors — stored as "A; B; C" in CSV
        raw_authors = csv_row.get("authors", "")
        if raw_authors:
            meta["authors"] = [a.strip() for a in str(raw_authors).split(";") if a.strip()]

        # Topics — stored as "topic1|topic2|topic3"
        raw_topics = csv_row.get("topics", "")
        if raw_topics:
            meta["topics"] = [t.strip() for t in str(raw_topics).split("|") if t.strip()]

    # --- Source 2: PDF built-in metadata (fills gaps only) ---
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pdf_meta = pdf.metadata or {}

            if not meta["title"] and pdf_meta.get("Title"):
                meta["title"] = pdf_meta["Title"].strip()

            if not meta["authors"] and pdf_meta.get("Author"):
                raw = pdf_meta["Author"].strip()
                meta["authors"] = [a.strip() for a in re.split(r"[;,]", raw) if a.strip()]

    except Exception:
        pass  # PDF metadata read failure is non-fatal

    # --- Source 3: Filename fallback ---
    if not meta["title"]:
        meta["title"] = meta["paper_id"]

    # Normalize year to int
    try:
        meta["year"] = int(meta["year"]) if meta["year"] else None
    except (ValueError, TypeError):
        meta["year"] = None

    return meta


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from each page using pdfplumber.
    Returns a list of {page_num, text} dicts.
    page_num is 1-indexed to match human-readable citations.
    """
    pages = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned  = clean_text(raw_text)

                # Skip pages that are essentially empty (covers, blank pages)
                if len(cleaned) < 50:
                    continue

                pages.append({
                    "page_num": i,
                    "text":     cleaned,
                })

    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF {pdf_path}: {e}") from e

    return pages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf(
    pdf_path: str,
    csv_row: Optional[pd.Series] = None,
) -> dict:
    """
    Main entry point. Parse a single PDF and return a structured document.

    Args:
        pdf_path : path to the PDF file
        csv_row  : row from papers_enriched.csv as a pd.Series (optional but recommended)

    Returns:
        {
            paper_id, title, authors, year, venue, doi,
            topics, abstract, pdf_path,
            pages: [{ page_num, text }, ...]
            page_count: int
        }
    """
    pdf_path = str(pdf_path)

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    meta  = extract_metadata(pdf_path, csv_row)
    pages = extract_pages(pdf_path)

    return {
        **meta,
        "pages":      pages,
        "page_count": len(pages),
    }


# ---------------------------------------------------------------------------
# Batch helper — used by /ingest endpoint
# ---------------------------------------------------------------------------

def parse_all_pdfs(
    csv_path: str = "data/papers_enriched.csv",
    pdf_dir:  str = "data/pdfs",
) -> list[dict]:
    """
    Parse all PDFs listed in papers_enriched.csv.
    Skips missing files with a warning instead of crashing.

    Returns a list of parsed document dicts.
    """
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        pdf_path = row.get("pdf_path") or f"{pdf_dir}/{row['paper_id']}.pdf"

        if not Path(pdf_path).exists():
            print(f"  SKIP  {pdf_path} not found")
            continue

        try:
            doc = parse_pdf(pdf_path, csv_row=row)
            documents.append(doc)
            print(f"  OK    {row['paper_id']}  ({doc['page_count']} pages)")
        except Exception as e:
            print(f"  FAIL  {row['paper_id']}  {e}")

    print(f"\nParsed {len(documents)}/{len(df)} PDFs successfully.")
    return documents


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    csv_path = "data/papers_enriched.csv"
    df       = pd.read_csv(csv_path)
    row      = df.iloc[0]
    pdf_path = row["pdf_path"]

    print(f"Testing with: {pdf_path}\n")
    doc = parse_pdf(pdf_path, csv_row=row)

    # Print summary
    print(f"paper_id   : {doc['paper_id']}")
    print(f"title      : {doc['title']}")
    print(f"authors    : {doc['authors']}")
    print(f"year       : {doc['year']}")
    print(f"topics     : {doc['topics']}")
    print(f"page_count : {doc['page_count']}")
    print(f"\nPage 1 preview (first 300 chars):")
    print(doc["pages"][0]["text"][:300] if doc["pages"] else "NO PAGES")