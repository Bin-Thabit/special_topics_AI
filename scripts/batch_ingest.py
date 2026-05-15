"""
scripts/batch_ingest.py
------------------------
Ingests all PDFs listed in data/papers_enriched.csv
by calling the /ingest endpoint one by one.

Usage:
    python scripts/batch_ingest.py
    python scripts/batch_ingest.py --limit 10   # test with 10 papers first
    python scripts/batch_ingest.py --workers 2  # parallel uploads

Requirements:
    API server must be running: python api/main.py
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import httpx
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL     = "http://localhost:8000"
INGEST_URL  = f"{API_URL}/ingest"
HEALTH_URL  = f"{API_URL}/health"
TIMEOUT     = 120   # seconds per PDF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_server() -> bool:
    """Check if the API server is running."""
    try:
        r = httpx.get(HEALTH_URL, timeout=5)
        data = r.json()
        print(f"Server alive — model: {data['model']} | chunks: {data['chunks']}")
        return True
    except Exception as e:
        print(f"Server not reachable: {e}")
        return False


def ingest_one(row: dict) -> dict:
    """
    Ingest a single PDF. Returns a result dict with status.
    """
    pdf_path = row.get("pdf_path", "")
    paper_id = str(row.get("paper_id", ""))

    if not Path(pdf_path).exists():
        return {
            "paper_id": paper_id,
            "status":   "skipped — file not found",
            "chunks":   0,
            "pages":    0,
            "elapsed":  0,
        }

    try:
        with open(pdf_path, "rb") as f:
            files    = {"file": (f"{paper_id}.pdf", f, "application/pdf")}
            response = httpx.post(INGEST_URL, files=files, timeout=TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            return {
                "paper_id": paper_id,
                "status":   data.get("status", "ok"),
                "chunks":   data.get("chunks_embedded", 0),
                "pages":    data.get("pages_parsed", 0),
                "elapsed":  data.get("elapsed_seconds", 0),
            }
        else:
            return {
                "paper_id": paper_id,
                "status":   f"error {response.status_code}",
                "chunks":   0,
                "pages":    0,
                "elapsed":  0,
            }

    except Exception as e:
        return {
            "paper_id": paper_id,
            "status":   f"exception: {e}",
            "chunks":   0,
            "pages":    0,
            "elapsed":  0,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def batch_ingest(
    csv_path : str,
    limit    : int | None,
    workers  : int,
    output   : str,
) -> None:

    # Health check
    if not check_server():
        print("Start the server first: python api/main.py")
        return

    # Load CSV
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    rows = df.to_dict(orient="records")
    print(f"\nIngesting {len(rows)} papers with {workers} worker(s)...\n")

    results  = []
    start_all = time.time()

    if workers == 1:
        # Sequential — simpler, easier to debug
        for row in tqdm(rows, desc="Ingesting"):
            result = ingest_one(row)
            results.append(result)
            tqdm.write(
                f"  {result['paper_id']:<25} "
                f"{result['status']:<35} "
                f"chunks={result['chunks']:<4} "
                f"pages={result['pages']:<4} "
                f"{result['elapsed']}s"
            )
    else:
        # Parallel — faster but watch server memory
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(ingest_one, row): row for row in rows}
            for future in tqdm(as_completed(futures), total=len(rows), desc="Ingesting"):
                result = future.result()
                results.append(result)
                tqdm.write(
                    f"  {result['paper_id']:<25} "
                    f"{result['status']:<35} "
                    f"chunks={result['chunks']}"
                )

    total_elapsed = round(time.time() - start_all, 1)

    # Summary
    ok       = [r for r in results if r["status"] == "ok"]
    skipped  = [r for r in results if "skipped" in r["status"]]
    errors   = [r for r in results if "error" in r["status"] or "exception" in r["status"]]

    total_chunks = sum(r["chunks"] for r in results)
    total_pages  = sum(r["pages"]  for r in results)

    print(f"\n{'='*55}")
    print(f"Batch ingest complete in {total_elapsed}s")
    print(f"{'='*55}")
    print(f"  Total papers : {len(rows)}")
    print(f"  OK           : {len(ok)}")
    print(f"  Skipped      : {len(skipped)}")
    print(f"  Errors       : {len(errors)}")
    print(f"  Total chunks : {total_chunks}")
    print(f"  Total pages  : {total_pages}")
    print(f"  Avg chunks   : {total_chunks // max(len(ok), 1)} per paper")

    if errors:
        print(f"\nFailed papers:")
        for r in errors:
            print(f"  {r['paper_id']}: {r['status']}")

    # Save results to JSON
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output}")

    # Final health check — show updated chunk count
    check_server()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch ingest all PDFs into the pipeline."
    )
    parser.add_argument("--csv",     default="data/papers_enriched.csv",    help="Path to enriched CSV")
    parser.add_argument("--limit",   type=int, default=None,                help="Limit number of papers (for testing)")
    parser.add_argument("--workers", type=int, default=1,                   help="Parallel workers (default: 1)")
    parser.add_argument("--output",  default="data/ingest_results.json",    help="Path to save results JSON")
    args = parser.parse_args()

    batch_ingest(
        csv_path=args.csv,
        limit   =args.limit,
        workers =args.workers,
        output  =args.output,
    )