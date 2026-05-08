import arxiv
import csv
import os
import time
import requests

SAVE_DIR = "data/pdfs"
CSV_PATH = "data/papers.csv"
QUERY    = "cat:cs.AI"
MAX      = 200

os.makedirs(SAVE_DIR, exist_ok=True)

client = arxiv.Client()
search = arxiv.Search(
    query=QUERY,
    max_results=MAX,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)

rows = []
print(f"Fetching {MAX} papers from arXiv cs.AI...")

for i, paper in enumerate(client.results(search)):
    paper_id = paper.entry_id.split("/")[-1]
    pdf_path = f"{SAVE_DIR}/{paper_id}.pdf"

    if not os.path.exists(pdf_path):
        try:
            r = requests.get(paper.pdf_url, timeout=30)
            with open(pdf_path, "wb") as f:
                f.write(r.content)
            print(f"[{i+1}/{MAX}] Downloaded {paper_id}")
            time.sleep(1)  # be polite to arXiv
        except Exception as e:
            print(f"[{i+1}/{MAX}] FAILED {paper_id}: {e}")
            pdf_path = ""
    else:
        print(f"[{i+1}/{MAX}] Already exists {paper_id}")

    rows.append({
        "paper_id": paper_id,
        "title":    paper.title.replace("\n", " "),
        "authors":  "; ".join(a.name for a in paper.authors[:5]),
        "venue":    "arXiv",
        "year":     paper.published.year,
        "pdf_path": pdf_path,
        "topics":   "cs.AI",
    })

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone! {len(rows)} papers saved to {CSV_PATH}")