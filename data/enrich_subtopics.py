"""
scripts/enrich_subtopics.py
----------------------------
Reads data/papers.csv and assigns 2-3 fine-grained subtopics to each paper
using sentence-transformers (fully local, no API, no cost).

How it works:
    1. Combine title + abstract into one text per paper
    2. Embed all combined texts with bge-small-en
    3. Embed all candidate subtopic descriptions
    4. For each paper, pick top-k subtopics by cosine similarity
    5. Save to data/papers_enriched.csv (original never touched)

Usage:
    python scripts/enrich_subtopics.py
    python scripts/enrich_subtopics.py --dry-run
    python scripts/enrich_subtopics.py --top-k 3 --threshold 0.25

Requirements:
    sentence-transformers (already in requirements.txt)
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Subtopic vocabulary — rich descriptions embed better than single words
# ---------------------------------------------------------------------------

SUBTOPICS = {
    # --- Core cs.AI (official scope) ---
    "knowledge_representation":   "modeling world knowledge ontologies and semantic networks",
    "automated_reasoning":        "logic-based deduction theorem proving and formal verification",
    "planning_and_scheduling":    "sequence of actions goal-directed reasoning and heuristics",
    "expert_systems":             "rule-based systems and domain-specific decision engines",
    "uncertainty_in_ai":          "probabilistic reasoning bayesian networks and fuzzy logic",
    "heuristic_search":           "state-space search algorithms A-star and combinatorial optimization",
    "symbolic_ai":                "classical AI foundations and high-level cognitive modeling",
    "logic_programming":          "declarative programming Prolog and constraint logic programming",
    "nonmonotonic_reasoning":     "belief revision and reasoning with incomplete or changing facts",
    "qualitative_reasoning":      "reasoning about physical systems without precise numerical data",
    "distributed_ai_foundations": "foundational structures for intelligent agents and coordination",
    "cognitive_architectures":    "computational models of human thought processes and memory",

    # --- ACM I.2.0 / I.2.11 additions ---
    "ai_ethics_and_safety":       "fairness accountability transparency and AI value alignment",
    "constraint_satisfaction":    "constraint satisfaction problems and combinatorial solving",
    "case_based_reasoning":       "solving new problems by adapting solutions from past cases",
    "ontology_engineering":       "building and maintaining formal ontologies and taxonomies",
    "common_sense_reasoning":     "reasoning about everyday knowledge and intuitive physics",
    "explainability":             "interpretable AI model explanations and transparency methods",
    "temporal_reasoning":         "reasoning about time events sequences and causal ordering",
    "ai_applications":            "applied AI systems in medicine law finance and engineering",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def combine_text(row: pd.Series) -> str:
    """
    Combine title + abstract into a single string for embedding.
    Title is repeated twice to give it slightly more weight.
    Abstract is truncated to 300 chars to keep token count manageable.
    """
    title    = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()

    if len(abstract) > 300:
        abstract = abstract[:300]

    if abstract and abstract.lower() != "nan":
        return f"{title}. {title}. {abstract}"
    return f"{title}. {title}"


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def enrich(input_path: str, output_path: str, top_k: int, threshold: float, dry_run: bool) -> None:
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(input_path)
    required_cols = {"paper_id", "title"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    has_abstract = "abstract" in df.columns
    if has_abstract:
        print("abstract column found — using title + abstract for richer embeddings.")
    else:
        print("No abstract column found — using title only.")

    if dry_run:
        print("DRY-RUN mode - processing first 5 rows only.\n")
        df = df.head(5)

    print("Loading model: mixedbread-ai/mxbai-embed-large-v1")
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

    # Build combined texts
    texts = [combine_text(row) for _, row in df.iterrows()]

    # Embed subtopic descriptions
    subtopic_keys  = list(SUBTOPICS.keys())
    subtopic_descs = list(SUBTOPICS.values())
    print(f"Embedding {len(subtopic_keys)} subtopic labels...")
    subtopic_embeds = model.encode(subtopic_descs, show_progress_bar=False, normalize_embeddings=True)

    # Embed paper texts
    print(f"Embedding {len(texts)} papers (title + abstract)...")
    paper_embeds = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Compute similarity and assign subtopics
    print("\nAssigning subtopics...")
    sim_matrix = cosine_similarity_matrix(paper_embeds, subtopic_embeds)

    assigned = []
    most_topic_assigned = []
    for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc="Papers")):
        sims     = sim_matrix[i]
        top_idx  = np.argsort(sims)[::-1]
        top_tags = []

        for idx in top_idx:
            if sims[idx] >= threshold:
                top_tags.append(subtopic_keys[idx])
            if len(top_tags) == top_k:
                break

        # Guarantee at least 2 tags even if below threshold
        if len(top_tags) < 2:
            for idx in np.argsort(sims)[::-1]:
                if subtopic_keys[idx] not in top_tags:
                    top_tags.append(subtopic_keys[idx])
                if len(top_tags) == 2:
                    break

        assigned.append(top_tags)
        
        # most_topic = the single highest similarity topic
        best_idx = int(np.argmax(sims))
        most_topic_assigned.append(subtopic_keys[best_idx])
        tqdm.write(f"  {str(row.paper_id):<20}  {' | '.join(top_tags)}")

    # Write enriched CSV — original never touched
    df["topics"] = ["|".join(tags) for tags in assigned]
    df["most_topic"] = most_topic_assigned
    df.to_csv(output_path, index=False)

    print(f"\nDone - original CSV untouched.")
    print(f"Enriched CSV saved to: {output_path}")

    # Summary
    all_tags = [t for tags in assigned for t in tags]
    top = Counter(all_tags).most_common(15)
    print(f"\nTop 15 subtopics across {len(df)} papers:")
    for tag, count in top:
        bar = "█" * count
        print(f"  {tag:<35} {count:>4}  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich papers.csv with subtopics using sentence-transformers (fully local)."
    )
    parser.add_argument("--input",     default="data/papers.csv",          help="Path to input CSV")
    parser.add_argument("--output",    default="data/papers_enriched.csv", help="Path to output CSV")
    parser.add_argument("--top-k",     type=int,   default=3,              help="Subtopics per paper (default: 3)")
    parser.add_argument("--threshold", type=float, default=0.25,           help="Min cosine similarity (default: 0.25)")
    parser.add_argument("--dry-run",   action="store_true",                help="Process only first 5 rows")
    args = parser.parse_args()

    enrich(
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
        threshold=args.threshold,
        dry_run=args.dry_run,
    )