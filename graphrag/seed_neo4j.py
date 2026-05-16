"""
graphrag/seed_neo4j.py
-------------------
Reads papers_enriched.csv and builds a Neo4j knowledge graph with:
  - Paper nodes
  - Author nodes
  - Topic nodes
  - Venue nodes
  - WROTE, ABOUT, PUBLISHED_IN relationships

Run:
    python graphrag/seed_neo4j.py

"""

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ── 1. CONFIG ────────────────────────────────────────────────────────────────
# We load sensitive values from a .env file so we never hardcode passwords.
# Your .env file should contain:
#   NEO4J_URI=bolt://localhost:7687
#   NEO4J_USER=neo4j
#   NEO4J_PASSWORD=1234
#   CSV_PATH=/Users/abdulla/Desktop/special_topics_AI/data/papers_enriched.csv

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",     "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",    "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD","1234")
CSV_PATH       = os.getenv("CSV_PATH", "../special_topics_AI/data/papers_enriched.csv")


# ── 2. CONNECT TO NEO4J ──────────────────────────────────────────────────────
# GraphDatabase.driver() opens a connection pool to Neo4j.
# We wrap everything in a class to keep things clean.

class GraphSeeder:

    def __init__(self, uri, user, password):
        # This creates the connection.
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"✅ Connected to Neo4j at {uri}")

    def close(self):
        # Always close the connection when done.
        self.driver.close()

    def seed(self, csv_path: str):
        """Main method — reads CSV and seeds the graph."""

        # ── 3. READ CSV ──────────────────────────────────────────────────────
        print(f"📂 Reading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Found {len(df)} papers")

        # ── 4. CREATE CONSTRAINTS (INDEXES) ─────────────────────────────────
        # Constraints do two things:
        #   1. Guarantee uniqueness — no duplicate paper_ids, author names, etc.
        #   2. Speed up MERGE — Neo4j uses the index to find existing nodes fast.
        # Think of it like creating a unique index in MongoDB.
        print("🔧 Creating constraints...")
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper)  REQUIRE p.paper_id IS UNIQUE")
            session.run("CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
            session.run("CREATE CONSTRAINT topic_name  IF NOT EXISTS FOR (t:Topic)  REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT venue_name  IF NOT EXISTS FOR (v:Venue)  REQUIRE v.name IS UNIQUE")
        print("   Constraints ready")

        # ── 5. SEED ROW BY ROW ───────────────────────────────────────────────
        # We loop over each paper and run one Cypher query per paper.
        # Each query uses MERGE everywhere — safe to re-run without duplicates.

        success = 0
        errors  = 0

        for _, row in df.iterrows():
            try:
                self._seed_paper(row)
                success += 1
            except Exception as e:
                print(f"   ⚠️  Error on paper {row.get('paper_id','?')}: {e}")
                errors += 1

        print(f"\n✅ Done! {success} papers seeded, {errors} errors")

    def _seed_paper(self, row):
        """Seeds one paper and all its relationships."""

        # ── Parse authors: "Omar El Khalifi; Thomas Rossi" → ["Omar El Khalifi", "Thomas Rossi"]
        raw_authors = str(row.get("authors", ""))
        authors = [a.strip() for a in raw_authors.split(";") if a.strip()]

        # ── Parse topics: "symbolic_ai|heuristic_search" → ["symbolic_ai", "heuristic_search"]
        raw_topics = str(row.get("topics", ""))
        topics = [t.strip() for t in raw_topics.split("|") if t.strip()]

        # ── Clean other fields
        paper_id = str(row.get("paper_id", "")).strip()
        title    = str(row.get("title",    "")).strip()
        venue    = str(row.get("venue",    "")).strip()
        year     = int(row.get("year", 0)) if pd.notna(row.get("year")) else 0
        pdf_path = str(row.get("pdf_path", "")).strip()
        abstract = str(row.get("abstract", "")).strip()

        # ── 6. THE CYPHER QUERY ──────────────────────────────────────────────
        # This single query does everything for one paper:
        #
        #   MERGE (p:Paper {paper_id: $paper_id})   ← find or create Paper
        #   SET p += {...}                           ← update its properties
        #   MERGE (v:Venue {name: $venue})           ← find or create Venue
        #   MERGE (p)-[:PUBLISHED_IN]->(v)           ← draw the arrow
        #
        # The WITH p part passes the paper node into the next section
        # where we loop over authors and topics using UNWIND.
        #
        # UNWIND is like a for-loop in Cypher:
        #   UNWIND ["Omar", "Thomas"] AS author_name
        #   → runs the next line twice, once per name

        cypher = """
        // ── Paper node ──────────────────────────────────────────────────────
        MERGE (p:Paper {paper_id: $paper_id})
        SET   p.title    = $title,
              p.year     = $year,
              p.pdf_path = $pdf_path,
              p.abstract = $abstract

        // ── Venue node + relationship ────────────────────────────────────────
        WITH p
        MERGE (v:Venue {name: $venue})
        MERGE (p)-[:PUBLISHED_IN]->(v)

        // ── Author nodes + WROTE relationships ──────────────────────────────
        // UNWIND expands the list → one Author node per name
        WITH p
        UNWIND $authors AS author_name
            MERGE (a:Author {name: author_name})
            MERGE (a)-[:WROTE]->(p)

        // ── Topic nodes + ABOUT relationships ───────────────────────────────
        WITH p
        UNWIND $topics AS topic_name
            MERGE (t:Topic {name: topic_name})
            MERGE (p)-[:ABOUT]->(t)
        """

        with self.driver.session() as session:
            session.run(
                cypher,
                paper_id = paper_id,
                title    = title,
                year     = year,
                pdf_path = pdf_path,
                abstract = abstract,
                venue    = venue,
                authors  = authors,
                topics   = topics,
            )


# ── 7. ENTRY POINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    seeder = GraphSeeder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        seeder.seed(CSV_PATH)
    finally:
        seeder.close()