"""
graphrag/seed_neo4j.py
----------------------
Reads papers_enriched.csv and builds a Neo4j knowledge graph.

Nodes        : Paper, Author, Topic, Venue
Relationships:
  WRITTEN_BY      — Paper → Author
  PUBLISHED_IN    — Paper → Venue
  HAS_TOPIC       — Paper → Topic  (all topics from the `topics` column)
  PRIMARY_TOPIC   — Paper → Topic  (only most_topic)

Query patterns:
  One topic per paper  → MATCH (p:Paper)-[:PRIMARY_TOPIC]->(t:Topic)
  All topics           → MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
  Filter by primary    → MATCH (p:Paper {most_topic: "symbolic_ai"})

Run:
    pip install neo4j pandas python-dotenv
    python graphrag/seed_neo4j.py
"""

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j123")
CSV_PATH       = os.getenv("CSV_PATH",       "data/papers_enriched.csv")


# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT paper_id_unique    IF NOT EXISTS FOR (p:Paper)  REQUIRE p.paper_id IS UNIQUE",
    "CREATE CONSTRAINT author_name_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.name     IS UNIQUE",
    "CREATE CONSTRAINT topic_name_unique  IF NOT EXISTS FOR (t:Topic)  REQUIRE t.name     IS UNIQUE",
    "CREATE CONSTRAINT venue_name_unique  IF NOT EXISTS FOR (v:Venue)  REQUIRE v.name     IS UNIQUE",
    "CREATE INDEX paper_title_index       IF NOT EXISTS FOR (p:Paper)  ON (p.title)",
    "CREATE INDEX paper_topic_index       IF NOT EXISTS FOR (p:Paper)  ON (p.most_topic)",
]

# ── Cypher ────────────────────────────────────────────────────────────────────
#
# Two topic relationship types are created independently:
#
#   HAS_TOPIC      — one rel per topic in the `topics` column (up to 3)
#                    use when you need the full topic coverage of a paper
#
#   PRIMARY_TOPIC  — exactly ONE rel, pointing to most_topic
#                    use when you need one topic per paper
#
# most_topic is ALSO stored as a property on the Paper node
# so you can filter without traversing a relationship at all:
#   MATCH (p:Paper {most_topic: "symbolic_ai"})
#
UPSERT_PAPER = """
MERGE (p:Paper {paper_id: $paper_id})
SET   p.title      = $title,
      p.year       = $year,
      p.pdf_path   = $pdf_path,
      p.abstract   = $abstract,
      p.most_topic = $most_topic

WITH p
MERGE (v:Venue {name: $venue})
MERGE (p)-[:PUBLISHED_IN]->(v)

WITH p
UNWIND $authors AS author_name
    MERGE (a:Author {name: author_name})
    MERGE (p)-[:WRITTEN_BY]->(a)

WITH p
UNWIND $all_topics AS topic_name
    MERGE (t:Topic {name: topic_name})
    MERGE (p)-[:HAS_TOPIC]->(t)

WITH p
MERGE (pt:Topic {name: $most_topic})
MERGE (p)-[:PRIMARY_TOPIC]->(pt)
"""


class GraphSeeder:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"✅ Connected to Neo4j at {uri}")

    def close(self):
        self.driver.close()

    def seed(self, csv_path: str):
        print(f"📂 Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Found {len(df)} papers")

        self._create_schema()
        self._import_papers(df)

    def _create_schema(self):
        print("🔧 Creating constraints and indexes …")
        with self.driver.session() as session:
            for stmt in SCHEMA_STATEMENTS:
                session.run(stmt)
        print("   Schema ready")

    def _import_papers(self, df: pd.DataFrame):
        success, errors = 0, 0

        with self.driver.session() as session:
            for _, row in df.iterrows():
                try:
                    params = self._build_params(row)
                    session.execute_write(self._write_paper, params)
                    success += 1
                except Exception as exc:
                    print(f"   ⚠️  Skipping {row.get('paper_id', '?')}: {exc}")
                    errors += 1

                if (success + errors) % 50 == 0:
                    print(f"   … {success + errors} processed")

        print(f"\n✅ Done — {success} imported, {errors} errors")

    @staticmethod
    def _build_params(row) -> dict:
        authors = [
            a.strip()
            for a in str(row.get("authors", "")).split(";")
            if a.strip()
        ]

        # All topics from the pipe-separated `topics` column
        all_topics = [
            t.strip()
            for t in str(row.get("topics", "")).split("|")
            if t.strip()
        ]

        most_topic = str(row.get("most_topic", "")).strip()

        # Safety: make sure most_topic is always in all_topics
        if most_topic and most_topic not in all_topics:
            all_topics.append(most_topic)

        return {
            "paper_id":   str(row.get("paper_id",  "")).strip(),
            "title":      str(row.get("title",     "")).strip(),
            "venue":      str(row.get("venue",     "")).strip(),
            "year":       int(row["year"]) if pd.notna(row.get("year")) else 0,
            "pdf_path":   str(row.get("pdf_path",  "")).strip(),
            "abstract":   str(row.get("abstract",  "")).strip(),
            "most_topic": most_topic,
            "authors":    authors,
            "all_topics": all_topics,
        }

    @staticmethod
    def _write_paper(tx, params: dict):
        tx.run(UPSERT_PAPER, **params)


if __name__ == "__main__":
    seeder = GraphSeeder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        seeder.seed(CSV_PATH)
    finally:
        seeder.close()