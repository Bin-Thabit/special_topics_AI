"""
stores/neo4j_store.py — read-only Neo4j query client for D3.

seed_neo4j.py (Abdullah) WRITES the graph via GraphSeeder.
This is its read-only companion: run a Cypher query, get rows back as dicts,
and select the relevant subgraph for a question (GraphRAG Step 1).

Same env vars / driver pattern as the seeder — nothing new to configure.
"""
from __future__ import annotations

import os
import logging

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SCHEMA — matched to Abdullah's seed_neo4j.py (verified).
#   Nodes : (:Paper {paper_id, title, year, abstract, pdf_path, most_topic})
#           (:Author {name})   (:Topic {name})   (:Venue {name})
#   Edges : (Paper)-[:WRITTEN_BY]->(Author)
#           (Paper)-[:HAS_TOPIC]->(Topic)        # every topic (up to 3)
#           (Paper)-[:PRIMARY_TOPIC]->(Topic)    # only most_topic
#           (Paper)-[:PUBLISHED_IN]->(Venue)
# Sanity-check the live graph anytime with Neo4jStore().schema_summary().
# ---------------------------------------------------------------------------


class Neo4jStore:
    """Lightweight read client. Open once, reuse for many queries, close at shutdown."""

    # Cypher for select_subgraph(), kept as a constant for readability.
    # Four candidate sources UNION-ed together, then aggregated per paper.
    _SUBGRAPH_CYPHER = """
    CALL () {
        // (a) the seed papers themselves — strongest signal
        MATCH (p:Paper) WHERE p.paper_id IN $seed_ids
        RETURN p AS paper, 'seed' AS reason
      UNION
        // (b) papers sharing a seed's PRIMARY topic (tight, high-signal)
        MATCH (s:Paper)-[:PRIMARY_TOPIC]->(:Topic)<-[:PRIMARY_TOPIC]-(p:Paper)
        WHERE s.paper_id IN $seed_ids AND NOT p.paper_id IN $seed_ids
        RETURN p AS paper, 'shared_primary_topic' AS reason
      UNION
        // (c) papers sharing an AUTHOR with a seed (co-authorship hop)
        MATCH (s:Paper)-[:WRITTEN_BY]->(:Author)<-[:WRITTEN_BY]-(p:Paper)
        WHERE s.paper_id IN $seed_ids AND NOT p.paper_id IN $seed_ids
        RETURN p AS paper, 'shared_author' AS reason
      UNION
        // (d) papers carrying ANY topic named in the question (broad recall)
        MATCH (p:Paper)-[:HAS_TOPIC]->(t:Topic)
        WHERE t.name IN $topics
        RETURN p AS paper, 'topic_match' AS reason
    }
    WITH paper, collect(DISTINCT reason) AS reasons
    OPTIONAL MATCH (paper)-[:WRITTEN_BY]->(a:Author)
    OPTIONAL MATCH (paper)-[:HAS_TOPIC]->(t:Topic)
    WITH paper, reasons,
         collect(DISTINCT a.name) AS authors,
         collect(DISTINCT t.name) AS topics
    RETURN paper.paper_id   AS paper_id,
           paper.title      AS title,
           paper.year       AS year,
           paper.most_topic AS primary_topic,
           authors,
           topics,
           reasons,
           reduce(sc = 0, r IN reasons |
                  sc + CASE r WHEN 'seed'                 THEN 3
                              WHEN 'shared_author'        THEN 2
                              WHEN 'shared_primary_topic' THEN 2
                              WHEN 'topic_match'          THEN 1
                              ELSE 0 END) AS score
    ORDER BY score DESC, paper_id
    LIMIT $max_papers
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "neo4j123")
        # None -> driver uses the default database ("neo4j")
        self.database = database or os.getenv("NEO4J_DATABASE")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            self.driver.verify_connectivity()
        except Exception as e:  # fail fast with a readable message
            raise RuntimeError(
                f"Could not connect to Neo4j at {uri}. Is the container up and "
                f"are the NEO4J_* env vars correct? ({e})"
            ) from e

    # --- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- generic read ------------------------------------------------------
    def run_query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """
        Run a read query and return rows as a list of dicts.
        Each dict is keyed by the names you used in the Cypher RETURN clause.
        Example: RETURN p.paper_id AS paper_id  ->  [{"paper_id": "..."}, ...]
        """
        params = params or {}
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    # --- GraphRAG Step 1 ---------------------------------------------------
    def select_subgraph(
        self,
        seed_paper_ids: list[str] | None = None,
        topics: list[str] | None = None,
        max_papers: int = 20,
        ids_only: bool = False,
    ) -> list[dict] | list[str]:
        """
        Pick the relevant slice of the graph for a question (GraphRAG Step 1).

        Anchors (use either or both):
          seed_paper_ids : top paper_ids from an initial vector search.
                           Expanded 1 hop via shared topic and shared author.
          topics         : topic names parsed from the question, e.g.
                           ["automated_reasoning", "knowledge_representation"].

        Each returned paper carries a `reasons` list explaining WHY it was
        included (seed / shared_primary_topic / shared_author / topic_match)
        and a `score` so the most-connected papers rank first. That trace is
        handy for the D3 report and ablation, and for debugging odd results.

        Returns
          list[dict] : {paper_id, title, year, primary_topic, authors,
                        topics, reasons, score}
          list[str]  : just paper_ids, when ids_only=True (feeds Step 2 directly)
        """
        seed_paper_ids = seed_paper_ids or []
        topics = topics or []
        if not seed_paper_ids and not topics:
            log.warning("select_subgraph: no seeds and no topics -> empty result")
            return []

        rows = self.run_query(
            self._SUBGRAPH_CYPHER,
            {
                "seed_ids": seed_paper_ids,
                "topics": topics,
                "max_papers": int(max_papers),
            },
        )
        if ids_only:
            return [r["paper_id"] for r in rows]
        return rows

    # --- sanity helper -----------------------------------------------------
    def schema_summary(self) -> dict:
        """
        Quick check that the live graph matches the schema assumptions above.
        Returns node-label counts, relationship-type counts, and one Paper's
        property keys. Run this once after seeding if select_subgraph() returns
        nothing — the usual culprit is a property-name mismatch.
        """
        labels = self.run_query(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC"
        )
        rels = self.run_query(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC"
        )
        sample = self.run_query("MATCH (p:Paper) RETURN keys(p) AS props LIMIT 1")
        return {
            "node_labels": labels,
            "relationships": rels,
            "paper_property_keys": sample[0]["props"] if sample else [],
        }


if __name__ == "__main__":
    # Smoke test: python -m stores.neo4j_store
    with Neo4jStore() as store:
        print("Schema:", store.schema_summary())
        demo = store.select_subgraph(
            seed_paper_ids=["2605.06667v1"],          # replace with a real paper_id
            topics=["planning_and_scheduling"],         # replace with a real topic
            max_papers=10,
        )
        for row in demo:
            print(f"[{row['score']}] {row['paper_id']}  {row['reasons']}  {row['title'][:60]}")