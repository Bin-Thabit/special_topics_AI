"""
stores/neo4j_store.py — read-only Neo4j query client for D3.

Your seed_neo4j.py has a GraphSeeder for WRITING the graph.
This is its read-only companion: run a Cypher query, get rows back as dicts.
Same env vars and driver pattern as the seeder, so nothing new to configure.
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


class Neo4jStore:
    """Lightweight read client. Open once, reuse for many queries, close at shutdown."""

    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None):
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "neo4j123")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def run_query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """
        Run a read query and return rows as a list of dicts.
        Each dict is keyed by the names you used in the Cypher RETURN clause.
        Example: RETURN p.paper_id AS paper_id  ->  [{"paper_id": "..."}, ...]
        """
        params = params or {}
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]
        