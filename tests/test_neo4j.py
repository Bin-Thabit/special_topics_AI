"""
tests/test_neo4j.py
-------------------
Pytest tests for the Neo4j knowledge graph.
Tests verify that the graph was seeded correctly and
that all 5 Cypher queries return expected results.

Run:
    pytest tests/test_neo4j.py -v
"""

import pytest
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# ── CONNECTION ────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j123")


# ── FIXTURE ───────────────────────────────────────────────────────────────────
# A fixture is a reusable setup that pytest runs before each test.
# Here it opens a Neo4j session and closes it after the test finishes.
# Think of it like setUp/tearDown in other test frameworks.

@pytest.fixture
def session():
    """Opens a Neo4j session for each test, closes it after."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        yield s
    driver.close()


# ── HELPER ────────────────────────────────────────────────────────────────────
def run(session, cypher, **params):
    """Shortcut to run a Cypher query and return all results as a list."""
    return list(session.run(cypher, **params))


# =============================================================================
# GROUP 1: Node count tests
# Verify the graph has the right number of nodes after seeding
# =============================================================================

def test_paper_count(session):
    """Graph must have exactly 200 Paper nodes."""
    result = run(session, "MATCH (p:Paper) RETURN count(p) AS total")
    assert result[0]["total"] == 200, f"Expected 200 papers, got {result[0]['total']}"


def test_author_count(session):
    """Graph must have at least 100 Author nodes (we know it has 745)."""
    result = run(session, "MATCH (a:Author) RETURN count(a) AS total")
    assert result[0]["total"] >= 100, f"Too few authors: {result[0]['total']}"


def test_topic_count(session):
    """Graph must have exactly 20 Topic nodes."""
    result = run(session, "MATCH (t:Topic) RETURN count(t) AS total")
    assert result[0]["total"] == 20, f"Expected 20 topics, got {result[0]['total']}"


def test_venue_count(session):
    """Graph must have at least 1 Venue node."""
    result = run(session, "MATCH (v:Venue) RETURN count(v) AS total")
    assert result[0]["total"] >= 1, f"Expected at least 1 venue, got {result[0]['total']}"


# =============================================================================
# GROUP 2: Relationship count tests
# Verify the graph has relationships between nodes
# =============================================================================

def test_wrote_relationships_exist(session):
    """There must be WROTE relationships connecting Authors to Papers."""
    result = run(session, "MATCH ()-[r:WROTE]->() RETURN count(r) AS total")
    assert result[0]["total"] > 0, "No WROTE relationships found"


def test_about_relationships_exist(session):
    """There must be ABOUT relationships connecting Papers to Topics."""
    result = run(session, "MATCH ()-[r:ABOUT]->() RETURN count(r) AS total")
    assert result[0]["total"] > 0, "No ABOUT relationships found"


def test_published_in_relationships_exist(session):
    """There must be PUBLISHED_IN relationships connecting Papers to Venues."""
    result = run(session, "MATCH ()-[r:PUBLISHED_IN]->() RETURN count(r) AS total")
    assert result[0]["total"] > 0, "No PUBLISHED_IN relationships found"


# =============================================================================
# GROUP 3: Node property tests
# Verify that nodes have the correct properties saved
# =============================================================================

def test_paper_has_required_properties(session):
    """Every Paper node must have paper_id, title, year, pdf_path, abstract."""
    result = run(session, """
        MATCH (p:Paper)
        WHERE p.paper_id IS NULL OR p.title IS NULL OR p.year IS NULL
        RETURN count(p) AS broken
    """)
    assert result[0]["broken"] == 0, f"{result[0]['broken']} papers have missing properties"


def test_author_has_name(session):
    """Every Author node must have a name property."""
    result = run(session, """
        MATCH (a:Author)
        WHERE a.name IS NULL OR a.name = ""
        RETURN count(a) AS broken
    """)
    assert result[0]["broken"] == 0, f"{result[0]['broken']} authors have missing names"


def test_topic_has_name(session):
    """Every Topic node must have a name property."""
    result = run(session, """
        MATCH (t:Topic)
        WHERE t.name IS NULL OR t.name = ""
        RETURN count(t) AS broken
    """)
    assert result[0]["broken"] == 0, f"{result[0]['broken']} topics have missing names"


# =============================================================================
# GROUP 4: Query tests
# Verify that all 5 Cypher queries return meaningful results
# =============================================================================

def test_query1_coauthor_finder(session):
    """Q1: Co-author finder must return at least 1 co-author for Omar El Khalifi."""
    result = run(session, """
        MATCH (a:Author {name: "Omar El Khalifi"})-[:WROTE]->(p:Paper)<-[:WROTE]-(coauthor:Author)
        WHERE a <> coauthor
        RETURN coauthor.name AS coauthor, p.title AS shared_paper
    """)
    assert len(result) > 0, "No co-authors found for Omar El Khalifi"
    # We know from seeding that Omar has 4 co-authors
    assert len(result) == 4, f"Expected 4 co-authors, got {len(result)}"


def test_query2_topic_cluster(session):
    """Q2: Topic cluster must return papers for symbolic_ai."""
    result = run(session, """
        MATCH (p:Paper)-[:ABOUT]->(t:Topic {name: "symbolic_ai"})
        RETURN p.title AS paper
    """)
    assert len(result) > 0, "No papers found for topic symbolic_ai"
    # We know from testing that symbolic_ai has 107 papers
    assert len(result) == 107, f"Expected 107 papers for symbolic_ai, got {len(result)}"


def test_query3_prolific_authors(session):
    """Q3: Prolific authors must return Tianyang Hu as top author with 4 papers."""
    result = run(session, """
        MATCH (a:Author)-[:WROTE]->(p:Paper)
        RETURN a.name AS author, count(p) AS paper_count
        ORDER BY paper_count DESC
        LIMIT 1
    """)
    assert result[0]["author"] == "Tianyang Hu", f"Expected Tianyang Hu, got {result[0]['author']}"
    assert result[0]["paper_count"] == 4, f"Expected 4 papers, got {result[0]['paper_count']}"


def test_query4_topic_overlap(session):
    """Q4: Topic overlap must return papers covering both symbolic_ai and explainability."""
    result = run(session, """
        MATCH (p:Paper)-[:ABOUT]->(t1:Topic {name: "symbolic_ai"})
        MATCH (p)-[:ABOUT]->(t2:Topic {name: "explainability"})
        RETURN p.title AS paper
    """)
    assert len(result) > 0, "No papers found covering both symbolic_ai and explainability"


def test_query5_author_topic_profile(session):
    """Q5: Author topic profile must return papers and topics for Tianyang Hu."""
    result = run(session, """
        MATCH (a:Author {name: "Tianyang Hu"})-[:WROTE]->(p:Paper)-[:ABOUT]->(t:Topic)
        RETURN collect(DISTINCT p.title) AS papers,
               collect(DISTINCT t.name)  AS topics
    """)
    assert len(result) > 0, "No results for Tianyang Hu"
    papers = result[0]["papers"]
    topics = result[0]["topics"]
    assert len(papers) == 4, f"Expected 4 papers for Tianyang Hu, got {len(papers)}"
    assert len(topics) > 0, "No topics found for Tianyang Hu"


# =============================================================================
# GROUP 5: Uniqueness tests
# Verify no duplicate nodes were created (constraints working correctly)
# =============================================================================

def test_no_duplicate_papers(session):
    """No two Paper nodes should have the same paper_id."""
    result = run(session, """
        MATCH (p:Paper)
        WITH p.paper_id AS pid, count(*) AS cnt
        WHERE cnt > 1
        RETURN count(*) AS duplicates
    """)
    assert result[0]["duplicates"] == 0, f"Found duplicate paper nodes!"


def test_no_duplicate_authors(session):
    """No two Author nodes should have the same name."""
    result = run(session, """
        MATCH (a:Author)
        WITH a.name AS name, count(*) AS cnt
        WHERE cnt > 1
        RETURN count(*) AS duplicates
    """)
    assert result[0]["duplicates"] == 0, "Found duplicate author nodes!"


def test_no_duplicate_topics(session):
    """No two Topic nodes should have the same name."""
    result = run(session, """
        MATCH (t:Topic)
        WITH t.name AS name, count(*) AS cnt
        WHERE cnt > 1
        RETURN count(*) AS duplicates
    """)
    assert result[0]["duplicates"] == 0, "Found duplicate topic nodes!"
    