// =============================================================================
// graph/queries_browser.cypher
// D2 Report — 5 Cypher Queries demonstrating Neo4j graph traversal
// These queries are written for the Neo4j Browser (hardcoded values)
// For Python usage see: graph/cypher_queries.py
// =============================================================================


// ── QUERY 1: Co-author Finder ─────────────────────────────────────────────────
// Find everyone who has co-authored a paper with a specific author.
// We follow the path: Author → Paper ← Author (arrow flips on the way back)
// WHERE a <> coauthor makes sure the author is not returned as their own co-author.

MATCH (a:Author {name: "Omar El Khalifi"})-[r1:WROTE]->(p:Paper)<-[r2:WROTE]-(coauthor:Author)
WHERE a <> coauthor
RETURN a, r1, p, r2, coauthor


// ── QUERY 2: Topic Cluster ────────────────────────────────────────────────────
// Find all papers and authors connected to a specific topic.
// We follow the full path: Author → Paper → Topic in one MATCH line.
// Change "symbolic_ai" to any topic name to explore different clusters.

MATCH (a:Author)-[r1:WROTE]->(p:Paper)-[r2:ABOUT]->(t:Topic {name: "symbolic_ai"})
RETURN a, r1, p, r2, t


// ── QUERY 3: Prolific Authors ─────────────────────────────────────────────────
// Show all authors and the papers they wrote in the corpus.
// LIMIT 50 keeps the visual graph readable — remove it to see all connections.
// To find the top authors by count, use the version in cypher_queries.py instead.

MATCH (a:Author)-[r:WROTE]->(p:Paper)
RETURN a, r, p
LIMIT 50


// ── QUERY 4: Topic Overlap ────────────────────────────────────────────────────
// Find papers that cover TWO topics at the same time.
// We use two MATCH lines with the same variable (p) — this acts as an AND filter.
// Only papers connected to BOTH symbolic_ai AND explainability are returned.

MATCH (p:Paper)-[r1:ABOUT]->(t1:Topic {name: "symbolic_ai"})
MATCH (p)-[r2:ABOUT]->(t2:Topic {name: "explainability"})
RETURN p, r1, t1, r2, t2


// ── QUERY 5: Author Topic Profile ────────────────────────────────────────────
// Show all papers and research topics for a specific author.
// We follow the full 3-hop path: Author → Paper → Topic.
// Useful in GraphRAG when a user asks "what does researcher X work on?"

MATCH (a:Author {name: "Tianyang Hu"})-[r1:WROTE]->(p:Paper)-[r2:ABOUT]->(t:Topic)
RETURN a, r1, p, r2, t
