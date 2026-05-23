
// ── QUERY 1: Most Popular Topic Full Neighborhood ─────────────────────────────
// Step 1: Find the topic that has the most papers connected to it
// Step 2: From that winning topic, expand outward to all its papers and authors
// Result: One topic node at the center, all its papers and authors around it
// The visual looks like a star — topic in the middle, everything else radiating out

MATCH (t:Topic)<-[:ABOUT]-(p:Paper)
WITH t, count(p) AS paper_count
ORDER BY paper_count DESC
LIMIT 1
MATCH path = (t)<-[:ABOUT]-(p:Paper)<-[:WROTE]-(a:Author)
RETURN path


// ── QUERY 2: Shortest Collaboration Path ─────────────────────────────────────
// Find the shortest chain of connections between two authors who never
// directly wrote a paper together.
// Neo4j automatically tries all possible paths and returns the shortest one.
// [*..10] means: try up to 10 hops — stop searching if no path found within 10
// Result: A chain of Author and Paper nodes showing how the two researchers
// are indirectly connected through the corpus

MATCH (a:Author {name: "Tianyang Hu"}),
      (b:Author {name: "Omar El Khalifi"})
MATCH path = shortestPath((a)-[*..10]-(b))
RETURN path


// ── QUERY 3: Hidden Collaboration Chain (Friend of a Friend) ─────────────────
// Find cases where: Author A worked with Author B, and Author B worked with
// Author C, but Author A and Author C never worked together directly.
// Author B is the hidden bridge between A and C.
// The first MATCH finds the CO_AUTHORED chain: A → B → C
// The second and third MATCH find the actual papers that created those connections
// NOT (a)-[:CO_AUTHORED]-(c) confirms A and C have no direct collaboration
// Result: A chain showing two researchers who are one handshake away from meeting

MATCH (a:Author)-[r1:CO_AUTHORED]->(b:Author)-[r2:CO_AUTHORED]->(c:Author)
WHERE NOT (a)-[:CO_AUTHORED]-(c)
  AND a <> c
WITH a, r1, b, r2, c
MATCH (a)-[r3:WROTE]->(p1:Paper)<-[r4:WROTE]-(b)
MATCH (b)-[r5:WROTE]->(p2:Paper)<-[r6:WROTE]-(c)
RETURN a, r1, b, r2, c, r3, p1, r4, r5, p2, r6
LIMIT 30


// ── QUERY 4: Top 3 Authors — Papers, Topics and Connections ──────────────────
// Step 1: Find the 3 most prolific authors by paper count
// Step 2: collect(a) saves them as a list so we can reference them later
// Step 3: UNWIND loops over the 3 authors one by one
// Step 4: For each author get all their papers and topics
// Step 5: OPTIONAL MATCH tries to find CO_AUTHORED between the top 3 authors
//         OPTIONAL means: if no connection exists, still return the author
//         WHERE b IN top_authors ensures we only show connections within the top 3
// Result: 3 author nodes with all their papers and topics, plus any
//         CO_AUTHORED arrows between them if they ever collaborated

MATCH (a:Author)-[:WROTE]->(p:Paper)
WITH a, count(p) AS paper_count
ORDER BY paper_count DESC
LIMIT 3
WITH collect(a) AS top_authors
UNWIND top_authors AS a
MATCH (a)-[r1:WROTE]->(p:Paper)-[r2:ABOUT]->(t:Topic)
OPTIONAL MATCH (a)-[r3:CO_AUTHORED]->(b:Author)
WHERE b IN top_authors
RETURN a, r1, p, r2, t, r3, b


// ── QUERY 5: Same Topic, Never Collaborated ───────────────────────────────────
// Find pairs of authors who research the same topic but never wrote together.
// The path goes: Author A → Paper1 → Topic ← Paper2 ← Author B
// The topic node sits in the middle as the bridge between the two authors.
// NOT (a)-[:CO_AUTHORED]-(b) confirms they have no direct collaboration
// No arrow direction on CO_AUTHORED check — we check both directions at once
// This is useful for GraphRAG to suggest potential new collaborations:
// "These two researchers work on the same topic but have never met"
// Result: Two author nodes connected through a shared topic with their papers

MATCH (a:Author)-[r1:WROTE]->(p1:Paper)-[r2:ABOUT]->(t:Topic)<-[r3:ABOUT]-(p2:Paper)<-[r4:WROTE]-(b:Author)
WHERE a <> b
  AND NOT (a)-[:CO_AUTHORED]-(b)
RETURN a, r1, p1, r2, t, r3, p2, r4, b
LIMIT 30
