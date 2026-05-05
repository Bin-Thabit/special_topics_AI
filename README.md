# special_topics_AI — PDF-Papers AI Agent

CSAI415 Course Project — Hybrid Retrieval + GraphRAG with Online Learning and AutoML

## Team
| Member | Responsibility |
|--------|---------------|
| You | Ingestion · Retrieval · GraphRAG · River · API |
| Abdullah | AutoML Baseline · Neo4j Graph · Evaluation · PEFT/QLoRA |

## One-command setup
```powershell
uv venv
uv pip install -r requirements.txt
cp .env.example .env
docker compose up -d
```

## Deliverables
| Tag | Week | Description |
|-----|------|-------------|
| v0.1.0-d1 | 5 | AutoML baseline + River online learner |
| v0.2.0-d2 | 7 | Retrieval stack + Neo4j graph |
| v0.3.0-d3 | 9 | GraphRAG executor + evaluation + safety |
| v1.0.0-d4 | 10/11 | PEFT/QLoRA tuning + final demo |

## Project structure
```
special_topics_AI/
├── ingestion/      # PDF parsing, chunking, embeddings
├── stores/         # MongoDB, Qdrant, Neo4j clients
├── retrieval/      # BM25, dense, hybrid fusion
├── graphrag/       # Cypher queries, subgraph expansion
├── agent/          # ReAct/LangGraph planner + tools
├── adaptation/     # River online learner, ADWIN drift
├── tuning/         # QLoRA fine-tuning scripts
├── evaluation/     # RAGAS, Recall@k, latency
├── api/            # FastAPI app
├── automl/         # Optuna/FLAML search
├── notebooks/      # D1 report notebook
├── data/           # Seed CSV, sample PDFs
├── tests/          # pytest smoke tests
└── docs/           # Architecture diagrams
```
