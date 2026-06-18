README.md

# Production RAG Engineering

> Engineering methodology for production-grade RAG systems: six core modules, two industry-specific implementations — from document ingestion to continuous evaluation.
> Not a tutorial. Not a demo. These are the decision records left behind after building and debugging real production systems.

---

## What's in This Repo

This repository contains two complete production-scenario implementations:

- **ESG Compliance Detection** (`esg/`): Full pipeline covering GRI standard knowledge base construction, corporate report compliance judgment, and evaluation loop closure.
- **Medical Terminology Standardization** (`medical/`): Clinical abbreviation disambiguation + NER entity recognition + SNOMED-CT standard code mapping, with two initialization options: Milvus standalone and Milvus + Neo4j dual-store.

---

## Technical Highlights

Each highlight below has a corresponding source implementation:

- **Multi-tool PDF Parsing Router** (`loading_service.py`): Routes across PyMuPDF / pypdf / pdfplumber / unstructured based on document type, with unified `page_map` output. Automatic fallback on parse failure. Table structure retention improved from 68% → 99%.

- **Four Chunking Strategies Routed by Document Type** (`chunking_service.py`): Per-page / fixed-size / paragraph-based / sentence-based strategies, with 95% document classification accuracy. Each chunk carries `chunk_id + page_range + word_count` metadata as positional anchors for downstream traceability.

- **Multi-provider Embedding + Batch Write** (`embedding_service.py`): Unified adapter for OpenAI / Bedrock / HuggingFace. OpenAI batched at 20 items/request. Four-layer metadata recorded on write (`chunk_id + page_range + embedding_model + vector_dimension + timestamp`). `chunk_id` links the full chain: chunking → retrieval → judgment.

- **Vector Retrieval + Dual-parameter Filtering** (`search_service.py`): Milvus IVF_FLAT index with `top_k + threshold` dual-parameter recall control. Supports switching between FLAT / IVF_FLAT / IVF_SQ8 / HNSW index types.

- **Multi-model Routing Judgment Engine** (`generation_service.py`): Three model categories routed by scenario — HuggingFace local (Llama2 / DeepSeek, data stays on-premise) / OpenAI GPT-4 (complex logic, 95% accuracy) / DeepSeek API (audit scenarios, exposes reasoning chain). Results stored in `05-generation-results/`.

- **Golden Test Set Evaluation + Dual-metric Scoring** (`evaluation_service.py`): Reads human-annotated CSV (`ID / Disclosure Requirement / Page Number / Compliance Status`), retrieves per record, and computes `score_hit` (precision) and `score_find` (recall). Results stored in `06-evaluation-result/`. Supports `top_k + threshold` parameter comparison testing.

- **Milvus + Neo4j Dual-store Initialization** (`create_milvus_db_with_graph.py`): SNOMED-CT concept nodes written to Milvus; Neo4j graph built in parallel (`ObjectConcept → HAS_DESCRIPTION → Description`). Supports querying all synonym descriptions by concept ID for graph-assisted two-stage disambiguation.

- **Embedding Factory Pattern** (`embedding_factory.py`): Unified interface adapting Bedrock / OpenAI / HuggingFace. Switch models without modifying business logic.

---

## Quantifiable Results

| Metric | Before | After | Measurement Basis |
|:---|:---|:---|:---|
| Compliance judgment accuracy | 70% | **93%** | 80-item golden test set, `score_hit + score_find` dual metrics |
| Terminology disambiguation error rate | 12% | **3%** | 200 ambiguous samples, manual review |
| Long-tail clause miss rate | 60% | **7%** | Golden test set, 3 iterations (chunking → retrieval → prompt) |
| Audit traceability response time | 2 hours | **5 minutes** | 50 audit challenge scenarios, re-tested |
| Cost per judgment | $0.58 | **$0.23** | Rule engine filters 60% of cases; model calls reduced from 58 → 23 |
| Incremental knowledge base update time | 2 hours | **30 minutes** | Average of 10 incremental update runs |
| Graph query latency | >2s | **<300ms** | P99 over 1,000 queries, after table partitioning + composite index |
| Manual review rate | 100% | **15%** | Confidence < 0.8 auto-flagged; remainder judged directly by system |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Document Input                          │
│                PDF / HTML / Structured Table                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                Document Ingestion Pipeline                    │
│                                                               │
│  loading_service  →  parsing_service  →  chunking_service    │
│  (4-tool routing)    (4 parse strategies) (4 chunk strategies)│
│       ↓                                                       │
│  embedding_service  →  Milvus (vector store)                 │
│  (batch write + 4-layer metadata)                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Hybrid Retrieval Layer                      │
│                                                               │
│  Vector Search   ──┐                                          │
│  Graph Traversal ──┼──→  top_k + threshold dual-param filter  │
│  (Neo4j synonym)  ─┘                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Compliance Judgment Engine                    │
│                                                               │
│  Layer 1: Rule Engine      → filters 60% simple cases        │
│       ↓                                                       │
│  Layer 2: Multi-LLM Router → Llama2(local) / GPT-4 / DeepSeek│
│       ↓                                                       │
│  Layer 3: NER Verification → element completeness check      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Full-Chain Traceability                      │
│                                                               │
│  chunk_id links: chunk metadata → retrieval result → judgment │
│  4-layer metadata: identity / position / technical / business │
│  Any conclusion traceable to source document fragment         │
│  (page_range + char_offset)                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 Evaluation & Iteration Loop                   │
│                                                               │
│  Golden Dataset CSV  →  score_hit + score_find               │
│  (annotated page + compliance status)  (precision + recall)  │
│                           ↓                                   │
│       top_k / threshold parameter comparison → regression gate│
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
production-rag-engineering/
├── esg/
│   ├── services/
│   │   ├── loading_service.py       # 4-tool PDF parsing router (PyMuPDF / pypdf / pdfplumber / unstructured)
│   │   ├── parsing_service.py       # 4 structuring strategies (full-text / per-page / by-heading / text+table)
│   │   ├── chunking_service.py      # 4 chunking strategies routed by document type
│   │   ├── embedding_service.py     # Batch write (20/batch) + 4-layer metadata recording
│   │   ├── search_service.py        # Milvus vector retrieval, top_k + threshold dual-param filtering
│   │   └── generation_service.py    # Multi-model routing (Llama2 / GPT-4 / DeepSeek)
│   ├── routers/
│   │   ├── evaluation.py            # Evaluation API entry point (FastAPI router)
│   │   └── evaluation_service.py    # Golden test set evaluation, score_hit + score_find dual metrics
│   └── utils/
│       └── config.py                # Milvus config (4 index types + IVF_FLAT nlist=1024)
└── medical/
    ├── services/
    │   ├── abbr_service.py          # Abbreviation disambiguation (pure LLM / LLM + vector re-ranking)
    │   ├── ner_service.py           # Clinical NER (Clinical-AI-Apollo/Medical-NER, with entity merge & dedup)
    │   └── std_service.py           # SNOMED-CT standard code vector retrieval
    ├── tools/
    │   ├── create_milvus_db.py               # SNOMED-CT single-store init (BGE-M3 + COSINE)
    │   └── create_milvus_db_with_graph.py    # Milvus + Neo4j dual-store init (concept nodes + synonym graph)
    └── utils/
        ├── embedding_config.py      # EmbeddingProvider enum + EmbeddingConfig dataclass
        └── embedding_factory.py     # Embedding factory pattern (Bedrock / OpenAI / HuggingFace unified interface)
```

---

## Engineering Deep-Dive Series

Six engineering retrospectives, one per core module:

| Part | Title | Source |
|:---|:---|:---|
| Part 1 | How do unstructured documents become a searchable knowledge base? Five key engineering decisions in the ingestion pipeline | `loading_service.py` / `parsing_service.py` |
| Part 2 | Why does one system need three chunking strategies — and why shouldn't one document type be chunked at all? | `chunking_service.py` |
| Part 3 | Vector retrieval for domain-specific terminology: from model selection to dual-validation in production | `embedding_service.py` / `search_service.py` |
| Part 4 | High semantic similarity score ≠ correct business conclusion: a three-layer judgment engine from retrieval to quantifiable decisions | `generation_service.py` |
| Part 5 | Installing a black box recorder for your RAG system: 4-layer metadata + 3-level verification for 5-minute root cause analysis | `embedding_service.py` / `evaluation_service.py` |
| Part 6 | RAG recall quality from 60% to 93% — not tuned by intuition | `evaluation_service.py` |