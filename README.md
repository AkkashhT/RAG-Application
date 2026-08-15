# LocalRAG

A fully local, privacy-first Retrieval-Augmented Generation application. **No data ever leaves your machine** — no OpenAI, no Anthropic, no cloud APIs of any kind. Every inference call, embedding, reranking step, and storage write happens on localhost.

```
React UI  ←SSE→  FastAPI backend
                      │
          ┌───────────┴────────────┐
          │                        │
   Document pipeline          Query router
   (Docling parse,            (docs / SQL / both)
    OCR fallback,                   │
    hybrid embed,          ┌────────┴────────┐
    Qdrant upsert)         │                 │
          │           Hybrid search       NL→SQL
          ▼           top-20 (Qdrant)     (schema + few-shot)
       Qdrant                │                 │
  (dense + BM25,        BGE reranker      EXPLAIN + reject
   RRF fusion)          top-5                  │
                             │                 │
                      Confidence gate (skip LLM if below threshold)
                             │
                     Ollama qwen3:14b (local, GPU)
                             │
                    Streamed answer + citations
```

---

## Hardware requirements

| Component | Requirement |
|-----------|-------------|
| NVIDIA GPU | 10 GB+ VRAM recommended (see model tiers below) |
| RAM | 16 GB+ |
| Disk | 30 GB+ (models + data) |
| OS | Linux (Docker + NVIDIA Container Toolkit) |

**Apple Silicon:** See [Mac / Apple Silicon](#mac--apple-silicon) section below.

### Model VRAM tiers

| VRAM | Generation model | Set in `.env` |
|------|-----------------|---------------|
| < 10 GB | `qwen3:8b` | `OLLAMA_GENERATION_MODEL=qwen3:8b` |
| 10–18 GB | `qwen3:14b` *(default)* | — |
| 24 GB+ | `qwen3:27b` | `OLLAMA_GENERATION_MODEL=qwen3:27b` |

Add ~2 GB for `nomic-embed-text` + `bge-reranker-v2-m3` running simultaneously.

---

## Quick start

### 1. Install prerequisites

**NVIDIA Container Toolkit** (required for GPU passthrough into Docker):

```bash
# Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`

### 2. Clone and configure

```bash
git clone <repo-url> localrag
cd localrag
cp .env.example .env   # edit if you want to change model or ports
```

### 3. Start services

```bash
make build    # build backend and frontend images (~10 min first time; reranker weights are baked in)
make up       # start Qdrant, Ollama, backend, frontend
```

### 4. Pull models into Ollama

```bash
make pull-models
# This pulls qwen3:14b (~9 GB) and nomic-embed-text (~270 MB)
```

### 5. Open the app

- **UI:** http://localhost:3000
- **API docs:** http://localhost:8000/docs
- **Health check:** `make health`

---

## Usage

### Uploading documents

1. Go to **Documents** in the left nav.
2. Drag and drop or click to upload PDF, DOCX, TXT, CSV, or MD files.
3. Ingestion runs in the background — watch the status badge change from *Ingesting…* to *Ready*.
4. Pages with no extractable text (scanned PDFs) are automatically OCR'd with Tesseract. Citations from OCR'd pages show a warning badge.

### Connecting a SQL database

1. Go to **Settings → SQL Database Connection**.
2. Enter a connection string. **We strongly recommend a read-only database user** — this is the primary safety boundary.

```bash
# Create a read-only Postgres role (example)
CREATE USER localrag_ro WITH PASSWORD 'yourpassword';
GRANT CONNECT ON DATABASE yourdb TO localrag_ro;
GRANT USAGE ON SCHEMA public TO localrag_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO localrag_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO localrag_ro;
```

Connection string formats:
```
postgresql://localrag_ro:pass@localhost:5432/yourdb
mysql+aiomysql://localrag_ro:pass@localhost:3306/yourdb
sqlite+aiosqlite:///./path/to/your.db
```

3. Click **Connect & Test**.

### Asking questions

Type any question in the chat. The app:
- Classifies intent (docs / SQL / both) using the local LLM
- Runs hybrid vector + BM25 search → BGE reranking → top-5 chunks
- Checks the confidence gate; returns "not found" if nothing is relevant
- Streams the answer with expandable citations

**If the SQL path was used**, the generated SQL query is always shown in a collapsible block beneath the answer. **Verify it looks correct** — the system validates that queries are syntactically valid and read-only, but cannot detect semantically wrong queries (e.g., filtering the wrong column). Showing the SQL to the user is the mitigation for this.

---

## Configuration

All settings are adjustable in the **Settings** page or via `.env`:

| Setting | Default | Notes |
|---------|---------|-------|
| `OLLAMA_GENERATION_MODEL` | `qwen3:14b` | See VRAM tiers above |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Requires re-indexing to change |
| `RERANKER_DEVICE` | `cuda` | `cpu` for CPU-only mode |
| `CONFIDENCE_THRESHOLD` | `0.4` | **Calibrate this** — see eval section |
| `RERANKER_INITIAL_TOP_K` | `20` | Candidates fetched before reranking |
| `RERANKER_TOP_K` | `5` | Chunks passed to LLM after reranking |
| `HYBRID_DENSE_WEIGHT` | `0.6` | Dense vs BM25 fusion weight |
| `CHUNK_SIZE_TOKENS` | `800` | Requires re-indexing to change |
| `CHUNK_OVERLAP_TOKENS` | `100` | Requires re-indexing to change |
| `LLM_TEMPERATURE` | `0.1` | Low = more deterministic answers |
| `MAX_CONCURRENT_LLM_CALLS` | `1` | Keeps at 1 to prevent VRAM OOM |

---

## Evaluation and threshold calibration

The `confidence_threshold` default (0.4) is a starting point, not a verified number. Run the eval script after loading your actual documents:

```bash
# Run the full eval suite
make eval

# Also output threshold calibration (recommended after first document load)
make eval-calibrate
```

The calibration report shows reranker score distributions for known-relevant vs. known-irrelevant questions and suggests a threshold. **Re-run this after loading your real documents** — thresholds tuned on synthetic test cases don't necessarily transfer.

Add your own test cases to `eval/test_set.json`:
```json
{
  "id": "my_001",
  "question": "What is the contract renewal date for Acme Corp?",
  "expected_source_type": "document",
  "expected_answer_gist": "March 31, 2025",
  "notes": "From the contracts/acme_2024.pdf document"
}
```

---

## Running tests

```bash
make test
```

Tests cover:
- Chunking logic (including multi-page table merging regression fixture)
- SQL safety validator (statement rejection + EXPLAIN validation)
- Query router source-selection (forced modes + auto fallback)
- Confidence gate threshold behavior (boundary conditions)
- Concurrency semaphore (overlapping request serialization)
- Ollama failure handling (unreachable, mid-stream crash)

---

## Mac / Apple Silicon

Apple Silicon GPUs cannot pass through to Docker containers. The recommended setup:

1. Install Ollama natively on the host:
   ```bash
   brew install ollama
   ollama serve &
   ollama pull qwen3:14b
   ollama pull nomic-embed-text
   ```

2. In `.env`, set:
   ```
   OLLAMA_BASE_URL=http://host.docker.internal:11434
   RERANKER_DEVICE=cpu
   ```

3. Use the CPU override compose file:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
   ```
   (Remove or comment out the `ollama` service — it's running natively.)

Performance will be significantly slower than GPU — `qwen3:8b` is recommended for Mac.

---

## Project structure

```
localrag/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routers (documents, chat, database, settings)
│   │   ├── core/         # config.py, pipeline.py (RAG + query router)
│   │   ├── db/           # SQLAlchemy models, session factory, SQL connector
│   │   ├── ingestion/    # parsers.py (Docling+OCR), chunker.py, embedder.py, service.py
│   │   └── retrieval/    # qdrant.py (hybrid search), reranker.py (BGE)
│   ├── tests/
│   │   └── test_all.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/        # ChatPage, DocumentsPage, SettingsPage
│   │   └── lib/api.ts    # typed API client
│   ├── Dockerfile
│   └── nginx.conf
├── eval/
│   ├── test_set.json     # hand-written Q&A pairs
│   └── run_eval.py       # RAGAS eval + threshold calibration
├── docker-compose.yml
├── docker-compose.cpu.yml
├── Makefile
└── README.md
```

---

## Cloud dependency audit

Every place where data *could* accidentally leave the machine is documented in the source code with a `CLOUD-DEPENDENCY AUDIT:` comment. Key points:

- **config.py** — `assert_no_cloud_keys()` runs at startup and hard-fails if any cloud LLM environment variables are set (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- **embedder.py** — only calls `OLLAMA_BASE_URL` (localhost). No fallback to any hosted API.
- **qdrant.py** — fastembed BM25 sparse encoding runs locally; documented with a comment explaining what to look for if it ever changes.
- **reranker.py** — weights downloaded from HuggingFace **at image build time**; runtime inference is offline.
- **pipeline.py** — direct Ollama API calls; no LangChain/LlamaIndex (which could silently route to cloud via default integrations).
- **eval/run_eval.py** — RAGAS default LLM (OpenAI) is explicitly overridden with a local Ollama wrapper; documented with a warning if it reverts.

---

## License
Free to use.
MIT


Very nice project.
