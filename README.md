# PayerPolicy RAG — Prior Authorization Access Score Analyzer

An end-to-end RAG pipeline that extracts **12 Prior Authorization (PA) parameters** from Plaque Psoriasis (PsO) payer policy PDFs and computes a **1–100 access score** per brand.

Now ships with a **full-stack web UI** — upload PDFs via drag-and-drop, watch live progress, and explore scores and parameters in a clean dashboard.

---

## Demo

| Upload | Processing | Results |
|--------|------------|---------|
| Drag & drop one or more policy PDFs | Live step-by-step progress per file | Per-brand score gauge + expandable 12-parameter table |

---

## Architecture

```
PDF Upload (browser)
       │
       ▼
 FastAPI Backend  ──────────────────────────────────────────────────
       │
       ├─ PyMuPDF          Strip headers, footers, credentials, references
       │
       ├─ OpenDataLoader   PDF → clean Markdown  (replaces Docling)
       │   (Java-backed)
       │
       ├─ Chunking         Recursive char split  700 chars / 100 overlap
       │
       ├─ ChromaDB         Ephemeral per-job vector store
       │
       ├─ Hybrid Retrieval BM25 + all-MiniLM-L6-v2 (384-dim) + RRF
       │                   + ms-marco-MiniLM-L-6-v2 cross-encoder rerank
       │
       ├─ Brand Detection  llama-3.1-8b-instant  (Groq)
       │
       ├─ Param Extraction llama-3.3-70b-versatile (Groq)
       │   └─ 12 PA parameters per brand
       │
       └─ Access Score     Weighted sum → 1–100
              │
              ▼
     Next.js Frontend  ─────────────────────────────────────────────
       Score gauge cards · Parameter detail table · Multi-PDF support
```

---

## Models

| Component | Model | Size |
|-----------|-------|------|
| Dense embeddings | `sentence-transformers/all-MiniLM-L6-v2` | ~22 MB |
| Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80 MB |
| Brand detection | `llama-3.1-8b-instant` (Groq API) | — |
| Parameter extraction | `llama-3.3-70b-versatile` (Groq API) | — |

---

## Access Score

Ranges **1–100**. Higher = easier patient access.

| Parameter | Weight |
|-----------|--------|
| Age | 20 |
| Initial Authorization Duration | 15 |
| TB Test required | 15 |
| Steps through Brands | 10 |
| Steps through Generic | 10 |
| Step through-Phototherapy | 5 |
| Step Therapy text | 5 |
| Reauthorization Required | 5 |
| Reauthorization Duration | 5 |
| Specialist Types | 4 |
| Reauth Requirements text | 3 |
| Quantity Limits | 3 |

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.13 | `brew install python@3.13` |
| Node.js | 18+ | `brew install node` |
| Java | 21 (for OpenDataLoader) | `brew install openjdk@21` |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/charlesaurav13/PayerPolicyRAG
cd PayerPolicyRAG
```

### 2. Python virtual environment

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 3. Java path (macOS)

```bash
export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"
export JAVA_HOME="/opt/homebrew/opt/openjdk@21"
```

Add both lines to `~/.zshrc` to make permanent.

### 4. API key

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and set:
# GROQ_API_KEY=gsk_...
```

Get a free key at [console.groq.com](https://console.groq.com).

### 5. Frontend dependencies

```bash
cd frontend && npm install && cd ..
```

### 6. Run

**Terminal 1 — Backend:**
```bash
./start_backend.sh
# → http://localhost:8001
```

**Terminal 2 — Frontend:**
```bash
./start_frontend.sh
# → http://localhost:3000
```

Open **http://localhost:3000** in your browser.

> **First run:** the embedding and reranker models (~100 MB total) are downloaded automatically and cached in `backend/model_cache/`.

---

## Web UI Usage

1. Drag one or more payer policy PDFs onto the upload zone (or click to browse).
2. Click **Analyze PDFs**.
3. Watch live progress — PDF conversion → chunking → brand detection → parameter extraction.
4. View per-brand score cards sorted by access score.
5. Click **Show all 12 parameters** on any card to expand the full detail table.
6. Click **← Analyze more PDFs** to start over.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/process` | Upload PDF(s), returns `job_id` |
| `GET` | `/api/status/{job_id}` | Poll processing status + progress |
| `GET` | `/api/results/{job_id}` | Fetch extracted brand results |
| `GET` | `/api/health` | Health check |

### Upload example

```bash
curl -X POST http://localhost:8001/api/process \
  -F "files=@policy.pdf"
# → {"job_id": "...", "total": 1}
```

### Poll status

```bash
curl http://localhost:8001/api/status/<job_id>
# → {"status": "processing", "step": "Detecting brands (8B model)...", ...}
```

### Get results

```bash
curl http://localhost:8001/api/results/<job_id>
# → {"results": [{"brand": "SOTYKTU", "access_score": "76", ...}]}
```

---

## Notebook (original batch pipeline)

The original Jupyter pipeline is preserved at `payer_policy_pipeline.ipynb`. It processes all PDFs in `Sample_PsO_ADS_Track/` and writes `submission.csv`.

```bash
# Install notebook deps (includes docling for the original pipeline)
pip install -r requirements.txt
jupyter notebook payer_policy_pipeline.ipynb
```

---

## Project Structure

```
PayerPolicyRAG/
├── backend/
│   ├── main.py              FastAPI server (jobs, upload, status, results)
│   ├── pipeline.py          RAG pipeline (OpenDataLoader + ChromaDB + Groq)
│   ├── requirements.txt     Python dependencies
│   └── .env.example         API key template
├── frontend/
│   ├── app/
│   │   ├── page.tsx         Full UI — upload, progress, results
│   │   ├── layout.tsx       Root layout
│   │   └── globals.css      Base styles
│   └── package.json
├── Sample_PsO_ADS_Track/    Sample PsO payer policy PDFs
├── payer_policy_pipeline.ipynb  Original batch notebook
├── requirements.txt         Notebook dependencies
├── start_backend.sh         One-command backend launcher
├── start_frontend.sh        One-command frontend launcher
└── README.md
```

---

## Rate Limits (Groq free tier)

| Model | RPM | TPM |
|-------|-----|-----|
| `llama-3.1-8b-instant` | 30 | 131,072 |
| `llama-3.3-70b-versatile` | 30 | 12,000 |

The pipeline's `RateLimiter` tracks requests and tokens in a 60-second sliding window and sleeps automatically when limits are approached.

---

## License

MIT
