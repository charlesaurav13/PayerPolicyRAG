# Payer Policy PA Parameter Extraction Pipeline

End-to-end RAG pipeline that extracts **12 Prior Authorization (PA) parameters** from Plaque Psoriasis (PsO) payer policy PDFs and generates a structured CSV containing an access score for each brand.

---

## Python Version

Requires **Python 3.13+**.

> Python 3.14+ is not currently supported due to tokenizer build compatibility issues.

---

## Installation

```bash
python --version

# macOS
brew install python@3.13

python3.13 -m venv py313
source py313/bin/activate

pip install -r requirements.txt
```

### API Key Setup

Create a `.env` file:

```bash
cp .env.example .env
```

or export directly:

```bash
export GROQ_API_KEY=gsk_...
```

For Kaggle:

* Add `GROQ_API_KEY` via **Add-ons → Secrets**
* Enable **Attach to Session**

---

## Running the Pipeline

Open `payer_policy_pipeline.ipynb` and execute all cells.

By default, the pipeline processes all PDFs in:

```text
pdfs/Sample_PsO_ADS_Track/
```

To limit the number of PDFs processed, configure `MAX_PDFS` in `.env`:

```env
MAX_PDFS=1    # Process 1 PDF
MAX_PDFS=5    # Process first 5 PDFs
MAX_PDFS=     # Process all PDFs
```

Results are written to:

```text
submission.csv
```

The pipeline automatically skips:

* PDFs whose markdown files already exist
* Brands already present in the output CSV

---

## Architecture

```text
PDF
 │
 ├─ PyMuPDF
 │   └─ Remove headers, footers, links, and credential text
 │
 ├─ Docling
 │   └─ Convert cleaned PDF → Markdown with table detection
 │
 ├─ Chunking
 │   └─ Recursive character splitting
 │      • Chunk Size: 700
 │      • Overlap: 100
 │
 ├─ ChromaDB
 │   └─ Store chunk embeddings
 │
 ├─ Hybrid Retrieval
 │   ├─ BM25 sparse retrieval
 │   ├─ Dense vector retrieval
 │   ├─ Reciprocal Rank Fusion (RRF)
 │   └─ Cross-encoder reranking
 │
 ├─ Brand Detection (Llama 3.1 8B)
 │   └─ Identify brands and collect anchor chunks
 │
 ├─ Parameter Extraction (Llama 3.3 70B)
 │   └─ Extract 12 PA parameters per brand
 │
 └─ CSV Output
     └─ filename, brand, parameters, access_score
```

---

## Key Design Decisions

* **PDF Cleaning:** PyMuPDF removes headers, footers, hyperlinks, and credential text before processing.
* **Markdown Conversion:** Docling converts cleaned PDFs into structured Markdown with table detection.
* **Complex Table Handling:** PyMuPDF is used as a fallback when additional table extraction is required.
* **Token Optimization:** Brand-specific anchor chunks (`brand`, `chunk_id`, `page_number`) are stored and passed to the extraction stage, reducing token usage and improving retrieval quality.
* **Rate Limiting:** Custom request and token rate limiting keeps usage within Groq free-tier constraints.
* **Efficient Embeddings:** Uses lightweight 384-dimensional embeddings for faster inference and lower memory usage.
* **Improved Retrieval:** Combines BM25, dense retrieval, RRF fusion, and reranking to maximize chunk relevance before extraction.

---

## LLMs

| Model                     | Purpose              | Reason                                                                                |
| ------------------------- | -------------------- | ------------------------------------------------------------------------------------- |
| `llama-3.1-8b-instant`    | Brand Detection      | Fast and cost-efficient identification of drug names and anchor sections              |
| `llama-3.3-70b-versatile` | Parameter Extraction | Handles complex reasoning across lengthy and sometimes contradictory policy documents |

Both models are served through the Groq API.

A custom `RateLimiter` tracks request volume and token consumption using a rolling 60-second window to remain within free-tier limits.

---

## Embeddings

| Component              | Model                     | Dimension | Purpose                                      |
| ---------------------- | ------------------------- | --------- | -------------------------------------------- |
| Dense Encoder          | `BAAI/bge-small-en-v1.5`  | 384       | Semantic retrieval of document chunks        |
| Cross-Encoder Reranker | `BAAI/bge-reranker-v2-m3` | —         | Re-ranks BM25 and dense retrieval candidates |

### Why 384 Dimensions?

* Faster embedding generation
* Lower memory consumption
* Open-source model
* Minimal accuracy loss compared to larger embedding models

---

## Access Score

The access score ranges from **1–100**.

Higher scores indicate easier patient access to a drug.

### Formula

```text
access_score = Σ(weight × parameter_score)
```

Maximum possible score:

```text
100
```

A higher score reflects:

* Fewer step therapy restrictions
* Lower age restrictions
* Longer authorization durations
* Fewer specialist requirements
* Reduced testing and reauthorization requirements

---

## Output Format

```csv
filename,
brand,
Age,
Step Therapy Requirements Documented in Policy,
Number of Steps through Brands,
Number of Steps through Generic,
Step through-Phototherapy,
TB Test required,
Initial Authorization Duration,
Reauthorization Duration,
Reauthorization Required,
Reauthorization Requirements Documented in Policy,
Specialist Types,
Quantity Limits,
access_score
```
