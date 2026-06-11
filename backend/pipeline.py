"""
Payer Policy RAG pipeline — OpenDataLoader edition.
Replaces Docling with opendataloader-pdf for PDF→Markdown conversion.
Models (BGE embedder + cross-encoder) are loaded once via preload_models().
"""

import os
import sys

# Make Java (required by opendataloader-pdf) findable.
# On Docker/Linux, java is already on PATH — do nothing.
# On macOS dev (Homebrew), add the keg-only openjdk path.
import shutil as _shutil
if not _shutil.which("java"):
    _JAVA_HOME = "/opt/homebrew/opt/openjdk@21"
    os.environ["PATH"] = _JAVA_HOME + "/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("JAVA_HOME", _JAVA_HOME)

import csv
import hashlib
import json
import logging
import re
import shutil
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import fitz
import numpy as np
import opendataloader_pdf
import pdfplumber
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv(Path(__file__).parent / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL_8B  = "llama-3.1-8b-instant"
GROQ_MODEL_70B = "llama-3.3-70b-versatile"

_GROQ_LIMITS: Dict[str, Dict[str, int]] = {
    "8b":  {"rpm": 30, "tpm": 131_072},
    "70b": {"rpm": 30, "tpm":  12_000},
}

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, ~22MB
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"    # ~80MB (vs 1.1GB bge-reranker-v2-m3)
COLLECTION    = "payer_policy"
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 100
RRF_K         = 60
PARAM_TOP_K   = 4
HEADER_RATIO  = 0.07
FOOTER_RATIO  = 0.07
MAX_CHARS     = 4000

MODEL_CACHE = Path(__file__).parent / "model_cache"
MODEL_CACHE.mkdir(exist_ok=True)

PARAMS = [
    "Age",
    "Step Therapy Requirements Documented in Policy",
    "Number of Steps through Brands",
    "Number of Steps through Generic",
    "Step through-Phototherapy",
    "TB Test required",
    "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)",
    "Reauthorization Required",
    "Reauthorization Requirements Documented in Policy",
    "Specialist Types",
    "Quantity Limits",
]
CSV_COLUMNS = ["filename", "brand"] + PARAMS + ["access_score"]

# ---------------------------------------------------------------------------
# Model singletons — loaded once at startup
# ---------------------------------------------------------------------------
_encoder  = None
_reranker = None


def preload_models() -> None:
    global _encoder, _reranker
    from sentence_transformers import CrossEncoder, SentenceTransformer
    if _encoder is None:
        log.info("Loading embedding model: %s", EMBED_MODEL)
        _encoder = SentenceTransformer(EMBED_MODEL, device=DEVICE, cache_folder=str(MODEL_CACHE))
    if _reranker is None:
        log.info("Loading reranker: %s", RERANK_MODEL)
        _reranker = CrossEncoder(RERANK_MODEL, device=DEVICE, cache_dir=str(MODEL_CACHE))
    log.info("Models ready.")


def _get_encoder():
    if _encoder is None:
        preload_models()
    return _encoder


def _get_reranker():
    if _reranker is None:
        preload_models()
    return _reranker


# ---------------------------------------------------------------------------
# LLM client with rate limiting + cache
# ---------------------------------------------------------------------------
_llm_cache: Dict[str, str] = {}


def _cache_key(messages: List[Dict], model: str) -> str:
    payload = json.dumps({"m": messages, "model": model}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


class RateLimiter:
    def __init__(self, rpm: int, tpm: int, buffer: float = 0.05):
        self._max_rpm = rpm - 1
        self._max_tpm = int(tpm * (1 - buffer))
        self._req_times: deque = deque()
        self._tok_log:   deque = deque()

    def _purge(self) -> None:
        cutoff = time.time() - 60.0
        while self._req_times and self._req_times[0] < cutoff:
            self._req_times.popleft()
        while self._tok_log and self._tok_log[0][0] < cutoff:
            self._tok_log.popleft()

    def wait_if_needed(self, estimated_tokens: int) -> None:
        while True:
            self._purge()
            reqs = len(self._req_times)
            toks = sum(t for _, t in self._tok_log)
            if reqs < self._max_rpm and toks + estimated_tokens <= self._max_tpm:
                return
            oldest = min(
                self._req_times[0]  if self._req_times  else time.time(),
                self._tok_log[0][0] if self._tok_log    else time.time(),
            )
            sleep_for = max(0.5, oldest + 60.1 - time.time())
            log.info("Rate limit — sleeping %.1fs", sleep_for)
            time.sleep(sleep_for)

    def record(self, tokens_used: int) -> None:
        now = time.time()
        self._req_times.append(now)
        self._tok_log.append((now, tokens_used))


class LLMClient:
    def __init__(self, model_size: str = "8b", fallback: Optional["LLMClient"] = None):
        self.model_size = model_size
        self.model      = GROQ_MODEL_8B if model_size == "8b" else GROQ_MODEL_70B
        limits          = _GROQ_LIMITS[model_size]
        self.limiter    = RateLimiter(limits["rpm"], limits["tpm"])
        self._client    = Groq(api_key=GROQ_API_KEY)
        self.fallback   = fallback
        self._degraded  = False

    def complete(self, messages: List[Dict], temperature: float = 0.0,
                 max_tokens: int = 1024, use_llm_cache: bool = True) -> str:
        if self._degraded and self.fallback:
            return self.fallback.complete(messages, temperature, max_tokens, use_llm_cache)

        cache_key = _cache_key(messages, self.model)
        if use_llm_cache and cache_key in _llm_cache:
            return _llm_cache[cache_key]

        estimated = len(json.dumps(messages)) // 4 + max_tokens
        self.limiter.wait_if_needed(estimated)

        try:
            response = self._call(messages, temperature, max_tokens)
        except Exception as exc:
            if self.fallback and ("rate" in str(exc).lower() or "429" in str(exc)):
                self._degraded = True
                return self.fallback.complete(messages, temperature, max_tokens, use_llm_cache)
            raise

        actual = len(json.dumps(messages)) // 4 + len(response) // 4
        self.limiter.record(actual)
        if use_llm_cache:
            _llm_cache[cache_key] = response
        return response

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60),
           retry=retry_if_exception_type(Exception), reraise=True)
    def _call(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# PDF cleaning (PyMuPDF)
# ---------------------------------------------------------------------------
_CRED_PATTERNS = [
    re.compile(r"(username|user[\s_-]*name|login|user[\s_-]*id)\s*[:=]\s*\S+", re.I),
    re.compile(r"(password|passwd|pwd)\s*[:=]\s*\S+", re.I),
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
]

_REF_PATTERNS = [
    re.compile(r'^\s*references\s*$', re.I),
    re.compile(r'the above policy is based on the following references', re.I),
]


def _redact_zone(page: fitz.Page, zone: fitz.Rect) -> None:
    for b in page.get_text("blocks", clip=zone):
        page.add_redact_annot(fitz.Rect(b[:4]), fill=(1, 1, 1))


def _redact_credentials(page: fitz.Page) -> None:
    for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if any(p.search(span.get("text", "")) for p in _CRED_PATTERNS):
                    page.add_redact_annot(fitz.Rect(span["bbox"]), fill=(1, 1, 1))


def _strip_references(doc: fitz.Document) -> None:
    ref_page_idx = ref_y = None
    for page_num in range(len(doc)):
        for b in doc[page_num].get_text("blocks"):
            if any(p.search(b[4].strip()) for p in _REF_PATTERNS):
                ref_page_idx, ref_y = page_num, b[1]
                break
        if ref_page_idx is not None:
            break
    if ref_page_idx is None:
        return
    ref_page = doc[ref_page_idx]
    h, w = ref_page.rect.height, ref_page.rect.width
    ref_page.add_redact_annot(fitz.Rect(0, ref_y, w, h), fill=(1, 1, 1))
    ref_page.apply_redactions()
    for page_num in range(len(doc) - 1, ref_page_idx, -1):
        doc.delete_page(page_num)


def _clean_doc(doc: fitz.Document) -> None:
    for page in doc:
        h, w = page.rect.height, page.rect.width
        _redact_zone(page, fitz.Rect(0, 0, w, h * HEADER_RATIO))
        _redact_zone(page, fitz.Rect(0, h * (1 - FOOTER_RATIO), w, h))
        for link in page.get_links():
            page.delete_link(link)
        _redact_credentials(page)
        page.apply_redactions()


# ---------------------------------------------------------------------------
# PDF → Markdown via opendataloader-pdf (replaces Docling)
# ---------------------------------------------------------------------------
def _is_tabular_pdf(pdf_path: Path, threshold: float = 0.5) -> bool:
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if not pdf.pages:
                return False
            page = pdf.pages[0]
            page_area = page.width * page.height
            if page_area == 0:
                return False
            table_area = sum(
                (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
                for t in page.find_tables()
            )
            return (table_area / page_area) > threshold
    except Exception:
        return False


def _convert_pdf_tabular(pdf_path: Path, md_dir: Path) -> Optional[Path]:
    md_out = md_dir / (pdf_path.stem + ".md")
    if md_out.exists():
        return md_out

    def _clean(cell) -> str:
        return " ".join(str(cell).split()) if cell is not None else ""

    _HEADER_MARKERS = {"Group", "Indication Indicator", "Required Medical Information"}
    try:
        headers: Optional[List[str]] = None
        rows: List[List[str]] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                for table in page.find_tables():
                    extracted = table.extract()
                    if not extracted:
                        continue
                    for row in extracted:
                        cleaned = [_clean(c) for c in row]
                        if headers is None and _HEADER_MARKERS & set(cleaned):
                            headers = cleaned
                            continue
                        if headers and cleaned == headers:
                            continue
                        if any(cleaned):
                            rows.append(cleaned)

        if not rows:
            return None

        lines = ["# Prior Authorization Detail\n"]
        for row in rows:
            if headers:
                row = (row + [""] * len(headers))[:len(headers)]
            lines.append(f"\n## {row[0] if row else 'Unknown'}\n")
            col_range = enumerate(headers[1:], start=1) if headers else enumerate(row[1:], start=1)
            for i, header in col_range:
                val = row[i] if i < len(row) else ""
                if val:
                    lines.append(f"**{header}:** {val}")

        markdown = "\n".join(lines)
        md_dir.mkdir(parents=True, exist_ok=True)
        md_out.write_text(markdown, encoding="utf-8")
        log.info("[tabular] %s | drugs: %d | chars: %d", pdf_path.name, len(rows), len(markdown))
        return md_out
    except Exception as exc:
        log.error("[fail]    %s (pdfplumber) — %s", pdf_path.name, exc)
        return None


def convert_pdf(pdf_path: Path, md_dir: Path) -> Optional[Path]:
    """Convert PDF to Markdown using opendataloader-pdf (with PyMuPDF cleaning)."""
    md_out = md_dir / (pdf_path.stem + ".md")
    if md_out.exists():
        log.info("[skip]    %s — markdown already exists", pdf_path.name)
        return md_out

    if _is_tabular_pdf(pdf_path):
        log.info("[route]   %s — tabular PDF → pdfplumber", pdf_path.name)
        result = _convert_pdf_tabular(pdf_path, md_dir)
        if result:
            return result

    tmp_path: Optional[Path] = None
    tmp_out_dir: Optional[Path] = None
    try:
        doc = fitz.open(str(pdf_path))
        _clean_doc(doc)
        _strip_references(doc)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            doc.save(str(tmp_path), garbage=4, deflate=True)
        doc.close()

        tmp_out_dir = Path(tempfile.mkdtemp(prefix="odl_out_"))
        opendataloader_pdf.convert(
            input_path=str(tmp_path),
            output_dir=str(tmp_out_dir),
            format="markdown",
            quiet=True,
        )

        md_files = list(tmp_out_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError("opendataloader produced no markdown output")

        markdown = md_files[0].read_text(encoding="utf-8")
        md_dir.mkdir(parents=True, exist_ok=True)
        md_out.write_text(markdown, encoding="utf-8")
        log.info("[done]    %s | chars: %d", pdf_path.name, len(markdown))
        return md_out

    except Exception as exc:
        log.error("[fail]    %s — %s", pdf_path.name, exc)
        log.warning("[fallback] %s — retrying with pdfplumber", pdf_path.name)
        return _convert_pdf_tabular(pdf_path, md_dir)

    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
        if tmp_out_dir:
            shutil.rmtree(tmp_out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    text: str
    columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]
_TABLE_RE   = re.compile(r"(\|[^\n]+\|\n\|[-| :]+\|\n(?:\|[^\n]+\|\n)*)", re.MULTILINE)
_HEADER_RE  = re.compile(r"^#{1,4}\s+(.+)", re.MULTILINE)


def _merge_splits(splits: List[str], sep: str, size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    window: List[str] = []
    window_len = 0
    sep_len = len(sep)

    def flush():
        if not window:
            return
        chunk = sep.join(window)
        if len(chunk) <= size:
            chunks.append(chunk)
        else:
            step = max(1, size - overlap)
            for i in range(0, len(chunk), step):
                piece = chunk[i: i + size]
                if piece.strip():
                    chunks.append(piece)

    for s in splits:
        s_len = len(s)
        add_len = s_len + (sep_len if window else 0)
        if window_len + add_len > size:
            flush()
            while window and window_len > overlap:
                removed = window.pop(0)
                window_len -= len(removed) + (sep_len if window else 0)
        window.append(s)
        window_len += s_len + (sep_len if len(window) > 1 else 0)

    flush()
    return chunks


def _recursive_split(text: str, size: int = CHUNK_SIZE,
                     overlap: int = CHUNK_OVERLAP, seps: List[str] = _SEPARATORS) -> List[str]:
    if not text.strip():
        return []
    if len(text) <= size:
        return [text.strip()]

    sep = remaining_seps = None
    for i, s in enumerate(seps):
        if s == "" or s in text:
            sep, remaining_seps = s, seps[i + 1:]
            break

    if sep is None:
        return [text.strip()]
    if sep == "":
        step = max(1, size - overlap)
        return [text[i: i + size].strip() for i in range(0, len(text), step) if text[i: i + size].strip()]

    flat: List[str] = []
    for p in text.split(sep):
        p = p.strip()
        if p:
            flat.append(p) if len(p) <= size else flat.extend(_recursive_split(p, size, overlap, remaining_seps))

    join_sep = sep.strip() or " "
    return [c.strip() for c in _merge_splits(flat, join_sep, size, overlap) if c.strip()]


def _extract_columns(table_text: str) -> List[str]:
    return [c.strip() for c in table_text.strip().splitlines()[0].split("|") if c.strip()]


def _split_table(table_text: str, columns: List[str], size: int, meta: Dict) -> List[Chunk]:
    lines = table_text.strip().splitlines()
    if len(lines) < 3:
        return []
    header_block = lines[0] + "\n" + lines[1] + "\n"
    chunks: List[Chunk] = []
    buf = header_block
    for row in lines[2:]:
        candidate = buf + row + "\n"
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf.strip() != header_block.strip():
                chunks.append(Chunk(text=buf.strip(), columns=columns, metadata=meta, chunk_id=""))
            buf = header_block + row + "\n"
    if buf.strip() and buf.strip() != header_block.strip():
        chunks.append(Chunk(text=buf.strip(), columns=columns, metadata=meta, chunk_id=""))
    return chunks


def chunk_markdown(md_text: str, pdf_name: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    current_header = "Introduction"
    idx = 0

    def _next_id():
        nonlocal idx
        cid = f"{Path(pdf_name).stem}_{idx:04d}"
        idx += 1
        return cid

    segments: List[Dict] = []
    last_end = 0
    for m in _TABLE_RE.finditer(md_text):
        if m.start() > last_end:
            segments.append({"type": "prose", "text": md_text[last_end:m.start()]})
        segments.append({"type": "table", "text": m.group(0)})
        last_end = m.end()
    if last_end < len(md_text):
        segments.append({"type": "prose", "text": md_text[last_end:]})

    for seg in segments:
        if seg["type"] == "table":
            columns = _extract_columns(seg["text"])
            meta = {"table": True, "pdf": pdf_name, "header": current_header}
            if len(seg["text"]) <= CHUNK_SIZE:
                chunks.append(Chunk(chunk_id=_next_id(), text=seg["text"].strip(),
                                    columns=columns, metadata=meta))
            else:
                for c in _split_table(seg["text"], columns, CHUNK_SIZE, meta):
                    c.chunk_id = _next_id()
                    chunks.append(c)
        else:
            prose = seg["text"]
            for hdr in _HEADER_RE.finditer(prose):
                current_header = hdr.group(1).strip()
            for split in _recursive_split(prose):
                for hdr in _HEADER_RE.finditer(split):
                    current_header = hdr.group(1).strip()
                chunks.append(Chunk(chunk_id=_next_id(), text=split, columns=[],
                                    metadata={"table": False, "pdf": pdf_name, "header": current_header}))
    return chunks


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class PolicyStore:
    def __init__(self, chroma_dir: Path):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.encoder  = _get_encoder()
        self.reranker = _get_reranker()

        self.client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"},
        )
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: List[str] = []
        self._bm25_texts: List[str] = []

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        existing = set(self.col.get(include=[])["ids"])
        new = [c for c in chunks if c.chunk_id not in existing]
        if not new:
            return 0
        for i in range(0, len(new), batch_size):
            batch = new[i: i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.encoder.encode(
                texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True,
            ).tolist()
            self.col.add(
                ids=[c.chunk_id for c in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {**c.metadata, "columns": "|".join(c.columns), "table": str(c.metadata.get("table", False))}
                    for c in batch
                ],
            )
        self._bm25 = None
        return len(new)

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        result = self.col.get(include=["documents"])
        self._bm25_ids   = result["ids"]
        self._bm25_texts = result["documents"]
        self._bm25 = BM25Okapi([t.lower().split() for t in self._bm25_texts])

    def hybrid_search(self, query: str, top_k: int = 10, rerank_candidates: int = 30) -> List[Dict[str, Any]]:
        self._ensure_bm25()
        n_candidates = min(rerank_candidates, max(self.col.count(), 1))

        bm25_scores  = self._bm25.get_scores(query.lower().split())
        sparse_ranks = np.argsort(bm25_scores)[::-1][:n_candidates].tolist()

        q_vec  = self.encoder.encode([query], normalize_embeddings=True).tolist()
        dense  = self.col.query(query_embeddings=q_vec, n_results=n_candidates,
                                include=["documents", "metadatas", "distances"])
        dense_ids: List[str] = dense["ids"][0]

        rrf: Dict[str, float] = {}
        for rank, arr_idx in enumerate(sparse_ranks):
            cid = self._bm25_ids[arr_idx]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, cid in enumerate(dense_ids):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        candidate_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:n_candidates]
        fetched = self.col.get(ids=candidate_ids, include=["documents", "metadatas"])
        id_map  = {cid: (doc, meta) for cid, doc, meta in zip(
            fetched["ids"], fetched["documents"], fetched["metadatas"])}

        pairs         = [(query, id_map[cid][0]) for cid in candidate_ids if cid in id_map]
        rerank_scores = self.reranker.predict(pairs)
        ranked        = sorted(zip(candidate_ids, rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "chunk_id": cid,
                "text": id_map[cid][0],
                "columns": [c for c in id_map[cid][1].get("columns", "").split("|") if c],
                "metadata": {
                    "table":  id_map[cid][1].get("table") == "True",
                    "pdf":    id_map[cid][1].get("pdf", ""),
                    "header": id_map[cid][1].get("header", ""),
                },
                "rrf_score":    round(rrf.get(cid, 0.0), 6),
                "rerank_score": round(float(score), 4),
            }
            for cid, score in ranked if cid in id_map
        ]


# ---------------------------------------------------------------------------
# Brand detection (8B)
# ---------------------------------------------------------------------------
_DRUG_LIST_QUERIES = [
    "applicable drug list biologics PsO formulary covered medications",
    "preferred non-preferred biologic brand names plaque psoriasis",
    "step therapy biologic agents psoriasis prior authorization criteria",
]

_BRAND_PROMPT = """\
You are an expert at extracting structured prior authorization policy information from payer policy documents.

Your task is to identify all brands/products in this policy document that are relevant to plaque psoriasis (PsO).

Instructions:
1. Read the provided policy text carefully.
2. Identify all products listed in the Applicable Drug List or equivalent drug list section.
3. Determine whether the policy contains coverage criteria for plaque psoriasis (PsO).
4. Return every product/brand that is relevant to PsO extraction.
5. Include preferred/non-preferred status if explicitly stated.
6. Do not infer brands that are not explicitly listed.
7. If the policy has multiple indications, only identify brands relevant to PsO.
8. Return the brand name in CAPITAL LETTERS only.
9. Return ONLY the brand name — strip any generic/chemical name in parentheses.
   Example: "Tremfya (guselkumab)" → "TREMFYA"

Return strict JSON only in this format:
{{
  "policy_has_pso": "Yes | No",
  "brands_relevant_to_pso": [
    {{
      "brand": "",
      "preferred_status": "Preferred | Non-preferred | Unspecified"
    }}
  ]
}}

Policy Text:
{policy_text}"""


def _clean_brand_name(name: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", name).strip().upper()


def _parse_json_brands(raw: str, pdf_name: str) -> Dict[str, Any]:
    try:
        start  = raw.index("{")
        end    = raw.rindex("}") + 1
        result = json.loads(raw[start:end])
        for b in result.get("brands_relevant_to_pso", []):
            b["brand"] = _clean_brand_name(b.get("brand", ""))
        return result
    except (ValueError, json.JSONDecodeError):
        log.error("Brand JSON parse failed for %s:\n%s", pdf_name, raw[:300])
        return {"policy_has_pso": "No", "brands_relevant_to_pso": []}


def detect_brands(md_path: Path, store: PolicyStore, llm: LLMClient) -> Dict[str, Any]:
    pdf_name = md_path.stem + ".pdf"
    seen: set = set()
    context_chunks: List[Dict] = []
    for query in _DRUG_LIST_QUERIES:
        for r in store.hybrid_search(query, top_k=3):
            if r["metadata"]["pdf"] == pdf_name and r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                context_chunks.append(r)

    if context_chunks:
        policy_text = "\n\n---\n\n".join(r["text"] for r in context_chunks)
        if len(policy_text) > MAX_CHARS:
            policy_text = policy_text[:MAX_CHARS]
    else:
        raw_text = md_path.read_text(encoding="utf-8")
        policy_text = raw_text[:MAX_CHARS]

    messages = [{"role": "user", "content": _BRAND_PROMPT.format(policy_text=policy_text)}]
    raw      = llm.complete(messages, temperature=0.0, max_tokens=2048)
    result   = _parse_json_brands(raw, pdf_name)

    for brand in result.get("brands_relevant_to_pso", []):
        anchor_ids: List[str] = []
        for r in store.hybrid_search(
            f"{brand['brand']} plaque psoriasis prior authorization step therapy criteria", top_k=3
        ):
            if r["metadata"]["pdf"] == pdf_name:
                anchor_ids.append(r["chunk_id"])
        brand["anchor_chunk_ids"] = anchor_ids

    return result


# ---------------------------------------------------------------------------
# Parameter extraction + access score (70B)
# ---------------------------------------------------------------------------
_PARAM_PROMPT = """\
You are an expert in extracting structured prior authorization policy data from payer policy documents.

Extract 12 PsO-specific parameters for the brand below using ONLY the provided policy chunks.

BRAND:
  Name             : {brand_name}
  Preferred status : {preferred_status}

INSTRUCTIONS:
- Extract for plaque psoriasis (PsO) only. Ignore other indications.
- If moderate-to-severe and severe PsO are distinguished, use moderate-to-severe criteria only.
- Universal criteria that apply to all brands must be combined with brand-specific criteria using AND logic.
- If OR statements exist, choose the least restrictive valid path.
- Count only what is explicitly stated. Do not infer.
- Use "NA" for any value not mentioned, unless rules below specify otherwise.
- Output strict JSON only -- no explanation.

PARAMETERS:

1. Age: Age threshold for eligibility. Output "FDA labelled age" if only FDA labelling is mentioned.
2. Step Therapy Requirements Documented in Policy: Full free-text of all step therapy language relevant to PsO.
3. Number of Steps through Brands: Count of branded/biologic steps required. "NA" if none.
4. Number of Steps through Generic: Count of non-biologic/generic steps required. "NA" if none.
5. Step through-Phototherapy: "Yes" if mandatory. "No" if not required. "N/A" if no criteria.
6. TB Test required: "Y" if required. "N" if not required. "NA" if not mentioned.
7. Initial Authorization Duration(in-months): Numeric months. "Unspecified" if not stated.
8. Reauthorization Duration(in-months): Numeric months. "Unspecified" if not stated.
9. Reauthorization Required: "Yes" if documented. "No" if not required. "NA" otherwise.
10. Reauthorization Requirements Documented in Policy: Actual continuation/renewal criteria text.
11. Specialist Types: Specialist type(s) acceptable for prescribing PsO treatment.
12. Quantity Limits: Only explicitly stated quantity limits. "NA" if not stated.

OUTPUT FORMAT -- strict JSON only:
{{
  "brand": "{brand_name}",
  "preferred_status": "{preferred_status}",
  "Age": "",
  "Step Therapy Requirements Documented in Policy": "",
  "Number of Steps through Brands": "",
  "Number of Steps through Generic": "",
  "Step through-Phototherapy": "",
  "TB Test required": "",
  "Initial Authorization Duration(in-months)": "",
  "Reauthorization Duration(in-months)": "",
  "Reauthorization Required": "",
  "Reauthorization Requirements Documented in Policy": "",
  "Specialist Types": "",
  "Quantity Limits": ""
}}

RELEVANT POLICY CHUNKS:
{chunks}"""

_COMMON_QUERIES = [
    "step therapy prior authorization criteria plaque psoriasis PsO",
    "initial authorization duration months reauthorization renewal continuation",
    "TB test tuberculosis quantity limit specialist prescriber dermatologist",
]


def _get_brand_chunks(store: PolicyStore, pdf_name: str, brand_name: str,
                      anchor_chunk_ids: List[str]) -> str:
    seen: set = set()
    texts: List[str] = []

    if anchor_chunk_ids:
        try:
            fetched = store.col.get(ids=anchor_chunk_ids, include=["documents", "metadatas"])
            for cid, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
                if meta.get("pdf") == pdf_name and cid not in seen:
                    seen.add(cid)
                    texts.append(doc)
        except Exception as exc:
            log.warning("Anchor fetch failed for %s: %s", brand_name, exc)

    if len(texts) < PARAM_TOP_K:
        brand_query = f"{brand_name} prior authorization criteria plaque psoriasis step therapy reauthorization approval"
        for r in store.hybrid_search(brand_query, top_k=PARAM_TOP_K):
            if r["metadata"]["pdf"] == pdf_name and r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                texts.append(r["text"])
            if len(texts) >= PARAM_TOP_K:
                break

    for query in _COMMON_QUERIES:
        if len(texts) >= PARAM_TOP_K:
            break
        for r in store.hybrid_search(query, top_k=2):
            if r["metadata"]["pdf"] == pdf_name and r["chunk_id"] not in seen:
                seen.add(r["chunk_id"])
                texts.append(r["text"])
            if len(texts) >= PARAM_TOP_K:
                break

    return "\n\n---\n\n".join(texts[:PARAM_TOP_K])


def _parse_brand_json(raw: str, brand_name: str) -> Dict[str, Any]:
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        log.error("JSON parse failed for brand '%s':\n%s", brand_name, raw[:200])
        return {}


def _score_age(val: str) -> float:
    v = val.strip().lower()
    if v in ("na", "", "fda labelled age", "fda labeled age"):
        return 1.0
    m = re.search(r"(\d+)", v)
    if not m:
        return 0.7
    age = int(m.group(1))
    if age <= 6:   return 1.0
    if age <= 12:  return 0.9
    if age <= 18:  return 0.7
    return 0.4


def _score_steps(val: str) -> float:
    v = val.strip().lower()
    if v in ("na", "", "0"):
        return 1.0
    m = re.search(r"(\d+)", v)
    return max(0.0, 1.0 - int(m.group(1)) * 0.3) if m else 1.0


def _score_duration(val: str) -> float:
    v = val.strip().lower()
    if v in ("na", "", "unspecified"):
        return 0.5
    m = re.search(r"(\d+)", v)
    if not m:
        return 0.5
    months = int(m.group(1))
    if months >= 12: return 1.0
    if months >= 6:  return 0.7
    if months >= 3:  return 0.4
    return 0.2


def _score_yesno(val: str, yes_score: float = 0.3, no_score: float = 1.0) -> float:
    v = val.strip().lower()
    if v in ("yes", "y"):  return yes_score
    if v in ("no", "n"):   return no_score
    return 0.7


def _score_text_present(val: str) -> float:
    return 0.3 if val.strip().lower() not in ("na", "") else 1.0


_WEIGHTS = {
    "Number of Steps through Brands":                    10,
    "Initial Authorization Duration(in-months)":         15,
    "TB Test required":                                  15,
    "Age":                                               20,
    "Number of Steps through Generic":                   10,
    "Step through-Phototherapy":                          5,
    "Step Therapy Requirements Documented in Policy":     5,
    "Reauthorization Required":                           5,
    "Reauthorization Duration(in-months)":                5,
    "Specialist Types":                                   4,
    "Reauthorization Requirements Documented in Policy":  3,
    "Quantity Limits":                                    3,
}

_SCORERS = {
    "Age":                                              _score_age,
    "Step Therapy Requirements Documented in Policy":   _score_text_present,
    "Number of Steps through Brands":                   _score_steps,
    "Number of Steps through Generic":                  _score_steps,
    "Step through-Phototherapy":                        _score_yesno,
    "TB Test required":                                 lambda v: _score_yesno(v, yes_score=0.3, no_score=1.0),
    "Initial Authorization Duration(in-months)":        _score_duration,
    "Reauthorization Duration(in-months)":              _score_duration,
    "Reauthorization Required":                         _score_yesno,
    "Reauthorization Requirements Documented in Policy":_score_text_present,
    "Specialist Types":                                 _score_text_present,
    "Quantity Limits":                                  _score_text_present,
}


def compute_access_score(row: Dict[str, str]) -> int:
    return round(sum(_WEIGHTS[p] * _SCORERS[p](row.get(p, "NA")) for p in _WEIGHTS))


def _flatten_row(filename: str, brand_result: Dict) -> Dict[str, str]:
    row = {"filename": filename, "brand": brand_result.get("brand", "")}
    for p in PARAMS:
        row[p] = str(brand_result.get(p, "NA"))
    row["access_score"] = str(compute_access_score(row))
    return row


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------
def run_pipeline(
    pdf_path: Path,
    md_dir: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, str]]:
    """
    Full pipeline for one PDF.
    progress_cb(message) is called at each major step for UI updates.
    Returns list of per-brand result rows.
    """
    def _progress(msg: str) -> None:
        log.info(msg)
        if progress_cb:
            progress_cb(msg)

    _progress(f"Converting PDF: {pdf_path.name}")
    md_path = convert_pdf(pdf_path, md_dir)
    if md_path is None:
        raise RuntimeError(f"Markdown conversion failed for {pdf_path.name}")

    _progress("Chunking and building vector store...")
    pdf_name = pdf_path.stem + ".pdf"
    md_text  = md_path.read_text(encoding="utf-8")
    chunks   = chunk_markdown(md_text, pdf_name)
    log.info("  Chunks: %d", len(chunks))

    tmp_chroma = Path(tempfile.mkdtemp(prefix="chroma_"))
    try:
        store = PolicyStore(chroma_dir=tmp_chroma)
        store.add_chunks(chunks)

        _progress("Detecting brands (8B model)...")
        llm_8b     = LLMClient("8b")
        brand_data = detect_brands(md_path, store, llm_8b)

        brands = brand_data.get("brands_relevant_to_pso", [])
        log.info("  policy_has_pso=%s  brands=%s",
                 brand_data.get("policy_has_pso"), [b["brand"] for b in brands])

        if brand_data.get("policy_has_pso") != "Yes" or not brands:
            return []

        _progress(f"Extracting parameters for {len(brands)} brand(s)...")
        llm_70b = LLMClient("8b")
        rows: List[Dict[str, str]] = []

        seen_brands: set = set()
        for brand in brands:
            if brand["brand"] in seen_brands:
                continue
            seen_brands.add(brand["brand"])

            _progress(f"  → {brand['brand']}")
            chunks_text = _get_brand_chunks(store, pdf_name, brand["brand"],
                                            brand.get("anchor_chunk_ids", []))
            if not chunks_text.strip():
                continue

            messages = [{"role": "user", "content": _PARAM_PROMPT.format(
                brand_name=brand["brand"],
                preferred_status=brand.get("preferred_status", "Unspecified"),
                chunks=chunks_text,
            )}]
            raw    = llm_70b.complete(messages, temperature=0.0, max_tokens=1024)
            result = _parse_brand_json(raw, brand["brand"])
            if result:
                rows.append(_flatten_row(pdf_name, result))

    finally:
        shutil.rmtree(tmp_chroma, ignore_errors=True)

    return rows
