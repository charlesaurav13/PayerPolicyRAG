"""
FastAPI backend for PayerPolicy RAG web interface.
"""

import logging
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import pipeline

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job store (in-memory)
# ---------------------------------------------------------------------------
jobs: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Preloading ML models at startup…")
    pipeline.preload_models()
    log.info("Models ready. Server accepting requests.")
    yield


app = FastAPI(title="PayerPolicy RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------
def _run_job(job_id: str, pdf_paths: List[Path], md_dir: Path, upload_dir: Path) -> None:
    job = jobs[job_id]
    job["status"] = "processing"
    all_results: List[Dict] = []

    def _on_progress(msg: str) -> None:
        job["step"] = msg

    try:
        for i, pdf_path in enumerate(pdf_paths):
            job["current_file"] = pdf_path.name
            job["current"]      = i + 1
            job["step"]         = f"Starting {pdf_path.name}"

            try:
                rows = pipeline.run_pipeline(pdf_path, md_dir, progress_cb=_on_progress)
                for row in rows:
                    all_results.append(row)
                    job["results"] = list(all_results)
            except Exception as exc:
                log.error("Pipeline error for %s: %s", pdf_path.name, exc)
                job["errors"].append({"file": pdf_path.name, "error": str(exc)})

        job["status"] = "done"
        job["step"]   = "Complete"

    except Exception as exc:
        log.error("Job %s failed: %s", job_id, exc)
        job["status"] = "error"
        job["step"]   = str(exc)

    finally:
        shutil.rmtree(upload_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/process")
async def process_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    pdf_files = [f for f in files if f.filename and f.filename.lower().endswith(".pdf")]
    if not pdf_files:
        raise HTTPException(status_code=400, detail="All uploaded files must be PDFs")

    job_id     = str(uuid.uuid4())
    upload_dir = Path(tempfile.mkdtemp(prefix=f"payer_upload_{job_id}_"))
    md_dir     = upload_dir / "markdown"
    md_dir.mkdir()

    pdf_paths: List[Path] = []
    for upload in pdf_files:
        dest = upload_dir / upload.filename
        content = await upload.read()
        dest.write_bytes(content)
        pdf_paths.append(dest)

    jobs[job_id] = {
        "status":       "queued",
        "total":        len(pdf_paths),
        "current":      0,
        "current_file": "",
        "step":         "Queued",
        "results":      [],
        "errors":       [],
    }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, pdf_paths, md_dir, upload_dir),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "total": len(pdf_paths)}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id":       job_id,
        "status":       job["status"],
        "total":        job["total"],
        "current":      job["current"],
        "current_file": job["current_file"],
        "step":         job["step"],
        "result_count": len(job["results"]),
        "errors":       job["errors"],
    }


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("done", "processing"):
        raise HTTPException(status_code=400, detail=f"Job status is '{job['status']}'")
    return {"job_id": job_id, "results": job["results"]}


@app.get("/api/health")
async def health():
    return {"status": "ok"}
