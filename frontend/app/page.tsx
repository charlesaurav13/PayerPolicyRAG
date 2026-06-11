"use client";

import { useCallback, useRef, useState } from "react";

const API = "http://localhost:8001";

const PARAMS = [
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
];

interface BrandResult {
  filename: string;
  brand: string;
  access_score: string;
  [key: string]: string | undefined;
}

interface JobStatus {
  status: "queued" | "processing" | "done" | "error";
  total: number;
  current: number;
  current_file: string;
  step: string;
  result_count: number;
  errors: { file: string; error: string }[];
}

// ── Score Gauge ────────────────────────────────────────────────────────────
function ScoreGauge({ score }: { score: number }) {
  const r = 40;
  const cx = 54;
  const cy = 54;
  const circumference = 2 * Math.PI * r;
  const dashOffset = circumference * (1 - score / 100);
  const color =
    score >= 70 ? "#22c55e" : score >= 40 ? "#f59e0b" : "#ef4444";

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="108" height="108" viewBox="0 0 108 108">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#e2e8f0" strokeWidth="10" />
        <circle
          cx={cx} cy={cy} r={r}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
        <text x={cx} y={cy + 7} textAnchor="middle" fontSize="22" fontWeight="700" fill={color}>
          {score}
        </text>
      </svg>
      <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">Access Score</span>
    </div>
  );
}

// ── Pill ───────────────────────────────────────────────────────────────────
function Pill({ label, value, suffix = "" }: { label: string; value: string; suffix?: string }) {
  return (
    <span className="inline-flex items-center gap-1 bg-slate-100 rounded-lg px-2.5 py-1 text-xs">
      <span className="text-slate-500">{label}:</span>
      <span className="font-semibold text-slate-700">
        {value}{value !== "—" && value !== "NA" ? suffix : ""}
      </span>
    </span>
  );
}

// ── Brand Card ─────────────────────────────────────────────────────────────
function BrandCard({ result }: { result: BrandResult }) {
  const [open, setOpen] = useState(false);
  const score = parseInt(result.access_score || "0", 10);
  const label =
    score >= 70
      ? { text: "Good Access",     cls: "bg-green-100 text-green-700" }
      : score >= 40
      ? { text: "Moderate Access", cls: "bg-amber-100 text-amber-700" }
      : { text: "Limited Access",  cls: "bg-red-100 text-red-700" };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden hover:shadow-md transition-shadow">
      <div className="p-5 flex items-center gap-5">
        <ScoreGauge score={score} />
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-lg font-bold text-slate-800 truncate">{result.brand}</h3>
            <span className={`shrink-0 text-xs font-semibold px-2 py-0.5 rounded-full ${label.cls}`}>
              {label.text}
            </span>
          </div>
          <p className="text-sm text-slate-500 mt-0.5 font-mono truncate">{result.filename}</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Pill label="Steps (Brand)" value={result["Number of Steps through Brands"] ?? "—"} />
            <Pill label="Init Auth"     value={result["Initial Authorization Duration(in-months)"] ?? "—"} suffix=" mo" />
            <Pill label="TB Test"       value={result["TB Test required"] ?? "—"} />
            <Pill label="Reauth"        value={result["Reauthorization Required"] ?? "—"} />
          </div>
        </div>
      </div>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full text-sm text-indigo-600 font-semibold py-2.5 border-t border-slate-100 hover:bg-indigo-50 transition-colors"
      >
        {open ? "Hide parameters ▲" : "Show all 12 parameters ▼"}
      </button>
      {open && (
        <div className="border-t border-slate-100 divide-y divide-slate-50">
          {PARAMS.map(p => (
            <div key={p} className="px-5 py-3 flex gap-4">
              <span className="text-xs font-semibold text-slate-400 w-52 shrink-0 pt-0.5">{p}</span>
              <span className="text-sm text-slate-700 break-words">{result[p] || "NA"}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Drop Zone ──────────────────────────────────────────────────────────────
function DropZone({ onFiles }: { onFiles: (f: File[]) => void }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handle = (files: FileList | null) => {
    if (!files) return;
    const pdfs = Array.from(files).filter(f => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length) onFiles(pdfs);
  };

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files); }}
      onClick={() => inputRef.current?.click()}
      className={`relative cursor-pointer rounded-2xl border-2 border-dashed transition-all py-16 flex flex-col items-center gap-4
        ${dragging ? "border-indigo-500 bg-indigo-50" : "border-slate-300 bg-white hover:border-indigo-400 hover:bg-slate-50"}`}
    >
      <input ref={inputRef} type="file" accept=".pdf" multiple className="hidden"
        onChange={e => handle(e.target.files)} />
      <div className="w-14 h-14 rounded-2xl bg-indigo-100 flex items-center justify-center">
        <svg className="w-7 h-7 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414A1 1 0 0119 9.414V19a2 2 0 01-2 2z" />
        </svg>
      </div>
      <div className="text-center">
        <p className="text-slate-700 font-semibold">
          Drop PDFs here or <span className="text-indigo-600">browse</span>
        </p>
        <p className="text-slate-400 text-sm mt-1">Multiple payer policy PDF files supported</p>
      </div>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────
export default function Home() {
  const [files,   setFiles]   = useState<File[]>([]);
  const [status,  setStatus]  = useState<JobStatus | null>(null);
  const [results, setResults] = useState<BrandResult[]>([]);
  const [error,   setError]   = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const addFiles = useCallback((incoming: File[]) => {
    setFiles(prev => {
      const existing = new Set(prev.map(f => f.name));
      return [...prev, ...incoming.filter(f => !existing.has(f.name))];
    });
    setError(null);
  }, []);

  const removeFile = (name: string) => setFiles(prev => prev.filter(f => f.name !== name));

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  const startPolling = (id: string) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API}/api/status/${id}`);
        if (!res.ok) return;
        const data: JobStatus = await res.json();
        setStatus(data);
        if (data.status === "done" || data.status === "error") {
          stopPolling();
          if (data.status === "done") {
            const rRes = await fetch(`${API}/api/results/${id}`);
            const rData = await rRes.json();
            setResults(rData.results ?? []);
          }
        }
      } catch { /* keep polling */ }
    }, 2000);
  };

  const analyze = async () => {
    if (!files.length) return;
    setError(null);
    setResults([]);
    setStatus(null);
    const form = new FormData();
    files.forEach(f => form.append("files", f));
    try {
      const res = await fetch(`${API}/api/process`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const { job_id } = await res.json();
      startPolling(job_id);
    } catch (e) {
      setError(String(e));
    }
  };

  const reset = () => {
    stopPolling();
    setFiles([]); setStatus(null); setResults([]); setError(null);
  };

  const isProcessing = status && !["done", "error"].includes(status.status);
  const isDone       = status?.status === "done";
  const grouped      = results.reduce<Record<string, BrandResult[]>>((acc, r) => {
    (acc[r.filename] ??= []).push(r);
    return acc;
  }, {});

  return (
    <div className="min-h-screen flex flex-col bg-slate-100">
      {/* Header */}
      <header className="bg-gradient-to-r from-indigo-700 to-indigo-600 shadow-md">
        <div className="max-w-5xl mx-auto px-6 py-5 flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-white/20 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414A1 1 0 0119 9.414V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">PayerPolicy RAG</h1>
            <p className="text-indigo-200 text-sm">Prior Authorization Access Score Analyzer</p>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-10 flex flex-col gap-8">

        {/* Upload section */}
        {!isProcessing && !isDone && (
          <section className="flex flex-col gap-5">
            <div>
              <h2 className="text-2xl font-bold text-slate-800">Analyze Payer Policies</h2>
              <p className="text-slate-500 mt-1">
                Upload one or more payer policy PDFs to extract PA parameters and compute access scores (1–100).
              </p>
            </div>

            <DropZone onFiles={addFiles} />

            {files.length > 0 && (
              <div className="flex flex-col gap-2">
                <p className="text-sm font-semibold text-slate-600">
                  {files.length} file{files.length > 1 ? "s" : ""} selected
                </p>
                <div className="flex flex-wrap gap-2">
                  {files.map(f => (
                    <div key={f.name}
                      className="flex items-center gap-1.5 bg-indigo-50 border border-indigo-200 rounded-lg px-3 py-1.5 text-sm text-indigo-700">
                      <svg className="w-3.5 h-3.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" />
                      </svg>
                      <span className="max-w-[220px] truncate">{f.name}</span>
                      <button onClick={() => removeFile(f.name)}
                        className="ml-1 text-indigo-400 hover:text-red-500 transition-colors leading-none">✕</button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700">
                <strong>Error:</strong> {error}
              </div>
            )}

            <button
              onClick={analyze}
              disabled={!files.length}
              className="self-start bg-indigo-600 text-white font-semibold px-7 py-3 rounded-xl
                hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shadow-sm"
            >
              Analyze {files.length > 0 ? `${files.length} PDF${files.length > 1 ? "s" : ""}` : "PDFs"}
            </button>
          </section>
        )}

        {/* Processing section */}
        {isProcessing && status && (
          <section className="bg-white rounded-2xl border border-slate-200 shadow-sm p-10 flex flex-col items-center gap-6 text-center">
            <div className="w-16 h-16 rounded-full border-4 border-indigo-100 border-t-indigo-600 animate-spin" />
            <div className="flex flex-col gap-1">
              <p className="text-xl font-bold text-slate-800">
                Processing {status.current} of {status.total} PDF{status.total > 1 ? "s" : ""}
              </p>
              {status.current_file && (
                <p className="text-slate-500 text-sm font-mono">{status.current_file}</p>
              )}
              <p className="text-slate-400 text-sm mt-2 max-w-md">{status.step}</p>
            </div>
            <div className="w-full max-w-xs bg-slate-100 rounded-full h-2 overflow-hidden">
              <div
                className="bg-indigo-500 h-2 rounded-full transition-all duration-700"
                style={{ width: `${status.total > 0 ? Math.max(5, (status.current / status.total) * 100) : 5}%` }}
              />
            </div>
            <p className="text-xs text-slate-400 max-w-sm">
              Each PDF runs brand detection (8B) and parameter extraction (70B) via Groq. This typically takes 1–2 minutes per file.
            </p>
          </section>
        )}

        {/* Results section */}
        {isDone && (
          <section className="flex flex-col gap-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <h2 className="text-2xl font-bold text-slate-800">Results</h2>
                <p className="text-slate-500 mt-1">
                  {results.length} brand{results.length !== 1 ? "s" : ""} extracted across{" "}
                  {Object.keys(grouped).length} file{Object.keys(grouped).length !== 1 ? "s" : ""}
                </p>
              </div>
              <button
                onClick={reset}
                className="text-sm text-indigo-600 font-semibold border border-indigo-200 rounded-xl px-4 py-2 hover:bg-indigo-50 transition-colors"
              >
                ← Analyze more PDFs
              </button>
            </div>

            {status?.errors && status.errors.length > 0 && (
              <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-800">
                <strong>Some files had errors:</strong>
                <ul className="mt-1.5 space-y-0.5 list-disc list-inside">
                  {status.errors.map((e, i) => (
                    <li key={i}><span className="font-mono">{e.file}</span>: {e.error}</li>
                  ))}
                </ul>
              </div>
            )}

            {results.length === 0 ? (
              <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center text-slate-400">
                No PsO brands found in the uploaded documents.
              </div>
            ) : (
              Object.entries(grouped).map(([filename, brands]) => (
                <div key={filename} className="flex flex-col gap-3">
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4 text-slate-400 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" />
                    </svg>
                    <h3 className="text-sm font-semibold text-slate-600 font-mono truncate">{filename}</h3>
                    <span className="shrink-0 text-xs bg-slate-200 text-slate-600 px-2 py-0.5 rounded-full">
                      {brands.length} brand{brands.length > 1 ? "s" : ""}
                    </span>
                  </div>
                  <div className="flex flex-col gap-3">
                    {[...brands]
                      .sort((a, b) => parseInt(b.access_score || "0") - parseInt(a.access_score || "0"))
                      .map((r, i) => <BrandCard key={i} result={r} />)}
                  </div>
                </div>
              ))
            )}
          </section>
        )}
      </main>

      <footer className="border-t border-slate-200 bg-white py-4">
        <p className="text-center text-xs text-slate-400">
          PayerPolicy RAG · OpenDataLoader PDF · Groq llama-3.1-8b + llama-3.3-70b · all-MiniLM-L6-v2 (384-dim)
        </p>
      </footer>
    </div>
  );
}
