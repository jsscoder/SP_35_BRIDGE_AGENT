"use client";

import { ChangeEvent, DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

type RunStatus = "uploaded" | "queued" | "running" | "completed" | "completed_with_warnings" | "failed" | "interrupted";
type Run = {
  id: string;
  created_at: string;
  updated_at: string;
  status: RunStatus;
  files: string[];
  tile: boolean;
  error: string | null;
  review_url: string | null;
  inference: {
    photos?: number;
    ok?: number;
    ok_reduced?: number;
    needs_batch_gpu?: number;
    error?: number;
    model?: { gpu?: string; dtype?: string; sam3?: string };
  } | null;
  review: {
    photos_with_successful_inference?: number;
    inference_status_counts?: Record<string, number>;
    detection_counts_by_class?: Record<string, number>;
    note?: string;
  } | null;
  stats: {
    photos: number;
    detections: number;
    classes: Record<string, number>;
    seconds_per_photo: number | null;
    peak_vram_gib: number | null;
  };
};

type Health = { status: string; local_only: boolean; checkpoint_present: boolean; worker_python_configured: boolean };

function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true as const };
  switch (name) {
    case "bridge": return <svg {...common}><path d="M3 19h18M5 19v-5m14 5v-5M5 14c3-6 11-6 14 0M8 14v5m8-5v5M3 11h18" /><path d="M12 8V4m-2 2h4" /></svg>;
    case "upload": return <svg {...common}><path d="M12 16V4m-5 5 5-5 5 5M4 16v4h16v-4" /></svg>;
    case "image": return <svg {...common}><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5L5 21" /></svg>;
    case "scan": return <svg {...common}><path d="M4 8V5a1 1 0 0 1 1-1h3m8 0h3a1 1 0 0 1 1 1v3m0 8v3a1 1 0 0 1-1 1h-3m-8 0H5a1 1 0 0 1-1-1v-3M4 12h16" /></svg>;
    case "chart": return <svg {...common}><path d="M4 19V5m0 14h17M8 15l3-4 3 2 5-7" /><path d="M16 6h3v3" /></svg>;
    case "clock": return <svg {...common}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></svg>;
    case "cpu": return <svg {...common}><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 1v3m6-3v3m-6 16v3m6-3v3M1 9h3m-3 6h3m16-6h3m-3 6h3M9 9h6v6H9z" /></svg>;
    case "arrow": return <svg {...common}><path d="M5 12h14m-6-6 6 6-6 6" /></svg>;
    case "external": return <svg {...common}><path d="M14 4h6v6m-11 3L20 4" /><path d="M18 13v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h5" /></svg>;
    case "check": return <svg {...common}><path d="m5 12 4 4L19 6" /></svg>;
    case "close": return <svg {...common}><path d="m18 6-12 12M6 6l12 12" /></svg>;
    default: return <svg {...common}><circle cx="12" cy="12" r="9" /><path d="M12 8v4m0 4h.01" /></svg>;
  }
}

function shortId(id: string) { return id.slice(0, 8).toUpperCase(); }
function formatDate(value: string) {
  return new Intl.DateTimeFormat("en", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}
function formatBytes(bytes: number) {
  return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
function statusLabel(status: RunStatus) {
  return ({ uploaded: "Ready to run", queued: "In queue", running: "Analysing", completed: "Complete", completed_with_warnings: "Finished with warnings", failed: "Failed", interrupted: "Interrupted" })[status];
}

export default function Home() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [tile, setTile] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [runResponse, healthResponse] = await Promise.all([
        fetch(`${API}/api/runs`, { cache: "no-store" }),
        fetch(`${API}/api/health`, { cache: "no-store" }),
      ]);
      if (!runResponse.ok || !healthResponse.ok) throw new Error("Could not connect to the local API.");
      const runBody = await runResponse.json() as { runs: Run[] };
      setRuns(runBody.runs);
      setHealth(await healthResponse.json() as Health);
      setError("");
    } catch {
      setHealth(null);
      setError(`Backend unavailable. Start the local API at ${API}.`);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    const timer = window.setInterval(() => void refresh(), 1800);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refresh]);

  const selected = runs.find((run) => run.id === selectedId) ?? runs[0] ?? null;
  const busyRuns = runs.some((run) => run.status === "queued" || run.status === "running");
  const totals = useMemo(() => ({
    photos: runs.reduce((sum, run) => sum + run.stats.photos, 0),
    detections: runs.reduce((sum, run) => sum + run.stats.detections, 0),
    complete: runs.filter((run) => run.status === "completed" || run.status === "completed_with_warnings").length,
  }), [runs]);

  function addFiles(list: FileList | File[]) {
    const incoming = Array.from(list);
    const accepted = incoming.filter((file) => ["image/jpeg", "image/png"].includes(file.type) || /\.(jpe?g|png)$/i.test(file.name));
    if (accepted.length !== incoming.length) setError("Only JPG, JPEG, and PNG images are supported.");
    setFiles((current) => {
      const next = [...current];
      for (const file of accepted) {
        if (!next.some((item) => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified)) next.push(file);
      }
      return next;
    });
  }

  function onInputChange(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) addFiles(event.target.files);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    addFiles(event.dataTransfer.files);
  }

  async function uploadFiles() {
    if (!files.length) return;
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      files.forEach((file) => form.append("files", file));
      const response = await fetch(`${API}/api/runs`, { method: "POST", body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "Upload failed.");
      setFiles([]);
      setSelectedId(body.id);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function startAnalysis(run: Run) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}/api/runs/${run.id}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tile }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "Could not start analysis.");
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not start analysis.");
    } finally {
      setBusy(false);
    }
  }

  const classes = Object.entries(selected?.stats.classes ?? {}).sort((a, b) => b[1] - a[1]).slice(0, 7);
  const maxClass = Math.max(1, ...classes.map(([, count]) => count));
  const active = selected?.status === "running" || selected?.status === "queued";
  return (
    <main className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="BridgeSight home">
          <span className="brand-mark"><Icon name="bridge" size={23} /></span>
          <span><strong>BridgeSight</strong><small>INSPECTION STUDIO</small></span>
        </a>
        <div className="sidebar-label">WORKSPACE</div>
        <button className="nav-item active"><span className="nav-icon"><Icon name="image" /></span>Photo review<span className="nav-count">{runs.length}</span></button>
        <div className="sidebar-label recent-label">RECENT RUNS</div>
        <div className="run-list">
          {runs.length ? runs.slice(0, 8).map((run) => (
            <button key={run.id} className={`run-nav ${selected?.id === run.id ? "selected" : ""}`} onClick={() => setSelectedId(run.id)}>
              <span className={`run-dot ${run.status}`} />
              <span className="run-nav-copy"><strong>{shortId(run.id)}</strong><small>{run.files.length} {run.files.length === 1 ? "photo" : "photos"} · {formatDate(run.created_at)}</small></span>
            </button>
          )) : <p className="empty-recent">Your recent analysis runs will appear here.</p>}
        </div>
        <div className="sidebar-bottom">
          <div className="local-card">
            <span className="local-lock"><span /></span>
            <div><strong>Local &amp; private</strong><small>Photos stay on this machine</small></div>
          </div>
          <div className="sidebar-footer"><span>SP:35 · Chapter 7</span><span className="version-tag">PREVIEW</span></div>
        </div>
      </aside>

      <section className="main-column">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> <strong>Photo review</strong></div>
          <div className="topbar-right">
            <div className={`api-indicator ${health ? "online" : "offline"}`}><span />{health ? "Local engine connected" : "Engine offline"}</div>
            <div className="avatar">BS</div>
          </div>
        </header>

        <div className="content">
          <section className="welcome-row">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> BRIDGE INSPECTION WORKSPACE</div>
              <h1>See the detail.<br /><em>Build the picture.</em></h1>
              <p className="welcome-copy">Review inspection photos with local AI assistance. Every detection is a proposal for an engineer to assess.</p>
            </div>
            <div className="bridge-illustration" aria-hidden="true">
              <div className="sun-disc" /><div className="horizon" />
              <svg viewBox="0 0 380 170" fill="none">
                <path d="M13 128h354M29 128V84m322 44V84M29 84c62-82 260-82 322 0M72 83v45m59-65v65m59-72v72m59-65v65m59-45v45" stroke="currentColor" strokeWidth="2" />
                <path d="M16 82h348M29 128l-10 15h32l-11-15m281 0-10 15h32l-11-15" stroke="currentColor" strokeWidth="2" />
                <path d="M0 145h380" stroke="currentColor" strokeOpacity=".22" />
              </svg>
              <span className="illustration-caption">FIELD NOTES · 01</span>
            </div>
          </section>

          {error && <div className="alert" role="alert"><Icon name="info" size={18} /><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><Icon name="close" size={16} /></button></div>}
          {health && (!health.checkpoint_present || !health.worker_python_configured) && (
            <div className="alert warn" role="status"><Icon name="info" size={18} /><span>{!health.checkpoint_present ? "SAM checkpoint not found. Uploads are available, but analysis will fail until the local model weights are installed." : "SAM worker Python is not configured."}</span></div>
          )}

          <section className="upload-card">
            <div className="section-heading">
              <div><span className="step-number">01</span><div><h2>Add inspection photos</h2><p>Choose one image or a batch. Analysis starts only when you press Run.</p></div></div>
              <span className="supported-tag"><span /> JPG · PNG</span>
            </div>
            <div className={`drop-zone ${dragging ? "dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
              <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,image/jpeg,image/png" multiple onChange={onInputChange} hidden />
              <div className="upload-icon"><Icon name="upload" size={23} /></div>
              <strong>Drop your photos here</strong>
              <span>or <button className="browse-button" onClick={() => inputRef.current?.click()}>browse files</button> from your device</span>
              <small>Up to 200 images · 25 MiB each · 500 MiB total · processed locally</small>
            </div>
            {files.length > 0 && (
              <div className="selected-files">
                <div className="selected-header"><strong>{files.length} selected</strong><span>{formatBytes(files.reduce((sum, file) => sum + file.size, 0))} total</span><button onClick={() => setFiles([])}>Clear all</button></div>
                <div className="file-chips">{files.slice(0, 4).map((file) => <span className="file-chip" key={`${file.name}-${file.lastModified}`}><Icon name="image" size={14} />{file.name}<small>{formatBytes(file.size)}</small><button aria-label={`Remove ${file.name}`} onClick={() => setFiles((current) => current.filter((item) => item !== file))}><Icon name="close" size={13} /></button></span>)}{files.length > 4 && <span className="more-files">+{files.length - 4} more</span>}</div>
                <button className="primary-button upload-button" disabled={busy || !health} onClick={() => void uploadFiles()}>{busy ? <span className="spinner" /> : <Icon name="upload" size={18} />}{busy ? "Uploading…" : `Upload ${files.length === 1 ? "photo" : `${files.length} photos`}`}<Icon name="arrow" size={17} /></button>
              </div>
            )}
          </section>

          <section className="workspace-grid">
            <div className="analysis-column">
              {selected && (
                <section className="run-card">
                  <div className="section-heading compact">
                    <div><span className="step-number">02</span><div><h2>Analysis run <span className="run-id">#{shortId(selected.id)}</span></h2><p>{selected.files.length} {selected.files.length === 1 ? "photo" : "photos"} · added {formatDate(selected.created_at)}</p></div></div>
                    <span className={`status-pill ${selected.status}`}><span />{statusLabel(selected.status)}</span>
                  </div>
                  {selected.status === "uploaded" || selected.status === "interrupted" ? (
                    <div className="run-ready">
                      <div className="ready-icon"><Icon name="scan" size={23} /></div>
                      <div className="ready-copy"><strong>Ready when you are</strong><span>Run SAM 3.1 to find candidate components and distress regions.</span></div>
                      <label className="tile-toggle"><input type="checkbox" checked={tile} onChange={(event) => setTile(event.target.checked)} /><span className="toggle-track" /><span>High-detail tile pass</span></label>
                      <button className="primary-button analyze-button" disabled={busy || !health?.checkpoint_present} onClick={() => void startAnalysis(selected)}><Icon name="scan" size={18} />Run analysis<Icon name="arrow" size={17} /></button>
                    </div>
                  ) : active ? (
                    <div className="progress-panel">
                      <div className="progress-ring"><span className="spinner" /></div>
                      <div className="progress-text"><strong>{selected.status === "queued" ? "Waiting for the GPU…" : "Reading the structure…"}</strong><span>{selected.status === "queued" ? "This run will start as soon as the current job finishes." : "SAM is generating proposals. Larger batches may take a while."}</span></div>
                      <span className="live-tag"><span /> LIVE</span>
                    </div>
                  ) : (
                    <div className="run-complete-banner">
                      <span className={`complete-icon ${selected.status}`}><Icon name={selected.status === "failed" ? "close" : "check"} size={19} /></span>
                      <div><strong>{selected.status === "failed" ? "Analysis needs attention" : selected.status === "completed_with_warnings" ? "Analysis finished with warnings" : "Analysis and plots are ready"}</strong><span>{selected.error}</span></div>
                      {selected.review_url && <a className="text-link" href={`${API}${selected.review_url}`} target="_blank" rel="noreferrer">Open plotting report <Icon name="external" size={15} /></a>}
                    </div>
                  )}
                  {selected.status === "failed" && <button className="retry-button" onClick={() => void startAnalysis(selected)}>Retry this run <Icon name="arrow" size={15} /></button>}
                </section>
              )}

              <section className="stats-section">
                <div className="stats-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> RUN OVERVIEW</div><h2>Analysis at a glance</h2></div><span className="stats-context">{selected ? `RUN #${shortId(selected.id)}` : "ALL RUNS"}</span></div>
                <div className="metric-grid">
                  <Metric icon="image" label="Photos reviewed" value={selected ? selected.stats.photos : totals.photos} foot={selected ? `of ${selected.files.length} uploaded` : "across all runs"} />
                  <Metric icon="scan" label="Proposals found" value={selected ? selected.stats.detections : totals.detections} foot={selected ? "components + distress masks" : "visual detections"} />
                  <Metric icon="clock" label="Avg. per photo" value={selected?.stats.seconds_per_photo == null ? "—" : `${selected.stats.seconds_per_photo}s`} foot="SAM processing time" />
                  <Metric icon="cpu" label="Peak VRAM" value={selected?.stats.peak_vram_gib == null ? "—" : `${selected.stats.peak_vram_gib.toFixed(2)} GB`} foot={selected?.inference?.model?.gpu ?? (selected ? "GPU memory allocated" : `${totals.complete} completed runs`)} />
                </div>
                <div className="charts-grid">
                  <div className="chart-card">
                    <div className="chart-title"><div><strong>Proposals by class</strong><span>{selected?.stats.detections ?? totals.detections} accepted detections</span></div><span className="chart-badge"><Icon name="chart" size={16} /></span></div>
                    {classes.length ? <div className="bar-list">{classes.map(([name, count], index) => <div className="bar-row" key={name}><span className="bar-label">{name.replaceAll("_", " ")}</span><div className="bar-track"><span style={{ width: `${Math.max(3, count / maxClass * 100)}%`, animationDelay: `${index * 55}ms` }} /></div><strong>{count}</strong></div>)}</div> : <div className="chart-empty"><span className="empty-chart-icon"><Icon name="chart" size={20} /></span><strong>No plotted results yet</strong><span>Run an analysis to see the class breakdown.</span></div>}
                    {classes.length > 0 && <div className="chart-footnote">Top {classes.length} classes · accepted masks only</div>}
                  </div>
                  <div className="chart-card quality-card">
                    <div className="chart-title"><div><strong>Run quality signals</strong><span>Processing outcomes</span></div><span className="chart-badge mint"><Icon name="check" size={16} /></span></div>
                    <div className="quality-list">
                      <QualityRow label="Successful photos" value={selected?.review?.inference_status_counts?.ok ?? selected?.stats.photos ?? 0} tone="green" />
                      <QualityRow label="Reduced-size retries" value={selected?.review?.inference_status_counts?.ok_reduced ?? selected?.inference?.ok_reduced ?? 0} tone="amber" />
                      <QualityRow label="Needs GPU follow-up" value={selected?.review?.inference_status_counts?.needs_batch_gpu ?? selected?.inference?.needs_batch_gpu ?? 0} tone="red" />
                      <QualityRow label="Inference errors" value={selected?.review?.inference_status_counts?.error ?? selected?.inference?.error ?? 0} tone="red" />
                    </div>
                    <div className="accuracy-note"><span className="note-mark">i</span><span>Visual review only. Accuracy needs engineer-labelled ground truth.</span></div>
                  </div>
                </div>
              </section>

              {selected?.review_url && <section className="report-card"><div className="report-head"><div><div className="eyebrow"><span className="eyebrow-line" /> PLOTTING AGENT</div><h2>Visual review report</h2></div><a className="secondary-button" href={`${API}${selected.review_url}`} target="_blank" rel="noreferrer">Open full report<Icon name="external" size={15} /></a></div><iframe title="Inference plotting report" src={`${API}${selected.review_url}`} loading="lazy" /></section>}
            </div>

            <aside className="side-column">
              <section className="queue-card">
                <div className="side-card-heading"><span className="side-icon"><Icon name="clock" /></span><div><h3>Run history</h3><span>{runs.length} saved {runs.length === 1 ? "run" : "runs"}</span></div><span className="queue-count">{runs.length}</span></div>
                <div className="history-list">
                  {runs.length ? runs.slice(0, 7).map((run) => (
                    <button className={`history-item ${selected?.id === run.id ? "chosen" : ""}`} key={run.id} onClick={() => setSelectedId(run.id)}>
                      <span className={`history-thumb ${run.status}`}><Icon name={run.status === "completed" || run.status === "completed_with_warnings" ? "image" : run.status === "failed" ? "close" : "scan"} size={18} /></span>
                      <span className="history-copy"><strong>Run #{shortId(run.id)}</strong><small>{run.files.length} {run.files.length === 1 ? "image" : "images"} · {formatDate(run.created_at)}</small></span>
                      <span className={`mini-status ${run.status}`} />
                    </button>
                  )) : <div className="history-empty"><span className="history-empty-icon"><Icon name="image" /></span><strong>No runs yet</strong><span>Upload a photo set to get started.</span></div>}
                </div>
                {runs.length > 0 && <div className="history-footer"><span>{busyRuns ? "GPU queue active" : "Runs saved on this device"}</span><span className={`queue-live ${busyRuns ? "active" : ""}`}><i />{busyRuns ? "PROCESSING" : "LOCAL"}</span></div>}
              </section>
              <section className="principles-card">
                <div className="principles-top"><span className="principles-spark">✳</span><span>BUILT FOR TRUST</span></div>
                <h3>AI proposes.<br />Engineers decide.</h3>
                <p>SAM detections are review candidates—not condition states or inspection conclusions.</p>
                <div className="principles-divider" />
                <div className="principle-row"><span>01</span><span>Runs locally on your machine</span><Icon name="check" size={15} /></div>
                <div className="principle-row"><span>02</span><span>No automatic condition scoring</span><Icon name="check" size={15} /></div>
                <div className="principle-row"><span>03</span><span>Human review stays in control</span><Icon name="check" size={15} /></div>
              </section>
              <div className="side-help"><span className="help-icon">?</span><span><strong>Need a refresher?</strong><small>See the project run guide in README.</small></span></div>
            </aside>
          </section>
          <footer className="page-footer"><span>BRIDGESIGHT · LOCAL INSPECTION WORKSPACE</span><span>SP:35 · HUMAN REVIEW REQUIRED</span></footer>
        </div>
      </section>
    </main>
  );
}

function Metric({ icon, label, value, foot }: { icon: string; label: string; value: string | number; foot: string }) {
  return <div className="metric-card"><div className="metric-top"><span>{label}</span><span className="metric-icon"><Icon name={icon} size={17} /></span></div><strong className="metric-value">{typeof value === "number" ? value.toLocaleString() : value}</strong><small>{foot}</small></div>;
}

function QualityRow({ label, value, tone }: { label: string; value: number; tone: string }) {
  return <div className="quality-row"><span className={`quality-dot ${tone}`} /><span>{label}</span><strong>{value}</strong></div>;
}
