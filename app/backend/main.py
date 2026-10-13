from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SP35_APP_DATA", ROOT / "data" / "app_runs")).resolve()
RUNS_ROOT = DATA_ROOT / "runs"
REPORTS_ROOT = DATA_ROOT / "reports"
DB_PATH = DATA_ROOT / "jobs.sqlite3"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILES_PER_RUN = 200
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_RUN_BYTES = 500 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_RUN_BYTES + 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 80_000_000
CHUNK_BYTES = 1024 * 1024
LOG_TAIL_CHARS = 3000
RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_filename(filename: str | None) -> str:
    raw = (filename or "").replace("\\", "/")
    name = PurePosixPath(raw).name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported image format: {suffix or 'no extension'}")
    stem = Path(name).stem
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip(" ._-")
    if not stem:
        stem = "photo"
    stem = stem[:120]
    return f"{stem}{suffix}"


class AnalysisOptions(BaseModel):
    tile: bool = False


class JobService:
    def __init__(self, data_root: Path = DATA_ROOT, worker_python: str | None = None):
        self.data_root = data_root.resolve()
        self.runs_root = self.data_root / "runs"
        self.reports_root = self.data_root / "reports"
        self.db_path = self.data_root / "jobs.sqlite3"
        self.worker_python = worker_python or os.environ.get("SP35_WORKER_PYTHON")
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sp35-inference")
        self._lock = threading.Lock()

    def initialize(self) -> None:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.reports_root.mkdir(parents=True, exist_ok=True)
        with self._database() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    files_json TEXT NOT NULL,
                    tile INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    stats_json TEXT NOT NULL DEFAULT '{}'
                )
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            if "stats_json" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN stats_json TEXT NOT NULL DEFAULT '{}'")
            db.execute(
                "UPDATE jobs SET status='interrupted', updated_at=?, "
                "error='Backend restarted before this queued or running job completed.' "
                "WHERE status IN ('queued', 'running')",
                (_now(),),
            )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=15000")
        return db

    @contextmanager
    def _database(self):
        db = self._connect()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _worker_executable(self) -> str:
        if self.worker_python:
            path = Path(self.worker_python)
            if not path.is_file():
                raise RuntimeError(f"Configured SAM worker Python does not exist: {path}")
            return str(path)
        windows_worker = ROOT / ".venv-win" / "Scripts" / "python.exe"
        return str(windows_worker) if windows_worker.is_file() else sys.executable

    def _validate_run_id(self, run_id: str) -> None:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise HTTPException(status_code=404, detail="Run not found")

    def _row(self, run_id: str) -> sqlite3.Row | None:
        self._validate_run_id(run_id)
        with self._database() as db:
            return db.execute("SELECT * FROM jobs WHERE id=?", (run_id,)).fetchone()

    def _summary(self, path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            return {"summary_error": f"Invalid summary JSON: {exc}"}
        return value if isinstance(value, dict) else {"summary_error": "Summary JSON must be an object"}

    def serialize(self, row: sqlite3.Row) -> dict[str, Any]:
        run_dir = self.runs_root / row["id"]
        inference = self._summary(run_dir / "inference" / "run_summary.json")
        review = self._summary(self.reports_root / row["id"] / "summary.json")
        stats = {
            "photos": 0,
            "photo_results": 0,
            "detections": 0,
            "classes": {},
            "seconds_per_photo": None,
            "peak_vram_gib": None,
            **json.loads(row["stats_json"] or "{}"),
        }
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "files": json.loads(row["files_json"]),
            "tile": bool(row["tile"]),
            "error": row["error"],
            "inference": inference,
            "review": review,
            "stats": stats,
            "review_url": f"/api/runs/{row['id']}/review"
            if (self.reports_root / row["id"] / "index.html").is_file() else None,
        }

    def _collect_stats(self, run_id: str) -> dict[str, Any]:
        inference_dir = self.runs_root / run_id / "inference"
        photo_count = 0
        seconds, peaks = [], []
        for path in inference_dir.glob("*.json"):
            if path.name == "run_summary.json":
                continue
            data = self._summary(path)
            if not data or not data.get("photo"):
                continue
            photo_count += 1
            if isinstance(data.get("sec"), (int, float)):
                seconds.append(float(data["sec"]))
            if isinstance(data.get("peak_alloc_gib"), (int, float)):
                peaks.append(float(data["peak_alloc_gib"]))
        review = self._summary(self.reports_root / run_id / "summary.json")
        class_counts = review.get("detection_counts_by_class", {}) if review else {}
        if not isinstance(class_counts, dict):
            class_counts = {}
        return {
            "photos": review.get("photos_with_successful_inference", 0) if review else 0,
            "photo_results": photo_count,
            "detections": sum(class_counts.values()),
            "classes": class_counts,
            "seconds_per_photo": round(sum(seconds) / len(seconds), 2) if seconds else None,
            "peak_vram_gib": max(peaks) if peaks else None,
        }

    def _persist_stats(self, run_id: str) -> None:
        stats = self._collect_stats(run_id)
        with self._database() as db:
            db.execute(
                "UPDATE jobs SET stats_json=?, updated_at=? WHERE id=?",
                (json.dumps(stats), _now(), run_id),
            )

    def list_jobs(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._database() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self.serialize(row) for row in rows]

    async def create_job(self, uploads: list[UploadFile]) -> dict[str, Any]:
        if not uploads:
            raise HTTPException(status_code=422, detail="Select at least one JPG, PNG, or JPEG image.")
        if len(uploads) > MAX_FILES_PER_RUN:
            raise HTTPException(status_code=413, detail=f"Upload limit is {MAX_FILES_PER_RUN} images per run.")

        self.data_root.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        run_dir = self.runs_root / run_id
        uploaded_names: list[str] = []
        total_bytes = 0

        try:
            with tempfile.TemporaryDirectory(prefix=".upload-", dir=self.data_root) as temp:
                input_dir = Path(temp) / "input"
                input_dir.mkdir()
                for index, upload in enumerate(uploads, start=1):
                    original_name = _safe_filename(upload.filename)
                    stored_name = f"{index:03d}_{original_name}"
                    destination = input_dir / stored_name
                    file_bytes = 0
                    try:
                        with destination.open("wb") as target:
                            while chunk := await upload.read(CHUNK_BYTES):
                                file_bytes += len(chunk)
                                total_bytes += len(chunk)
                                if file_bytes > MAX_FILE_BYTES:
                                    raise HTTPException(
                                        status_code=413,
                                        detail=f"{original_name} exceeds the 25 MiB per-image limit.",
                                    )
                                if total_bytes > MAX_RUN_BYTES:
                                    raise HTTPException(
                                        status_code=413,
                                        detail="This upload exceeds the 500 MiB total limit.",
                                    )
                                target.write(chunk)
                    finally:
                        await upload.close()
                    if not file_bytes:
                        raise HTTPException(status_code=422, detail=f"{original_name} is empty.")
                    try:
                        with Image.open(destination) as image:
                            if image.width * image.height > MAX_IMAGE_PIXELS:
                                raise HTTPException(
                                    status_code=413,
                                    detail=f"{original_name} exceeds the 80 megapixel safety limit.",
                                )
                            image.verify()
                    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                        raise HTTPException(status_code=422, detail=f"{original_name} is not a valid safe image.") from exc
                    uploaded_names.append(stored_name)

                run_dir.mkdir(parents=True)
                shutil.move(str(input_dir), run_dir / "input")
            created = _now()
            with self._database() as db:
                db.execute(
                    "INSERT INTO jobs(id, created_at, updated_at, status, files_json) VALUES (?, ?, ?, 'uploaded', ?)",
                    (run_id, created, created, json.dumps(uploaded_names)),
                )
            row = self._row(run_id)
            assert row is not None
            return self.serialize(row)
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

    def enqueue(self, run_id: str, options: AnalysisOptions) -> dict[str, Any]:
        self._validate_run_id(run_id)
        with self._lock, self._database() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Run not found")
            if row["status"] not in {"uploaded", "interrupted"}:
                raise HTTPException(status_code=409, detail=f"Run cannot start from status '{row['status']}'.")
            updated = _now()
            db.execute(
                "UPDATE jobs SET status='queued', updated_at=?, tile=?, error=NULL WHERE id=?",
                (updated, int(options.tile), run_id),
            )
        try:
            self.executor.submit(self._execute, run_id)
        except RuntimeError as exc:
            self._set_status(run_id, "failed", f"Could not queue inference: {exc}")
            raise HTTPException(status_code=503, detail="Inference queue is unavailable.") from exc
        row = self._row(run_id)
        assert row is not None
        return self.serialize(row)

    def _set_status(self, run_id: str, status: str, error: str | None = None) -> None:
        with self._database() as db:
            db.execute(
                "UPDATE jobs SET status=?, updated_at=?, error=? WHERE id=?",
                (status, _now(), error, run_id),
            )

    def _run_command(self, run_id: str, args: list[str], log_path: Path) -> int:
        command = [self._worker_executable(), *args]
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(command)}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return completed.returncode

    def _execute(self, run_id: str) -> None:
        run_dir = self.runs_root / run_id
        input_dir = run_dir / "input"
        inference_dir = run_dir / "inference"
        review_dir = self.reports_root / run_id
        log_path = run_dir / "job.log"
        started = time.monotonic()
        try:
            row = self._row(run_id)
            if row is None:
                return
            self._set_status(run_id, "running")
            infer_args = ["-m", "vision.infer", "--folder", str(input_dir),
                          "--outdir", str(inference_dir)]
            if row["tile"]:
                infer_args.append("--tile")
            infer_code = self._run_command(run_id, infer_args, log_path)
            photo_results = [p for p in inference_dir.glob("*.json")
                             if p.name != "run_summary.json"]
            if not photo_results:
                self._persist_stats(run_id)
                self._set_status(run_id, "failed", self._log_tail(log_path) or
                                 f"Image inference exited with code {infer_code} and wrote no photo results.")
                return

            review_code = self._run_command(
                run_id,
                ["-m", "tools.inference_review", "--predictions", str(inference_dir),
                 "--images", str(input_dir), "--outdir", str(review_dir)],
                log_path,
            )
            self._persist_stats(run_id)
            elapsed = round(time.monotonic() - started, 1)
            if review_code != 0:
                error = f"Analysis finished, but plotting failed (exit code {review_code}). {self._log_tail(log_path)}"
                self._set_status(run_id, "completed_with_warnings", error[:LOG_TAIL_CHARS])
            elif infer_code != 0:
                error = f"Some photos had inference warnings (exit code {infer_code}). {self._log_tail(log_path)}"
                self._set_status(run_id, "completed_with_warnings", error[:LOG_TAIL_CHARS])
            else:
                self._set_status(run_id, "completed", f"Finished in {elapsed}s.")
        except Exception as exc:
            self._set_status(run_id, "failed", f"{type(exc).__name__}: {exc}; {self._log_tail(log_path)}"[:LOG_TAIL_CHARS])

    @staticmethod
    def _log_tail(path: Path) -> str:
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - LOG_TAIL_CHARS * 4))
                return stream.read().decode("utf-8", errors="replace")[-LOG_TAIL_CHARS:].strip()
        except FileNotFoundError:
            return ""


def create_app(service: JobService | None = None) -> FastAPI:
    jobs = service or JobService()
    jobs.initialize()
    api = FastAPI(
        title="SP:35 Bridge Inspection API",
        version="0.1.0",
        description="Local photo intake, SAM inference jobs, and plotting reports.",
    )
    api.state.jobs = jobs
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @api.middleware("http")
    async def limit_upload_request(request: Request, call_next):
        if request.method == "POST" and request.url.path == "/api/runs":
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > MAX_REQUEST_BYTES:
                        return JSONResponse(status_code=413, content={"detail": "Upload request exceeds the total-size limit."})
                except ValueError:
                    return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
        return await call_next(request)

    @api.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            worker_exists = Path(jobs._worker_executable()).is_file()
        except RuntimeError:
            worker_exists = False
        return {
            "status": "ok",
            "local_only": True,
            "worker_python_configured": worker_exists,
            "checkpoint_present": (ROOT / "data" / "weights" / "sam3.1" / "sam3.1_multiplex.pt").is_file(),
        }

    @api.post("/api/runs", status_code=201)
    async def create_run(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        return await jobs.create_job(files)

    @api.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": jobs.list_jobs()}

    @api.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        row = jobs._row(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return jobs.serialize(row)

    @api.post("/api/runs/{run_id}/analyze", status_code=202)
    def analyze_run(run_id: str, options: AnalysisOptions = AnalysisOptions()) -> dict[str, Any]:
        return jobs.enqueue(run_id, options)

    @api.get("/api/runs/{run_id}/review")
    def open_review(run_id: str) -> RedirectResponse:
        row = jobs._row(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        report = jobs.reports_root / run_id / "index.html"
        if not report.is_file():
            raise HTTPException(status_code=409, detail="Plot report is not ready for this run.")
        return RedirectResponse(url=f"/api/review-files/{run_id}/index.html")

    api.mount("/api/review-files", StaticFiles(directory=jobs.reports_root), name="review-files")
    return api


app = create_app()
