"""Tests for local upload intake, persistent jobs, and plotting-report access."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.backend.main import JobService, create_app


def jpeg_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), (220, 180, 90))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def test_upload_single_and_batch_images_store_safe_names(tmp_path: Path) -> None:
    service = JobService(tmp_path, worker_python="python")
    with TestClient(create_app(service)) as client:
        one = client.post("/api/runs", files=[("files", ("bridge.jpg", jpeg_bytes(), "image/jpeg"))])
        assert one.status_code == 201
        assert one.json()["status"] == "uploaded"
        assert one.json()["files"] == ["001_bridge.jpg"]

        batch = client.post(
            "/api/runs",
            files=[
                ("files", ("pier.jpg", jpeg_bytes(), "image/jpeg")),
                ("files", ("deck.png", jpeg_bytes(), "image/jpeg")),
            ],
        )
        assert batch.status_code == 201
        assert batch.json()["files"] == ["001_pier.jpg", "002_deck.png"]


def test_upload_rejects_bad_extension_and_corrupt_image(tmp_path: Path) -> None:
    with TestClient(create_app(JobService(tmp_path, worker_python="python"))) as client:
        unsupported = client.post("/api/runs", files=[("files", ("notes.txt", b"hello", "text/plain"))])
        assert unsupported.status_code == 415
        corrupt = client.post("/api/runs", files=[("files", ("bridge.jpg", b"not an image", "image/jpeg"))])
        assert corrupt.status_code == 422


def test_upload_rejects_oversized_dimensions(tmp_path: Path) -> None:
    image = Image.new("1", (9000, 9000))
    output = io.BytesIO()
    image.save(output, format="PNG")
    with TestClient(create_app(JobService(tmp_path, worker_python="python"))) as client:
        response = client.post("/api/runs", files=[("files", ("huge.png", output.getvalue(), "image/png"))])
    assert response.status_code == 413


def test_analysis_is_explicit_and_duplicate_start_is_rejected(tmp_path: Path) -> None:
    service = JobService(tmp_path, worker_python="python")
    app = create_app(service)
    with TestClient(app) as client:
        uploaded = client.post("/api/runs", files=[("files", ("bridge.jpg", jpeg_bytes(), "image/jpeg"))])
        run_id = uploaded.json()["id"]
        assert uploaded.json()["status"] == "uploaded"

        service._execute = lambda run_id: time.sleep(0.2)  # type: ignore[method-assign]
        started = client.post(f"/api/runs/{run_id}/analyze", json={"tile": False})
        assert started.status_code == 202
        duplicate = client.post(f"/api/runs/{run_id}/analyze", json={"tile": False})
        assert duplicate.status_code == 409


def test_analysis_runs_existing_cli_then_builds_report(tmp_path: Path, monkeypatch) -> None:
    service = JobService(tmp_path, worker_python="python")
    with TestClient(create_app(service)) as client:
        uploaded = client.post("/api/runs", files=[("files", ("bridge.jpg", jpeg_bytes(), "image/jpeg"))])
        run_id = uploaded.json()["id"]
        commands = []

        def fake_command(command_run_id, args, log_path):
            commands.append(args)
            if args[1] == "vision.infer":
                output = Path(args[args.index("--outdir") + 1])
                output.mkdir(parents=True)
                (output / "001_bridge.json").write_text(
                    json.dumps({"photo": "001_bridge.jpg", "status": "ok", "sec": 1.5,
                                "peak_alloc_gib": 2.2, "components": [], "distresses": []}),
                    encoding="utf-8",
                )
                (output / "run_summary.json").write_text(
                    json.dumps({"photos": 1, "ok": 1, "error": 0}), encoding="utf-8"
                )
            else:
                report = Path(args[args.index("--outdir") + 1])
                report.mkdir(parents=True)
                (report / "summary.json").write_text(
                    json.dumps({"photos_with_successful_inference": 1,
                                "detection_counts_by_class": {"pier": 2},
                                "inference_status_counts": {"ok": 1}}),
                    encoding="utf-8",
                )
                (report / "index.html").write_text("<html>review</html>", encoding="utf-8")
            log_path.write_text("ok", encoding="utf-8")
            return 0

        monkeypatch.setattr(service, "_run_command", fake_command)
        started = client.post(f"/api/runs/{run_id}/analyze", json={"tile": True})
        assert started.status_code == 202
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            response = client.get(f"/api/runs/{run_id}").json()
            if response["status"] not in {"queued", "running"}:
                break
            time.sleep(0.02)

        assert response["status"] == "completed"
        assert response["stats"]["photos"] == 1
        assert response["stats"]["detections"] == 2
        assert response["stats"]["seconds_per_photo"] == 1.5
        assert response["stats"]["peak_vram_gib"] == 2.2
        assert len(commands) == 2
        assert "--tile" in commands[0]
        assert commands[1][1] == "tools.inference_review"


def test_restart_marks_inflight_jobs_interrupted(tmp_path: Path) -> None:
    service = JobService(tmp_path, worker_python="python")
    with TestClient(create_app(service)) as client:
        response = client.post("/api/runs", files=[("files", ("bridge.jpg", jpeg_bytes(), "image/jpeg"))])
        run_id = response.json()["id"]
        with service._connect() as db:
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (run_id,))
        restarted = JobService(tmp_path, worker_python="python")
        restarted.initialize()
        assert restarted._row(run_id)["status"] == "interrupted"


def test_review_route_requires_generated_report(tmp_path: Path) -> None:
    service = JobService(tmp_path, worker_python="python")
    with TestClient(create_app(service)) as client:
        response = client.post("/api/runs", files=[("files", ("bridge.jpg", jpeg_bytes(), "image/jpeg"))])
        run_id = response.json()["id"]
        assert client.get(f"/api/runs/{run_id}/review").status_code == 409
        report = service.reports_root / run_id / "index.html"
        report.parent.mkdir()
        report.write_text("<html>ready</html>", encoding="utf-8")
        private_photo = service.runs_root / run_id / "input" / "001_bridge.jpg"
        assert private_photo.is_file()
        assert client.get(f"/api/review-files/{run_id}/input/001_bridge.jpg").status_code == 404
        opened = client.get(f"/api/runs/{run_id}/review")
        assert opened.status_code == 200
        assert "ready" in opened.text
