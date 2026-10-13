"""Prepare visual review cards and sort accepted distress proposals by review verdict."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from tools.inference_review import blend_masks, class_color, decode_rle, resize_mask

VERDICTS = ("apt", "misclassified", "uncertain")
CONFIDENCE = ("high", "medium", "low")
QUEUE_FIELDS = (
    "id", "photo", "distress_type", "score", "prompts", "candidate_image",
    "verdict", "confidence", "rationale",
)
OUTPUT_FIELDS = (
    "id", "photo", "distress_type", "score", "confidence", "rationale", "candidate_image",
)
CARD_MAX_SIDE = 900


def _photo_name(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).name


def _review_card(image: Image.Image, mask: Any, distress_type: str, score: Any) -> Image.Image:
    scale = min(1.0, CARD_MAX_SIDE / max(image.size))
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    photo = image.convert("RGB")
    if photo.size != size:
        photo = photo.resize(size, Image.Resampling.LANCZOS)
    shown_mask = resize_mask(mask, size)
    overlay = blend_masks(photo, [{"class": distress_type, "mask": shown_mask}], lambda row: class_color(row["class"]))
    header_height = 34
    card = Image.new("RGB", (size[0] * 2, size[1] + header_height), "white")
    draw = ImageDraw.Draw(card)
    heading = f"Distress type: {distress_type} | SAM score: {float(score):.3f} | mask overlay shown in right panel"
    draw.text((8, 10), heading, fill="black")
    card.paste(photo, (0, header_height))
    card.paste(overlay, (size[0], header_height))
    return card


def prepare_queue(predictions: Path, images: Path, outdir: Path) -> dict[str, int]:
    predictions, images, outdir = predictions.resolve(), images.resolve(), outdir.resolve()
    if outdir == predictions or predictions in outdir.parents:
        raise ValueError("output folder must be separate from the inference predictions folder")
    queue_csv = outdir / "decisions.csv"
    if queue_csv.exists():
        raise FileExistsError(f"{queue_csv} already exists; use a new output folder to prepare another review")

    outdir.mkdir(parents=True, exist_ok=True)
    queue_dir = outdir / "review_queue"
    queue_dir.mkdir(exist_ok=True)
    rows: list[dict[str, str]] = []
    photos_reviewed = 0
    proposal_index = 0

    for prediction_file in sorted(predictions.glob("*.json")):
        if prediction_file.name == "run_summary.json":
            continue
        data = json.loads(prediction_file.read_text(encoding="utf-8"))
        if data.get("status") not in ("ok", "ok_reduced"):
            continue
        photo_name = _photo_name(data.get("photo") or "")
        if not photo_name:
            raise ValueError(f"{prediction_file} has no source photo name")
        photo_path = images / photo_name
        if not photo_path.is_file():
            raise FileNotFoundError(f"source photo for {prediction_file.name} does not exist: {photo_path}")
        with Image.open(photo_path) as opened:
            photo = ImageOps.exif_transpose(opened).convert("RGB")
        photos_reviewed += 1

        distress_items = data.get("distresses", [])
        if not isinstance(distress_items, list):
            raise ValueError(f"{prediction_file} has a non-list distresses field")
        for item in distress_items:
            if not isinstance(item, dict) or not item.get("class") or not isinstance(item.get("rle"), dict):
                raise ValueError(f"{prediction_file} contains a distress without a class or RLE mask")
            distress_type = str(item["class"])
            mask = decode_rle(item["rle"])
            proposal_index += 1
            proposal_id = f"D{proposal_index:06d}"
            card_name = f"{proposal_id}.png"
            card_path = queue_dir / card_name
            card = _review_card(photo, mask, distress_type, item.get("score", 0.0))
            card.save(card_path)
            rows.append({
                "id": proposal_id,
                "photo": photo_name,
                "distress_type": distress_type,
                "score": f"{float(item.get('score', 0.0)):.6f}",
                "prompts": json.dumps(item.get("prompts", []), ensure_ascii=False),
                "candidate_image": f"review_queue/{card_name}",
                "verdict": "",
                "confidence": "",
                "rationale": "",
            })

    with queue_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=QUEUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"photos_reviewed": photos_reviewed, "distress_proposals": len(rows)}


def apply_decisions(outdir: Path, decisions: Path | None = None) -> dict[str, Any]:
    outdir = outdir.resolve()
    decisions = (decisions or outdir / "decisions.csv").resolve()
    queue_csv = outdir / "decisions.csv"
    if not queue_csv.is_file():
        raise FileNotFoundError(f"review queue does not exist: {queue_csv}")
    if not decisions.is_file():
        raise FileNotFoundError(f"decisions CSV does not exist: {decisions}")

    with queue_csv.open(newline="", encoding="utf-8") as stream:
        queue_rows = list(csv.DictReader(stream))
    with decisions.open(newline="", encoding="utf-8") as stream:
        decision_rows = list(csv.DictReader(stream))
    expected_ids = {row["id"] for row in queue_rows}
    decisions_by_id: dict[str, dict[str, str]] = {}
    for row in decision_rows:
        proposal_id = (row.get("id") or "").strip()
        if proposal_id in decisions_by_id:
            raise ValueError(f"duplicate decision for {proposal_id}")
        decisions_by_id[proposal_id] = {key: (value or "").strip() for key, value in row.items()}
    unknown_ids = set(decisions_by_id) - expected_ids
    if unknown_ids:
        raise ValueError(f"decisions contain unknown proposal IDs: {', '.join(sorted(unknown_ids))}")
    missing = expected_ids - set(decisions_by_id)
    if missing:
        raise ValueError(f"missing verdict for proposal IDs: {', '.join(sorted(missing)[:10])}")

    by_id = {row["id"]: row for row in queue_rows}
    for proposal_id, row in decisions_by_id.items():
        verdict = row.get("verdict", "").lower()
        confidence = row.get("confidence", "").lower()
        if not verdict:
            raise ValueError(f"missing verdict for {proposal_id}")
        if verdict not in VERDICTS:
            raise ValueError(f"invalid verdict for {proposal_id}: {verdict!r}; expected one of {VERDICTS}")
        if confidence not in CONFIDENCE:
            raise ValueError(f"invalid confidence for {proposal_id}: {confidence!r}; expected one of {CONFIDENCE}")
        if not row.get("rationale"):
            raise ValueError(f"missing visual rationale for {proposal_id}")
        row["verdict"] = verdict
        row["confidence"] = confidence
        candidate_rel = Path(by_id[proposal_id]["candidate_image"])
        candidate_path = (outdir / candidate_rel).resolve()
        if outdir not in candidate_path.parents or not candidate_path.is_file():
            raise ValueError(f"invalid or missing review image for {proposal_id}: {candidate_rel}")

    destination_dirs = {verdict: outdir / verdict for verdict in VERDICTS}
    for directory in destination_dirs.values():
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"refusing to overwrite existing review output: {directory}")
        directory.mkdir(exist_ok=True)

    counts = {verdict: 0 for verdict in VERDICTS}
    bucket_rows: dict[str, list[dict[str, str]]] = {verdict: [] for verdict in VERDICTS}
    for proposal_id in sorted(expected_ids):
        source_row = by_id[proposal_id]
        decision = decisions_by_id[proposal_id]
        verdict = decision["verdict"]
        candidate_path = (outdir / source_row["candidate_image"]).resolve()
        filename = f"{proposal_id}.png"
        shutil.copy2(candidate_path, destination_dirs[verdict] / filename)
        bucket_rows[verdict].append({
            "id": proposal_id,
            "photo": source_row["photo"],
            "distress_type": source_row["distress_type"],
            "score": source_row["score"],
            "confidence": decision["confidence"],
            "rationale": decision["rationale"],
            "candidate_image": filename,
        })
        counts[verdict] += 1

    for verdict, rows in bucket_rows.items():
        with (destination_dirs[verdict] / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "counts": counts,
        "total_distress_proposals": len(queue_rows),
        "classification_scope": "distress type only; component labels are not used",
        "interpretation": (
            "Visual triage of SAM proposals, not ground-truth accuracy or engineer confirmation. "
            "Uncertain items are kept separate for human review."
        ),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visually triage accepted SAM distress proposals into review folders.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="render proposal cards and create a decisions CSV")
    prepare.add_argument("--predictions", type=Path, default=Path("data/pilot/infer_all_v1"))
    prepare.add_argument("--images", type=Path, default=Path("../yugmi_images"))
    prepare.add_argument("--outdir", type=Path, default=Path("data/pilot/infer_all_v1_bifurcated"))
    apply = subparsers.add_parser("apply", help="copy reviewed proposal cards into verdict folders")
    apply.add_argument("--outdir", type=Path, default=Path("data/pilot/infer_all_v1_bifurcated"))
    apply.add_argument("--decisions", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_queue(args.predictions, args.images, args.outdir)
        print(json.dumps(result, indent=2))
    else:
        result = apply_decisions(args.outdir, args.decisions)
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
