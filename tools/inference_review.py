"""Create local visual summaries of SAM inference and optionally compare COCO masks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps


def build_defect_coverage(detection_counts: Counter, knowledge_dir: Path,
                          prompts_path: Path, source_photo_count: int,
                          inference_result_photo_count: int,
                          successful_photo_count: int) -> dict:
    """Compare observed SAM distress proposals with SP:35 rows and prompt mappings."""
    catalogs = []
    for filename in ("distress_bridge.yaml", "distress_approach.yaml"):
        document = yaml.safe_load((knowledge_dir / filename).read_text(encoding="utf-8"))
        catalogs.extend(document["distresses"])

    prompt_data = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))
    prompts_by_row: dict[str, list[str]] = defaultdict(list)
    for prompt in prompt_data.get("distresses", []):
        row_id = prompt.get("sp35_row")
        if row_id:
            prompts_by_row[row_id].append(prompt["id"])

    known_rows = {row["id"] for row in catalogs}
    unknown_rows = set(prompts_by_row) - known_rows
    if unknown_rows:
        raise ValueError(f"Prompt library references unknown SP:35 distress rows: {sorted(unknown_rows)}")

    not_from_photos = {
        item["sp35_row"]: item.get("reason", "")
        for item in prompt_data.get("not_from_photos", [])
    }
    mapped_classes: set[str] = set()
    coverage_rows = []
    for row in catalogs:
        row_id = row["id"]
        prompt_classes = prompts_by_row.get(row_id, [])
        mapped_classes.update(prompt_classes)
        proposals = sum(int(detection_counts.get(class_name, 0)) for class_name in prompt_classes)
        coverage_rows.append({
            "table": row_id.split("-", 1)[0].removeprefix("T"),
            "id": row_id,
            "name": row["name"],
            "prompt_classes": prompt_classes,
            "prompted": bool(prompt_classes),
            "proposal_count": proposals,
            "observed_in_dataset": proposals > 0,
            "not_attempted_reason": not_from_photos.get(row_id),
        })

    by_table: dict[str, dict] = {}
    for row in coverage_rows:
        table = row["table"]
        totals = by_table.setdefault(table, {
            "catalog_rows": 0,
            "prompted_rows": 0,
            "observed_prompted_rows": 0,
            "proposal_count": 0,
        })
        totals["catalog_rows"] += 1
        totals["prompted_rows"] += int(row["prompted"])
        totals["observed_prompted_rows"] += int(row["prompted"] and row["observed_in_dataset"])
        totals["proposal_count"] += row["proposal_count"]

    prompted_rows = sum(row["prompted"] for row in coverage_rows)
    observed_rows = sum(row["prompted"] and row["observed_in_dataset"] for row in coverage_rows)
    return {
        "source": {
            "bridge_catalog": str(knowledge_dir / "distress_bridge.yaml"),
            "approach_catalog": str(knowledge_dir / "distress_approach.yaml"),
            "prompt_library": str(prompts_path),
        },
        "dataset": {
            "source_photos": source_photo_count,
            "inference_result_photos": inference_result_photo_count,
            "successful_inference_photos": successful_photo_count,
        },
        "catalog_row_count": len(coverage_rows),
        "catalog_rows_by_table": by_table,
        "prompted_catalog_rows": prompted_rows,
        "observed_prompted_rows": observed_rows,
        "observed_prompted_row_fraction": observed_rows / prompted_rows if prompted_rows else None,
        "proposal_count": sum(int(count) for count in detection_counts.values()),
        "proposal_counts_by_class": dict(sorted(detection_counts.items())),
        "unmapped_proposal_counts": {
            label: int(count) for label, count in sorted(detection_counts.items())
            if label not in mapped_classes
        },
        "rows": coverage_rows,
        "interpretation": (
            "Observed means at least one accepted SAM proposal mapped to this SP:35 row in the dataset. "
            "This is prompt/category coverage, not the number of true defects present, unique physical defects, "
            "recall, or accuracy. Counts are accepted proposal masks and may include repeat views of the same defect. "
            "Ground-truth annotations are required to measure missed defects."
        ),
    }


def render_defect_coverage_html(coverage: dict) -> str:
    table_rows = []
    for row in coverage["rows"]:
        classes = ", ".join(row["prompt_classes"]) if row["prompt_classes"] else "Not prompted"
        if row["not_attempted_reason"] and not row["prompted"]:
            classes += f" — {row['not_attempted_reason']}"
        outcome = str(row["proposal_count"]) if row["prompted"] else "—"
        table_rows.append(
            "<tr>" + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (row["table"], row["id"], row["name"], classes, outcome)
            ) + "</tr>"
        )
    table_summary_rows = []
    for table, counts in sorted(
        coverage["catalog_rows_by_table"].items(),
        key=lambda item: tuple(int(part) for part in item[0].split(".")),
    ):
        table_summary_rows.append(
            "<tr>" + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (
                    f"Table {table}",
                    counts["catalog_rows"],
                    counts["prompted_rows"],
                    counts["observed_prompted_rows"],
                    counts["proposal_count"],
                )
            ) + "</tr>"
        )
    fraction = coverage["observed_prompted_row_fraction"]
    observed = (f"{coverage['observed_prompted_rows']} of {coverage['prompted_catalog_rows']} "
                f"prompted SP:35 rows had proposals ({fraction:.0%})"
                if fraction is not None else "No SP:35 rows are mapped to image prompts.")
    unmapped = coverage["unmapped_proposal_counts"]
    unmapped_html = (
        "<p>Proposal classes without an SP:35 row mapping: "
        + html.escape(", ".join(f"{label}: {count}" for label, count in unmapped.items()))
        + ".</p>"
        if unmapped else ""
    )
    return (
        "<section><h2>SP:35 defect catalogue vs dataset proposals</h2>"
        f"<p>SP:35 catalogue rows in Tables 7.2, 7.3 and supplementary Table 7.4: "
        f"{coverage['catalog_row_count']}. Directly prompted rows: {coverage['prompted_catalog_rows']}; "
        f"accepted distress proposal masks: {coverage['proposal_count']}. "
        f"Image folder: {coverage['dataset']['source_photos']} images found; "
        f"inference output: {coverage['dataset']['inference_result_photos']} photos "
        f"({coverage['dataset']['successful_inference_photos']} successful). "
        f"{html.escape(observed)}.</p>"
        f"<p>{html.escape(coverage['interpretation'])}</p>"
        f"{unmapped_html}"
        "<table><thead><tr><th>Catalogue</th><th>Rows</th><th>Prompted rows</th>"
        "<th>Prompted rows with proposals</th><th>Proposals</th></tr></thead><tbody>"
        f"{''.join(table_summary_rows)}</tbody></table>"
        "<details><summary>All SP:35 distress rows and mapped proposal counts</summary>"
        "<table><thead><tr><th>Table</th><th>SP:35 row</th><th>Defect</th>"
        "<th>Prompt coverage</th><th>Proposals</th></tr></thead><tbody>"
        f"{''.join(table_rows)}</tbody></table></details></section>"
    )


def decode_rle(rle: dict) -> np.ndarray:
    """Decode COCO RLE; uncompressed counts work without pycocotools."""
    size = rle.get("size")
    counts = rle.get("counts")
    if not isinstance(size, list) or len(size) != 2 or not all(isinstance(v, int) and v > 0 for v in size):
        raise ValueError("RLE size must contain positive [height, width] integers")
    height, width = size
    if isinstance(counts, list):
        flat = np.zeros(height * width, dtype=bool)
        offset, foreground = 0, False
        for run in counts:
            if not isinstance(run, int) or run < 0 or offset + run > flat.size:
                raise ValueError("invalid uncompressed COCO RLE counts")
            if foreground:
                flat[offset:offset + run] = True
            offset += run
            foreground = not foreground
        if offset != flat.size:
            raise ValueError("COCO RLE counts do not cover the mask")
        return flat.reshape((height, width), order="F")
    if isinstance(counts, str):
        rle = {"size": size, "counts": counts.encode("ascii")}
    elif isinstance(counts, bytes):
        rle = {"size": size, "counts": counts}
    else:
        raise ValueError("COCO RLE counts must be a list or compressed string")
    try:
        from pycocotools import mask as cocomask
    except ImportError as exc:
        raise RuntimeError("decoding compressed COCO RLE requires pycocotools; use the .venv-win environment") from exc
    return np.asarray(cocomask.decode(rle), dtype=bool)


def decode_coco_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        mask = decode_rle(segmentation)
    elif isinstance(segmentation, list):
        try:
            from pycocotools import mask as cocomask
        except ImportError as exc:
            raise RuntimeError("decoding COCO polygons requires pycocotools; use the .venv-win environment") from exc
        polygons = [p for p in segmentation if isinstance(p, list) and len(p) >= 6]
        if not polygons:
            raise ValueError("COCO polygon segmentation has no valid polygons")
        encoded = cocomask.frPyObjects(polygons, height, width)
        mask = np.asarray(cocomask.decode(cocomask.merge(encoded)), dtype=bool)
    else:
        raise ValueError("COCO segmentation must be an RLE object or polygon list")
    if mask.shape != (height, width):
        raise ValueError(f"mask size {mask.shape} does not match image size {(height, width)}")
    return mask


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"cannot compare masks with different sizes: {left.shape} and {right.shape}")
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def match_masks(predictions: list[dict], ground_truth: list[dict], threshold: float) -> dict:
    """Greedily match highest-scoring predictions one-to-one within each class."""
    ordered = sorted(predictions, key=lambda item: item["score"], reverse=True)
    unmatched = set(range(len(ground_truth)))
    matches = []
    for pred in ordered:
        candidates = [(mask_iou(pred["mask"], ground_truth[i]["mask"]), i) for i in unmatched]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= threshold:
            unmatched.remove(best_index)
            matches.append((pred, ground_truth[best_index], best_iou))
        else:
            matches.append((pred, None, best_iou))
    tp = sum(gt is not None for _, gt, _ in matches)
    fp = len(matches) - tp
    fn = len(unmatched)
    matched_ious = [iou for _, gt, iou in matches if gt is not None]
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "matched_mask_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None,
        "matches": matches,
    }


def class_color(label: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return tuple(80 + (channel % 160) for channel in digest[:3])


def blend_masks(image: Image.Image, records: list[dict], color_for_record) -> Image.Image:
    result = image.convert("RGB")
    for record in records:
        overlay = Image.new("RGB", result.size, color_for_record(record))
        mask = Image.fromarray(record["mask"].astype(np.uint8) * 255, mode="L")
        result.paste(Image.blend(result, overlay, 0.45), mask=mask)
    return result


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask.shape == (size[1], size[0]):
        return mask
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 0


def mask_box(mask: np.ndarray) -> list[int] | None:
    """XYXY extent of a mask (x1, y1 exclusive), as vision.postprocess stores `bbox`."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return None
    return [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]


def prediction_record(item: dict, kind: str, size: tuple[int, int]) -> dict:
    """An accepted component or distress, mask and box scaled to the photo's `size`.
    The stored bbox is in the frame of its RLE (processed photo, or native for tiles)."""
    mask = decode_rle(item["rle"])
    frame_h, frame_w = mask.shape
    box = item.get("bbox") or mask_box(mask)
    sx, sy = size[0] / frame_w, size[1] / frame_h
    return {"class": item["class"], "kind": kind, "score": float(item.get("score", 0.0)),
            "bbox": [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy] if box else None,
            "mask": resize_mask(mask, size)}


# accepted components: thin blue box; distress proposals: thicker red box, drawn on top
BOX_STYLE = {"component": ((0, 170, 255), 2), "distress": ((255, 48, 0), 3)}


def draw_boxes(panel: Image.Image, records: list[dict]) -> Image.Image:
    draw = ImageDraw.Draw(panel)
    for record in sorted(records, key=lambda r: r.get("kind") == "distress"):
        if record.get("bbox") is None or record.get("kind") not in BOX_STYLE:
            continue
        color, width = BOX_STYLE[record["kind"]]
        x0, y0, x1, y1 = (round(v) for v in record["bbox"])
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=width)
        label = f"{record['class']} {record['score']:.2f}"
        if record["kind"] == "distress":
            label += " (proposed)"
        height = 12
        ty = y0 - height if y0 >= height else y0
        draw.rectangle((x0, ty, x0 + draw.textlength(label) + 4, ty + height), fill=color)
        draw.text((x0 + 2, ty), label, fill="white" if record["kind"] == "distress" else "black")
    return panel


def render_panels(image: Image.Image, predictions: list[dict], ground_truth: list[dict] | None) -> Image.Image:
    scale = min(1.0, 1600 / max(image.size))
    display_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    display = image.convert("RGB")
    if display.size != display_size:
        display = display.resize(display_size, Image.Resampling.LANCZOS)
    sx, sy = display_size[0] / image.width, display_size[1] / image.height
    shown_predictions = [{"class": item["class"], "kind": item.get("kind"), "score": item.get("score", 0.0),
                          "bbox": ([item["bbox"][0] * sx, item["bbox"][1] * sy,
                                    item["bbox"][2] * sx, item["bbox"][3] * sy] if item.get("bbox") else None),
                          "mask": resize_mask(item["mask"], display_size)}
                         for item in predictions]
    sam_panel = blend_masks(display, shown_predictions, lambda r: class_color(r["class"]))
    panels = [("Photo", display), ("SAM detections", draw_boxes(sam_panel, shown_predictions))]
    if ground_truth is not None:
        shown_ground_truth = [{"class": item["class"], "mask": resize_mask(item["mask"], display_size)}
                              for item in ground_truth]
        panels.append(("Ground truth", blend_masks(display, shown_ground_truth, lambda r: class_color(r["class"]))))
    panel_w, panel_h = display.width, display.height + 28
    canvas = Image.new("RGB", (panel_w * len(panels), panel_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        x = index * panel_w
        draw.text((x + 6, 6), title, fill="black")
        canvas.paste(panel, (x, 28))
    return canvas


def render_count_chart(counts: Counter, path: Path) -> None:
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    label_width = min(300, max(100, max(len(label) for label, _ in rows) * 8 + 12))
    chart_width, row_height, top = 920, 32, 12
    chart = Image.new("RGB", (chart_width, top * 2 + row_height * len(rows)), "white")
    draw = ImageDraw.Draw(chart)
    max_count = max(count for _, count in rows)
    for index, (label, count) in enumerate(rows):
        y = top + index * row_height
        draw.text((8, y + 5), label, fill="black")
        bar_width = int((chart_width - label_width - 70) * count / max_count)
        x = label_width
        draw.rectangle((x, y + 3, x + bar_width, y + row_height - 4), fill=class_color(label))
        draw.text((x + bar_width + 8, y + 5), str(count), fill="black")
    chart.save(path)


def _key(path: str) -> str:
    return Path(path.replace("\\", "/")).name.casefold()


def load_ground_truth(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    image_by_id = {item["id"]: item for item in data.get("images", [])}
    categories = {item["id"]: item["name"] for item in data.get("categories", [])}
    by_filename: dict[str, dict] = {}
    for image in image_by_id.values():
        key = _key(image["file_name"])
        if key in by_filename:
            raise ValueError(f"duplicate COCO image basename: {image['file_name']}")
        by_filename[key] = image
    annotations: dict[str, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        image = image_by_id.get(ann["image_id"])
        if image is None:
            raise ValueError(f"COCO annotation references unknown image_id {ann['image_id']}")
        class_name = categories.get(ann["category_id"])
        if class_name is None:
            raise ValueError(f"COCO annotation references unknown category_id {ann['category_id']}")
        h, w = int(image["height"]), int(image["width"])
        if "segmentation" not in ann:
            raise ValueError(f"COCO annotation {ann.get('id')} has no segmentation mask")
        annotations[_key(image["file_name"])].append({
            "class": class_name,
            "segmentation": ann["segmentation"],
            "size": (h, w),
        })
    return by_filename, annotations


def write_report(prediction_dir: Path, image_dir: Path, output_dir: Path,
                 ground_truth_path: Path | None, iou_threshold: float) -> dict:
    gt_images, gt_annotations = load_ground_truth(ground_truth_path) if ground_truth_path else ({}, {})
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [path for path in image_dir.rglob("*") if path.suffix.lower() in (".jpg", ".jpeg", ".png")]
    image_lookup = {str(path.relative_to(image_dir)).replace("\\", "/").casefold(): path for path in image_paths}
    basename_lookup: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        basename_lookup[path.name.casefold()].append(path)
    report_images = []
    statuses = Counter()
    detection_counts = Counter()
    distress_detection_counts = Counter()
    totals: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "matched": 0})
    evaluated, successful = 0, 0
    inferred_keys = set()
    found_result = False
    used_plot_names = set()
    for result_file in sorted(prediction_dir.glob("*.json")):
        data = json.loads(result_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "photo" not in data:
            continue
        found_result = True
        name = data["photo"]
        status = data.get("status", "unknown")
        statuses[status] += 1
        row = {"photo": name, "status": status, "plot": None, "prediction_count": None, "ground_truth_count": None}
        if status not in ("ok", "ok_reduced"):
            report_images.append(row)
            continue
        successful += 1
        key = _key(name)
        inferred_keys.add(key)
        relative_name = name.replace("\\", "/").casefold()
        source = image_lookup.get(relative_name)
        if source is None:
            candidates = basename_lookup.get(key, [])
            if len(candidates) > 1:
                raise ValueError(f"multiple source photos named {name}; use unique file names in the inference run")
            source = candidates[0] if candidates else None
        if source is None:
            raise FileNotFoundError(f"could not find source photo {name} under {image_dir}")
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        native_size = data.get("native_size")
        if native_size and native_size != [image.width, image.height]:
            raise ValueError(f"{name}: recorded native size {native_size} does not match source photo "
                             f"{[image.width, image.height]}")
        # accepted detections only: suppressors and rejected proposals are never drawn
        preds = []
        for group, kind in (("components", "component"), ("distresses", "distress")):
            for item in data.get(group, []):
                if item.get("rle"):
                    preds.append(prediction_record(item, kind, image.size))
        for item in data.get("tiled", {}).get("distresses", []):
            if item.get("rle"):
                preds.append(prediction_record(item, "distress", image.size))
        detection_counts.update(item["class"] for item in preds)
        distress_detection_counts.update(item["class"] for item in preds if item["kind"] == "distress")
        gts = None
        if ground_truth_path and key in gt_images:
            gt_image = gt_images[key]
            if (int(gt_image["height"]), int(gt_image["width"])) != (image.height, image.width):
                raise ValueError(f"{name}: COCO image dimensions do not match source photo")
            gts = [{"class": item["class"],
                    "mask": decode_coco_segmentation(item["segmentation"], *item["size"])}
                   for item in gt_annotations.get(key, [])]
            evaluated += 1
            labels = {r["class"] for r in preds} | {r["class"] for r in gts}
            for label in labels:
                pred_cls = [r for r in preds if r["class"] == label]
                gt_cls = [r for r in gts if r["class"] == label]
                result = match_masks(pred_cls, gt_cls, iou_threshold)
                for _, matched_gt, iou in result["matches"]:
                    if matched_gt is not None:
                        totals[label]["iou_sum"] += iou
                        totals[label]["matched"] += 1
                for metric in ("tp", "fp", "fn"):
                    totals[label][metric] += result[metric]
        plot_name = f"{Path(name).stem}_review.png"
        if plot_name.casefold() in used_plot_names:
            suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
            plot_name = f"{Path(name).stem}_{suffix}_review.png"
        used_plot_names.add(plot_name.casefold())
        render_panels(image, preds, gts).save(output_dir / plot_name)
        row.update(plot=plot_name, prediction_count=len(preds),
                   ground_truth_count=len(gts) if gts is not None else None)
        report_images.append(row)

    if not found_result:
        raise ValueError(f"no per-photo inference JSON files found in {prediction_dir}")
    class_rows = []
    if ground_truth_path:
        for label, result in sorted(totals.items()):
            tp, fp, fn = result["tp"], result["fp"], result["fn"]
            class_rows.append({
                "class": label, "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "matched_mask_iou": result["iou_sum"] / result["matched"] if result["matched"] else None,
            })

    repo_root = Path(__file__).resolve().parents[1]
    defect_coverage = build_defect_coverage(
        distress_detection_counts,
        repo_root / "kb" / "sp35_2024",
        repo_root / "kb" / "prompts" / "sam3_prompts.yaml",
        len(image_paths),
        sum(statuses.values()),
        successful,
    )
    summary = {
        "photos_with_successful_inference": successful,
        "inference_status_counts": dict(statuses),
        "detection_counts_by_class": dict(sorted(detection_counts.items())),
        "distress_proposal_counts_by_class": dict(sorted(distress_detection_counts.items())),
        "sp35_defect_coverage": defect_coverage,
        "photos_with_ground_truth_match": evaluated,
        "ground_truth_images_without_inference": len(set(gt_images) - inferred_keys),
        "ground_truth_file": str(ground_truth_path) if ground_truth_path else None,
        "iou_threshold": iou_threshold if ground_truth_path else None,
        "comparison_method": "score-descending, greedy one-to-one mask matching per class at the stated IoU threshold"
        if ground_truth_path else None,
        "note": "These are segmentation review metrics, not mask AP or SP:35 ratings."
        if ground_truth_path else "No ground-truth annotations supplied; no accuracy metrics were calculated.",
        "plot_note": "Plots are resized to at most 1600 pixels on the longest side; metrics use full-resolution masks.",
        "classes": class_rows,
        "images": report_images,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if ground_truth_path:
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["class", "tp", "fp", "fn", "precision", "recall", "matched_mask_iou"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(class_rows)
    chart_html = ""
    if detection_counts:
        render_count_chart(detection_counts, output_dir / "detection_counts.png")
        chart_html = '<h2>Detection counts by class</h2><img src="detection_counts.png" alt="Detection counts by class">'
    defect_coverage_html = render_defect_coverage_html(defect_coverage)
    rows_html = "\n".join(
        (f'<li><a href="{html.escape(row["plot"])}">{html.escape(row["photo"])}</a>: '
         f'{row["prediction_count"]} SAM masks'
         + (f', {row["ground_truth_count"]} ground-truth masks'
            if row["ground_truth_count"] is not None else "")
         + f' ({html.escape(row["status"])})</li>')
        if row["plot"] else f'<li>{html.escape(row["photo"])}: {html.escape(row["status"])}</li>'
        for row in summary["images"]
    )
    metric_html = ""
    if ground_truth_path:
        metric_rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row[field] if row[field] is not None else 'n/a'))}</td>"
                             for field in ("class", "tp", "fp", "fn", "precision", "recall", "matched_mask_iou"))
            + "</tr>" for row in class_rows
        )
        metric_html = (
            f"<p>Greedy one-to-one matches at mask IoU ≥ {iou_threshold:.2f}. "
            "Precision/recall and mean matched-mask IoU are not mask AP.</p>"
            "<table><thead><tr><th>Class</th><th>TP</th><th>FP</th><th>FN</th>"
            "<th>Precision</th><th>Recall</th><th>Matched IoU</th></tr></thead>"
            f"<tbody>{metric_rows}</tbody></table>"
        )
    title = "SAM Inference Review"
    html_text = ("<!doctype html><html><head><meta charset=\"utf-8\"><title>SAM review</title>"
                 "<style>body{font:16px sans-serif;max-width:1100px;margin:2rem auto}img{max-width:100%;height:auto}"
                 "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.35rem}</style></head><body>"
                 f"<h1>{title}</h1><p>{html.escape(summary['note'])}</p>"
                 f"<p>{html.escape(summary['plot_note'])}</p>"
                 f"<p>Successful photos: {successful}; statuses: {html.escape(json.dumps(dict(statuses)))}</p>"
                 f"{defect_coverage_html}{chart_html}{metric_html}<h2>Photos</h2><ul>{rows_html}</ul>"
                 "</body></html>")
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Folder of per-photo inference JSON files")
    parser.add_argument("--images", type=Path, required=True, help="Folder containing the original photos")
    parser.add_argument("--ground-truth", type=Path, help="Optional COCO instance-segmentation annotations JSON")
    parser.add_argument("--outdir", type=Path, default=Path("data/pilot/review"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0 < args.iou_threshold <= 1:
        parser.error("--iou-threshold must be greater than 0 and at most 1")
    summary = write_report(args.predictions, args.images, args.outdir, args.ground_truth, args.iou_threshold)
    print(json.dumps({key: summary[key] for key in (
        "photos_with_successful_inference", "inference_status_counts", "photos_with_ground_truth_match",
        "ground_truth_file", "iou_threshold")}, indent=2))
    print(f"Review report: {args.outdir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
