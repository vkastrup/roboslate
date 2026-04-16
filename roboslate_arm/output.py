"""
Result formatting and output for RoboSlate-arm.

Builds the final JSON result structure from merged detection data and
handles writing to file, CSV, and printing a terminal summary.

Status categories:
  found         — slate detected, fields extracted
  not_found     — no slate detected after full scan
  unreadable    — slate detected but all fields returned null
  error         — processing failed
"""

import csv
import json
import os
import re
from datetime import datetime, timezone

from roboslate_arm import __version__
from roboslate_arm.merge import SLATE_FIELDS


# ---------------------------------------------------------------------------
# Roll disambiguation from filename
# ---------------------------------------------------------------------------

_REEL_PATTERN = re.compile(r"^([A-Z]\d{3})", re.IGNORECASE)


def extract_reel_from_filename(filename):
    """
    Extract the reel/roll identifier from a camera filename.

    Handles common patterns:
      A051C002_230614_A4LG.mp4  → A051
      A001_C001_230614.mov      → A001
      B042C003.mxf              → B042

    Returns the reel string (e.g. "A051") or None if not found.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = _REEL_PATTERN.match(stem)
    return m.group(1).upper() if m else None


def disambiguate_roll(merged_fields, filename):
    """
    When the roll field contains multiple comma-separated values (multi-camera slate),
    use the filename reel to select the correct one.

    Modifies merged_fields in place.
    """
    reel = extract_reel_from_filename(filename)
    if not reel:
        return

    roll_field = merged_fields.get("roll", {})
    if not isinstance(roll_field, dict):
        return

    value = roll_field.get("value") or ""
    if "," not in value:
        if value.upper() == reel:
            roll_field["confidence"] = "high"
            roll_field["filename_confirmed"] = True
        return

    candidates = [v.strip() for v in value.split(",")]
    match = next((c for c in candidates if c.upper() == reel), None)

    if match:
        roll_field["value"] = match
        roll_field["confidence"] = "high"
        roll_field["conflict"] = False
        roll_field["conflict_values"] = None
        roll_field["filename_confirmed"] = True


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------

def classify_status(merged):
    """
    Classify a merged result into one of four status categories.

    Returns: "found" | "not_found" | "unreadable"
    """
    if not merged.get("slate_detected"):
        return "not_found"

    fields = merged.get("fields") or {}
    has_values = any(
        isinstance(v, dict) and v.get("value") is not None
        for v in fields.values()
    )
    if not has_values:
        return "unreadable"

    return "found"


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def build_result(video_path, merged, scan_metadata, shot_info=None):
    """
    Assemble the final result dict from merged detection data.

    Args:
        video_path: Absolute path to the source video file.
        merged: Output of merge.merge_detections().
        scan_metadata: Dict with: duration_seconds, frames_scanned,
                       ocr_calls_made, phases_run, scan_log.
        shot_info: Optional dict with SCRATCH shot metadata.

    Returns:
        Complete result dict ready for JSON serialisation.
    """
    shot = shot_info or {}

    if video_path and merged.get("fields"):
        disambiguate_roll(merged["fields"], os.path.basename(video_path))

    status = classify_status(merged)
    best = merged.get("best_frame") or {}

    result = {
        "roboslate_version": __version__,
        "processed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,

        "source": {
            "file": os.path.abspath(video_path) if video_path else None,
            "filename": os.path.basename(video_path) if video_path else None,
            "duration_seconds": scan_metadata.get("duration_seconds"),
            "scratch_shot_uuid": shot.get("uuid"),
            "scratch_shot_name": shot.get("name"),
            "scratch_project": shot.get("project"),
            "scratch_construct": shot.get("construct"),
        },

        "result": {
            "status": status,
            "slate_found": merged.get("slate_detected", False),
            "overall_confidence": merged.get("overall_confidence", "none"),
            "detected_at_seconds": best.get("timestamp") if merged.get("slate_detected") else None,
            "detection_phase": best.get("phase") if merged.get("slate_detected") else None,
            "frames_scanned": scan_metadata.get("frames_scanned", merged.get("frame_count", 0)),
            "ocr_calls_made": scan_metadata.get("ocr_calls_made", 0),
            "needs_review": merged.get("needs_review", False),
            "conflicts": merged.get("conflicts", []),
        },

        "slate": merged.get("fields") if merged.get("slate_detected") else None,

        "scan_log": scan_metadata.get("scan_log", []),
    }

    return result


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def get_default_output_path(video_path):
    """
    Return default JSON path: same directory as video, .roboslate.json suffix.
    e.g. /media/A001_C007.mov → /media/A001_C007.roboslate.json
    """
    stem = os.path.splitext(video_path)[0]
    return stem + ".roboslate-arm.json"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_json(result, output_path):
    """Write result dict as pretty-printed JSON."""
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


def write_csv_row(result, csv_path, fields=None):
    """
    Append a single flat row to a CSV file.
    Creates the file with a header row if it does not exist.
    """
    active_fields = fields if fields is not None else SLATE_FIELDS
    src = result.get("source", {})
    res = result.get("result", {})
    slate = result.get("slate") or {}

    row = {
        "filename":             src.get("filename"),
        "status":               result.get("status"),
        "overall_confidence":   res.get("overall_confidence"),
        "detected_at_seconds":  res.get("detected_at_seconds"),
        "needs_review":         res.get("needs_review"),
        "conflicts":            "|".join(res.get("conflicts", [])),
    }

    for field in active_fields:
        field_data = slate.get(field, {})
        if isinstance(field_data, dict):
            row[field] = field_data.get("value")
            row[f"{field}_confidence"] = field_data.get("confidence", "low")
        else:
            row[field] = field_data
            row[f"{field}_confidence"] = "low"

    row.update({
        "frames_scanned":     res.get("frames_scanned"),
        "ocr_calls_made":     res.get("ocr_calls_made"),
        "processed_at":       result.get("processed_at"),
        "source_file":        src.get("file"),
    })

    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def print_summary(result, fields=None):
    """Print a human-readable summary to stdout."""
    src = result.get("source", {})
    res = result.get("result", {})
    slate = result.get("slate") or {}
    status = result.get("status", res.get("status", "unknown"))

    print(f"\n{'='*56}")
    print(f"  {src.get('filename', 'unknown')}")
    print(f"{'='*56}")

    status_labels = {
        "found":      "Slate found",
        "not_found":  "No slate detected",
        "unreadable": "Slate found but unreadable",
        "error":      "Processing error",
    }
    print(f"  Status         : {status_labels.get(status, status)}")

    if status == "found":
        print(f"  Confidence     : {res.get('overall_confidence', '?')}")
        print(f"  Detected at    : {res.get('detected_at_seconds')}s [{res.get('detection_phase')}]")

        if res.get("needs_review"):
            conflicts = res.get("conflicts", [])
            if conflicts:
                print(f"  !! REVIEW      : conflicts in: {', '.join(conflicts)}")
            else:
                print(f"  !! REVIEW      : low confidence fields")

    print(f"  Frames scanned : {res.get('frames_scanned')} ({res.get('ocr_calls_made')} OCR calls)")

    if status == "found" and slate:
        print()
        all_field_labels = [
            ("Scene",      "scene"),
            ("Take",       "take"),
            ("Slate #",    "slate_number"),
            ("Roll/Mag",   "roll"),
            ("Camera",     "camera"),
            ("Director",   "director"),
            ("DoP",        "dop"),
            ("Production", "production"),
            ("Date",       "date"),
            ("FPS",        "fps"),
            ("Format",     "format"),
            ("Notes",      "notes"),
        ]
        active_set = set(fields) if fields is not None else None
        field_labels = [
            (label, key) for label, key in all_field_labels
            if active_set is None or key in active_set
        ]
        for label, key in field_labels:
            field_data = slate.get(key, {})
            if not isinstance(field_data, dict):
                continue
            value = field_data.get("value")
            confidence = field_data.get("confidence", "low")
            conflict = field_data.get("conflict", False)

            if value is not None:
                flag = ""
                if conflict:
                    flag = f"  !! CONFLICT: {field_data.get('conflict_values', [])}"
                elif confidence == "low":
                    flag = "  !! low confidence"
                elif confidence == "medium":
                    flag = "  ~ medium"
                print(f"  {label:<12} : {value}{flag}")

    print(f"{'='*56}\n")
