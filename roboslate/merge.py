"""
Multi-frame detection merging.

When multiple frames of the same slate are scanned, this module consolidates
the readings into a single authoritative result:

  - Fields consistent across frames → promoted to high confidence
  - Fields where all frames agree on None → remain None
  - Fields with conflicting readings → flagged as conflicts, kept at low confidence
  - Fields from a single frame → inherit that frame's per-field confidence
"""

from collections import Counter


CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, None: 0}
SLATE_FIELDS = [
    "scene", "take", "slate_number", "roll", "camera",
    "director", "dop", "production",
    "date", "fps", "format", "notes",
]


# ---------------------------------------------------------------------------
# Field-level merging
# ---------------------------------------------------------------------------

def _merge_field(readings):
    """
    Merge a list of per-frame readings for a single field.

    Each reading is a dict: {"value": str|None, "confidence": "high"|"medium"|"low"}
    or just None (field absent from that frame's response).

    Returns:
        dict: {"value": ..., "confidence": ..., "conflict": bool, "conflict_values": list|None}
    """
    # Normalise: treat missing/None-value readings as absent
    non_null = [r for r in readings if r and r.get("value") is not None]

    if not non_null:
        return {"value": None, "confidence": "low", "conflict": False, "conflict_values": None}

    # Collect unique values (case-insensitive comparison, preserve original case)
    value_counter = Counter(r["value"].strip() for r in non_null if r.get("value"))
    unique_values = list(value_counter.keys())

    if len(unique_values) == 1:
        # All frames agree on this value
        value = unique_values[0]
        # Promote confidence: if multiple frames agree, cap at high
        if len(non_null) >= 2:
            confidence = "high"
        else:
            confidence = non_null[0].get("confidence", "low")
        return {
            "value": value,
            "confidence": confidence,
            "conflict": False,
            "conflict_values": None,
        }
    else:
        # Conflict: multiple different values seen
        # Use the most common value, but mark as low confidence
        most_common = value_counter.most_common(1)[0][0]
        return {
            "value": most_common,
            "confidence": "low",
            "conflict": True,
            "conflict_values": unique_values,
        }


# ---------------------------------------------------------------------------
# Detection-level merging
# ---------------------------------------------------------------------------

def merge_detections(detections, fields=None):
    """
    Merge a list of per-frame detection dicts into a single consolidated result.

    Args:
        detections: List of result dicts from vision.detect_and_extract(),
                    each augmented with 'phase', 'timestamp', 'frame_file'.
        fields: List of field names to include. None → use all SLATE_FIELDS.

    Returns:
        Dict with keys:
          - slate_detected (bool)
          - overall_confidence ("high"|"medium"|"low"|"none")
          - fields (dict of field_name → {value, confidence, conflict, conflict_values})
          - frame_count (int)
          - detection_frames (list of timestamps where slate was detected)
          - conflicts (list of field names with conflicting readings)
          - needs_review (bool) — True if any field has a conflict or is low confidence
          - best_frame (dict) — the single frame result with highest overall confidence
    """
    active_fields = fields if fields is not None else SLATE_FIELDS

    # Filter to frames that detected a slate
    slate_frames = [d for d in detections if d.get("slate_detected")]

    if not slate_frames:
        return {
            "slate_detected": False,
            "overall_confidence": "none",
            "fields": {f: {"value": None, "confidence": "low", "conflict": False, "conflict_values": None} for f in active_fields},
            "frame_count": len(detections),
            "detection_frames": [],
            "conflicts": [],
            "needs_review": False,
            "best_frame": None,
        }

    # Find the best individual frame (highest overall_confidence, then earliest timestamp)
    def _frame_rank(d):
        return (
            CONFIDENCE_RANK.get(d.get("overall_confidence", d.get("confidence")), 0),
            -d.get("timestamp", 0),
        )
    best_frame = max(slate_frames, key=_frame_rank)

    # Merge each field across all slate-detecting frames
    merged_fields = {}
    for field in active_fields:
        readings = []
        for d in slate_frames:
            field_data = (d.get("fields") or {}).get(field)
            if isinstance(field_data, dict):
                readings.append(field_data)
            elif field_data is not None:
                # Legacy format: plain string
                readings.append({"value": field_data, "confidence": "low"})
            else:
                readings.append(None)
        merged_fields[field] = _merge_field(readings)

    # Cross-field cleanup: if scene was read as "57-226" and slate_number is "226",
    # the slate number crept into the scene field — strip it.
    _clean_scene_field(merged_fields)

    conflicts = [f for f, v in merged_fields.items() if v.get("conflict")]

    # Overall confidence: downgrade if there are conflicts or low-confidence fields
    if conflicts:
        overall = "low"
    else:
        confidences = [v["confidence"] for v in merged_fields.values() if v["value"] is not None]
        if not confidences:
            overall = "low"
        elif all(c == "high" for c in confidences):
            overall = "high"
        elif any(c == "high" for c in confidences):
            overall = "medium"
        else:
            overall = "low"

    needs_review = bool(conflicts) or overall == "low"

    return {
        "slate_detected": True,
        "overall_confidence": overall,
        "fields": merged_fields,
        "frame_count": len(detections),
        "detection_frames": [d.get("timestamp") for d in slate_frames],
        "conflicts": conflicts,
        "needs_review": needs_review,
        "best_frame": best_frame,
    }


# ---------------------------------------------------------------------------
# Flat field extractor (for downstream use)
# ---------------------------------------------------------------------------

def _clean_scene_field(fields):
    """
    Remove slate_number contamination from scene.

    Some frames are read as scene="57-226" when scene=57 and slate_number=226.
    If scene ends with "-{slate_number}", strip the suffix in place.
    Also removes conflict values that were just the contaminated form.
    """
    scene = fields.get("scene", {})
    slate = fields.get("slate_number", {})

    slate_val = slate.get("value") if isinstance(slate, dict) else None
    if not slate_val:
        return

    def _strip(val):
        if val and isinstance(val, str):
            suffix = f"-{slate_val}"
            if val.endswith(suffix):
                return val[: -len(suffix)]
        return val

    if isinstance(scene, dict):
        scene["value"] = _strip(scene.get("value"))
        if scene.get("conflict_values"):
            cleaned = list({_strip(v) for v in scene["conflict_values"]})
            scene["conflict_values"] = cleaned
            if len(cleaned) == 1:
                scene["value"] = cleaned[0]
                scene["conflict"] = False
                scene["conflict_values"] = None


