"""
Local vision backend for RoboSlate using PaddleOCR.

Drop-in replacement for roboslate/vision.py — identical public interface,
no Claude API required. Uses PaddleOCR for text detection and recognition,
then maps extracted text blobs to slate fields via spatial proximity.

Accuracy is lower than the Claude backend, especially for:
  - Unusual slate layouts
  - Low-contrast or backlit stickers
  - Handwritten entries
  - Partially visible slates

Use this backend for cost-free local testing:
    python roboslate.py --file clip.mov --backend local
"""

import logging
import os

from roboslate.merge import SLATE_FIELDS
from roboslate import preprocessing

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label keyword map — maps field names to their common slate label variants
# ---------------------------------------------------------------------------

# Each list contains lowercase strings to match against (after stripping punctuation)
LABEL_MAP = {
    "scene":        ["scene", "sc", "sce", "scn"],
    "take":         ["take", "tk", "t"],
    "slate_number": ["slate", "slate#", "slate #", "board", "slat"],
    "roll":         ["roll", "mag", "magazine", "cam roll", "mag.", "reel", "rulle"],
    "camera":       ["camera", "cam", "cam.", "kamera", "camara"],
    "director":     ["director", "dir", "dir.", "instruktor", "regie"],
    "dop":          ["dop", "dp", "d.p.", "d.o.p", "cinematographer", "dop.", "fotograf", "fotogr"],
    "production":   ["production", "prod", "prod.", "produktion", "film"],
    "date":         ["date", "dato", "dag"],
    "fps":          ["fps", "frame rate", "framerate", "fr", "billedfrekvens"],
    "format":       ["format", "fmt"],
    "notes":        ["notes", "note", "remarks", "bemærkning"],
}

# Flat lookup: lowercase label text → field name
_LABEL_LOOKUP = {
    alias: field
    for field, aliases in LABEL_MAP.items()
    for alias in aliases
}

# Minimum number of label keywords that must appear for slate detection
SLATE_KEYWORD_THRESHOLD = 3

# OCR confidence → RoboSlate confidence mapping thresholds
OCR_HIGH   = 0.90
OCR_MEDIUM = 0.70


# ---------------------------------------------------------------------------
# Fallback result (no slate detected or processing error)
# ---------------------------------------------------------------------------

def _fallback(preprocessed=False, escalated=False, pass_num=1):
    return {
        "slate_detected":    False,
        "overall_confidence": "low",
        "partially_visible": False,
        "fields": {f: {"value": None, "confidence": "low"} for f in SLATE_FIELDS},
        "extraction_notes":  None,
        "parse_error":       False,
        "model_used":        "paddleocr",
        "preprocessed":      preprocessed,
        "escalated":         escalated,
        "pass":              pass_num,
        "needs_review":      True,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_client(api_key=None):
    """
    Initialise and return a PaddleOCR instance.

    api_key is accepted for interface compatibility but ignored.
    PaddleOCR models are downloaded on first use (~100 MB total).
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise RuntimeError(
            "PaddleOCR is not installed. Run:\n"
            "  venv/bin/pip install paddlepaddle paddleocr\n"
            "or install from requirements-local.txt:\n"
            "  venv/bin/pip install -r requirements-local.txt"
        )

    # Suppress PaddleOCR's connectivity check and verbose output
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    # use_textline_orientation=True handles rotated/angled text on slates
    ocr = PaddleOCR(use_textline_orientation=True, lang="en")
    log.info("PaddleOCR initialised (local backend)")
    return ocr


def scan_frames(frame_entries, client, cfg, reel_hint=None):
    """
    Scan a list of frames with the local OCR backend.

    Args:
        frame_entries: List of (phase, timestamp, file_path) tuples.
        client: PaddleOCR instance from build_client().
        cfg: Config dict.
        reel_hint: Camera reel string from filename (e.g. "A051").

    Returns:
        List of detection dicts (same schema as vision.scan_frames()).
    """
    max_frames          = int(cfg.get("MAX_FRAMES", 40))
    early_stop_conf     = cfg.get("EARLY_STOP_CONFIDENCE", "high")
    consistent_stop     = int(cfg.get("CONSISTENT_READINGS_STOP", 5))

    results = []
    consecutive_slate   = 0
    total_api_calls     = 0  # OCR calls in local mode

    for phase, timestamp, frame_path in frame_entries[:max_frames]:
        if not os.path.isfile(str(frame_path)):
            log.warning(f"Frame file missing: {frame_path}")
            continue

        detection = detect_with_escalation(frame_path, client, cfg, reel_hint=reel_hint)
        total_api_calls += detection.get("_ocr_calls", 1)

        detection["phase"]      = phase
        detection["timestamp"]  = timestamp
        detection["frame_file"] = os.path.basename(str(frame_path))
        detection["_total_api_calls"] = total_api_calls

        results.append(detection)

        if detection.get("slate_detected"):
            consecutive_slate += 1
        else:
            consecutive_slate = 0

        # Early exit: enough consecutive high-confidence slate readings
        if (
            consecutive_slate >= consistent_stop
            and early_stop_conf != "none"
            and detection.get("overall_confidence") == early_stop_conf
        ):
            log.info(
                f"Early stop after {len(results)} frames "
                f"({consecutive_slate} consecutive {early_stop_conf}-confidence slate reads)"
            )
            break

    return results


def detect_with_escalation(image_path, client, cfg, reel_hint=None):
    """
    Run the local OCR pipeline on a single frame.

    Pass 1: OCR on raw frame.
    Pass 2: OCR on preprocessed frame (if pass 1 is low confidence).

    Args:
        image_path: Absolute path to JPEG frame.
        client: PaddleOCR instance.
        cfg: Config dict.
        reel_hint: Camera reel string from filename (e.g. "A051").

    Returns:
        Detection dict with all required keys.
    """
    max_size = int(cfg.get("MAX_IMAGE_SIZE", 1568))
    camera_letter = reel_hint[0].upper() if reel_hint else None

    # --- Pass 1: raw frame ---
    result = _run_ocr(image_path, client, max_size, preprocessed=False, pass_num=1)
    ocr_calls = 1

    # --- Pass 2: preprocess and retry if low confidence ---
    if not result["slate_detected"] or result["overall_confidence"] == "low":
        try:
            preprocessed_path = preprocessing.preprocess_frame_file(image_path)
            result2 = _run_ocr(preprocessed_path, client, max_size, preprocessed=True, pass_num=2)
            ocr_calls += 1

            # Keep whichever pass found more fields
            fields1 = _count_extracted(result)
            fields2 = _count_extracted(result2)
            if fields2 > fields1 or (not result["slate_detected"] and result2["slate_detected"]):
                result = result2
                result["escalated"] = True
        except Exception as e:
            log.warning(f"Preprocessing/pass 2 failed: {e}")

    result["_ocr_calls"] = ocr_calls

    # --- reel_hint roll correction ---
    if camera_letter and result.get("slate_detected"):
        roll_field = result["fields"].get("roll", {})
        roll_value = (roll_field.get("value") or "")
        roll_wrong = roll_value and not roll_value.upper().startswith(camera_letter)

        if roll_value is None or roll_wrong:
            if reel_hint:
                result["fields"]["roll"] = {
                    "value":      reel_hint.upper(),
                    "confidence": "medium",
                    "source":     "filename",
                }
                log.info(f"Roll overridden from filename: {reel_hint.upper()}")

    return result


# ---------------------------------------------------------------------------
# Core OCR → field extraction
# ---------------------------------------------------------------------------

def _run_ocr(image_path, ocr_client, max_size, preprocessed=False, pass_num=1):
    """
    Run PaddleOCR on a single image file and extract slate fields.

    Returns a full detection dict.
    """
    try:
        raw = ocr_client.predict(image_path)
    except Exception as e:
        log.warning(f"PaddleOCR failed on {image_path}: {e}")
        result = _fallback(preprocessed=preprocessed, pass_num=pass_num)
        result["parse_error"] = True
        return result

    # New PaddleOCR API returns: [ {rec_texts, rec_scores, rec_polys, ...} ]
    blobs = _parse_ocr_output(raw)

    if not blobs:
        return _fallback(preprocessed=preprocessed, pass_num=pass_num)

    # Classify blobs as labels or values
    label_blobs, value_map = _classify_blobs(blobs)

    # Slate detection heuristic
    n_keywords = len(label_blobs)
    avg_score  = (
        sum(b["score"] for b in blobs) / len(blobs)
        if blobs else 0.0
    )
    slate_detected = n_keywords >= SLATE_KEYWORD_THRESHOLD

    if not slate_detected:
        return _fallback(preprocessed=preprocessed, pass_num=pass_num)

    # Overall confidence
    if n_keywords >= 5 and avg_score >= OCR_HIGH:
        overall_conf = "high"
    elif n_keywords >= 3 and avg_score >= OCR_MEDIUM:
        overall_conf = "medium"
    else:
        overall_conf = "low"

    # Build per-field results
    fields = {}
    for field in SLATE_FIELDS:
        entry = value_map.get(field)
        if entry:
            conf_str = _score_to_conf(entry["score"])
            fields[field] = {"value": entry["text"], "confidence": conf_str}
        else:
            fields[field] = {"value": None, "confidence": "low"}

    # Split combined "scene-slate" values (e.g. "64-353" → scene=64, slate_number=353)
    # Slates sometimes show scene and slate number joined with a hyphen in one field
    _split_scene_slate(fields)

    extraction_notes = (
        f"PaddleOCR pass {pass_num}: {n_keywords} label(s) found, "
        f"avg score {avg_score:.2f}"
    )

    return {
        "slate_detected":    True,
        "overall_confidence": overall_conf,
        "partially_visible": n_keywords < 5,
        "fields":            fields,
        "extraction_notes":  extraction_notes,
        "parse_error":       False,
        "model_used":        "paddleocr",
        "preprocessed":      preprocessed,
        "escalated":         False,
        "pass":              pass_num,
        "needs_review":      overall_conf != "high",
    }


def _parse_ocr_output(raw):
    """
    Flatten new-style PaddleOCR predict() output into a list of blob dicts.

    New API returns: [ {"rec_texts": [...], "rec_scores": [...], "rec_polys": [...], ...} ]

    Each blob: {"text": str, "score": float,
                "x0": float, "y0": float, "x1": float, "y1": float,
                "cx": float, "cy": float}
    """
    blobs = []
    if not raw:
        return blobs

    page = raw[0] if raw else {}
    texts  = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    polys  = page.get("rec_polys", [])

    for text, score, poly in zip(texts, scores, polys):
        if not text or not text.strip():
            continue

        # poly is a numpy array of shape (N, 2) or list of [x, y] points
        try:
            xs = [float(pt[0]) for pt in poly]
            ys = [float(pt[1]) for pt in poly]
        except (TypeError, IndexError):
            continue

        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        blobs.append({
            "text":  text.strip(),
            "score": float(score),
            "x0": x0, "y0": y0,
            "x1": x1, "y1": y1,
            "cx": (x0 + x1) / 2,
            "cy": (y0 + y1) / 2,
        })

    return blobs


# Regex to split "Label: value" or "Label value" inline blobs
import re as _re
_INLINE_SPLIT = _re.compile(r"^(.+?)[\s:]+(.+)$")


def _classify_blobs(blobs):
    """
    Split blobs into recognised label→value pairs.

    Handles two cases:
    1. Inline combined blob: "SCENE: 64" or "Take:02" → split on colon/space
    2. Separate blobs: label blob + nearest value blob (spatial proximity)

    Returns:
        label_blobs: list of blobs identified as labels
        value_map: dict of {field_name: best_value_blob}
    """
    label_blobs = []
    value_blobs = []
    label_fields = {}  # blob index → field name
    value_map = {}

    # --- Pass 1: inline "Label: value" blobs ---
    remaining_blobs = []
    for blob in blobs:
        text_norm = blob["text"].strip()
        matched = False

        # Try splitting on colon separator first
        if ":" in text_norm:
            parts = text_norm.split(":", 1)
            label_part = parts[0].strip().lower().rstrip(" .")
            value_part = parts[1].strip()
            field = _LABEL_LOOKUP.get(label_part)
            if field and value_part:
                label_blobs.append(blob)
                vblob = dict(blob)
                vblob["text"] = value_part
                if field not in value_map:
                    value_map[field] = vblob
                matched = True

        if not matched:
            remaining_blobs.append(blob)

    # --- Pass 2: classify remaining blobs as labels or pure values ---
    for i, blob in enumerate(remaining_blobs):
        normalised = blob["text"].lower().strip(" :.-")
        field = _LABEL_LOOKUP.get(normalised)
        if field:
            label_blobs.append(blob)
            label_fields[i] = field
        else:
            value_blobs.append(blob)

    # --- Pass 3: spatial proximity for standalone label blobs ---
    for idx, field in label_fields.items():
        if field in value_map:
            continue  # already found inline

        label     = remaining_blobs[idx]
        best_blob = None
        best_dist = float("inf")
        label_h   = label["y1"] - label["y0"]

        for vblob in value_blobs:
            to_right = vblob["x0"] >= label["x1"] - 5
            below    = (
                vblob["y0"] >= label["y1"] - 5
                and abs(vblob["cx"] - label["cx"]) < label_h * 3
            )
            if not (to_right or below):
                continue

            dy   = abs(vblob["cy"] - label["cy"])
            dx   = max(0.0, vblob["x0"] - label["x1"])
            dist = dx + dy * 2

            if dist < best_dist:
                best_dist = dist
                best_blob = vblob

        if best_blob:
            existing = value_map.get(field)
            if existing is None or best_dist < existing.get("_dist", float("inf")):
                best_blob = dict(best_blob)
                best_blob["_dist"] = best_dist
                value_map[field] = best_blob

    return label_blobs, value_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_scene_slate(fields):
    """
    If scene has a value like "64-353" and slate_number is absent, split it.
    Modifies fields in place.
    """
    scene_field = fields.get("scene", {})
    slate_field = fields.get("slate_number", {})

    scene_val = scene_field.get("value") if isinstance(scene_field, dict) else None
    slate_val = slate_field.get("value") if isinstance(slate_field, dict) else None

    if scene_val and slate_val is None and "-" in scene_val:
        parts = scene_val.split("-", 1)
        if parts[0].strip() and parts[1].strip():
            conf = scene_field.get("confidence", "medium")
            fields["scene"]        = {"value": parts[0].strip(), "confidence": conf}
            fields["slate_number"] = {"value": parts[1].strip(), "confidence": conf}


def _score_to_conf(score):
    if score >= OCR_HIGH:
        return "high"
    if score >= OCR_MEDIUM:
        return "medium"
    return "low"


def _count_extracted(result):
    """Count fields with non-None values in a detection result."""
    fields = result.get("fields") or {}
    return sum(1 for f in fields.values() if isinstance(f, dict) and f.get("value") is not None)
