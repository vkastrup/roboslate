"""
Apple Vision OCR backend for RoboSlate-arm.

Uses VNRecognizeTextRequest (Neural Engine) to detect and read slate text.
No API key or network access required — runs entirely on-device.

Drop-in replacement interface: build_client() + scan_frames()
"""

import logging
import os
import re

from roboslate_arm.merge import SLATE_FIELDS
from roboslate_arm import preprocessing

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR text normalisation
# ---------------------------------------------------------------------------

# Apple Vision occasionally reads Cyrillic lookalike characters instead of
# their Latin equivalents (e.g. Т→T, А→A, К→K). This breaks label lookup.
# Translate them to ASCII before any matching.
_CYRILLIC_TO_LATIN = str.maketrans({
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H',
    'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'R', 'Т': 'T',
    'Х': 'X', 'а': 'a', 'е': 'e', 'о': 'o', 'р': 'r',
    'с': 'c', 'х': 'x', 'у': 'y',
})

def _normalize_ocr(text):
    """Translate Cyrillic lookalikes to Latin, lowercase, strip punctuation."""
    return text.translate(_CYRILLIC_TO_LATIN).lower().strip(" :.-")


def _levenshtein(a, b):
    """Edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[len(b)]


def _fuzzy_label_lookup(text):
    """
    Match OCR text to a field name.

    1. Normalize Cyrillic lookalikes + lowercase.
    2. Exact lookup.
    3. If no match and text is short (≤ 8 chars): fuzzy match with edit distance ≤ 1.

    Returns field name or None.
    """
    normalized = _normalize_ocr(text)

    # Exact match
    field = _LABEL_LOOKUP.get(normalized)
    if field:
        return field

    # Fuzzy match — only for short tokens that could plausibly be labels.
    # Skip if the text is all digits — numbers are values, never field labels.
    if len(normalized) <= 8 and not normalized.isdigit():
        best_field = None
        best_dist = 2  # threshold: edit distance must be < 2 (i.e. ≤ 1)
        for alias, field_name in _LABEL_LOOKUP.items():
            if abs(len(normalized) - len(alias)) > 1:
                continue  # skip if length differs too much
            dist = _levenshtein(normalized, alias)
            if dist < best_dist:
                best_dist = dist
                best_field = field_name
        if best_field:
            return best_field

    return None


# Pattern for camera roll/mag sticker values: letter + exactly 3 digits (e.g. A013, B042)
_ROLL_STICKER_RE = re.compile(r'^[A-Z]\d{3}$', re.IGNORECASE)

# Per-field value sanity checks — reject proximity matches that don't fit the field's
# expected format. Prevents long strings (names, production titles) landing in
# numeric/code fields when the label is detected but the correct value is absent.
_FIELD_VALUE_RE = {
    "scene":        re.compile(r'^[A-Z\d]{1,6}$', re.IGNORECASE),
    "take":         re.compile(r'^[\d]{1,3}[A-Z]?$', re.IGNORECASE),
    "slate_number": re.compile(r'^[\d]{1,4}[A-Z]?$', re.IGNORECASE),
    # Roll sticker: exactly 1 letter + 3 digits (e.g. "A022", "B013")
    "roll":         re.compile(r'^[A-Z]\d{3}$', re.IGNORECASE),
    # Camera: single letter or digit
    "camera":       re.compile(r'^[A-Z\d]{1,2}$', re.IGNORECASE),
    "fps":          re.compile(r'^\d{2,3}(\.\d+)?$'),
}


# ---------------------------------------------------------------------------
# Label keyword map — maps field names to their common slate label variants
# ---------------------------------------------------------------------------

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

# Apple Vision confidence → RoboSlate confidence thresholds
OCR_HIGH   = 0.90
OCR_MEDIUM = 0.70

# Regex to split "Label: value" or "Label value" inline blobs
_INLINE_SPLIT = re.compile(r"^(.+?)[\s:]+(.+)$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_client(api_key=None):
    """
    Verify Apple Vision is available and return a sentinel client object.

    api_key is accepted for interface compatibility but ignored.
    Apple Vision runs on-device — no network, no key.

    Returns:
        True (a truthy sentinel; all actual Vision calls are made inline).

    Raises ImportError if pyobjc-framework-Vision is not installed.
    """
    try:
        import Vision   # noqa: F401
        import Quartz   # noqa: F401
    except ImportError:
        raise ImportError(
            "pyobjc-framework-Vision is not installed.\n"
            "Run: venv/bin/pip install pyobjc-framework-Vision pyobjc-framework-Quartz"
        )
    log.info("Apple Vision OCR backend ready")
    return True  # sentinel; Vision is imported fresh per call in _run_ocr()


def scan_frames(frame_entries, client, cfg, reel_hint=None):
    """
    Scan a list of frames with Apple Vision OCR.

    Args:
        frame_entries: List of (phase, timestamp, file_path) tuples.
        client: Sentinel from build_client() — not used directly.
        cfg: Config dict.
        reel_hint: Camera reel string from filename (e.g. "A051").

    Returns:
        List of detection dicts (same schema as vision_local.scan_frames()).
    """
    max_frames      = int(cfg.get("MAX_FRAMES", 40))
    early_stop_conf = cfg.get("EARLY_STOP_CONFIDENCE", "high")
    consistent_stop = int(cfg.get("CONSISTENT_READINGS_STOP", 5))

    results = []
    consecutive_slate = 0
    total_ocr_calls   = 0

    for phase, timestamp, frame_path in frame_entries[:max_frames]:
        if not os.path.isfile(str(frame_path)):
            log.warning(f"Frame file missing: {frame_path}")
            continue

        detection = detect_with_escalation(frame_path, cfg, reel_hint=reel_hint)
        total_ocr_calls += detection.get("_ocr_calls", 1)

        detection["phase"]             = phase
        detection["timestamp"]         = timestamp
        detection["frame_file"]        = os.path.basename(str(frame_path))
        detection["_total_ocr_calls"]  = total_ocr_calls

        results.append(detection)

        if detection.get("slate_detected"):
            consecutive_slate += 1
        else:
            consecutive_slate = 0

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


# ---------------------------------------------------------------------------
# Single-frame escalation pipeline
# ---------------------------------------------------------------------------

def detect_with_escalation(image_path, cfg, reel_hint=None):
    """
    Run Apple Vision OCR on a single frame with a preprocessing fallback.

    Pass 1: OCR on raw frame.
    Pass 2: OCR on preprocessed frame (only if pass 1 is low confidence).

    Returns:
        Detection dict with all required keys.
    """
    camera_letter = reel_hint[0].upper() if reel_hint else None

    # --- Pass 1: raw frame ---
    result = _run_ocr(image_path, preprocessed=False, pass_num=1)
    ocr_calls = 1

    # --- Pass 2: preprocess and retry if low confidence ---
    if not result["slate_detected"] or result["overall_confidence"] == "low":
        try:
            preprocessed_path = preprocessing.preprocess_frame_file(image_path)
            result2 = _run_ocr(preprocessed_path, preprocessed=True, pass_num=2)
            ocr_calls += 1

            # Keep whichever pass found more fields.
            # Prefer pass 2 on equal count — the preprocessed image is specifically
            # tuned for low-confidence frames, so equal quality means pass 2 is safer.
            if _count_extracted(result2) >= _count_extracted(result) or (
                not result["slate_detected"] and result2["slate_detected"]
            ):
                result = result2
                result["escalated"] = True
        except Exception as e:
            log.warning(f"Preprocessing/pass 2 failed for {image_path}: {e}")

    # Note: VLM escalation (Pass 3) is handled post-merge in roboslate-arm.py,
    # not per-frame here. Running VLM per-frame causes hallucinated values to
    # conflict with OCR readings from other frames, inflating needs_review counts.

    result["_ocr_calls"] = ocr_calls

    # --- reel_hint roll correction ---
    if camera_letter and result.get("slate_detected"):
        roll_field = result["fields"].get("roll", {})
        roll_value = (roll_field.get("value") or "") if isinstance(roll_field, dict) else ""
        roll_wrong = roll_value and not roll_value.upper().startswith(camera_letter)

        if not roll_value or roll_wrong:
            if reel_hint:
                result["fields"]["roll"] = {
                    "value":      reel_hint.upper(),
                    "confidence": "medium",
                    "source":     "filename",
                }
                log.info(f"Roll overridden from filename: {reel_hint.upper()}")

    return result


# ---------------------------------------------------------------------------
# Core OCR
# ---------------------------------------------------------------------------

def _run_ocr(image_path, preprocessed=False, pass_num=1):
    """
    Run VNRecognizeTextRequest on a single image file and extract slate fields.

    Returns a full detection dict.
    """
    import Vision
    import Quartz

    try:
        image_url = Quartz.NSURL.fileURLWithPath_(image_path)

        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        # Language correction OFF — we want exact text (slates have codes like "A051", "INT 42")
        request.setUsesLanguageCorrection_(False)

        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(image_url, {})
        success, error = handler.performRequests_error_([request], None)

        if not success:
            log.warning(f"Vision OCR failed on {image_path}: {error}")
            result = _fallback(preprocessed=preprocessed, pass_num=pass_num)
            result["parse_error"] = True
            return result

    except Exception as e:
        log.warning(f"Vision OCR exception on {image_path}: {e}")
        result = _fallback(preprocessed=preprocessed, pass_num=pass_num)
        result["parse_error"] = True
        return result

    # Parse observations into normalised blob dicts
    blobs = []
    for obs in (request.results() or []):
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        best = candidates[0]
        text = best.string()
        if not text or not text.strip():
            continue
        score = float(best.confidence())

        # CoreGraphics bbox: bottom-left origin, normalised → flip to top-left
        cg = obs.boundingBox()
        x   = float(cg.origin.x)
        cg_y = float(cg.origin.y)
        w   = float(cg.size.width)
        h   = float(cg.size.height)
        y   = 1.0 - cg_y - h  # flip to top-left origin

        # Convert normalised coords to pixel-space for spatial proximity logic.
        # We use a nominal 1000x1000 space — only relative distances matter.
        x0 = x * 1000
        y0 = y * 1000
        x1 = (x + w) * 1000
        y1 = (y + h) * 1000

        blobs.append({
            "text":  text.strip(),
            "score": score,
            "x0": x0, "y0": y0,
            "x1": x1, "y1": y1,
            "cx": (x0 + x1) / 2,
            "cy": (y0 + y1) / 2,
        })

    if not blobs:
        return _fallback(preprocessed=preprocessed, pass_num=pass_num)

    # Classify blobs as labels or values
    label_blobs, value_map = _classify_blobs(blobs)

    # Slate detection heuristic: require enough recognised field labels
    n_keywords = len(label_blobs)
    avg_score  = sum(b["score"] for b in blobs) / len(blobs)
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

    # Build a raw per-field dict (no validation yet) with Cyrillic normalised
    raw_fields = {}
    raw_conf = {}
    for field in SLATE_FIELDS:
        entry = value_map.get(field)
        if entry:
            raw_fields[field] = {
                "value": entry["text"].translate(_CYRILLIC_TO_LATIN),
                "confidence": _score_to_conf(entry["score"]),
            }
        else:
            raw_fields[field] = {"value": None, "confidence": "low"}

    # Split combined "scene-slate" values BEFORE validation so that
    # values like "-30153" (no separator) can be split into scene=30, slate=153
    # rather than being rejected outright by the scene format check.
    _split_scene_slate(raw_fields)

    # Apply per-field format validation to the (possibly split) raw values
    fields = {}
    for field in SLATE_FIELDS:
        entry = raw_fields[field]
        value = entry.get("value")
        if value is not None:
            pattern = _FIELD_VALUE_RE.get(field)
            # For scene, strip leading OCR artefacts before the format check
            check_val = value.strip().lstrip("-|") if field == "scene" else value.strip()
            if pattern and not pattern.match(check_val):
                fields[field] = {"value": None, "confidence": "low"}
            else:
                fields[field] = entry
        else:
            fields[field] = {"value": None, "confidence": "low"}

    # Cross-field sanity: detect slate_number + take merge.
    # When OCR reads adjacent SLATE and TAKE boxes as one blob (e.g. "88" + "5"
    # → "885"), the merged value passes take format validation as a 3-digit take.
    # Heuristic: if take starts with a 2+ digit slate_number AND the remainder
    # is 1–2 digits, the merged value is almost certainly wrong — split it.
    _fix_take_slate_merge(fields)

    extraction_notes = (
        f"Apple Vision pass {pass_num}: {n_keywords} label(s) found, "
        f"avg score {avg_score:.2f}"
    )

    return {
        "slate_detected":     True,
        "overall_confidence": overall_conf,
        "partially_visible":  n_keywords < 5,
        "fields":             fields,
        "extraction_notes":   extraction_notes,
        "parse_error":        False,
        "model_used":         "apple_vision",
        "preprocessed":       preprocessed,
        "escalated":          False,
        "pass":               pass_num,
        "needs_review":       overall_conf != "high",
    }


# ---------------------------------------------------------------------------
# Blob classification: label → value mapping
# ---------------------------------------------------------------------------

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
    label_blobs  = []
    value_blobs  = []
    label_fields = {}  # remaining_blobs index → field name
    value_map    = {}

    # --- Pass 1: inline "Label: value" blobs ---
    remaining_blobs = []
    for blob in blobs:
        text_norm = blob["text"].strip()
        matched = False

        if ":" in text_norm:
            parts      = text_norm.split(":", 1)
            label_part = parts[0].strip()
            value_part = parts[1].strip()
            field      = _fuzzy_label_lookup(label_part)
            if field and value_part:
                label_blobs.append(blob)
                vblob = dict(blob)
                vblob["text"] = value_part
                if field not in value_map:
                    value_map[field] = vblob
                matched = True

        if not matched:
            remaining_blobs.append(blob)

    # --- Pass 2: classify remaining blobs as pure labels or pure values ---
    for i, blob in enumerate(remaining_blobs):
        field = _fuzzy_label_lookup(blob["text"])
        if field:
            label_blobs.append(blob)
            label_fields[i] = field
        else:
            value_blobs.append(blob)

    # --- Pass 3: spatial proximity for standalone label blobs ---
    for idx, field in label_fields.items():
        if field in value_map:
            continue

        label   = remaining_blobs[idx]
        best_blob = None
        best_dist = float("inf")
        label_h = label["y1"] - label["y0"]

        for vblob in value_blobs:
            to_right = vblob["x0"] >= label["x1"] - 5
            below    = (
                # Allow values whose top starts up to 1.5× the label's own
                # height above the label's bottom. This handles large combined
                # blobs (e.g. "-30153" spanning SCENE+SLATE rows) whose top
                # sits slightly above the label bottom while still rejecting
                # blobs from rows clearly above the label.
                vblob["y0"] >= label["y1"] - label_h * 1.5
                # Value must horizontally overlap or be right of the label —
                # prevents roll stickers to the LEFT of SCENE from winning
                and vblob["x1"] >= label["x0"]
                and abs(vblob["cx"] - label["cx"]) < label_h * 3
            )
            if not (to_right or below):
                continue

            # Deprioritise roll-sticker-pattern values (e.g. A013) for non-roll fields
            # by adding a large penalty. They still win if there's genuinely nothing else.
            is_roll_sticker = bool(_ROLL_STICKER_RE.match(vblob["text"].strip()))
            roll_penalty = 500 if (is_roll_sticker and field != "roll") else 0

            dy   = abs(vblob["cy"] - label["cy"])
            dx   = max(0.0, vblob["x0"] - label["x1"])
            dist = dx + dy * 2 + roll_penalty

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

def _fallback(preprocessed=False, escalated=False, pass_num=1):
    return {
        "slate_detected":     False,
        "overall_confidence": "low",
        "partially_visible":  False,
        "fields":             {f: {"value": None, "confidence": "low"} for f in SLATE_FIELDS},
        "extraction_notes":   None,
        "parse_error":        False,
        "model_used":         "apple_vision",
        "preprocessed":       preprocessed,
        "escalated":          escalated,
        "pass":               pass_num,
        "needs_review":       True,
    }


_SCENE_SLATE_SPLIT_RE = re.compile(
    r'^-?([A-Z]?\d+[A-Z]*)[\s\-|]+(\d+)$',
    re.IGNORECASE,
)

# Fallback: no separator — e.g. "-30153" or "120090".
# Scene is 1-3 digits with optional leading/trailing letters, slate is last 3 digits.
_SCENE_SLATE_NOSEP_RE = re.compile(
    r'^-?([A-Z]?\d{1,3}[A-Z]*)(\d{3})$',
    re.IGNORECASE,
)

def _split_scene_slate(fields):
    """
    Split a combined scene-slate value into separate fields when slate_number is absent.

    Handles separators: hyphen, space, pipe, leading minus (OCR artefact).
    Also handles no-separator concatenations: "-30153" → scene=30, slate=153.
    Supports multi-letter scene suffixes: "90AB-144" → scene=90AB, slate=144.
    Examples: "64-353", "120 90", "120|90", "-120 90", "-30153", "120090", "90AB-144".
    Modifies fields in place.
    """
    scene_field = fields.get("scene", {})
    slate_field = fields.get("slate_number", {})

    scene_val = scene_field.get("value") if isinstance(scene_field, dict) else None
    slate_val = slate_field.get("value") if isinstance(slate_field, dict) else None

    if not scene_val:
        return

    # Always attempt the split — the regex patterns are the gating condition.
    # A misread slate_val (e.g. "1" from a TAKE label) must not block splitting
    # a combined scene value like "30157". If neither regex matches, nothing changes.
    conf = scene_field.get("confidence", "medium")
    # Strip leading OCR artefacts ("-", "|") before trying to split
    scene_clean = scene_val.strip().lstrip("-|")
    m = _SCENE_SLATE_SPLIT_RE.match(scene_clean)
    if m:
        scene_part = m.group(1).strip()
        slate_part = m.group(2).strip()
        # Separator split can be fooled by "30157 1" → scene="30157", slate="1".
        # If the scene part is long enough to be a combined value (≥5 chars, e.g. "30157"),
        # try a nosep re-split. Skip for short artifacts like "1120" from "1120|91".
        m_nosep = _SCENE_SLATE_NOSEP_RE.match(scene_part) if len(scene_part) >= 5 else None
        if m_nosep:
            fields["scene"]        = {"value": m_nosep.group(1).strip(), "confidence": conf}
            fields["slate_number"] = {"value": m_nosep.group(2).strip(), "confidence": conf}
        else:
            fields["scene"]        = {"value": scene_part, "confidence": conf}
            fields["slate_number"] = {"value": slate_part, "confidence": conf}
        return

    # Fallback: no separator between digits (e.g. "-30153" → scene=30, slate=153)
    m2 = _SCENE_SLATE_NOSEP_RE.match(scene_clean)
    if m2:
        fields["scene"]        = {"value": m2.group(1).strip(), "confidence": conf}
        fields["slate_number"] = {"value": m2.group(2).strip(), "confidence": conf}


def _fix_take_slate_merge(fields):
    """
    Detect and fix slate_number + take box merge.

    Two variants of the same OCR artefact:

    Case 1 — same blob assigned to both fields (take == slate_number):
      The blob spans both boxes; spatial proximity assigns it to both.
      e.g. slate_number="885", take="885" → take=null (let VLM fill).

    Case 2 — slate and take boxes read as one blob (take starts with slate_number):
      The boundary between boxes is invisible; values concatenate.
      e.g. slate_number="88", take="885" → take="5" (low confidence).

    In both cases, pass 2 (preprocessed) usually reads the fields correctly;
    the fix ensures pass 2 wins the count comparison in detect_with_escalation.
    """
    slate_val = (fields.get("slate_number") or {}).get("value")
    take_val  = (fields.get("take") or {}).get("value")

    if not slate_val or not take_val:
        return
    if len(slate_val) < 2:
        return

    # Case 1: same value — same blob assigned to both fields
    if take_val == slate_val:
        log.debug(
            f"take/slate same-blob: take=slate_number={slate_val!r} — clearing take"
        )
        fields["take"] = {"value": None, "confidence": "low"}
        return

    # Case 2: take is slate_number with 1–2 extra digits appended
    if not take_val.startswith(slate_val):
        return
    remainder = take_val[len(slate_val):]
    if not remainder or not remainder.isdigit() or len(remainder) > 2:
        return

    log.debug(
        f"take/slate prefix merge: take={take_val!r}, slate_number={slate_val!r}"
        f" — correcting take to {remainder!r}"
    )
    fields["take"] = {"value": remainder, "confidence": "low"}


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
