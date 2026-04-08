"""
Claude Vision API interface for slate detection and field extraction.

Each frame goes through an escalation pipeline:
  1. Haiku, full frame — fast, cheap
  2. If low confidence: Haiku on preprocessed frame
  3. If still low: Sonnet on preprocessed frame
  4. If still low: flagged for human review

Per-field confidence is a first-class concept in the JSON schema:
  {"scene": {"value": "57", "confidence": "high"}, "roll": {"value": "A033", "confidence": "low"}}
"""

import base64
import io
import json
import logging
import os
import re
import tempfile
import time

log = logging.getLogger(__name__)

from roboslate.merge import SLATE_FIELDS

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a specialized vision system for the film and television industry. "
    "Your job is to analyse video frames and detect clapperboards (also called slates). "
    "A clapperboard is a board with printed fields like Scene, Take, Roll, Camera, Director, "
    "and a clapper stick at the top. They appear at the start of takes, held in front of camera. "
    "You must respond only with valid JSON. No prose, no markdown, no explanation."
)

# Minimal detection-only prompt — used with a fast/cheap model to decide
# whether a frame is worth sending to the extraction model.
# Response must be tiny: {"slate": true} or {"slate": false}
DETECTION_SYSTEM_PROMPT = (
    "You detect clapperboards in video frames. "
    "Respond only with valid JSON: {\"slate\": true} or {\"slate\": false}. "
    "Nothing else."
)

DETECTION_USER_PROMPT = (
    "Does this video frame contain a clapperboard or slate? "
    "Respond with only: {\"slate\": true} or {\"slate\": false}"
)

# Per-field disambiguation notes — included in the prompt only when the field is active.
_FIELD_DISAMBIG = {
    "slate_number": (
        '    slate_number = the sequential board/slate number (e.g. "215", "42B") — '
        'labelled "Slate", "Slate #", or displayed as a large number on the board'
    ),
    "roll": (
        '    roll         = the camera magazine or roll identifier(s) — labelled "Roll", "Mag", '
        '"MAG", "Magazine", or "Cam Roll". Often printed on small adhesive stickers. '
        'Read each sticker very carefully character by character. If there are multiple cameras '
        'with separate stickers (e.g. one for camera A and one for camera B), list ALL of them '
        'comma-separated in a single string (e.g. "A051, B042"). Do not omit any.'
    ),
    "camera": (
        '    camera       = the camera unit letter or identifier (e.g. "A", "B", "CAM A") — '
        "often in a small coloured box in the corner of the slate, NOT a person's name"
    ),
    "dop": (
        "    dop          = the director of photography or camera operator — a person's name. "
        'On some slates the "CAMERA" label refers to this person, not a camera unit letter.'
    ),
}


def _build_user_prompt(fields=None):
    """
    Build the Sonnet extraction prompt for the given list of field names.
    fields=None → use all SLATE_FIELDS.
    """
    active = fields if fields is not None else SLATE_FIELDS

    # Build the fields JSON block
    field_lines = "\n".join(
        f'    "{f}": {{"value": null or string, "confidence": "high" or "medium" or "low"}},'
        for f in active
    )
    # Strip trailing comma from last line
    if field_lines.endswith(","):
        field_lines = field_lines[:-1]

    # Build disambiguation section — only for active fields that have notes
    disambig_lines = [v for k, v in _FIELD_DISAMBIG.items() if k in active]
    disambig_block = ""
    if disambig_lines:
        disambig_block = (
            "- IMPORTANT field definitions — these are different things, do not conflate them:\n"
            + "\n".join(disambig_lines)
            + "\n"
        )

    return f"""\
Analyse this video frame.

Is a clapperboard (slate) visible and legible in this image?
A slate may be: traditional wooden/acrylic board, digital slate, smart slate, or any
variation where production metadata fields are displayed.

Respond with exactly this JSON structure — no deviations:

{{
  "slate_detected": true or false,
  "overall_confidence": "high" or "medium" or "low",
  "partially_visible": true or false,
  "bbox": null or [x1, y1, x2, y2],
  "fields": {{
{field_lines}
  }},
  "extraction_notes": null or string
}}

bbox: if a slate is detected, provide the bounding box of the slate board as [x1, y1, x2, y2]
where each value is a 0.0–1.0 fraction of image width/height (top-left origin). Set to null if no slate.

Rules:
- If no slate is visible, set slate_detected to false, overall_confidence to "low", and all field values to null.
- If a slate is visible but a field is unreadable, set that field's value to null.
- Do NOT guess or infer. Only include what is clearly legible.
- Preserve exact text as written (e.g. "14A", "INT 42", "CAM B").
- Per-field confidence:
    "high"   — text is clearly legible, no ambiguity
    "medium" — mostly legible but 1-2 characters uncertain
    "low"    — barely legible, significant uncertainty
- Set value to null (not low confidence) if a field is truly absent from the slate.
{disambig_block}- extraction_notes: any relevant observation (e.g. "digital slate", "clapper closed", "heavy motion blur").
"""


def _fallback_result(fields=None):
    """Return a fallback result dict for the given field subset."""
    active = fields if fields is not None else SLATE_FIELDS
    return {
        "slate_detected": False,
        "overall_confidence": "low",
        "partially_visible": False,
        "fields": {f: {"value": None, "confidence": "low"} for f in active},
        "extraction_notes": None,
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def build_client(api_key):
    if anthropic is None:
        raise ImportError(
            "The 'anthropic' package is not installed.\n"
            "Run ./install.sh to set up the virtual environment."
        )
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def _encode_image(img_or_path, max_size):
    """
    Encode an image (PIL Image or file path) as base64 JPEG.
    Resizes to max_size on the long edge first.

    Returns base64 string.
    """
    if Image is None:
        raise ImportError("Pillow is not installed. Run ./install.sh.")

    if isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert("RGB")
    else:
        img = img_or_path.convert("RGB")

    w, h = img.size
    if max(w, h) > max_size:
        if w >= h:
            new_size = (max_size, int(h * max_size / w))
        else:
            new_size = (int(w * max_size / h), max_size)
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_response(text):
    """Parse Claude's JSON response, handling markdown fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    stripped = re.sub(r"\s*```$", "", stripped.strip(), flags=re.MULTILINE)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Single-frame detection
# ---------------------------------------------------------------------------

def detect_and_extract(img_or_path, client, model, max_tokens, max_image_size=1568, fields=None):
    """
    Send a single frame to Claude and return a detection result dict.

    Args:
        img_or_path: File path string or PIL Image.
        client: anthropic.Anthropic instance.
        model: Model ID string.
        max_tokens: Max response tokens.
        max_image_size: Resize long edge to this before sending.
        fields: List of field names to extract. None → all SLATE_FIELDS.

    Returns:
        Dict with keys: slate_detected, overall_confidence, partially_visible,
        fields (dict of field→{value,confidence}), extraction_notes, parse_error.
    """
    active_fields = fields if fields is not None else SLATE_FIELDS
    prompt = _build_user_prompt(active_fields)
    fallback = _fallback_result(active_fields)
    image_data = _encode_image(img_or_path, max_image_size)

    retries = 2
    delay = 1.0

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=int(max_tokens),
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            break
        except Exception as e:
            if anthropic is not None and isinstance(e, anthropic.RateLimitError) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise

    raw_text = response.content[0].text if response.content else ""
    parsed = _parse_response(raw_text)

    if parsed is None:
        result = dict(fallback)
        result["parse_error"] = True
        result["raw_response"] = raw_text
        return result

    result = dict(fallback)
    result.update(parsed)

    # Normalise fields: ensure all expected keys exist with proper structure
    if "fields" not in result or not isinstance(result["fields"], dict):
        result["fields"] = {f: {"value": None, "confidence": "low"} for f in active_fields}
    else:
        # Keep only active fields; normalise format
        normalised = {}
        for f in active_fields:
            fd = result["fields"].get(f)
            if fd is None:
                normalised[f] = {"value": None, "confidence": "low"}
            elif not isinstance(fd, dict):
                normalised[f] = {"value": fd, "confidence": "low"}
            else:
                normalised[f] = fd
        result["fields"] = normalised

    result["parse_error"] = False
    return result


# ---------------------------------------------------------------------------
# Overall confidence helper
# ---------------------------------------------------------------------------

def _overall_confidence(result):
    """Return the overall_confidence of a result, normalised."""
    oc = result.get("overall_confidence", "low")
    if oc not in ("high", "medium", "low"):
        # Derive from fields if missing
        values = [
            v.get("confidence", "low")
            for v in (result.get("fields") or {}).values()
            if isinstance(v, dict) and v.get("value") is not None
        ]
        if not values:
            return "low"
        if all(c == "high" for c in values):
            return "high"
        if any(c == "high" for c in values):
            return "medium"
        return "low"
    return oc


# ---------------------------------------------------------------------------
# Targeted roll/mag sticker extraction
# ---------------------------------------------------------------------------

ROLL_STICKER_PROMPT = """\
Look carefully at this film production slate image.

There may be one or more small adhesive stickers showing camera roll or magazine numbers.
Each sticker is typically:
- A small coloured sticker (green, yellow, or white) in the top-left corner of the slate
- Labelled "MAG", "Roll", or unlabelled
- Contains an alphanumeric code like "A051", "B042", etc. — a letter followed by 3 digits

This is a multi-camera production — there may be TWO stickers, one per camera (e.g. "A051" and "B042").
Look very carefully for ALL stickers. Count them.

Respond with exactly this JSON — nothing else:
{{"roll": "A051"}} for one camera, {{"roll": "A051, B042"}} for two cameras,
or {{"roll": null}} if you genuinely cannot find any sticker.
"""

ROLL_STICKER_PROMPT_TARGETED = """\
Look carefully at this film production slate image.

We are looking for camera {letter}'s MAG/roll sticker specifically.
There are multiple cameras on this production. Camera A's sticker may be more prominent —
ignore it. We need camera {letter}'s sticker.

The sticker for camera {letter} will:
- Start with the letter {letter} (e.g. "{letter}042", "{letter}015")
- Be a small coloured adhesive label, likely in the top-left area of the slate
- Contain a letter followed by 3 digits

What does camera {letter}'s roll/mag sticker say?

Respond with exactly this JSON — nothing else:
{{"roll": "{letter}042"}} or {{"roll": null}} if you genuinely cannot find it.
"""


def _extract_roll_sticker(image_path, client, model, max_image_size=1568, camera_letter=None):
    """
    Focused single-question pass to extract the roll/mag sticker value.

    Crops to the top half of the frame (where the sticker usually lives)
    to increase effective resolution on that area, then asks specifically
    about the sticker.

    Returns a dict {"value": str|None, "confidence": str} or None on failure.
    """
    if Image is None:
        return None

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            # Crop to top 55% of frame — sticker is always in the upper portion
            top_half = img.crop((0, 0, w, int(h * 0.55)))
            # Scale up to max_image_size for better resolution on the small sticker
            long_edge = max(top_half.size)
            if long_edge < max_image_size:
                scale = max_image_size / long_edge
                new_size = (int(top_half.width * scale), int(top_half.height * scale))
                top_half = top_half.resize(new_size, Image.LANCZOS)

            buf = io.BytesIO()
            top_half.convert("RGB").save(buf, format="JPEG", quality=90)
            image_data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        log.warning("roll sticker image prep failed for %s: %s", image_path, e, exc_info=True)
        return None

    try:
        response = client.messages.create(
            model=model,
            max_tokens=30,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data},
                        },
                        {"type": "text", "text": ROLL_STICKER_PROMPT_TARGETED.format(letter=camera_letter) if camera_letter else ROLL_STICKER_PROMPT},
                    ],
                }
            ],
        )
    except Exception as e:
        log.warning("roll sticker API call failed for %s: %s", image_path, e, exc_info=True)
        return None

    raw = response.content[0].text if response.content else ""
    parsed = _parse_response(raw)
    if not parsed or not isinstance(parsed, dict):
        return None

    value = parsed.get("roll")
    if value is None:
        return {"value": None, "confidence": "low"}
    return {"value": str(value).strip(), "confidence": "medium"}


# ---------------------------------------------------------------------------
# Detection call (fast) — returns slate presence + optional bounding box
# ---------------------------------------------------------------------------

def detect_slate(image_path, client, model, max_image_size=1568):
    """
    Ask the model one binary question: is there a slate in this frame?
    Fast and cheap — max_tokens=20, expects only {"slate": true/false}.

    Returns:
        (slate_present: bool, bbox: None)
        bbox is always None here; bbox comes from the Sonnet extraction pass.
    """
    image_data = _encode_image(image_path, max_image_size)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=20,
            system=DETECTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data},
                        },
                        {"type": "text", "text": DETECTION_USER_PROMPT},
                    ],
                }
            ],
        )
    except Exception as e:
        log.warning("detect_slate API call failed for %s: %s", image_path, e, exc_info=True)
        return False, None

    raw = response.content[0].text if response.content else ""
    parsed = _parse_response(raw)
    if parsed and isinstance(parsed, dict):
        return bool(parsed.get("slate", False)), None
    return ("true" in raw.lower()), None


def _parse_bbox(raw_bbox):
    """
    Validate and normalise a raw bbox value from the model.
    Returns (x1, y1, x2, y2) tuple with values clamped to [0, 1], or None.
    """
    if not raw_bbox or not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in raw_bbox]
    except (TypeError, ValueError):
        return None
    # Ensure correct ordering before size check
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    # Reject degenerate boxes (too small to be useful — less than 5% in either dimension)
    if (x2 - x1) < 0.05 or (y2 - y1) < 0.05:
        return None
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Slate crop helper
# ---------------------------------------------------------------------------

def _crop_to_bbox(image_path, bbox, padding=0.03):
    """
    Crop an image to a bounding box and write to a temp JPEG.

    Args:
        image_path: Source image path.
        bbox: (x1, y1, x2, y2) normalised fractions, or None.
        padding: Extra margin to add around the box (fraction of image size).

    Returns:
        Path to cropped temp file, or image_path if crop is skipped.
        Caller is responsible for deleting the temp file.
    """
    if bbox is None or Image is None:
        return image_path

    try:
        img = Image.open(image_path)
        w, h = img.size
        x1, y1, x2, y2 = bbox

        # Apply padding
        px = padding * w
        py = padding * h
        left   = max(0, int(x1 * w - px))
        top    = max(0, int(y1 * h - py))
        right  = min(w, int(x2 * w + px))
        bottom = min(h, int(y2 * h + py))

        # Skip if crop is basically the whole image
        if (right - left) >= w * 0.9 and (bottom - top) >= h * 0.9:
            return image_path

        cropped = img.crop((left, top, right, bottom))

        tmp = tempfile.NamedTemporaryFile(suffix="_crop.jpg", delete=False)
        tmp.close()
        cropped.save(tmp.name, "JPEG", quality=92)
        return tmp.name

    except Exception as e:
        log.warning("crop_to_bbox failed for %s: %s", image_path, e, exc_info=True)
        return image_path


# ---------------------------------------------------------------------------
# Escalation pipeline helpers
# ---------------------------------------------------------------------------

def _apply_roll_correction(result, image_path, client, extraction_model, max_size, camera_letter, reel_hint):
    """
    Run a targeted roll-sticker pass on image_path and update result in place.
    Falls back to reel_hint (filename) if the sticker pass returns nothing.
    """
    roll_field = (result.get("fields") or {}).get("roll", {})
    roll_value = roll_field.get("value") if isinstance(roll_field, dict) else roll_field
    roll_wrong = camera_letter and roll_value and not roll_value.upper().startswith(camera_letter)

    if roll_value is None or roll_wrong:
        sticker = _extract_roll_sticker(image_path, client, extraction_model, max_size, camera_letter)
        if sticker and sticker.get("value"):
            result.setdefault("fields", {})["roll"] = sticker
        elif reel_hint and (roll_wrong or roll_value is None):
            result.setdefault("fields", {})["roll"] = {
                "value": reel_hint.upper(),
                "confidence": "medium",
                "source": "filename",
            }
        result["_roll_sticker_pass"] = True


# ---------------------------------------------------------------------------
# Escalation pipeline
# ---------------------------------------------------------------------------

def detect_with_escalation(image_path, client, cfg, reel_hint=None):
    """
    Full escalation pipeline for a single frame.

    Pass 1: Haiku, raw frame
    Pass 2: Haiku, preprocessed frame  (if pass 1 is not high confidence)
    Pass 3: Sonnet, preprocessed frame (if pass 2 is not high confidence)

    Returns:
        result dict augmented with 'model_used', 'preprocessed', 'escalated', 'needs_review'
    """
    from roboslate.preprocessing import preprocess_frame_file

    haiku = cfg.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    sonnet = cfg.get("ESCALATION_MODEL", "claude-sonnet-4-6")
    max_tokens = cfg.get("CLAUDE_MAX_TOKENS", "512")
    max_size = int(cfg.get("MAX_IMAGE_SIZE", 1568))
    early_stop = cfg.get("EARLY_STOP_CONFIDENCE", "high")
    camera_letter = reel_hint[0].upper() if reel_hint else None
    fields = cfg.get("_enabled_fields")  # None → all fields (backwards-compatible)

    preprocessed_path = None
    crop_path = None

    try:
        # --- Pass 1: Haiku detection ---
        # Binary "is there a slate?" — fast and cheap (max_tokens=20).
        # Non-slate frames exit here without an expensive Sonnet call.
        slate_present, _ = detect_slate(image_path, client, haiku, max_size)

        if not slate_present:
            result = _fallback_result(fields)
            result.update({
                "model_used": haiku,
                "preprocessed": False,
                "escalated": False,
                "pass": 1,
                "needs_review": False,
                "parse_error": False,
            })
            return result

        # --- Pass 2: Sonnet full extraction (full frame) ---
        # Sonnet also returns a bbox for the slate so we can crop for later passes.
        extraction_model = sonnet if (sonnet and sonnet != haiku) else haiku
        result = detect_and_extract(image_path, client, extraction_model, max_tokens, max_size, fields=fields)
        result["model_used"] = extraction_model
        result["preprocessed"] = False
        result["escalated"] = extraction_model != haiku
        result["pass"] = 2

        # --- Crop to slate region for subsequent passes ---
        # Use the bbox Sonnet returned alongside the field values.
        # Tighter crop → text is larger relative to image → better extraction accuracy
        # for the roll sticker pass and any preprocessing retry.
        sonnet_bbox = _parse_bbox(result.get("bbox"))
        crop_path = _crop_to_bbox(image_path, sonnet_bbox)
        extract_src = crop_path  # roll sticker + pass 3 use the crop

        # Record the bbox we ended up using (for debugging)
        result["bbox"] = sonnet_bbox

        # --- Roll sticker targeted pass (only when roll field is active) ---
        roll_active = fields is None or "roll" in (fields or [])
        if roll_active:
            _apply_roll_correction(result, extract_src, client, extraction_model, max_size, camera_letter, reel_hint)

        oc = _overall_confidence(result)
        if oc in ("high", "medium"):
            result["needs_review"] = False
            return result

        # --- Pass 3: Sonnet on preprocessed crop (low confidence only) ---
        preprocessed_path = preprocess_frame_file(extract_src)
        result2 = detect_and_extract(preprocessed_path, client, extraction_model, max_tokens, max_size, fields=fields)
        result2["model_used"] = extraction_model
        result2["preprocessed"] = True
        result2["escalated"] = True
        result2["pass"] = 3
        result2["bbox"] = sonnet_bbox

        if roll_active:
            _apply_roll_correction(result2, preprocessed_path, client, extraction_model, max_size, camera_letter, reel_hint)

        oc2 = _overall_confidence(result2)
        result2["needs_review"] = oc2 not in ("high", "medium")
        return result2

    finally:
        # Clean up temp files (crop and preprocessed)
        for tmp in (preprocessed_path, crop_path):
            if tmp and tmp != image_path and os.path.isfile(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Multi-frame scan
# ---------------------------------------------------------------------------

def scan_frames(frame_entries, client, cfg, reel_hint=None):
    """
    Scan a list of frames with the full escalation pipeline.

    Args:
        frame_entries: List of (phase, timestamp, file_path) tuples.
        client: anthropic.Anthropic instance.
        cfg: Config dict.

    Returns:
        List of result dicts, each with 'phase', 'timestamp', 'frame_file' added.
    """
    early_stop = cfg.get("EARLY_STOP_CONFIDENCE", "high")
    max_frames = int(cfg.get("MAX_FRAMES", 40))

    results = []
    sent = 0
    api_calls = 0
    # Track consecutive slate readings to stop once we have enough
    slate_readings = 0
    CONSISTENT_READINGS_STOP = int(cfg.get("CONSISTENT_READINGS_STOP", 5))

    for phase, timestamp, frame_path in frame_entries:
        if sent >= max_frames:
            break
        if not os.path.isfile(frame_path):
            continue

        result = detect_with_escalation(frame_path, client, cfg, reel_hint=reel_hint)
        result["phase"] = phase
        result["timestamp"] = timestamp
        result["frame_file"] = os.path.basename(frame_path)
        results.append(result)
        sent += 1

        # Count actual API calls (detection + extraction if slate found)
        api_calls += 1  # Haiku detection
        if result.get("slate_detected"):
            api_calls += 1  # Sonnet extraction
            if result.get("pass", 1) >= 3:
                api_calls += 1  # preprocessing retry
            slate_readings += 1
        else:
            slate_readings = 0  # reset if we see non-slate frame

        # Early exit: confidence at or above the configured threshold
        if result.get("slate_detected") and _overall_confidence(result) == early_stop:
            break

        # Early exit: enough consistent slate readings (even if low confidence)
        if slate_readings >= CONSISTENT_READINGS_STOP:
            break

    # Store actual API call count on the last result for reporting
    if results:
        results[-1]["_total_api_calls"] = api_calls

    return results


# ---------------------------------------------------------------------------
# Cost estimation (dry run)
# ---------------------------------------------------------------------------

# Approximate token counts — actual varies with image size and content
_APPROX_IMAGE_TOKENS = 1600      # ~1568px JPEG
_APPROX_PROMPT_TOKENS = 400
_APPROX_RESPONSE_TOKENS = 300

# Prices per million tokens (as of 2026-04; update as needed)
_PRICING = {
    "haiku":  {"input": 0.25,  "output": 1.25},
    "sonnet": {"input": 3.00,  "output": 15.00},
    "opus":   {"input": 15.00, "output": 75.00},
}


def _model_tier(model_id):
    model_id = model_id.lower()
    if "haiku" in model_id:
        return "haiku"
    if "opus" in model_id:
        return "opus"
    return "sonnet"


def estimate_cost(frame_count, cfg, assume_escalation_rate=0.05):
    """
    Estimate API cost for processing a set of frames.

    Args:
        frame_count: Number of frames to scan.
        cfg: Config dict.
        assume_escalation_rate: Fraction of frames expected to escalate to Sonnet (0–1).

    Returns:
        Dict with haiku_calls, sonnet_calls, estimated_cost_usd (string).
    """
    # Detection-only Haiku calls for all frames; Sonnet extraction for detected slates
    haiku_calls = frame_count
    sonnet_calls = int(frame_count * assume_escalation_rate)

    haiku_tier = _model_tier(cfg.get("CLAUDE_MODEL", "haiku"))
    sonnet_tier = _model_tier(cfg.get("ESCALATION_MODEL", "claude-sonnet-4-6"))

    def call_cost(tier, n):
        p = _PRICING.get(tier, _PRICING["sonnet"])
        input_cost = (n * (_APPROX_IMAGE_TOKENS + _APPROX_PROMPT_TOKENS) / 1_000_000) * p["input"]
        output_cost = (n * _APPROX_RESPONSE_TOKENS / 1_000_000) * p["output"]
        return input_cost + output_cost

    total = call_cost(haiku_tier, haiku_calls) + call_cost(sonnet_tier, sonnet_calls)

    return {
        "frame_count": frame_count,
        "haiku_calls": haiku_calls,
        "sonnet_calls_estimated": sonnet_calls,
        "estimated_cost_usd": f"${total:.4f}",
        "note": f"Estimate assumes ~{int(assume_escalation_rate*100)}% escalation rate. Actual cost may vary.",
    }
