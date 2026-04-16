"""
MLX VLM backend for RoboSlate-arm — Pass 3 escalation.

Uses a local vision-language model (default: gemma-4-26b-a4b-it-4bit) to extract
slate fields when Apple Vision OCR (passes 1+2) returns low confidence.

Runs entirely on-device via Apple Silicon Neural Engine / GPU. No API key
or network access required after the one-time model download.

Enable in config.env:
    ENABLE_VLM_ESCALATION=true
    VLM_MODEL=mlx-community/gemma-4-26b-a4b-it-4bit
    VLM_MAX_TOKENS=300
"""

import json
import logging
import os
import re
import sys

from roboslate_arm.merge import SLATE_FIELDS

log = logging.getLogger(__name__)

# Module-level model cache — loaded once per process, reused across frames.
_MODEL     = None
_PROCESSOR = None
_CONFIG    = None
_LOADED_PATH = None  # track which model is loaded


# ---------------------------------------------------------------------------
# Model download helpers
# ---------------------------------------------------------------------------

def _is_hf_repo_id(path: str) -> bool:
    """Return True if path looks like a HuggingFace repo ID (owner/name) vs a local path."""
    return (
        not os.path.isabs(path)
        and not path.startswith(("./", "~/", ".\\"))
        and "/" in path
        and not os.path.exists(path)
    )


def _local_model_path(repo_id: str, models_dir: str) -> str:
    """Return the expected local directory for a downloaded HF model."""
    return os.path.join(models_dir, repo_id.split("/")[-1])


def _ensure_model_downloaded(repo_id: str, models_dir: str) -> str:
    """
    Ensure a HF model is available locally in models_dir/<name>/.

    Interactive context: prompts the user for confirmation before downloading.
    Non-interactive context (e.g. SCRATCH pipe): raises RuntimeError immediately.

    Returns the local model path.
    """
    from huggingface_hub import snapshot_download

    dest = _local_model_path(repo_id, models_dir)
    if os.path.isdir(dest) and any(True for _ in os.scandir(dest)):
        return dest  # already present

    model_name = repo_id.split("/")[-1]

    if sys.stdin.isatty() and sys.stdout.isatty():
        print(f"\nVLM model not found locally.")
        print(f"  Repo:        {repo_id}")
        print(f"  Destination: {dest}")
        ans = input("Download now? [y/N] ").strip().lower()
        if ans != "y":
            raise RuntimeError(
                "VLM model download declined.\n"
                "Set ENABLE_VLM_ESCALATION=false in config.env to suppress this prompt."
            )
        os.makedirs(dest, exist_ok=True)
        print(f"Downloading {model_name} …")
        snapshot_download(repo_id=repo_id, local_dir=dest)
        print(f"Done. Tip: update config.env to skip future checks:")
        print(f"  VLM_MODEL={dest}")
        return dest
    else:
        raise RuntimeError(
            f"VLM model not available: {repo_id}\n"
            f"Run interactively once to download it, or set a local path in config.env:\n"
            f"  VLM_MODEL=/path/to/local/model"
        )


def _load_vlm_model(model_path: str, cfg: dict):
    """
    Load (or return cached) mlx-vlm model + processor.

    If model_path is a HuggingFace repo ID, the model is downloaded to
    mlx/models/<name>/ inside the project directory on first use (interactive
    prompt). Subsequent calls use the cached model objects.
    """
    global _MODEL, _PROCESSOR, _CONFIG, _LOADED_PATH

    # Resolve HF repo ID → local path (download if needed)
    if _is_hf_repo_id(model_path):
        models_dir = cfg.get("MLX_MODELS_DIR") or os.path.join(
            cfg.get("_PROJECT_DIR", "."), "mlx", "models"
        )
        model_path = _ensure_model_downloaded(model_path, models_dir)

    if _MODEL is not None and _LOADED_PATH == model_path:
        return _MODEL, _PROCESSOR, _CONFIG

    try:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
    except ImportError:
        raise ImportError(
            "mlx-vlm is not installed.\n"
            "Run: venv/bin/pip install mlx-vlm"
        )

    log.info(f"Loading VLM model: {model_path}  (first use — may take a moment)")
    _MODEL, _PROCESSOR = load(model_path)
    _CONFIG = load_config(model_path)
    _LOADED_PATH = model_path
    log.info("VLM model loaded and cached")
    return _MODEL, _PROCESSOR, _CONFIG


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_USER_PROMPT = (
    "Read this film clapperboard. The slate has a top row with three boxes:\n"
    "  SCENE (short code: \"30\",\"44B\",\"120\") | SLATE# (3-digit number: \"153\",\"088\",\"86\") | TAKE (small number: \"1\",\"5\")\n"
    "Below are rows for: DIRECTOR (person name), DOP (person name), PRODUCTION (title), DATE.\n"
    "A camera roll sticker (letter+3digits: \"A022\",\"A013\") may appear in a corner.\n"
    "CAMERA is a single letter (\"A\",\"B\").\n\n"
    "Return ONLY valid JSON — no comments, no explanation, nothing else:\n"
    "{\"scene\":null,\"take\":null,\"slate_number\":null,\"roll\":null,\"camera\":null,"
    "\"director\":null,\"dop\":null,\"production\":null,\"date\":null,"
    "\"fps\":null,\"format\":null,\"notes\":null}\n"
    "Use null for anything not clearly legible."
)


# ---------------------------------------------------------------------------
# Per-field format validation (mirrors vision_apple._FIELD_VALUE_RE)
# ---------------------------------------------------------------------------

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

# Cyrillic → Latin normalisation (same table as vision_apple and merge)
_CYRILLIC_TO_LATIN = str.maketrans({
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H',
    'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'R', 'Т': 'T',
    'Х': 'X', 'а': 'a', 'е': 'e', 'о': 'o', 'р': 'r',
    'с': 'c', 'х': 'x', 'у': 'y',
})


def _parse_vlm_response(text: str) -> dict:
    """
    Extract and validate field values from VLM JSON response.

    Returns a dict of {field: {"value": ..., "confidence": "medium"}} for all
    SLATE_FIELDS.  Fields the VLM could not read are set to
    {"value": None, "confidence": "low"}.
    """
    fields = {f: {"value": None, "confidence": "low"} for f in SLATE_FIELDS}

    # Pull the first {...} block out of the response (VLM may add preamble)
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not m:
        log.debug(f"VLM response had no JSON object: {text!r}")
        return fields

    raw_json = m.group()

    # Strip inline comments that some models add (e.g. "key": value,  # comment)
    raw_json = re.sub(r'#[^\n"]*(?=\n|$)', '', raw_json)
    # Remove trailing commas before } or ] (common model error)
    raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json)

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        log.debug(f"VLM JSON parse error: {e}  raw={raw_json!r}")
        return fields

    for field in SLATE_FIELDS:
        raw = data.get(field)
        if raw is None or not isinstance(raw, str) or not raw.strip():
            continue

        # Normalise Cyrillic lookalikes
        value = raw.strip().translate(_CYRILLIC_TO_LATIN)

        # Apply format validation for structured fields
        check_val = value.lstrip("-|") if field == "scene" else value
        pattern = _FIELD_VALUE_RE.get(field)
        if pattern and not pattern.match(check_val):
            log.debug(f"VLM: {field}={value!r} failed format check, discarded")
            continue

        fields[field] = {"value": value, "confidence": "medium"}

    return fields


def _fallback(pass_num: int = 3) -> dict:
    return {
        "slate_detected":     False,
        "overall_confidence": "low",
        "partially_visible":  False,
        "fields":             {f: {"value": None, "confidence": "low"} for f in SLATE_FIELDS},
        "extraction_notes":   f"VLM pass {pass_num}: no slate detected",
        "parse_error":        False,
        "model_used":         "mlx_vlm",
        "preprocessed":       False,
        "escalated":          False,
        "pass":               pass_num,
        "needs_review":       True,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_vlm_pass(image_path: str, cfg: dict, pass_num: int = 3) -> dict:
    """
    Run a VLM inference on a single frame and return a detection dict.

    The returned dict uses the same schema as vision_apple._run_ocr() so it
    can be dropped into the existing escalation / merge pipeline unchanged.

    Confidence is capped at "medium" — VLM reads are best-effort fallbacks;
    only multi-frame merge agreement can promote fields to "high".

    Args:
        image_path: Path to a JPEG frame file.
        cfg: Config dict (keys: VLM_MODEL, VLM_MAX_TOKENS).
        pass_num: Pass label for logging (default 3).

    Returns:
        Detection dict.
    """
    model_path = cfg.get("VLM_MODEL", "mlx-community/gemma-4-26b-a4b-it-4bit")
    max_tokens  = int(cfg.get("VLM_MAX_TOKENS", 300))

    try:
        model, processor, config = _load_vlm_model(model_path, cfg)
    except ImportError as e:
        log.error(str(e))
        return _fallback(pass_num)

    try:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted_prompt = apply_chat_template(
            processor,
            config,
            _USER_PROMPT,
            num_images=1,
        )

        result = generate(
            model,
            processor,
            prompt=formatted_prompt,
            image=image_path,
            max_tokens=max_tokens,
            temperature=0.0,
            repetition_penalty=1.2,
            verbose=False,
        )
        response_text = result.text if hasattr(result, "text") else str(result)

    except Exception as e:
        log.warning(f"VLM inference failed on {image_path}: {e}")
        result_dict = _fallback(pass_num)
        result_dict["parse_error"] = True
        return result_dict

    log.debug(f"VLM raw response: {response_text!r}")

    fields = _parse_vlm_response(response_text)

    # Count non-null fields to decide slate_detected
    n_extracted = sum(1 for v in fields.values() if v.get("value") is not None)
    slate_detected = n_extracted >= 2  # need at least 2 fields to count as a detection

    if slate_detected:
        overall_conf = "medium"
    else:
        overall_conf = "low"

    return {
        "slate_detected":     slate_detected,
        "overall_confidence": overall_conf,
        "partially_visible":  n_extracted < 5,
        "fields":             fields,
        "extraction_notes":   f"VLM pass {pass_num}: {n_extracted} field(s) extracted",
        "parse_error":        False,
        "model_used":         "mlx_vlm",
        "preprocessed":       False,
        "escalated":          False,
        "pass":               pass_num,
        "needs_review":       True,  # VLM reads always flagged for review
    }
