"""
Configuration loader for RoboSlate.

Reads a key=value config file (config.env). No third-party dependencies.

Search order:
  1. Explicit path passed to load_config()
  2. config.env in the same directory as roboslate.py
  3. ~/.config/roboslate/config.env
"""

import os
import sys

# Defaults applied when keys are absent from config.env
DEFAULTS = {
    "CLAUDE_MODEL":            "claude-haiku-4-5-20251001",
    "ESCALATION_MODEL":        "claude-sonnet-4-6",
    "CLAUDE_MAX_TOKENS":       "512",
    "SCAN_PHASE1_DURATION":    "60",
    "SCAN_PHASE1_FPS":         "2",
    "SCAN_PHASE2_DURATION":    "30",
    "SCAN_PHASE2_FPS":         "2",
    "SCAN_PHASE3_INTERVAL":    "5",
    "MAX_FRAMES":              "16",
    "CONSISTENT_READINGS_STOP": "5",
    "EARLY_STOP_CONFIDENCE":   "high",
    "MAX_IMAGE_SIZE":          "1568",
    "FRAME_JPEG_QUALITY":      "3",
    "VISION_BACKEND":          "claude",
    "LOG_PATH":                "",
}
# Note: FIELD_* keys are NOT in DEFAULTS — they are opt-in via config.env.
# Absence of all FIELD_* keys → all fields enabled (backwards-compatible).
# See get_enabled_fields() below.


def _parse_env_file(path):
    """Parse a key=value file. Lines starting with # are comments. Returns dict."""
    result = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def load_config(explicit_path=None, project_dir=None):
    """
    Load configuration. Returns a dict with all known keys populated.

    Args:
        explicit_path: Path to config.env passed via --config CLI flag.
        project_dir: Directory of roboslate.py (used to find sibling config.env).

    Exits with a helpful message if ANTHROPIC_API_KEY is missing.
    """
    search_paths = []

    if explicit_path:
        search_paths.append(explicit_path)

    if project_dir:
        search_paths.append(os.path.join(project_dir, "config.env"))

    search_paths.append(
        os.path.expanduser("~/.config/roboslate/config.env")
    )

    cfg = dict(DEFAULTS)
    loaded_from = None

    for path in search_paths:
        if os.path.isfile(path):
            try:
                parsed = _parse_env_file(path)
                cfg.update(parsed)
                loaded_from = path
                break
            except Exception as e:
                print(f"WARNING: Could not read config file {path}: {e}", file=sys.stderr)

    # Also allow env var override for API key (common CI/CD pattern)
    if os.environ.get("ANTHROPIC_API_KEY"):
        cfg["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

    # API key is only required for the Claude backend
    if cfg.get("VISION_BACKEND", "claude") == "claude":
        if not cfg.get("ANTHROPIC_API_KEY") or cfg["ANTHROPIC_API_KEY"].startswith("sk-ant-..."):
            print(
                "\nERROR: ANTHROPIC_API_KEY is not set.\n"
                "\nAdd your key to config.env:\n"
                "  ANTHROPIC_API_KEY=sk-ant-your-key-here\n"
                "\nOr set it as an environment variable:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-your-key-here\n"
                "\nGet a key at: https://console.anthropic.com/\n",
                file=sys.stderr,
            )
            sys.exit(1)

    return cfg, loaded_from


def get_int(cfg, key):
    """Return cfg[key] as int. Raises ValueError with a clear message on failure."""
    try:
        return int(cfg[key])
    except (KeyError, ValueError) as e:
        raise ValueError(f"Config key {key!r} must be an integer, got {cfg.get(key)!r}") from e


def get_float(cfg, key):
    """Return cfg[key] as float."""
    try:
        return float(cfg[key])
    except (KeyError, ValueError) as e:
        raise ValueError(f"Config key {key!r} must be a number, got {cfg.get(key)!r}") from e


def get_enabled_fields(cfg):
    """
    Return the list of slate field names that are enabled in the config.

    Each field is controlled by a FIELD_<name>=on line in config.env.
    Commenting out the line disables that field.
    If no FIELD_* keys are present at all, all fields are enabled (backwards-compatible).

    Example config.env snippet:
        FIELD_scene=on
        FIELD_take=on
        FIELD_slate_number=on
        FIELD_roll=on
        FIELD_camera=on
        FIELD_director=on
        FIELD_dop=on
        #FIELD_production=on   ← commented out = disabled
        #FIELD_date=on
        FIELD_fps=on
        #FIELD_format=on
        FIELD_notes=on
    """
    from roboslate.merge import SLATE_FIELDS
    # If no FIELD_* keys present at all → enable everything (backwards-compatible)
    has_any = any(f"FIELD_{f}" in cfg for f in SLATE_FIELDS)
    if not has_any:
        return list(SLATE_FIELDS)
    return [f for f in SLATE_FIELDS if cfg.get(f"FIELD_{f}", "off").lower() == "on"]
