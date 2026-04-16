"""
RoboSlate-arm — DaVinci Resolve integration script.

Place this file in Resolve's Scripts folder and run via:
  Workspace → Scripts → Utility → RoboSlate-arm

On macOS the Scripts folder is:
  ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/

Clip selection — the script checks in this order:
  1. Selected clips on the current timeline (Resolve 18+ only)
  2. All clips on the current timeline        (if PROCESS_ALL_TIMELINE_CLIPS = True)
  3. All clips in current Media Pool folder   (if PROCESS_ALL_MEDIA_POOL_FOLDER = True)

Sidecar caching: if a .roboslate-arm.json sidecar already exists next to the source
file and FORCE_REPROCESS is False, the existing result is used without re-running
OCR. Set FORCE_REPROCESS = True to always re-scan.

Output goes to:
  - Resolve's console (Workspace → Console)
  - {ROBOSLATE_ARM_DIR}/logs/resolve.log  (always written)
"""

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ── Edit these settings to match your installation ─────────────────────────
ROBOSLATE_ARM_DIR = "/path/to/roboslate2-proto"

# Set to True to process ALL clips on the current timeline (recommended).
# Control which clips to process by cueing a timeline before running.
PROCESS_ALL_TIMELINE_CLIPS = True

# Set to True to process ALL video clips in the current Media Pool folder.
# (Resolve's API does not expose which Media Pool items are selected,
# so this always processes the entire current folder.)
PROCESS_ALL_MEDIA_POOL_FOLDER = False

# Number of clips to analyse in parallel. Each worker is a subprocess call;
# Resolve API writes happen on the main thread afterwards.
#
# Default is 1 (safe). Set RESOLVE_MAX_WORKERS in config.env to increase.
# NOTE: When ENABLE_VLM_ESCALATION=true this is always forced to 1 regardless
# of this setting. Each subprocess loads the full MLX model (~13 GB for the
# default 26B 4-bit model); running multiple in parallel fills RAM instantly.
MAX_WORKERS = 1

# Set to True to re-run OCR even if a .roboslate-arm.json sidecar already exists.
# False (default) reuses existing sidecars — much faster on large batches.
FORCE_REPROCESS = False
# ───────────────────────────────────────────────────────────────────────────

VENV_PYTHON    = os.path.join(ROBOSLATE_ARM_DIR, "venv", "bin", "python")
ROBOSLATE2_PY  = os.path.join(ROBOSLATE_ARM_DIR, "roboslate-arm.py")
LOG_FILE       = os.path.join(ROBOSLATE_ARM_DIR, "logs", "resolve.log")

# Sentinel file written by RoboSlate-arm_Kill.py to cancel a running batch.
# Checked before each clip is dispatched — deleted at the start of each run.
_CANCEL_FILE   = "/tmp/roboslate-arm_cancel"


def _read_config_env():
    """Read config.env from ROBOSLATE_ARM_DIR. Returns a dict of raw key→value strings."""
    for path in (
        os.path.join(ROBOSLATE_ARM_DIR, "config.env"),
        os.path.expanduser("~/.config/roboslate-arm/config.env"),
    ):
        if not os.path.isfile(path):
            continue
        cfg = {}
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    cfg[k.strip()] = v.strip()
        except Exception:
            pass
        return cfg
    return {}


def _effective_workers():
    """
    Return the number of parallel workers to use.

    RESOLVE_MAX_WORKERS in config.env overrides the script-level MAX_WORKERS.
    Always forced to 1 when ENABLE_VLM_ESCALATION=true: each subprocess
    independently loads the full MLX model into memory, so multiple concurrent
    workers exhaust RAM and hard-crash the machine.
    """
    cfg = _read_config_env()
    if cfg.get("ENABLE_VLM_ESCALATION", "false").lower() == "true":
        return 1
    val = cfg.get("RESOLVE_MAX_WORKERS")
    if val is not None:
        try:
            return max(1, int(val))
        except ValueError:
            pass
    return MAX_WORKERS


# Resolve metadata keys written per clip.
# scene  → "Scene"  (Resolve built-in)
# slate# → "Shot"   (Resolve built-in)
# take   → "Take"   (Resolve built-in)
# All three are handled in _build_metadata().


# ---------------------------------------------------------------------------
# Logging — writes to both console and log file
# ---------------------------------------------------------------------------

def _log(msg):
    if threading.current_thread() is threading.main_thread():
        print(msg)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_install():
    """Verify ROBOSLATE_ARM_DIR and venv are set up correctly."""
    if ROBOSLATE_ARM_DIR == "/path/to/roboslate2-proto":
        _log("ERROR: ROBOSLATE_ARM_DIR is not set.")
        _log("  Edit resolve/RoboSlate-arm.py and set ROBOSLATE_ARM_DIR to the full path")
        _log("  of your RoboSlate-arm installation directory.")
        return False
    if not os.path.isfile(VENV_PYTHON):
        _log(f"ERROR: venv Python not found at: {VENV_PYTHON}")
        _log("  Run setup.sh first to create the virtual environment.")
        return False
    if not os.path.isfile(ROBOSLATE2_PY):
        _log(f"ERROR: roboslate-arm.py not found at: {ROBOSLATE2_PY}")
        return False
    return True


def _get_media_pool_clips(project):
    """Return (clip_name, file_path, MediaPoolItem) tuples for video clips
    in the current Media Pool folder."""
    try:
        media_pool = project.GetMediaPool()
        if not media_pool:
            return []
        folder = media_pool.GetCurrentFolder()
        if not folder:
            return []
        all_clips = folder.GetClipList() or []
    except Exception as e:
        _log(f"  (Media Pool unavailable: {e})")
        return []

    result = []
    for clip in all_clips:
        try:
            props = clip.GetClipProperty() or {}
            if "Video" not in props.get("Type", ""):
                continue
            file_path = props.get("File Path", "")
            clip_name = props.get("Clip Name", "") or os.path.basename(file_path)
            if file_path:
                result.append((clip_name, file_path, clip))
        except Exception:
            continue
    return result


def _get_selected_timeline_clips(project):
    """Return (clips, reason) for clips selected on the current timeline.

    clips  — list of (clip_name, file_path, MediaPoolItem); empty if none found.
    reason — None on success, or a short string explaining why clips is empty.
             "none selected" means the user ran with nothing selected (intentional).
             Any other reason means the selection was attempted but failed.
    Requires Resolve 18+.
    """
    try:
        timeline = project.GetCurrentTimeline()
        if not timeline:
            return [], "no active timeline"
        selected = timeline.GetSelectedClips()
        if not selected:
            return [], "none selected"
        result = []
        for item in selected:
            mpi = item.GetMediaPoolItem()
            if not mpi:
                continue
            props = mpi.GetClipProperty() or {}
            file_path = props.get("File Path", "")
            clip_name = props.get("Clip Name", "") or os.path.basename(file_path)
            if file_path:
                result.append((clip_name, file_path, mpi))
        if not result:
            return [], "selected items had no resolvable media"
        return result, None
    except (AttributeError, TypeError):
        # AttributeError: method doesn't exist on this Resolve version.
        # TypeError ('NoneType' object is not callable): method exists in the
        # API bindings but is a null stub — same effective result.
        return [], "none selected"
    except Exception as e:
        return [], f"selection API error: {e}"


def _get_all_timeline_clips(project):
    """Return (clip_name, file_path, MediaPoolItem) tuples for all clips
    across all video tracks on the current timeline.

    Deduplicates by file path — the same Media Pool item can appear on
    multiple tracks or multiple times on the same track.
    """
    try:
        timeline = project.GetCurrentTimeline()
        if not timeline:
            return []
        seen = set()
        result = []
        track_count = timeline.GetTrackCount("video")
        for track_idx in range(1, track_count + 1):
            items = timeline.GetItemListInTrack("video", track_idx) or []
            for item in items:
                mpi = item.GetMediaPoolItem()
                if not mpi:
                    continue
                props = mpi.GetClipProperty() or {}
                file_path = props.get("File Path", "")
                if not file_path or file_path in seen:
                    continue
                seen.add(file_path)
                clip_name = props.get("Clip Name", "") or os.path.basename(file_path)
                result.append((clip_name, file_path, mpi))
        return result
    except Exception as e:
        _log(f"  (Timeline error: {e})")
        return []


def _sidecar_path(file_path):
    base, _ = os.path.splitext(file_path)
    return base + ".roboslate-arm.json"


def _read_sidecar(file_path):
    """Return parsed sidecar JSON if it exists and is readable, else None."""
    sidecar = _sidecar_path(file_path)
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar) as f:
            return json.load(f)
    except Exception:
        return None


def _run_roboslate(file_path):
    """
    Run RoboSlate-arm on a single file and return the parsed result dict,
    or None on failure.

    If a sidecar already exists and FORCE_REPROCESS is False, the sidecar
    is read directly without launching a subprocess.
    """
    if not FORCE_REPROCESS:
        cached = _read_sidecar(file_path)
        if cached is not None:
            return cached, True  # (data, from_cache)

    cmd = [VENV_PYTHON, ROBOSLATE2_PY, "--file", file_path, "--stdout", "--quiet"]
    if FORCE_REPROCESS:
        cmd.append("--force")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        _log("    ERROR: timed out (>10 min)")
        return None, False
    except Exception as e:
        _log(f"    ERROR: could not launch RoboSlate-arm: {e}")
        return None, False

    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        msg = lines[-1] if lines else "unknown error"
        _log(f"    ERROR: {msg}")
        return None, False

    try:
        return json.loads(proc.stdout), False
    except json.JSONDecodeError:
        _log("    ERROR: could not parse RoboSlate-arm output")
        if proc.stdout:
            _log(f"    Raw output: {proc.stdout[:200]}")
        return None, False


def _build_metadata(data):
    """
    Build a Resolve metadata dict from a RoboSlate-arm result dict.
    Returns an empty dict if no usable fields were found.
    """
    slate = data.get("slate") or {}
    metadata = {}

    scene_val = (slate.get("scene") or {}).get("value")
    slate_num = (slate.get("slate_number") or {}).get("value")
    take_val  = (slate.get("take") or {}).get("value")

    if scene_val:
        metadata["Scene"] = str(scene_val)
    if slate_num:
        metadata["Shot"] = str(slate_num)
    if take_val:
        metadata["Take"] = str(take_val)

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _log("")
    _log("=" * 56)
    _log("  RoboSlate-arm — DaVinci Resolve")
    _log("=" * 56)

    if not _check_install():
        return

    try:
        resolve = bmd.scriptapp("Resolve")  # noqa: F821 — injected by Resolve
    except Exception as e:
        _log(f"ERROR: Could not connect to DaVinci Resolve: {e}")
        return

    if not resolve:
        _log("ERROR: Could not connect to DaVinci Resolve.")
        _log("  Make sure Resolve is open and scripting is enabled.")
        return

    try:
        pm = resolve.GetProjectManager()
        project = pm.GetCurrentProject()
    except Exception as e:
        _log(f"ERROR: Could not access project: {e}")
        return

    if not project:
        _log("ERROR: No project is currently open in DaVinci Resolve.")
        return

    _log(f"  Project: {project.GetName()}")

    # Clip discovery — priority order
    clips, sel_reason = _get_selected_timeline_clips(project)
    if clips:
        _log(f"  Source:  {len(clips)} selected clip(s) on timeline")
    elif sel_reason and sel_reason != "none selected":
        # Selection was attempted but failed — stop rather than silently processing all clips.
        _log(f"  ERROR: Could not read timeline selection — {sel_reason}")
        _log("  No clips processed. Fix the issue above and try again.")
        _log("  (To process all clips regardless of selection, set PROCESS_ALL_TIMELINE_CLIPS = True)")
        _log("")
        return
    else:
        # sel_reason == "none selected": nothing was selected, apply fallback settings.
        if PROCESS_ALL_TIMELINE_CLIPS:
            clips = _get_all_timeline_clips(project)
            if clips:
                _log(f"  Source:  All clips on timeline ({len(clips)}, deduplicated)")
        elif PROCESS_ALL_MEDIA_POOL_FOLDER:
            clips = _get_media_pool_clips(project)
            if clips:
                _log(f"  Source:  Media Pool current folder ({len(clips)} video clip(s))")

    if not clips:
        _log("")
        _log("  No clips to process. Options:")
        _log("    • Select clips on the timeline and run again (requires Resolve 18+)")
        _log("    • Set PROCESS_ALL_TIMELINE_CLIPS = True  to process the whole timeline")
        _log("    • Set PROCESS_ALL_MEDIA_POOL_FOLDER = True  to process current MP folder")
        _log("  (Edit the flags at the top of this script)")
        _log("")
        return

    workers = _effective_workers()
    _log(f"  Reuse sidecars: {'no (FORCE_REPROCESS=True)' if FORCE_REPROCESS else 'yes'}")
    if workers > 1:
        _log(f"  Workers: {workers} parallel")
    elif MAX_WORKERS > 1:
        _log(f"  Workers: 1 (forced — VLM escalation is enabled; parallel workers would exhaust RAM)")
    _log("")

    # Clear any leftover cancel sentinel from a previous run.
    try:
        os.remove(_CANCEL_FILE)
    except FileNotFoundError:
        pass

    # --- Run RoboSlate-arm on each clip (parallel subprocess calls) ---
    # Resolve API writes happen on the main thread afterwards.

    def _analyse_clip(clip_tuple):
        if os.path.exists(_CANCEL_FILE):
            return clip_tuple, None, False, "cancelled"
        clip_name, file_path, mpi = clip_tuple
        if not os.path.isfile(file_path):
            return clip_tuple, None, False, "offline"
        data, from_cache = _run_roboslate(file_path)
        return clip_tuple, data, from_cache, "ok"

    analysis_results = []
    cancelled = False
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_analyse_clip, clip): clip for clip in clips}
            for future in as_completed(futures):
                analysis_results.append(future.result())
        clip_order = {(c[0], c[1]): i for i, c in enumerate(clips)}
        analysis_results.sort(key=lambda r: clip_order.get((r[0][0], r[0][1]), 0))
    else:
        for clip in clips:
            result = _analyse_clip(clip)
            analysis_results.append(result)
            if result[3] == "cancelled":
                cancelled = True
                break

    # --- Write metadata back (main thread only) ---
    found = not_found = skipped = errors = cached = 0

    for clip_tuple, data, from_cache, run_status in analysis_results:
        clip_name, file_path, mpi = clip_tuple

        if run_status == "cancelled":
            _log(f"  [{clip_name}]  Cancelled.")
            skipped += 1
            continue

        _log(f"  [{clip_name}]")

        if run_status == "offline":
            _log(f"    Skipping — media offline: {file_path}")
            skipped += 1
            continue

        if data is None:
            errors += 1
            continue

        status = data.get("status", "unknown")

        if status == "not_found":
            _log("    No slate detected.")
            not_found += 1
            continue

        if status == "unreadable":
            _log("    Slate detected but unreadable.")
            not_found += 1
            continue

        if status != "found":
            _log(f"    Unexpected status: {status}")
            errors += 1
            continue

        metadata = _build_metadata(data)
        if not metadata:
            _log("    No metadata extracted.")
            not_found += 1
            continue

        try:
            ok = mpi.SetMetadata(metadata)
        except Exception as e:
            _log(f"    ERROR writing metadata: {e}")
            errors += 1
            continue

        if not ok:
            _log(f"    ERROR: SetMetadata returned False — keys: {list(metadata.keys())}")
            errors += 1
            continue

        res = data.get("result", {})
        conf   = res.get("overall_confidence", "?")
        review = "  !! needs review" if res.get("needs_review") else ""
        cache_tag = " [cached]" if from_cache else ""
        _log(f"    Written  [{conf} confidence]{review}{cache_tag}")

        slate = data.get("slate") or {}
        scene = metadata.get("Scene")
        shot  = metadata.get("Shot")
        take  = (slate.get("take") or {}).get("value")
        roll  = (slate.get("roll") or {}).get("value")
        parts = [f"Scene {scene}" if scene else None,
                 f"Shot {shot}"   if shot  else None,
                 f"Take {take}"   if take  else None,
                 f"Roll {roll}"   if roll  else None]
        summary = "  |  ".join(p for p in parts if p)
        if summary:
            _log(f"    {summary}")

        if from_cache:
            cached += 1
        found += 1

    _log("")
    _log("=" * 56)
    if cancelled:
        _log("  !! Cancelled by RoboSlate-arm_Kill")
    parts = [f"{found} found", f"{not_found} not found", f"{skipped} skipped", f"{errors} errors"]
    if cached:
        parts.append(f"{cached} from cache")
    _log(f"  {'Stopped' if cancelled else 'Done'}.  {'  |  '.join(parts)}")
    _log(f"  Log: {LOG_FILE}")
    _log("=" * 56)
    _log("")


main()
