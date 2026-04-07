"""
RoboSlate — DaVinci Resolve integration script.

Place this file in Resolve's Scripts folder and run via:
  Workspace → Scripts → Utility → RoboSlate

On macOS the Scripts folder is:
  ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/

Or install automatically with:
  ./install.sh --resolve   (from the RoboSlate project directory)

Clip selection — the script checks in this order:
  1. All video clips in the current Media Pool folder
  2. All clips on the current Timeline (fallback if Media Pool folder is empty)

Output goes to:
  - Resolve's console (Workspace → Console)
  - {ROBOSLATE_DIR}/logs/resolve.log  (always written, check this if nothing seems to happen)
"""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# ── Edit these settings to match your installation ─────────────────────────
ROBOSLATE_DIR = "/path/to/RoboSlate"

# Set to True to process ALL video clips in the current Media Pool folder.
# When False (default), only timeline clips are processed — select clips on
# a timeline and run the script from there.
# Note: Resolve's API does not expose which Media Pool items are selected,
# so enabling this always processes the entire current folder.
PROCESS_ALL_MEDIA_POOL_FOLDER = False

# Set to True to process ALL clips on the current timeline.
# This is the recommended default — control which clips are processed by
# putting only the clips you want on the current timeline before running.
# (Resolve's selection API is unreliable across versions.)
PROCESS_ALL_TIMELINE_CLIPS = True

# Number of clips to analyse in parallel. Each worker is a subprocess call
# to roboslate.py — Resolve API writes still happen on the main thread after.
# Set to 1 to process sequentially.
MAX_WORKERS = 4
# ───────────────────────────────────────────────────────────────────────────

VENV_PYTHON  = os.path.join(ROBOSLATE_DIR, "venv", "bin", "python")
ROBOSLATE_PY = os.path.join(ROBOSLATE_DIR, "roboslate.py")
LOG_FILE     = os.path.join(ROBOSLATE_DIR, "logs", "resolve.log")

# Map RoboSlate field names → Resolve metadata keys.
# scene and slate_number are combined into "Scene" (handled separately below).
FIELD_MAP = {
    "take":       "Take",
    "roll":       "Roll",
    "camera":     "Camera",
    "director":   "Director",
    "dop":        "DOP",
    "production": "Production",
    "date":       "Shoot Date",
    "fps":        "Frame Rate",
    "format":     "Format",
    "notes":      "Comments",
}


# ---------------------------------------------------------------------------
# Logging — writes to both console and log file
# ---------------------------------------------------------------------------

def _log(msg):
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
    """Verify ROBOSLATE_DIR and venv are set up correctly."""
    if ROBOSLATE_DIR == "/path/to/RoboSlate":
        _log("ERROR: ROBOSLATE_DIR is not set.")
        _log("  Edit resolve/RoboSlate.py and set ROBOSLATE_DIR to the full path")
        _log("  of your RoboSlate installation, or run: ./install.sh --resolve")
        return False
    if not os.path.isfile(VENV_PYTHON):
        _log(f"ERROR: venv Python not found at: {VENV_PYTHON}")
        _log("  Run ./install.sh first to set up the virtual environment.")
        return False
    if not os.path.isfile(ROBOSLATE_PY):
        _log(f"ERROR: roboslate.py not found at: {ROBOSLATE_PY}")
        return False
    return True


def _get_media_pool_clips(project):
    """
    Return (clip_name, file_path, MediaPoolItem) tuples for video clips
    in the current Media Pool folder.
    """
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
            clip_type = props.get("Type", "")
            # Accept Video and Video+Audio clips; skip audio-only, stills, etc.
            if "Video" not in clip_type:
                continue
            file_path = props.get("File Path", "")
            clip_name = props.get("Clip Name", "") or os.path.basename(file_path)
            if file_path:
                result.append((clip_name, file_path, clip))
        except Exception:
            continue

    return result


def _get_selected_timeline_clips(project):
    """
    Return (clip_name, file_path, MediaPoolItem) tuples for selected clips
    on the current timeline. Requires Resolve 18+.
    Returns an empty list if selection is unavailable or nothing is selected.
    """
    try:
        timeline = project.GetCurrentTimeline()
        if not timeline:
            return []
        selected = timeline.GetSelectedClips()
        if not selected:
            return []
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
        return result
    except Exception:
        return []


def _get_all_timeline_clips(project):
    """
    Return (clip_name, file_path, MediaPoolItem) tuples for all clips
    across all video tracks on the current timeline.
    """
    try:
        timeline = project.GetCurrentTimeline()
        if not timeline:
            return []
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
                clip_name = props.get("Clip Name", "") or os.path.basename(file_path)
                if file_path:
                    result.append((clip_name, file_path, mpi))
        return result
    except Exception as e:
        _log(f"  (Timeline error: {e})")
        return []


def _run_roboslate(file_path):
    """
    Run RoboSlate on a single file and return the parsed JSON result dict,
    or None on failure.
    """
    try:
        proc = subprocess.run(
            [VENV_PYTHON, ROBOSLATE_PY, "--file", file_path, "--stdout", "--quiet"],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        _log("    ERROR: timed out (>10 min)")
        return None
    except Exception as e:
        _log(f"    ERROR: could not launch RoboSlate: {e}")
        return None

    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        msg = lines[-1] if lines else "unknown error"
        _log(f"    ERROR: {msg}")
        return None

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        _log("    ERROR: could not parse RoboSlate output")
        if proc.stdout:
            _log(f"    Raw output: {proc.stdout[:200]}")
        return None


def _build_metadata(data):
    """
    Build a Resolve metadata dict from a RoboSlate result dict.
    Returns an empty dict if no usable fields were found.
    """
    slate = data.get("slate") or {}
    metadata = {}

    # Resolve uses separate fields: Scene → scene, Shot → slate_number, Take → take
    scene_val = (slate.get("scene") or {}).get("value")
    slate_num = (slate.get("slate_number") or {}).get("value")
    if scene_val:
        metadata["Scene"] = str(scene_val)
    if slate_num:
        metadata["Shot"] = str(slate_num)

    # Standard field mapping
    for rs_field, resolve_key in FIELD_MAP.items():
        field_data = slate.get(rs_field)
        if isinstance(field_data, dict):
            value = field_data.get("value")
            if value is not None:
                metadata[resolve_key] = str(value)

    # RoboSlate provenance tags
    res = data.get("result", {})
    metadata["RoboSlate_Confidence"]  = res.get("overall_confidence", "")
    metadata["RoboSlate_NeedsReview"] = "yes" if res.get("needs_review") else "no"
    metadata["RoboSlate_ProcessedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _log("")
    _log("=" * 56)
    _log("  RoboSlate — DaVinci Resolve")
    _log("=" * 56)

    if not _check_install():
        return

    # Connect to Resolve
    try:
        resolve = bmd.scriptapp("Resolve")  # noqa: F821 — provided by Resolve's runtime
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

    # --- Get clips ---
    # Priority order:
    #   1. Selected clips on the current timeline (Resolve 18+ only)
    #   2. All clips on the current timeline  (requires PROCESS_ALL_TIMELINE_CLIPS = True)
    #   3. All clips in current Media Pool folder (requires PROCESS_ALL_MEDIA_POOL_FOLDER = True)
    clips = _get_selected_timeline_clips(project)
    if clips:
        _log(f"  Source : {len(clips)} selected clip(s) on timeline")
    elif PROCESS_ALL_TIMELINE_CLIPS:
        clips = _get_all_timeline_clips(project)
        if clips:
            _log(f"  Source : All clips on timeline ({len(clips)})")
    elif PROCESS_ALL_MEDIA_POOL_FOLDER:
        clips = _get_media_pool_clips(project)
        if clips:
            _log(f"  Source : Media Pool current folder ({len(clips)} video clip(s))")

    if not clips:
        _log("")
        _log("  No clips to process. Options:")
        _log("    • Select clips on the timeline and run again (requires Resolve 18+)")
        _log("    • Set PROCESS_ALL_TIMELINE_CLIPS = True  to process the whole timeline")
        _log("    • Set PROCESS_ALL_MEDIA_POOL_FOLDER = True  to process current MP folder")
        _log("  (Edit the flags at the top of this script)")
        _log("")
        return

    _log("")

    if MAX_WORKERS > 1:
        _log(f"  Workers: {MAX_WORKERS} parallel")
        _log("")

    # --- Run RoboSlate on each clip (parallel subprocess calls) ---
    # Only the subprocess I/O is parallelised; Resolve API writes happen
    # on the main thread afterwards to avoid any thread-safety concerns.

    def _analyse_clip(clip_tuple):
        clip_name, file_path, mpi = clip_tuple
        if not os.path.isfile(file_path):
            return clip_tuple, None, "offline"
        data = _run_roboslate(file_path)
        return clip_tuple, data, "ok"

    analysis_results = []
    if MAX_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_analyse_clip, clip): clip for clip in clips}
            for future in as_completed(futures):
                analysis_results.append(future.result())
        # Restore original clip order for predictable log output
        clip_order = {(c[0], c[1]): i for i, c in enumerate(clips)}
        analysis_results.sort(key=lambda r: clip_order.get((r[0][0], r[0][1]), 0))
    else:
        for clip in clips:
            analysis_results.append(_analyse_clip(clip))

    # --- Write metadata back (main thread) ---
    found = not_found = skipped = errors = 0

    for clip_tuple, data, run_status in analysis_results:
        clip_name, file_path, mpi = clip_tuple
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
            _log(f"    Status: {status}")
            errors += 1
            continue

        metadata = _build_metadata(data)
        if not metadata:
            _log("    No metadata extracted.")
            not_found += 1
            continue

        try:
            existing = mpi.GetMetadata() or {}
            _log(f"    Existing metadata keys: {list(existing.keys())}")
            _log(f"    Writing keys: {list(metadata.keys())}")
            ok = mpi.SetMetadata(metadata)
            _log(f"    SetMetadata returned: {ok!r}")
            if not ok:
                # Try writing keys individually to find which ones are accepted
                for k, v in metadata.items():
                    r = mpi.SetMetadata(k, v)
                    _log(f"      SetMetadata({k!r}, {v!r}) → {r!r}")
        except Exception as e:
            _log(f"    ERROR writing metadata: {e}")
            errors += 1
            continue

        res = data.get("result", {})
        conf   = res.get("overall_confidence", "?")
        review = "  !! needs review" if res.get("needs_review") else ""
        _log(f"    Written  [{conf} confidence]{review}")

        # Show key fields inline
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

        found += 1

    # Summary
    _log("")
    _log("=" * 56)
    _log(f"  Done.  {found} found  |  {not_found} not found  |  {skipped} skipped  |  {errors} errors")
    _log(f"  Log: {LOG_FILE}")
    _log("=" * 56)
    _log("")


main()
