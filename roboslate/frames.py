"""
Frame extraction from video files using ffmpeg/ffprobe.

All ffmpeg/ffprobe calls are made via subprocess — no Python wrapper needed.
ffmpeg must be installed and available in PATH (or in standard Homebrew locations).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile


# ---------------------------------------------------------------------------
# Resolve ffmpeg / ffprobe binaries
# ---------------------------------------------------------------------------
# SCRATCH and other app launchers start scripts with a minimal PATH that often
# excludes /usr/local/bin and /opt/homebrew/bin. Resolve once at import time.

def _find_binary(name):
    """Return the full path to a binary, checking PATH then Homebrew prefixes."""
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/usr/local/bin", "/opt/homebrew/bin", "/opt/local/bin"):
        candidate = os.path.join(prefix, name)
        if os.path.isfile(candidate):
            return candidate
    return name  # fall back to bare name; will fail with clear error at call time

FFPROBE = _find_binary("ffprobe")
FFMPEG  = _find_binary("ffmpeg")


# ---------------------------------------------------------------------------
# Duration detection
# ---------------------------------------------------------------------------

def get_video_duration(video_path):
    """
    Return the duration of a video file in seconds (float).

    Uses ffprobe to read the first video stream. Falls back to container
    duration if stream duration is unavailable.

    Raises RuntimeError if ffprobe fails or duration cannot be determined.
    """
    cmd = [
        FFPROBE,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"ffprobe not found (tried: {FFPROBE}). "
            "Install ffmpeg (includes ffprobe): brew install ffmpeg"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timed out on: {video_path}")

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed on {video_path!r}:\n{result.stderr.strip()}"
        )

    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise RuntimeError(f"No video streams found in: {video_path}")

        stream = streams[0]

        # Prefer explicit duration field
        if "duration" in stream:
            return float(stream["duration"])

        # Fall back to nb_frames / frame_rate
        if "nb_frames" in stream and "r_frame_rate" in stream:
            nb_frames = int(stream["nb_frames"])
            num, den = stream["r_frame_rate"].split("/")
            fps = float(num) / float(den)
            return nb_frames / fps

    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Could not parse ffprobe output: {e}")

    raise RuntimeError(f"Could not determine duration of: {video_path}")


# ---------------------------------------------------------------------------
# Scan schedule
# ---------------------------------------------------------------------------

def build_scan_schedule(duration, cfg):
    """
    Build a list of (phase, timestamp) tuples for the three-phase scan strategy.

    Phase 1: first SCAN_PHASE1_DURATION seconds at SCAN_PHASE1_FPS fps
    Phase 2: last SCAN_PHASE2_DURATION seconds at SCAN_PHASE2_FPS fps
             (deduplicated against Phase 1)
    Phase 3: caller-triggered fallback — not included here; see phase3_timestamps()

    Args:
        duration: Video duration in seconds.
        cfg: Config dict from config.py.

    Returns:
        List of (phase_name, timestamp_float) tuples, sorted by timestamp.
    """
    p1_dur = min(float(cfg["SCAN_PHASE1_DURATION"]), duration)
    p1_fps = float(cfg["SCAN_PHASE1_FPS"])
    p1_interval = 1.0 / p1_fps

    p2_dur = float(cfg["SCAN_PHASE2_DURATION"])
    p2_fps = float(cfg["SCAN_PHASE2_FPS"])
    p2_interval = 1.0 / p2_fps
    p2_start = max(0.0, duration - p2_dur)

    # Phase 1 timestamps
    p1_times = set()
    t = 0.0
    while t <= p1_dur:
        p1_times.add(round(t, 3))
        t += p1_interval

    # Phase 2 timestamps (skip anything already in Phase 1)
    p2_entries = []
    t = p2_start
    while t <= duration:
        ts = round(t, 3)
        if ts not in p1_times:
            p2_entries.append(("phase2", ts))
        t += p2_interval

    p1_entries = [("phase1", ts) for ts in sorted(p1_times)]

    return p1_entries + sorted(p2_entries, key=lambda x: x[1])


def phase3_timestamps(duration, cfg):
    """
    Return Phase 3 fallback timestamps: every SCAN_PHASE3_INTERVAL seconds.

    Phase 3 is only used if Phases 1 and 2 found nothing.
    """
    interval = float(cfg["SCAN_PHASE3_INTERVAL"])
    timestamps = []
    t = 0.0
    while t <= duration:
        timestamps.append(("phase3", round(t, 3)))
        t += interval
    return timestamps


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_phase_range(video_path, start_time, duration_secs, fps, output_dir, prefix, jpeg_quality=3):
    """
    Extract a range of frames from a video using one ffmpeg call.

    Args:
        video_path: Path to source video.
        start_time: Start time in seconds.
        duration_secs: How many seconds to extract.
        fps: Frames per second to extract.
        output_dir: Directory to write JPEG frames into.
        prefix: Filename prefix (e.g. "phase1").

    Returns:
        List of absolute paths to extracted frame files (sorted).

    Raises RuntimeError on ffmpeg failure.
    """
    output_pattern = os.path.join(output_dir, f"{prefix}_%04d.jpg")

    cmd = [
        FFMPEG,
        "-ss", f"{start_time:.3f}",
        "-i", video_path,
        "-t", f"{duration_secs:.3f}",
        "-r", str(fps),
        "-vframes", str(int(duration_secs * fps) + 2),  # +2 safety margin
        "-q:v", str(jpeg_quality),
        "-vf", "scale='min(1568,iw)':-2",
        output_pattern,
        "-y",
        "-loglevel", "error",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out extracting range from: {video_path}")

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed extracting range from {video_path!r}:\n{result.stderr.strip()}"
        )

    # Return sorted list of files matching the pattern
    frames = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith(prefix + "_") and f.endswith(".jpg")
    )
    return frames


def extract_single_frame(video_path, timestamp, output_dir, jpeg_quality=3):
    """
    Extract a single frame at a specific timestamp.

    Used for Phase 3 sparse scanning where batch extraction would be wasteful.

    Returns:
        Absolute path to the extracted JPEG file.

    Raises RuntimeError on ffmpeg failure.
    """
    filename = f"phase3_T{timestamp:.3f}.jpg"
    output_path = os.path.join(output_dir, filename)

    cmd = [
        FFMPEG,
        "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", str(jpeg_quality),
        "-vf", "scale='min(1568,iw)':-2",
        output_path,
        "-y",
        "-loglevel", "error",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg timed out at timestamp {timestamp} for: {video_path}")

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed at {timestamp}s in {video_path!r}:\n{result.stderr.strip()}"
        )

    if not os.path.isfile(output_path):
        raise RuntimeError(
            f"ffmpeg produced no output at {timestamp}s for: {video_path}"
        )

    return output_path


def cleanup_frames(frame_paths):
    """Delete a list of frame files. Silently skips missing files."""
    for path in frame_paths:
        try:
            os.remove(path)
        except OSError:
            pass


def make_temp_dir():
    """Create and return a temporary directory for frame storage."""
    return tempfile.mkdtemp(prefix="roboslate_")
