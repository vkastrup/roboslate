#!/usr/bin/env python3
# Bootstrap: re-exec with the project venv Python if not already using it.
# This ensures the correct interpreter is used regardless of how SCRATCH
# (or any other launcher) invokes the script.
import os as _os, sys as _sys
_venv_python = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "venv", "bin", "python")
if _os.path.isfile(_venv_python) and not _sys.executable.startswith(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "venv")
):
    _os.execv(_venv_python, [_venv_python] + _sys.argv)
"""
RoboSlate — Clapperboard detection and metadata extraction
Uses Claude Vision to detect slates in video and extract production metadata.

Usage (CLI):
    venv/bin/python roboslate.py --file /path/to/clip.mov
    venv/bin/python roboslate.py --batch /path/to/folder --format csv
    venv/bin/python roboslate.py --help

Usage (Assimilate SCRATCH custom command):
    Called automatically by SCRATCH — receives input and output XML paths as positional args.
    python roboslate.py /path/to/input.xml /path/to/output.xml
"""

import argparse
import json
import logging
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------

try:
    import anthropic  # noqa: F401
except ImportError:
    print(
        "\nERROR: Required packages are not installed.\n"
        "\nRun the installer first:\n"
        "  cd /path/to/RoboSlate\n"
        "  ./install.sh\n"
        "\nThen run using the project's virtual environment:\n"
        "  venv/bin/python roboslate.py --help\n",
        file=sys.stderr,
    )
    sys.exit(1)

from roboslate.config import load_config, get_enabled_fields
from roboslate import frames as frm
from roboslate import vision
from roboslate import output as out
from roboslate import scratch as scr
from roboslate import merge as mrg


def _get_vision_mod(args, cfg):
    """Return the vision module to use based on --backend flag or VISION_BACKEND config."""
    backend = getattr(args, "backend", None) or cfg.get("VISION_BACKEND", "claude")
    if backend == "local":
        from roboslate import vision_local
        return vision_local
    return vision

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path, quiet):
    level = logging.WARNING if quiet else logging.INFO
    handlers = []

    default_log = log_path or os.path.join(PROJECT_DIR, "logs", "roboslate.log")
    os.makedirs(os.path.dirname(default_log), exist_ok=True)
    handlers.append(logging.FileHandler(default_log))

    if not quiet:
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Core single-clip processing
# ---------------------------------------------------------------------------

def process_file(video_path, cfg, args, shot_info=None):
    """
    Run the full pipeline on a single video file.

    Returns:
        (result_dict, status_string) where status is:
        "found" | "not_found" | "unreadable" | "error"
    """
    log = logging.getLogger(__name__)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    log.info(f"Processing: {video_path}")

    duration = frm.get_video_duration(video_path)
    log.info(f"Duration: {duration:.1f}s")

    frames_dir = getattr(args, "frames_dir", None) or frm.make_temp_dir()
    os.makedirs(frames_dir, exist_ok=True)
    own_frames_dir = getattr(args, "frames_dir", None) is None

    all_frame_entries = []  # (phase, timestamp, path)
    detections = []

    try:
        schedule = frm.build_scan_schedule(duration, cfg)

        if getattr(args, "phase1_only", False):
            schedule = [(p, t) for p, t in schedule if p == "phase1"]

        # Dry-run cost estimate
        if getattr(args, "dry_run", False):
            total_frames = min(len(schedule), int(cfg.get("MAX_FRAMES", 40)))
            est = vision.estimate_cost(total_frames, cfg)
            print(f"\n  Dry run: {total_frames} frames would be sent to Claude")
            print(f"  Estimated cost: {est['estimated_cost_usd']}")
            print(f"  ({est['note']})")
            print(f"  Frames at timestamps:")
            for phase, ts in schedule[:total_frames]:
                print(f"    {phase:8s}  {ts:.3f}s")
            return None, "dry_run"

        jpeg_quality = int(cfg.get("FRAME_JPEG_QUALITY", 3))

        # --- Phase 1: batch extract + scan ---
        p1_schedule = [(p, t) for p, t in schedule if p == "phase1"]
        if p1_schedule:
            p1_dur = min(float(cfg["SCAN_PHASE1_DURATION"]), duration)
            p1_fps = float(cfg["SCAN_PHASE1_FPS"])
            if not getattr(args, "quiet", False):
                print(f"  Phase 1: extracting {p1_dur:.0f}s at {p1_fps}fps...", end=" ", flush=True)

            p1_files = frm.extract_phase_range(video_path, 0.0, p1_dur, p1_fps, frames_dir, "phase1", jpeg_quality=jpeg_quality)
            for i, (phase, ts) in enumerate(p1_schedule):
                if i < len(p1_files):
                    all_frame_entries.append((phase, ts, p1_files[i]))

            if not getattr(args, "quiet", False):
                print(f"{len(p1_files)} frames extracted.")

        # --- Phase 2: batch extract ---
        p2_schedule = [(p, t) for p, t in schedule if p == "phase2"]
        if p2_schedule and not getattr(args, "phase1_only", False):
            p2_dur = float(cfg["SCAN_PHASE2_DURATION"])
            p2_fps = float(cfg["SCAN_PHASE2_FPS"])
            p2_start = max(0.0, duration - p2_dur)
            actual_p2_dur = duration - p2_start
            if not getattr(args, "quiet", False):
                print(f"  Phase 2: extracting last {actual_p2_dur:.0f}s at {p2_fps}fps...", end=" ", flush=True)

            p2_files = frm.extract_phase_range(video_path, p2_start, actual_p2_dur, p2_fps, frames_dir, "phase2", jpeg_quality=jpeg_quality)
            for i, (phase, ts) in enumerate(p2_schedule):
                if i < len(p2_files):
                    all_frame_entries.append((phase, ts, p2_files[i]))

            if not getattr(args, "quiet", False):
                print(f"{len(p2_files)} frames extracted.")

        max_frames = int(cfg.get("MAX_FRAMES", 40))
        entries_to_scan = all_frame_entries[:max_frames]

        # Apply model override
        scan_cfg = dict(cfg)
        if getattr(args, "model", None):
            scan_cfg["CLAUDE_MODEL"] = args.model
        if getattr(args, "full_scan", False):
            scan_cfg["EARLY_STOP_CONFIDENCE"] = "none"

        # Compute enabled fields once; store in scan_cfg for vision module
        enabled_fields = get_enabled_fields(scan_cfg)
        scan_cfg["_enabled_fields"] = enabled_fields

        vision_mod = _get_vision_mod(args, scan_cfg)
        client = vision_mod.build_client(scan_cfg.get("ANTHROPIC_API_KEY"))

        from roboslate.output import extract_reel_from_filename
        reel_hint = extract_reel_from_filename(video_path)

        if not getattr(args, "quiet", False):
            reel_note = f" [reel hint: {reel_hint}]" if reel_hint else ""
            backend_label = "PaddleOCR (local)" if vision_mod is not vision else scan_cfg.get("CLAUDE_MODEL", "claude")
            print(f"  Scanning {len(entries_to_scan)} frames with {backend_label}{reel_note}...")

        detections = vision_mod.scan_frames(entries_to_scan, client, scan_cfg, reel_hint=reel_hint)

        # --- Phase 3 fallback ---
        slate_found_so_far = any(d.get("slate_detected") for d in detections)
        if not slate_found_so_far and not getattr(args, "phase1_only", False):
            if not getattr(args, "quiet", False):
                print("  No slate in Phase 1/2. Trying sparse Phase 3 scan...")
            p3_timestamps = frm.phase3_timestamps(duration, scan_cfg)
            p3_entries = []
            for phase, ts in p3_timestamps:
                frame_path = frm.extract_single_frame(video_path, ts, frames_dir, jpeg_quality=jpeg_quality)
                p3_entries.append((phase, ts, frame_path))
                all_frame_entries.append((phase, ts, frame_path))

            remaining = max_frames - len(detections)
            if remaining > 0:
                p3_detections = vision_mod.scan_frames(p3_entries[:remaining], client, scan_cfg, reel_hint=reel_hint)
                detections.extend(p3_detections)

        # --- Merge multi-frame detections ---
        merged = mrg.merge_detections(detections, fields=enabled_fields)
        status = out.classify_status(merged)

        scan_log = [
            {
                "timestamp": d.get("timestamp"),
                "phase": d.get("phase"),
                "frame_file": d.get("frame_file"),
                "slate_detected": d.get("slate_detected", False),
                "overall_confidence": d.get("overall_confidence", "low"),
                "model_used": d.get("model_used"),
                "escalated": d.get("escalated", False),
                "needs_review": d.get("needs_review", False),
            }
            for d in detections
        ]

        scan_meta = {
            "duration_seconds": duration,
            "frames_scanned": len(detections),
            "api_calls_made": detections[-1].get("_total_api_calls", len(detections)) if detections else 0,
            "phases_run": list({d.get("phase") for d in detections}),
            "scan_log": scan_log,
        }

        result = out.build_result(video_path, merged, scan_meta, shot_info)
        return result, status

    finally:
        if own_frames_dir and not getattr(args, "keep_frames", False):
            all_paths = [e[2] for e in all_frame_entries if os.path.isfile(str(e[2]))]
            frm.cleanup_frames(all_paths)
            try:
                shutil.rmtree(frames_dir, ignore_errors=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_outputs(result, video_path, args, csv_path=None, cfg=None):
    output_format = getattr(args, "format", "json") or "json"
    stdout = getattr(args, "stdout", False)
    quiet = getattr(args, "quiet", False)
    output_path = getattr(args, "output", None) or out.get_default_output_path(video_path)
    fields = get_enabled_fields(cfg) if cfg else None

    if output_format in ("json", "both"):
        try:
            out.write_json(result, output_path)
            if not quiet:
                print(f"  Written: {output_path}")
        except (PermissionError, OSError) as e:
            if not quiet:
                print(f"  WARNING: Could not write JSON sidecar (skipping): {e}", file=sys.stderr)

    if output_format in ("csv", "both"):
        _csv = csv_path or os.path.splitext(output_path)[0] + ".csv"
        out.write_csv_row(result, _csv, fields=fields)
        if not quiet:
            print(f"  Appended: {_csv}")

    if stdout:
        print(json.dumps(result, indent=2))

    if not quiet and not stdout:
        out.print_summary(result, fields=fields)


# ---------------------------------------------------------------------------
# SCRATCH mode
# ---------------------------------------------------------------------------

def run_scratch_mode(input_xml, output_xml, cfg, args):
    enabled_fields = get_enabled_fields(cfg)
    """
    Process all shots from a SCRATCH XML selection.

    Reads clip list from input_xml, processes each, writes results back
    to output_xml and alongside each clip as .roboslate.json files.
    """
    log = logging.getLogger(__name__)
    quiet = getattr(args, "quiet", False)

    log.info(f"SCRATCH mode: parsing {input_xml}")
    scratch_input = scr.parse_scratch_xml(input_xml)
    shots = scratch_input.get("shots", [])

    if not shots:
        print("RoboSlate: No shots found in SCRATCH selection.", file=sys.stderr)
        sys.exit(0)

    if not quiet:
        print(f"\nRoboSlate: {len(shots)} shot(s) selected in SCRATCH.")
        print(f"  Project: {scratch_input.get('project', '?')}")
        print(f"  Construct: {scratch_input.get('construct', '?')}\n")

    clip_results = []  # for SCRATCH writeback and summary

    for i, shot in enumerate(shots, 1):
        name = shot.get("name") or os.path.basename(shot.get("file", f"shot_{i}"))
        media = scr.resolve_media_path(shot)

        if not quiet:
            print(f"[{i}/{len(shots)}] {name}")

        if media.get("offline"):
            print(f"  WARNING: media offline — {shot.get('file', '?')}", file=sys.stderr)
            clip_results.append({
                "uuid": shot.get("uuid", ""),
                "name": name,
                "status": "error",
                "error": "media offline",
                "merged": None,
            })
            continue

        shot_info = {
            "uuid":      shot.get("uuid"),
            "name":      name,
            "project":   scratch_input.get("project"),
            "construct": scratch_input.get("construct"),
        }

        try:
            result, status = process_file(media["path"], cfg, args, shot_info=shot_info)

            if result is not None:
                # Write JSON alongside the source clip (best-effort — RAID volumes may be read-only)
                json_path = out.get_default_output_path(media["path"])
                try:
                    out.write_json(result, json_path)
                    log.info(f"JSON written: {json_path}")
                except (PermissionError, OSError) as json_err:
                    log.warning(f"Could not write JSON sidecar (skipping): {json_err}")

            if not quiet:
                out.print_summary(result, fields=enabled_fields)

            merged = mrg.merge_detections(
                result.get("scan_log", []) if result else []
            ) if result else None
            # Actually, we already have the merged result baked into the result
            # Rebuild a lightweight version for SCRATCH writeback
            merged_for_scratch = {
                "slate_detected": (result or {}).get("result", {}).get("slate_found", False),
                "overall_confidence": (result or {}).get("result", {}).get("overall_confidence", "none"),
                "needs_review": (result or {}).get("result", {}).get("needs_review", False),
                "conflicts": (result or {}).get("result", {}).get("conflicts", []),
                "fields": (result or {}).get("slate") or {},
            }

            clip_results.append({
                "uuid":   shot.get("uuid", ""),
                "slot":   shot.get("slot", ""),
                "layer":  shot.get("layer", "0"),
                "name":   name,
                "status": status,
                "merged": merged_for_scratch,
                "result": result,
            })

        except Exception as e:
            log.error(f"Failed to process {name}: {e}", exc_info=True)
            print(f"  ERROR: {e}", file=sys.stderr)
            clip_results.append({
                "uuid":   shot.get("uuid", ""),
                "slot":   shot.get("slot", ""),
                "layer":  shot.get("layer", "0"),
                "name":   name,
                "status": "error",
                "error":  str(e),
                "merged": None,
            })

    # Batch summary
    scr.print_batch_summary(clip_results, quiet=quiet)

    # Write SCRATCH output XML
    if output_xml:
        written = scr.build_output_xml(scratch_input, clip_results, output_xml)
        if written and not quiet:
            found_count = sum(1 for r in clip_results if r.get("status") == "found")
            print(f"  SCRATCH metadata written for {found_count} shot(s): {output_xml}\n")


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="roboslate",
        description="Detect clapperboards in video and extract production metadata using Claude Vision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --file /path/to/clip.mov
  %(prog)s --file clip.mov --stdout --model claude-sonnet-4-6
  %(prog)s --batch /media/rushes --format csv
  %(prog)s --file clip.mov --dry-run           (estimate cost, no API calls)
  %(prog)s --file clip.mov --full-scan --keep-frames

SCRATCH usage (called automatically by SCRATCH custom command):
  %(prog)s /path/to/scratch_input.xml /path/to/scratch_output.xml
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file",  metavar="PATH", help="Video file to analyse.")
    input_group.add_argument("--batch", metavar="DIR",  help="Process all video files in a directory.")

    parser.add_argument("--output",  metavar="PATH",                        help="JSON output path (default: next to source video).")
    parser.add_argument("--format",  choices=["json", "csv", "both"], default="json", help="Output format.")
    parser.add_argument("--stdout",  action="store_true",                   help="Print JSON to stdout.")
    parser.add_argument("--quiet",   action="store_true",                   help="Suppress progress output.")
    parser.add_argument("--ext",     default=".mp4,.mov,.mxf,.r3d,.ari,.braw", help="Extensions for --batch.")
    parser.add_argument("--csv",     metavar="PATH",                        help="CSV path for --batch.")
    parser.add_argument("--full-scan",   action="store_true",               help="Disable early exit; scan all frames.")
    parser.add_argument("--phase1-only", action="store_true",               help="Only scan first 60s.")
    parser.add_argument("--max-frames",  type=int, metavar="N",             help="Override MAX_FRAMES config.")
    parser.add_argument("--keep-frames", action="store_true",               help="Keep extracted frame files.")
    parser.add_argument("--frames-dir",  metavar="DIR",                     help="Temp frames directory.")
    parser.add_argument("--dry-run",     action="store_true",               help="Estimate cost only; no API calls.")
    parser.add_argument("--config",      metavar="PATH",                    help="Path to config.env.")
    parser.add_argument("--model",       metavar="MODEL",                   help="Override CLAUDE_MODEL.")
    parser.add_argument("--force",       action="store_true",               help="Re-process clips even if a sidecar JSON already exists.")
    parser.add_argument("--workers",     type=int, metavar="N",             help="Parallel workers for --batch (default: BATCH_WORKERS in config, or 1).")
    parser.add_argument(
        "--backend",
        choices=["claude", "local"],
        default=None,
        help="Vision backend: 'claude' (default) uses Claude API; 'local' uses PaddleOCR (no API key needed).",
    )

    return parser


def run_cli(args, cfg):
    log = logging.getLogger(__name__)

    if args.max_frames:
        cfg = dict(cfg)
        cfg["MAX_FRAMES"] = str(args.max_frames)

    if args.file:
        sidecar = out.get_default_output_path(args.file)
        if os.path.isfile(sidecar) and not getattr(args, "force", False):
            print(f"  Skipping (already processed): {sidecar}")
            print(f"  Use --force to re-process.")
        else:
            result, status = process_file(args.file, cfg, args)
            if result is not None:
                write_outputs(result, args.file, args, cfg=cfg)

    elif args.batch:
        if not os.path.isdir(args.batch):
            print(f"ERROR: --batch directory not found: {args.batch}", file=sys.stderr)
            sys.exit(1)

        extensions = {e.strip().lower() for e in args.ext.split(",")}
        video_files = [
            os.path.join(args.batch, f)
            for f in sorted(os.listdir(args.batch))
            if os.path.splitext(f)[1].lower() in extensions
        ]

        if not video_files:
            print(f"No video files found in: {args.batch}", file=sys.stderr)
            sys.exit(0)

        def _summary_entry(name, result, status):
            return {"name": name, "status": status, "merged": {
                "slate_detected":    (result or {}).get("result", {}).get("slate_found", False),
                "overall_confidence":(result or {}).get("result", {}).get("overall_confidence", "none"),
                "needs_review":      (result or {}).get("result", {}).get("needs_review", False),
                "conflicts":         (result or {}).get("result", {}).get("conflicts", []),
                "fields":            (result or {}).get("slate") or {},
            }}

        # Determine parallelism: --workers flag > BATCH_WORKERS config > 1
        n_workers = getattr(args, "workers", None) or int(cfg.get("BATCH_WORKERS", 1))
        n_workers = max(1, n_workers)

        print(f"Found {len(video_files)} file(s) to process.")
        if n_workers > 1:
            print(f"Running with {n_workers} parallel workers.\n")
        else:
            print()

        clip_results = []
        csv_lock = threading.Lock()

        def _process_one_batch(vf):
            name = os.path.basename(vf)
            sidecar = out.get_default_output_path(vf)
            if os.path.isfile(sidecar) and not getattr(args, "force", False):
                if not args.quiet:
                    print(f"  [{name}] Skipping (already processed)")
                try:
                    with open(sidecar) as _f:
                        cached = json.load(_f)
                    return name, cached, cached.get("status", "found"), None
                except Exception:
                    pass  # fall through to re-process if sidecar is unreadable

            try:
                result, status = process_file(vf, cfg, args)
                if result is not None:
                    if args.format in ("csv", "both"):
                        # CSV appends must be serialised across threads
                        with csv_lock:
                            write_outputs(result, vf, args, csv_path=args.csv, cfg=cfg)
                    else:
                        write_outputs(result, vf, args, csv_path=args.csv, cfg=cfg)
                return name, result, status, None
            except Exception as e:
                log.error(f"Failed: {vf}: {e}")
                return name, None, "error", str(e)

        if n_workers == 1:
            # Sequential path — preserves ordered output with counters
            for i, vf in enumerate(video_files, 1):
                name = os.path.basename(vf)
                print(f"[{i}/{len(video_files)}] {name}")
                name, result, status, err = _process_one_batch(vf)
                if err:
                    print(f"  ERROR: {err}", file=sys.stderr)
                clip_results.append(_summary_entry(name, result, status))
        else:
            # Parallel path
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_process_one_batch, vf): vf for vf in video_files}
                raw_results = {}
                for future in as_completed(futures):
                    vf = futures[future]
                    name, result, status, err = future.result()
                    if err:
                        print(f"  ERROR [{name}]: {err}", file=sys.stderr)
                    raw_results[vf] = (name, result, status)

            for vf in video_files:
                name, result, status = raw_results[vf]
                clip_results.append(_summary_entry(name, result, status))

        # Batch summary
        scr.print_batch_summary(clip_results, quiet=args.quiet)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # SCRATCH mode: detect positional XML args before argparse
    if scr.is_scratch_mode():
        input_xml, output_xml = scr.get_scratch_args()
        cfg, _ = load_config(project_dir=PROJECT_DIR)
        setup_logging(cfg.get("LOG_PATH"), quiet=False)
        log = logging.getLogger(__name__)

        # Log sys.argv so we can verify what SCRATCH passed
        log.info(f"SCRATCH mode: sys.argv = {sys.argv}")
        log.info(f"  input_xml  = {input_xml}")
        log.info(f"  output_xml = {output_xml}")
        if output_xml is None:
            log.warning(
                "No output XML path provided (sys.argv has only 1 arg). "
                "SCRATCH will not receive metadata back. "
                "Check that the command is configured to wait for the script to finish."
            )

        from types import SimpleNamespace
        _Args = SimpleNamespace(
            dry_run=False, full_scan=False, phase1_only=False,
            keep_frames=False, frames_dir=None, stdout=False,
            quiet=False, output=None, format="json", model=None,
            max_frames=None, backend=None,
        )

        run_scratch_mode(input_xml, output_xml, cfg, _Args)
        return

    # CLI mode
    parser = build_parser()
    args = parser.parse_args()

    cfg, _ = load_config(explicit_path=args.config, project_dir=PROJECT_DIR)
    setup_logging(cfg.get("LOG_PATH"), quiet=args.quiet)

    try:
        run_cli(args, cfg)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        logging.getLogger(__name__).error(str(e), exc_info=True)
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
