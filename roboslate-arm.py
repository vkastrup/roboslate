#!/usr/bin/env python3
# Bootstrap: re-exec with the project venv Python if not already using it.
import os as _os, sys as _sys
_venv_python = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "venv", "bin", "python")
if _os.path.isfile(_venv_python) and not _sys.executable.startswith(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "venv")
):
    _os.execv(_venv_python, [_venv_python] + _sys.argv)
"""
RoboSlate-arm — Local clapperboard detection using Apple Vision OCR.

Detects slates in video files and extracts production metadata entirely
on-device — no API key or network access required.

Usage:
    python3 roboslate-arm.py --file /path/to/clip.mov
    python3 roboslate-arm.py --batch /path/to/folder --format csv
    python3 roboslate-arm.py --help
"""

import argparse
import json
import logging
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from roboslate_arm.config import load_config, get_enabled_fields
from roboslate_arm import frames as frm
from roboslate_arm import vision_apple as vision
from roboslate_arm import output as out
from roboslate_arm import merge as mrg
from roboslate_arm import scratch as scr

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path, quiet):
    level = logging.WARNING if quiet else logging.INFO
    handlers = []

    default_log = log_path or os.path.join(PROJECT_DIR, "logs", "roboslate-arm.log")
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

    frames_dir = getattr(args, "frames_dir", None) or frm.make_temp_dir(cfg.get("TEMP_DIR"))
    os.makedirs(frames_dir, exist_ok=True)
    own_frames_dir = getattr(args, "frames_dir", None) is None

    all_frame_entries = []  # (phase, timestamp, path)
    detections = []

    try:
        schedule = frm.build_scan_schedule(duration, cfg)

        if getattr(args, "phase1_only", False):
            schedule = [(p, t) for p, t in schedule if p == "phase1"]

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

        # Compute enabled fields once; store in scan_cfg for vision module
        scan_cfg = dict(cfg)
        if getattr(args, "full_scan", False):
            scan_cfg["EARLY_STOP_CONFIDENCE"] = "none"

        enabled_fields = get_enabled_fields(scan_cfg)
        scan_cfg["_enabled_fields"] = enabled_fields

        client = vision.build_client()

        reel_hint = out.extract_reel_from_filename(video_path)
        if not getattr(args, "quiet", False):
            reel_note = f" [reel hint: {reel_hint}]" if reel_hint else ""
            print(f"  Scanning {len(entries_to_scan)} frames with Apple Vision OCR{reel_note}...")

        detections = vision.scan_frames(entries_to_scan, client, scan_cfg, reel_hint=reel_hint)

        # --- Phase 3 fallback: sparse scan if nothing found in Phase 1/2 ---
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
                p3_detections = vision.scan_frames(p3_entries[:remaining], client, scan_cfg, reel_hint=reel_hint)
                detections.extend(p3_detections)

        # --- Merge multi-frame detections ---
        merged = mrg.merge_detections(detections, fields=enabled_fields)

        # --- Pass 3a: VLM not_found fallback (optional) ---
        # When OCR finds no slate at all (dark/blurry frames, missing labels),
        # try VLM on frames in order as a last resort.  Stops early once both
        # scene and slate_number are found.  Capped at 5 frames to avoid long
        # runtimes on clips where the slate genuinely isn't there.
        if (
            scan_cfg.get("ENABLE_VLM_ESCALATION", "false").lower() == "true"
            and not merged.get("slate_detected")
            and os.path.isdir(frames_dir)
        ):
            try:
                from roboslate_arm import vision_mlx as vlm_mod
                _vlm_best = None
                _vlm_best_count = 0
                _vlm_tried = 0
                for _fname in sorted(os.listdir(frames_dir)):
                    if not (_fname.endswith(".jpg") and not _fname.endswith("_pre.jpg")):
                        continue
                    if _vlm_tried >= 5:
                        break
                    _vlm_tried += 1
                    _fp = os.path.join(frames_dir, _fname)
                    _vr = vlm_mod.run_vlm_pass(_fp, scan_cfg, pass_num=3)
                    if not _vr.get("slate_detected"):
                        continue
                    _vf = _vr["fields"]
                    _n = sum(1 for v in _vf.values() if v.get("value") is not None)
                    if _n > _vlm_best_count:
                        _vlm_best = _vr
                        _vlm_best_count = _n
                    if _vf.get("scene", {}).get("value") and _vf.get("slate_number", {}).get("value"):
                        break  # core fields found — no need to try more frames

                if _vlm_best:
                    merged = {
                        "slate_detected":    True,
                        "needs_review":      True,
                        "overall_confidence": "low",
                        "fields":            _vlm_best["fields"],
                        "conflicts":         [],
                        "conflict_values":   {},
                        "best_frame":        None,
                        "detection_frames":  1,
                    }
                    log.debug(f"VLM not_found fallback: slate detected ({_vlm_best_count} fields)")
            except Exception as _e:
                log.warning(f"VLM not_found fallback failed: {_e}")

        # --- Pass 3b: VLM post-merge supplement/correction (optional) ---
        # Runs ONCE on the best frame after all OCR frames are merged.
        # Running post-merge (rather than per-frame) prevents VLM hallucinations
        # from conflicting with valid OCR readings across frames.
        #
        # Two modes:
        #   _VLM_SUPPLEMENT_FIELDS — fill fields that are null across all frames
        #   _VLM_OVERRIDE_FIELDS   — correct fields with "low" OCR confidence
        #                            (handles misreads like 3→0, B→8 in scene/slate#)
        #
        # Fields eligible for VLM supplement (null → fill):
        #   roll/camera — sticker values OCR misses when partially occluded
        #   take        — single digits occasionally lost to proximity mismatch
        #   dop         — handwritten name below a ruled line, often missed spatially
        # Fields eligible for VLM correction (low-confidence OCR → override):
        #   scene / slate_number — corrects character misreads (3→0, B→8)
        _VLM_SUPPLEMENT_FIELDS = {"roll", "camera", "take", "dop"}
        _VLM_OVERRIDE_FIELDS   = {"scene", "slate_number"}
        if (
            scan_cfg.get("ENABLE_VLM_ESCALATION", "false").lower() == "true"
            and merged.get("slate_detected")
        ):
            fields_data = merged.get("fields", {})
            missing = [
                f for f in _VLM_SUPPLEMENT_FIELDS
                if f in enabled_fields
                and fields_data.get(f, {}).get("value") is None
            ]
            low_conf = [
                f for f in _VLM_OVERRIDE_FIELDS
                if f in enabled_fields
                and fields_data.get(f, {}).get("confidence") == "low"
            ]
            fields_to_try = list(set(missing + low_conf))
            if fields_to_try:
                best = merged.get("best_frame") or {}
                best_frame_file = best.get("frame_file")
                best_frame_path = os.path.join(frames_dir, best_frame_file) if best_frame_file else None
                if best_frame_path and os.path.isfile(best_frame_path):
                    try:
                        from roboslate_arm import vision_mlx as vlm_mod
                        vlm_result = vlm_mod.run_vlm_pass(best_frame_path, scan_cfg, pass_num=3)
                        supplemented = []
                        for field in fields_to_try:
                            vlm_entry = vlm_result.get("fields", {}).get(field, {})
                            if vlm_entry.get("value") is not None:
                                merged["fields"][field] = vlm_entry
                                merged["needs_review"] = True
                                supplemented.append(field)
                        if supplemented:
                            log.debug(f"VLM post-merge supplemented/corrected {supplemented}")
                    except Exception as e:
                        log.warning(f"VLM post-merge pass failed: {e}")

        status = out.classify_status(merged)

        scan_log = [
            {
                "timestamp":          d.get("timestamp"),
                "phase":              d.get("phase"),
                "frame_file":         d.get("frame_file"),
                "slate_detected":     d.get("slate_detected", False),
                "overall_confidence": d.get("overall_confidence", "low"),
                "model_used":         d.get("model_used"),
                "escalated":          d.get("escalated", False),
                "needs_review":       d.get("needs_review", False),
            }
            for d in detections
        ]

        scan_meta = {
            "duration_seconds": duration,
            "frames_scanned":   len(detections),
            "ocr_calls_made":   detections[-1].get("_total_ocr_calls", len(detections)) if detections else 0,
            "phases_run":       list({d.get("phase") for d in detections}),
            "scan_log":         scan_log,
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
    stdout        = getattr(args, "stdout", False)
    quiet         = getattr(args, "quiet", False)
    output_path   = getattr(args, "output", None) or out.get_default_output_path(video_path)
    fields        = get_enabled_fields(cfg) if cfg else None

    if output_format in ("json", "both"):
        try:
            out.write_json(result, output_path)
            if not quiet:
                print(f"  Written: {output_path}")
        except (PermissionError, OSError) as e:
            if not quiet:
                print(f"  WARNING: Could not write JSON sidecar: {e}", file=sys.stderr)

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

def run_scratch_mode(input_xml, output_xml, cfg):
    from types import SimpleNamespace
    args = SimpleNamespace(
        full_scan=False, phase1_only=False, keep_frames=False,
        frames_dir=None, quiet=False, force=False,
    )

    scratch_input = scr.parse_scratch_xml(input_xml)
    shots = scratch_input["shots"]
    print(f"SCRATCH mode: {len(shots)} shot(s) from project '{scratch_input['project']}'")
    print()

    clip_results = []
    for i, shot in enumerate(shots, 1):
        name = shot.get("name") or shot.get("uuid") or f"shot-{i}"
        print(f"[{i}/{len(shots)}] {name}")

        media = scr.resolve_media_path(shot)
        if media["offline"]:
            print(f"  OFFLINE / not found: {media['path']}", file=sys.stderr)
            clip_results.append({
                "uuid":   shot["uuid"],
                "slot":   shot.get("slot", "0"),
                "layer":  shot.get("layer", "0"),
                "name":   name,
                "status": "error",
                "result": None,
                "error":  "offline",
            })
            continue

        shot_info = {
            "uuid":      shot["uuid"],
            "name":      shot["name"],
            "project":   scratch_input["project"],
            "construct": scratch_input["construct"],
        }

        try:
            result, status = process_file(media["path"], cfg, args, shot_info=shot_info)
            sidecar = out.get_default_output_path(media["path"])
            try:
                out.write_json(result, sidecar)
            except (PermissionError, OSError):
                pass
            clip_results.append({
                "uuid":   shot["uuid"],
                "slot":   shot.get("slot", "0"),
                "layer":  shot.get("layer", "0"),
                "name":   name,
                "status": status,
                "result": result,
            })
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed: {media['path']}: {e}")
            print(f"  ERROR: {e}", file=sys.stderr)
            clip_results.append({
                "uuid":   shot["uuid"],
                "slot":   shot.get("slot", "0"),
                "layer":  shot.get("layer", "0"),
                "name":   name,
                "status": "error",
                "result": None,
                "error":  str(e),
            })

    if output_xml:
        written = scr.build_output_xml(scratch_input, clip_results, output_xml)
        if written:
            print(f"\nOutput XML written: {output_xml}")
        else:
            print("\nNo slates found — output XML not written.")

    scr.print_batch_summary(clip_results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="roboslate-arm",
        description="Detect clapperboards in video and extract production metadata using Apple Vision OCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --file /path/to/clip.mov
  %(prog)s --file clip.mov --format csv
  %(prog)s --batch /media/rushes --format csv
  %(prog)s --file clip.mov --full-scan --keep-frames
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file",  metavar="PATH", help="Video file to analyse.")
    input_group.add_argument("--batch", metavar="DIR",  help="Process all video files in a directory.")

    parser.add_argument("--output",       metavar="PATH",                        help="JSON output path (default: next to source video).")
    parser.add_argument("--format",       choices=["json", "csv", "both"], default="json", help="Output format.")
    parser.add_argument("--stdout",       action="store_true",                   help="Print JSON to stdout.")
    parser.add_argument("--quiet",        action="store_true",                   help="Suppress progress output.")
    parser.add_argument("--ext",          default=".mp4,.mov,.mxf,.r3d,.ari,.braw", help="Extensions for --batch.")
    parser.add_argument("--csv",          metavar="PATH",                        help="CSV path for --batch.")
    parser.add_argument("--full-scan",    action="store_true",                   help="Disable early exit; scan all frames.")
    parser.add_argument("--phase1-only",  action="store_true",                   help="Only scan first 60s.")
    parser.add_argument("--max-frames",   type=int, metavar="N",                 help="Override MAX_FRAMES config.")
    parser.add_argument("--keep-frames",  action="store_true",                   help="Keep extracted frame files.")
    parser.add_argument("--frames-dir",   metavar="DIR",                         help="Temp frames directory.")
    parser.add_argument("--config",       metavar="PATH",                        help="Path to config.env.")
    parser.add_argument("--force",        action="store_true",                   help="Re-process clips even if a sidecar JSON already exists.")
    parser.add_argument("--workers",      type=int, metavar="N",                 help="Parallel workers for --batch (default: BATCH_WORKERS in config, or 1).")

    return parser


def run_cli(args, cfg):
    log = logging.getLogger(__name__)

    if args.max_frames:
        cfg = dict(cfg)
        cfg["MAX_FRAMES"] = str(args.max_frames)

    if args.file:
        sidecar = out.get_default_output_path(args.file)
        if os.path.isfile(sidecar) and not getattr(args, "force", False):
            if getattr(args, "stdout", False):
                with open(sidecar) as _f:
                    print(_f.read(), end="")
            else:
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
                "slate_detected":     (result or {}).get("result", {}).get("slate_found", False),
                "overall_confidence": (result or {}).get("result", {}).get("overall_confidence", "none"),
                "needs_review":       (result or {}).get("result", {}).get("needs_review", False),
                "conflicts":          (result or {}).get("result", {}).get("conflicts", []),
                "fields":             (result or {}).get("slate") or {},
            }}

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
                    return name, vf, cached, cached.get("status", "found"), None
                except Exception:
                    pass

            try:
                result, status = process_file(vf, cfg, args)
                if result is not None:
                    if args.format in ("csv", "both"):
                        with csv_lock:
                            write_outputs(result, vf, args, csv_path=args.csv, cfg=cfg)
                    else:
                        write_outputs(result, vf, args, csv_path=args.csv, cfg=cfg)
                return name, vf, result, status, None
            except Exception as e:
                log.error(f"Failed: {vf}: {e}")
                return name, vf, None, "error", str(e)

        if n_workers == 1:
            for i, vf in enumerate(video_files, 1):
                name = os.path.basename(vf)
                print(f"[{i}/{len(video_files)}] {name}")
                name, vf, result, status, err = _process_one_batch(vf)
                if err:
                    print(f"  ERROR: {err}", file=sys.stderr)
                clip_results.append(_summary_entry(name, result, status))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_process_one_batch, vf): vf for vf in video_files}
                raw_results = {}
                for future in as_completed(futures):
                    orig_vf = futures[future]
                    name, vf, result, status, err = future.result()
                    if err:
                        print(f"  ERROR [{name}]: {err}", file=sys.stderr)
                    raw_results[orig_vf] = (name, vf, result, status)
                for orig_vf in video_files:
                    name, vf, result, status = raw_results[orig_vf]
                    clip_results.append(_summary_entry(name, result, status))

        _print_batch_summary(clip_results, quiet=args.quiet)


def _print_batch_summary(clip_results, quiet=False):
    if quiet:
        return
    found     = sum(1 for r in clip_results if r.get("status") == "found")
    not_found = sum(1 for r in clip_results if r.get("status") == "not_found")
    errors    = sum(1 for r in clip_results if r.get("status") == "error")
    review    = sum(1 for r in clip_results if (r.get("merged") or {}).get("needs_review"))

    print(f"\n{'='*56}")
    print(f"  Batch complete: {len(clip_results)} clip(s)")
    print(f"  Slates found   : {found}")
    print(f"  Not found      : {not_found}")
    if errors:
        print(f"  Errors         : {errors}")
    if review:
        print(f"  Needs review   : {review}")
    print(f"{'='*56}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # SCRATCH mode: positional XML args — must check before argparse runs
    if scr.is_scratch_mode():
        input_xml, output_xml = scr.get_scratch_args()
        cfg, _ = load_config(project_dir=PROJECT_DIR)
        cfg["_PROJECT_DIR"] = PROJECT_DIR
        cfg["TEMP_DIR"] = os.path.join(PROJECT_DIR, "tmp")
        os.makedirs(cfg["TEMP_DIR"], exist_ok=True)
        setup_logging(cfg.get("LOG_PATH"), quiet=False)
        try:
            run_scratch_mode(input_xml, output_xml, cfg)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        return

    parser = build_parser()
    args = parser.parse_args()

    cfg, _ = load_config(explicit_path=args.config, project_dir=PROJECT_DIR)
    cfg["_PROJECT_DIR"] = PROJECT_DIR
    cfg["TEMP_DIR"] = os.path.join(PROJECT_DIR, "tmp")
    os.makedirs(cfg["TEMP_DIR"], exist_ok=True)
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
