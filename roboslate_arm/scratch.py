"""
SCRATCH integration for RoboSlate-arm.

SCRATCH custom commands receive TWO positional arguments:
  sys.argv[1] — path to the input XML (read-only; contains selected shots)
  sys.argv[2] — path to the output XML (write here to update SCRATCH)

The input XML structure (selection mode):
  <scratch project="..." watch_folder="...">
    <selection group="..." construct="...">
      <shot uuid="..." frame_no="..." frame_file="...">
        <file>...</file>
        <name>...</name>
        <fps>...</fps>
        <length>...</length>
        <metadata>
          <dataitem><key>...</key><value>...</value></dataitem>
        </metadata>
      </shot>
    </selection>
  </scratch>

RoboSlate-arm reads the shot file paths, runs the detection pipeline on each,
then writes an update XML with the extracted metadata written back to each shot.
"""

import os
import sys
from datetime import datetime, timezone
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# SCRATCH mode detection
# ---------------------------------------------------------------------------

def is_scratch_mode():
    """
    Return True if called by SCRATCH (first positional arg is an XML file).
    """
    if len(sys.argv) < 2:
        return False
    candidate = sys.argv[1]
    if candidate.startswith("-"):
        return False
    return candidate.lower().endswith(".xml") and os.path.isfile(candidate)


def get_scratch_args():
    """Return (input_xml_path, output_xml_path|None)."""
    input_xml = sys.argv[1] if len(sys.argv) >= 2 else None
    output_xml = sys.argv[2] if len(sys.argv) >= 3 else None
    return input_xml, output_xml


# ---------------------------------------------------------------------------
# Input XML parsing
# ---------------------------------------------------------------------------

def parse_scratch_xml(xml_path):
    """
    Parse the SCRATCH input XML and return a structured dict.

    Returns:
        {
            "project": str,
            "project_path": str,
            "watch_folder": str,
            "group": str,
            "construct": str,
            "shots": [
                {
                    "uuid": str,
                    "file": str,         # media file path
                    "name": str,
                    "fps": str,
                    "length": str,
                    "frame_no": str,
                    "frame_file": str,   # path to current frame
                    "reel_id": str,
                    "timecode": str,
                    "metadata": {key: value},  # existing SCRATCH metadata
                },
                ...
            ]
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {
        "project":      root.get("project", ""),
        "project_path": root.get("project_path", ""),
        "watch_folder": root.get("watch_folder", ""),
        "datetime":     root.get("datetime", ""),
        "version":      root.get("version", ""),
        "group":        "",
        "construct":    "",
        "shots":        [],
    }

    selection = root.find("selection")
    if selection is not None:
        result["group"]     = selection.get("group", "")
        result["construct"] = selection.get("construct", "")
        shot_elements = selection.findall("shot")
    else:
        shot_elements = root.findall(".//shot")

    for shot_el in shot_elements:
        shot = {
            "uuid":       shot_el.get("uuid", ""),
            "slot":       shot_el.get("slot", ""),
            "layer":      shot_el.get("layer", "0"),
            "frame_no":   shot_el.get("frame_no", "0"),
            "frame_file": shot_el.get("frame_file", ""),
            "file":       _text(shot_el, "file"),
            "name":       _text(shot_el, "name"),
            "reel_id":    _text(shot_el, "reel_id"),
            "fps":        _text(shot_el, "fps"),
            "length":     _text(shot_el, "length"),
            "timecode":   _text(shot_el, "timecode"),
            "metadata":   {},
        }

        for item in shot_el.findall(".//metadata/dataitem"):
            key = _text(item, "key")
            val = _text(item, "value")
            if key:
                shot["metadata"][key] = val or ""

        if not shot["file"] and shot["frame_file"]:
            shot["file"] = shot["frame_file"]

        result["shots"].append(shot)

    return result


def _text(element, tag):
    """Return the text of a child element, or empty string if absent."""
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return ""


# ---------------------------------------------------------------------------
# Media path resolution
# ---------------------------------------------------------------------------

def resolve_media_path(shot):
    """
    Resolve a shot's file path to a usable video/media path.

    Handles:
      - Video containers (.mov, .mp4, .mxf, .r3d, etc.) — returned as-is
      - Frame sequences (e.g. /media/A001.0001.exr) — returns directory

    Returns:
        {
            "path": str,               # usable path for ffmpeg
            "is_sequence": bool,
            "sequence_dir": str|None,  # directory if image sequence
            "offline": bool,           # True if file not found on disk
        }
    """
    file_path = shot.get("file") or shot.get("frame_file") or ""

    if not file_path:
        return {"path": "", "is_sequence": False, "sequence_dir": None, "offline": True}

    video_exts = {".mov", ".mp4", ".mxf", ".avi", ".r3d", ".braw", ".ari", ".arx", ".mkv"}
    _, ext = os.path.splitext(file_path.lower())

    offline = not os.path.exists(file_path)

    if ext in video_exts:
        return {
            "path": file_path,
            "is_sequence": False,
            "sequence_dir": None,
            "offline": offline,
        }

    # Image sequence — use the directory
    directory = os.path.dirname(file_path)
    return {
        "path": directory if os.path.isdir(directory) else file_path,
        "is_sequence": True,
        "sequence_dir": directory,
        "offline": not os.path.isdir(directory),
    }


# ---------------------------------------------------------------------------
# Output XML generation
# ---------------------------------------------------------------------------

# Map RoboSlate-arm field names → SCRATCH metadata key names.
# "scene" and "slate_number" are handled separately in _write_slate_metadata()
# because SCRATCH combines them as "scene-slate" in the Scene field.
FIELD_TO_SCRATCH_KEY = {
    "take":       "Take",
    "roll":       "Roll",
    "camera":     "Camera",
    "director":   "Director",
    "dop":        "DOP",
    "production": "Production",
    "date":       "ShootDate",
    "fps":        "FrameRate",
    "format":     "Format",
    "notes":      "Notes",
}


def build_output_xml(scratch_input, clip_results, output_xml_path):
    """
    Write a SCRATCH update XML that writes extracted slate metadata back to shots.

    Uses the slot-based update hierarchy that SCRATCH expects:
        <scratch action="update">
            <groups>
                <group name="...">
                    <constructs>
                        <construct name="...">
                            <slots>
                                <slot index="...">
                                    <shots>
                                        <shot layer="...">
                                            <metadata>...</metadata>
                                        </shot>
                                    </shots>
                                </slot>
                            </slots>
                        </construct>
                    </constructs>
                </group>
            </groups>
        </scratch>

    Args:
        scratch_input:   Parsed SCRATCH input dict (from parse_scratch_xml).
        clip_results:    List of dicts, one per shot, each with:
                         - "uuid":   shot UUID
                         - "slot":   timeline slot index
                         - "layer":  layer within slot
                         - "status": "found" | "not_found" | "unreadable" | "error"
                         - "result": v2 result dict (has result["slate"] and result["result"])
        output_xml_path: Where to write the output XML.

    Returns:
        Path to written XML, or None if no found clips to write.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    root = ET.Element("scratch")
    root.set("action", "update")

    group_name     = scratch_input.get("group", "")
    construct_name = scratch_input.get("construct", "")

    groups_el     = ET.SubElement(root, "groups")
    group_el      = ET.SubElement(groups_el, "group")
    group_el.set("name", group_name)
    constructs_el = ET.SubElement(group_el, "constructs")
    construct_el  = ET.SubElement(constructs_el, "construct")
    construct_el.set("name", construct_name)
    slots_el = ET.SubElement(construct_el, "slots")

    wrote_any = False

    for cr in clip_results:
        if cr.get("status") != "found":
            continue

        v2_result = cr.get("result") or {}
        fields    = v2_result.get("slate") or {}
        res_meta  = v2_result.get("result") or {}

        if not fields:
            continue

        slot_el  = ET.SubElement(slots_el, "slot")
        slot_el.set("index", str(cr.get("slot", "0")))
        shots_el = ET.SubElement(slot_el, "shots")
        shot_el  = ET.SubElement(shots_el, "shot")
        shot_el.set("layer", str(cr.get("layer", "0")))

        metadata_el = ET.SubElement(shot_el, "metadata")

        confidence = res_meta.get("overall_confidence", "low")
        _add_dataitem(metadata_el, "RoboSlate_Confidence",  confidence)
        _add_dataitem(metadata_el, "RoboSlate_ProcessedAt", timestamp)
        _add_dataitem(metadata_el, "RoboSlate_NeedsReview", "yes" if res_meta.get("needs_review") else "no")

        _write_slate_metadata(metadata_el, fields)
        wrote_any = True

    if not wrote_any:
        return None

    _indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)
    return output_xml_path


def build_standalone_xml(clip_results, output_path):
    """
    Write a SCRATCH update XML from a list of (file_path, result_dict) pairs,
    where result_dict is the v2 sidecar JSON structure.

    Uses the filename stem as the construct name so SCRATCH can match clips by name.
    Only clips with status "found" are included.

    Args:
        clip_results: List of (file_path, result_dict) tuples.
        output_path:  Where to write the XML file.

    Returns:
        Path to written XML, or None if there are no found clips to write.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    root = ET.Element("scratch")
    root.set("action", "update")
    groups_el     = ET.SubElement(root, "groups")
    group_el      = ET.SubElement(groups_el, "group")
    group_el.set("name", "RoboSlate")
    constructs_el = ET.SubElement(group_el, "constructs")

    wrote_any = False

    for file_path, result in clip_results:
        if (result or {}).get("status") != "found":
            continue

        fields   = (result.get("slate") or {})
        res_meta = (result.get("result") or {})

        construct_name = os.path.splitext(os.path.basename(file_path))[0]
        construct_el = ET.SubElement(constructs_el, "construct")
        construct_el.set("name", construct_name)
        slots_el = ET.SubElement(construct_el, "slots")
        slot_el  = ET.SubElement(slots_el, "slot")
        slot_el.set("index", "0")
        shots_el = ET.SubElement(slot_el, "shots")
        shot_el  = ET.SubElement(shots_el, "shot")
        shot_el.set("layer", "0")
        metadata_el = ET.SubElement(shot_el, "metadata")

        _add_dataitem(metadata_el, "RoboSlate_Confidence",  res_meta.get("overall_confidence", ""))
        _add_dataitem(metadata_el, "RoboSlate_ProcessedAt", timestamp)
        _add_dataitem(metadata_el, "RoboSlate_NeedsReview", "yes" if res_meta.get("needs_review") else "no")

        _write_slate_metadata(metadata_el, fields)
        wrote_any = True

    if not wrote_any:
        return None

    _indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    return output_path


def _write_slate_metadata(metadata_el, fields):
    """Write slate field dataitems to a SCRATCH metadata element.

    All fields are always written, even when no value was extracted — SCRATCH
    creates the metadata key so the user can type the value directly without
    having to add the field themselves. Empty values are written as "".
    """
    scene_value = (fields.get("scene") or {}).get("value")
    slate_value = (fields.get("slate_number") or {}).get("value")
    if scene_value and slate_value:
        _add_dataitem(metadata_el, "Scene", f"{scene_value}-{slate_value}")
    elif scene_value:
        _add_dataitem(metadata_el, "Scene", scene_value)
    elif slate_value:
        _add_dataitem(metadata_el, "Scene", slate_value)
    else:
        _add_dataitem(metadata_el, "Scene", "")
    _add_dataitem(metadata_el, "SlateNumber", slate_value or "")

    for rs_field, scratch_key in FIELD_TO_SCRATCH_KEY.items():
        field_data = fields.get(rs_field) or {}
        if not isinstance(field_data, dict):
            field_data = {}
        value = field_data.get("value")
        _add_dataitem(metadata_el, scratch_key, str(value) if value is not None else "")
        if value is not None and field_data.get("confidence", "low") == "low":
            _add_dataitem(metadata_el, f"{scratch_key}_LowConfidence", "yes")


def _add_dataitem(parent, key, value):
    item = ET.SubElement(parent, "dataitem")
    k = ET.SubElement(item, "key")
    k.text = key
    v = ET.SubElement(item, "value")
    v.text = value


def _indent_xml(elem, level=0):
    """Add pretty-print indentation to an ElementTree in place."""
    indent = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():  # noqa: F821
            child.tail = indent  # noqa: F821
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


# ---------------------------------------------------------------------------
# Batch summary
# ---------------------------------------------------------------------------

def print_batch_summary(clip_results, quiet=False):
    """
    Print a summary of SCRATCH batch processing results.
    """
    if quiet:
        return

    total      = len(clip_results)
    found      = sum(1 for r in clip_results if r.get("status") == "found")
    not_found  = sum(1 for r in clip_results if r.get("status") == "not_found")
    unreadable = sum(1 for r in clip_results if r.get("status") == "unreadable")
    errors     = sum(1 for r in clip_results if r.get("status") == "error")
    needs_review = sum(
        1 for r in clip_results
        if (r.get("result") or {}).get("result", {}).get("needs_review")
    )

    print(f"\n{'='*60}")
    print(f"  RoboSlate-arm — SCRATCH Batch ({total} shots)")
    print(f"{'='*60}")
    print(f"  Slate found          : {found}")
    print(f"  No slate detected    : {not_found}")
    if unreadable:
        print(f"  Slate unreadable     : {unreadable}")
    print(f"  Errors               : {errors}")
    if needs_review:
        print(f"  Flagged for review   : {needs_review}  <- low/conflicting confidence")
    print(f"{'='*60}")

    if errors:
        print("\n  ERRORS:")
        for r in clip_results:
            if r.get("status") == "error":
                print(f"    [{r.get('name', '?')}] {r.get('error', 'unknown error')}")

    if needs_review:
        print("\n  NEEDS REVIEW:")
        for r in clip_results:
            res_meta = (r.get("result") or {}).get("result", {})
            if res_meta.get("needs_review"):
                conflicts = res_meta.get("conflicts", [])
                flag = f" [CONFLICT: {', '.join(conflicts)}]" if conflicts else " [low confidence]"
                print(f"    [{r.get('name', '?')}]{flag}")

    print()
