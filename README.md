# RoboSlate

Detects clapperboards in video files and extracts production metadata — scene, take, roll, camera, director, DoP, and more — using Claude Vision or paddleOCR. Works as a standalone CLI tool, as an Assimilate SCRATCH custom command or  DaVinci Resolve script, that writes metadata directly back to your shots.

---

## How it works

Each video is scanned by extracting JPEG frames at configurable intervals. Each frame goes through a two-pass pipeline:

1. **Haiku** — fast binary detection: "is there a slate in this frame?" Frames without a slate are skipped cheaply.
2. **Sonnet** — full extraction on frames where a slate is detected. Returns per-field values with confidence scores (`high` / `medium` / `low`).

If confidence is still low after pass 2, a third pass runs on a preprocessed (contrast-enhanced) version of the frame.

Multiple frames of the same slate are merged into one result. Fields that agree across frames are promoted to high confidence; conflicting readings are flagged for review.

---

## Requirements

- Python 3.9 or later
- ffmpeg (with ffprobe) — for frame extraction
- An Anthropic API key — for text extraction

Install ffmpeg on macOS:
```bash
brew install ffmpeg
```

---

## Installation

```bash
git clone <repo-url>
cd RoboSlate
./install.sh
```

The installer:
- Creates a self-contained `venv/` inside the project directory
- Installs `anthropic` and `Pillow` (the only runtime dependencies)
- Copies `config.env.example` → `config.env` if it doesn't exist yet

**Add your API key to `config.env` before running:**
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## CLI usage

```bash
# Single file
venv/bin/python roboslate.py --file /path/to/clip.mov

# Print JSON to terminal instead of writing a file
venv/bin/python roboslate.py --file clip.mov --stdout

# Process an entire folder
venv/bin/python roboslate.py --batch /media/rushes --format csv

# Estimate API cost without making any calls
venv/bin/python roboslate.py --file clip.mov --dry-run

# Force Sonnet for all frames (no early exit)
venv/bin/python roboslate.py --file clip.mov --model claude-sonnet-4-6 --full-scan

# Keep extracted frames for inspection
venv/bin/python roboslate.py --file clip.mov --keep-frames --frames-dir /tmp/frames
```

### All CLI flags

| Flag | Description |
|---|---|
| `--file PATH` | Single video file to process |
| `--batch DIR` | Process all video files in a directory |
| `--output PATH` | JSON output path (default: alongside source video) |
| `--format json\|csv\|both` | Output format (default: `json`) |
| `--stdout` | Print JSON to stdout |
| `--quiet` | Suppress progress output |
| `--model MODEL` | Override `CLAUDE_MODEL` for this run |
| `--backend claude\|local` | Override vision backend for this run |
| `--full-scan` | Disable early exit; scan all frames |
| `--phase1-only` | Only scan the first 60 seconds |
| `--max-frames N` | Override `MAX_FRAMES` for this run |
| `--dry-run` | Estimate cost only; no API calls |
| `--keep-frames` | Keep extracted frame JPEGs after processing |
| `--frames-dir DIR` | Directory for temporary frame files |
| `--config PATH` | Path to a custom config file |
| `--force` | Re-process even if a `.roboslate.json` sidecar already exists |
| `--workers N` | Parallel workers for `--batch` (overrides `BATCH_WORKERS` in config) |

---

## Output

By default, results are written as a `.roboslate.json` sidecar file next to each source video.

```json
{
  "status": "found",
  "result": {
    "overall_confidence": "high",
    "needs_review": false,
    "conflicts": []
  },
  "slate": {
    "scene":        { "value": "57",   "confidence": "high" },
    "take":         { "value": "3",    "confidence": "high" },
    "slate_number": { "value": "215",  "confidence": "high" },
    "roll":         { "value": "A051", "confidence": "high" },
    "camera":       { "value": "A",    "confidence": "high" },
    "director":     { "value": "Jane Smith", "confidence": "medium" }
  }
}
```

Status values:
- `found` — slate detected and fields extracted
- `not_found` — no slate detected after full scan
- `unreadable` — slate detected but no legible fields
- `error` — processing failed (offline media, corrupt file, etc.)

---

## Configuration (`config.env`)

All settings live in `config.env` in the project root. Copy `config.env.example` as a starting point.

### API and models

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Fast/cheap model for binary slate detection
CLAUDE_MODEL=claude-haiku-4-5-20251001

# Model for field extraction (escalated to when Haiku detects a slate)
# Set equal to CLAUDE_MODEL to disable escalation and use only Haiku
ESCALATION_MODEL=claude-sonnet-4-6

# Max response tokens (512 is plenty for slate JSON)
CLAUDE_MAX_TOKENS=512
```

### Scanning strategy

```env
# Phase 1: scan first N seconds at X frames per second
SCAN_PHASE1_DURATION=60
SCAN_PHASE1_FPS=2

# Phase 2: scan last N seconds at X fps (catches slates near end of clip)
SCAN_PHASE2_DURATION=30
SCAN_PHASE2_FPS=2

# Phase 3 (fallback): if nothing found, one frame every N seconds across the whole video
SCAN_PHASE3_INTERVAL=5

# Hard cap on frames sent to Claude per video (cost control)
MAX_FRAMES=16

# Stop scanning once this many consecutive slate frames have been read
CONSISTENT_READINGS_STOP=5

# Exit early as soon as a result at this confidence is found
# Values: high, medium
EARLY_STOP_CONFIDENCE=high
```

### Batch processing

```env
# Number of clips to process in parallel when using --batch.
# Each worker makes its own API calls concurrently.
# Set to 1 to process sequentially. Can also be overridden with --workers N.
BATCH_WORKERS=4
```

### Image settings

```env
# Max image dimension (long edge in pixels) before sending to Claude
MAX_IMAGE_SIZE=1568

# JPEG quality for extracted frames (ffmpeg -q:v scale; 2=best, 5=smallest)
FRAME_JPEG_QUALITY=3
```

### Choosing which fields to extract

List only the fields you need. Comment out any line to disable that field — it will be excluded from the Claude prompt, the merged result, the terminal summary, JSON output, and CSV columns.

```env
FIELD_scene=on
FIELD_take=on
FIELD_slate_number=on
FIELD_roll=on
FIELD_camera=on
FIELD_director=on
FIELD_dop=on
#FIELD_production=on   ← commented out = not extracted
#FIELD_date=on
#FIELD_fps=on
#FIELD_format=on
#FIELD_notes=on
```

If none of these lines are present, all 12 fields are extracted (backwards-compatible default).

---

## Assimilate SCRATCH integration

RoboSlate runs as a SCRATCH custom command. Select one or more shots on the CONstruct. In the menu click **Tools → Custom Commands → RoboSlate** (or whatever title you give it). SCRATCH waits while the script runs, then receives the extracted metadata and writes it back to each shot automatically.

### Setup

Register RoboSlate as a custom command in SCRATCH's System Settings:

1. Open **System Settings → Custom Commands**
2. Click **Add**
3. Fill in the fields:
   - **Title:** RoboSlate (or any name you like)
   - **Type:** Application
   - **File:** `/path/to/RoboSlate/roboslate.py` — click **Set…** to browse
   - **Wait till Finished:** On
   - **Require Shot Selection:** On
   - **Export:** On — **XML:** Selection
4. Click **OK**

The custom command will appear in the **Tools** menu in CONstruct.

### What SCRATCH receives

RoboSlate writes the following metadata keys back to each shot:

| SCRATCH key | Source field |
|---|---|
| `Scene` | `scene-slate` combined (e.g. `57-215`) |
| `SlateNumber` | `slate_number` |
| `Take` | `take` |
| `Roll` | `roll` |
| `Camera` | `camera` |
| `Director` | `director` |
| `DOP` | `dop` |
| `Production` | `production` |
| `ShootDate` | `date` |
| `FrameRate` | `fps` |
| `Format` | `format` |
| `RoboSlate_Confidence` | `high` / `medium` / `low` |
| `RoboSlate_NeedsReview` | `yes` / `no` |
| `RoboSlate_ProcessedAt` | ISO timestamp |

Only fields that are enabled in `config.env` and have a non-null value are written.

I will update this in a later version to match the existing metadata fields

---

## DaVinci Resolve integration

RoboSlate runs as a Resolve script. Open a timeline, optionally select clips, then go to **Workspace → Scripts → Utility → RoboSlate**. The script processes each clip, calls RoboSlate on the source file, and writes the extracted metadata back to the clip in the Media Pool.

### Setup

**1. Install the script**

```bash
./install.sh --resolve
```

This copies `resolve/RoboSlate.py` to Resolve's Scripts folder and injects the correct path to your RoboSlate installation automatically.

**Manual install:** Copy `resolve/RoboSlate.py` to:
```
~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/
```
Then open the file and set `ROBOSLATE_DIR` to the full path of your RoboSlate project.

**2. Restart DaVinci Resolve**

The script will appear under **Workspace → Scripts → Utility → RoboSlate**.

**3. Run it**

With a timeline open, select clips you want to process (or leave nothing selected to process the whole timeline), then run the script. If no timeline is open or timeline has no clips, the current mediapool bin will be processed. Progress is printed to the console. When done, metadata appears in each clip's Clip Attributes under the Metadata tab.

### What metadata is written

| Resolve key | Source |
|---|---|
| `Scene` | `scene` |
| `Shot` | `slate_number` |
| `Take` | `take` |
| `Reel Number` | `roll` |
| `Camera #` | `camera` |
| `Production Name` | `production` |
| `Date Recorded` | `date` |
| `Camera FPS` | `fps` |

Only fields enabled in `config.env` and with a non-null value are written.

---

## Local (offline) backend

An optional PaddleOCR backend runs entirely on-device — no API key or internet connection required.

Install the extra dependencies:
```bash
venv/bin/pip install -r requirements-local.txt
```

Use it:
```bash
venv/bin/python roboslate.py --file clip.mov --backend local
```

Or set it as the default in `config.env`:
```env
VISION_BACKEND=local
```

**Note:** The local backend is significantly slower (~18–33 seconds per frame on CPU) and less accurate than Claude, especially on unusual slate layouts or poor lighting. A fast GPU is strongly recommended for practical use.

---

## Project structure

```
RoboSlate/
├── roboslate.py            # Entry point — CLI and SCRATCH mode
├── config.env              # Your local config (gitignored)
├── config.env.example      # Template to copy from
├── install.sh              # Installer script
├── resolve/
│   └── RoboSlate.py        # DaVinci Resolve script (place in Resolve's Scripts folder)
└── roboslate/
    ├── config.py           # Config loader and field selection
    ├── frames.py           # ffmpeg frame extraction
    ├── vision.py           # Claude Vision API — detection and extraction
    ├── vision_local.py     # PaddleOCR local backend
    ├── preprocessing.py    # Image contrast enhancement for low-confidence frames
    ├── merge.py            # Multi-frame result merging
    ├── output.py           # JSON, CSV, and terminal summary formatting
    └── scratch.py          # SCRATCH XML parsing and writeback
```
