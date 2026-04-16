# RoboSlate-arm

**Automatic clapperboard reading for macOS.** Point it at your rushes and get scene, take, roll, camera, and a dozen other slate fields written to a JSON sidecar — ready for DaVinci Resolve or Assimilate SCRATCH.

Powered by Apple Vision OCR and runs entirely on-device. Apple Silicon preferred (Neural Engine accelerated), but Intel Macs work too.

For best results, pair it with [Gemma 4](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) (26B, 4-bit, ~15 GB) running locally via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — a second-pass correction layer that fills fields OCR missed and fixes character misreads. Downloaded automatically on first use. Still fully on-device.

---

## What it extracts

| Field | Example |
|-------|---------|
| Scene | `57` |
| Take | `03` |
| Slate number | `215` |
| Roll | `A051` |
| Camera | `A` |
| Director | `Jane Smith` |
| DoP | `Anders Nielsen` |
| Production | `PROJEKT X` |
| Date | `23/5 2023` |
| FPS | `25` |
| Format | `4K` |
| Notes | — |

Each field carries a confidence level (`high` / `medium` / `low`) and a `needs_review` flag for anything RoboSlate-arm isn't certain about.

---

## Requirements

- **macOS 12 or later**
- **Python 3.9+** (ships with macOS; `python3 --version` to check)
- **ffmpeg** — install with [Homebrew](https://brew.sh): `brew install ffmpeg`

---

## Installation

```bash
git clone https://github.com/your-org/roboslate-arm.git
cd roboslate-arm
pip3 install pyobjc-framework-Vision pyobjc-framework-Quartz Pillow
cp config.env.example config.env
```

That's it. No virtual environment needed for most users.

> **Using a virtual environment** (recommended if you have multiple Python projects):
> ```bash
> ./setup.sh
> ```
> `setup.sh` creates a venv, installs dependencies, and copies the default config. When using the venv, prefix all commands below with `venv/bin/` — e.g. `python3 roboslate-arm.py`.

---

## Usage

### Single clip

```bash
python3 roboslate-arm.py --file /path/to/clip.mov
```

Writes a `.roboslate.json` file next to the source video.

### Batch — whole folder

```bash
python3 roboslate-arm.py --batch /Volumes/Rushes/Day01
```

Processes every supported video file (`.mp4 .mov .mxf .r3d .ari .braw`) in the folder. Writes a sidecar next to each clip.

### Batch with a combined CSV report

```bash
python3 roboslate-arm.py --batch /Volumes/Rushes/Day01 --format csv --csv day01.csv
```

Produces a single spreadsheet-ready CSV alongside the per-clip sidecars.

### Common options

| Flag | What it does |
|------|-------------|
| `--force` | Re-scan clips that already have a sidecar |
| `--full-scan` | Check every frame, not just the first confident hit |
| `--stdout` | Print JSON to the terminal instead of writing a file |
| `--quiet` | Suppress progress output |
| `--workers N` | Process N clips in parallel during `--batch` |
| `--keep-frames` | Keep the JPEG frames extracted from the video (useful for debugging) |

Full option list: `python3 roboslate-arm.py --help`

---

## Output format

Every processed clip gets a `.roboslate.json` sidecar written alongside the source video:

```json
{
  "status": "found",
  "result": {
    "overall_confidence": "high",
    "detected_at_seconds": 1.5,
    "needs_review": false,
    "conflicts": []
  },
  "slate": {
    "scene":        { "value": "57",             "confidence": "high"   },
    "take":         { "value": "03",             "confidence": "high"   },
    "slate_number": { "value": "215",            "confidence": "high"   },
    "roll":         { "value": "A051",           "confidence": "high"   },
    "camera":       { "value": "A",              "confidence": "high"   },
    "director":     { "value": "Jane Smith",     "confidence": "high"   },
    "dop":          { "value": "Anders Nielsen", "confidence": "medium" },
    "production":   { "value": "PROJEKT X",      "confidence": "high"   },
    "date":         { "value": "23/5 2023",      "confidence": "medium" },
    "fps":          { "value": null,             "confidence": "low"    },
    "format":       { "value": null,             "confidence": "low"    },
    "notes":        { "value": null,             "confidence": "low"    }
  }
}
```

`status` is `"found"`, `"not_found"`, or `"skipped"` (already processed, `--force` not set).

---

## DaVinci Resolve integration

Copy the script to Resolve's Scripts folder:

```bash
cp resolve/RoboSlate-arm.py \
  ~/Library/Application\ Support/Blackmagic\ Design/DaVinci\ Resolve/Fusion/Scripts/Utility/
```

Open the script in a text editor and set the path to this project at the top:

```python
ROBOSLATE_ARM_DIR = "/path/to/roboslate-arm"
```

Then run it from **Workspace → Scripts → Utility → RoboSlate-arm**.

**What it does:** writes **Scene**, **Shot** (slate number), and **Take** into Resolve's built-in metadata fields — no custom columns needed. If `.roboslate.json` sidecars already exist from a CLI batch run, they are read directly (no re-scanning).

**Clip selection order:**
1. Selected clips on the current timeline (Resolve 18+)
2. All clips on the current timeline — set `PROCESS_ALL_TIMELINE_CLIPS = True` in the script
3. All clips in the current Media Pool folder — set `PROCESS_ALL_MEDIA_POOL_FOLDER = True`

---

## Assimilate SCRATCH integration

Register RoboSlate-arm as a Custom Command in SCRATCH. In the Custom Commands editor, set the executable to:

```
/path/to/roboslate-arm/roboslate-arm.py
```

SCRATCH passes the input and output XML paths automatically. When triggered on a selection, RoboSlate-arm reads each shot, runs the detection pipeline, and writes slate metadata back into SCRATCH. JSON sidecars are also written alongside the media files.

---

## Configuration

Adjust `config.env` to tune scan behaviour. The defaults work well out of the box.

| Setting | Default | Description |
|---------|---------|-------------|
| `SCAN_PHASE1_DURATION` | `60` | Seconds to scan from the start of the clip |
| `SCAN_PHASE1_FPS` | `2` | Frames per second during Phase 1 |
| `SCAN_PHASE2_DURATION` | `30` | Seconds to scan from the end of the clip |
| `MAX_FRAMES` | `16` | Maximum frames to check per clip |
| `EARLY_STOP_CONFIDENCE` | `high` | Stop as soon as a confident slate is found |
| `BATCH_WORKERS` | `1` | Parallel workers for `--batch` |

To extract only specific fields, uncomment the relevant `FIELD_*` lines in `config.env`. If none are set, all fields are extracted.

---

## Gemma 4 VLM escalation (Apple Silicon only, recommended)

Apple Vision OCR works well on clean slates but frequently misses roll stickers, hand-written take numbers, and partially visible fields. Enabling Gemma 4 as a second pass significantly improves results — on our test footage, a large portion of clips needed VLM correction to get complete metadata.

The model runs on-device via the Neural Engine. No API key or internet connection required after the one-time download.

**Install:**

```bash
pip3 install mlx-vlm
```

**Enable in `config.env`:**

```
ENABLE_VLM_ESCALATION=true
VLM_MODEL=mlx-community/gemma-4-26b-a4b-it-4bit
```

On first run, RoboSlate-arm will download Gemma 4 automatically (~15 GB) and save it for reuse. If you have it already downloaded locally, point `VLM_MODEL` at the absolute path instead.

The VLM runs once per clip, only when OCR left fields incomplete or uncertain. Any field it fills is flagged `medium` confidence and the clip marked `needs_review` — it never silently overrides a confident OCR read.

---

## Supported video formats

`.mp4` `.mov` `.mxf` `.r3d` `.ari` `.braw` — and anything else ffmpeg can decode. Use `--ext` to add more:

```bash
python3 roboslate-arm.py --batch /Volumes/Rushes --ext .mp4,.mov,.mkv
```

---

## Debugging OCR

To inspect what Apple Vision sees on a still image:

```bash
python3 vision_test.py /path/to/frame.jpg
```

Saves an annotated copy to `output/` with bounding boxes and confidence scores overlaid.

---

## How it works

1. **Frame extraction** — ffmpeg pulls frames from the first 60 seconds and last 30 seconds of the clip. If nothing is found, a sparse fallback scan covers the rest.
2. **OCR** — Apple Vision reads each frame in accurate mode. A contrast-enhanced retry is attempted on any frame that scores low confidence.
3. **Field classification** — text blobs are matched against known slate field labels (English and Danish). Combined values like `"30-157"` or `"90AB144"` are split automatically.
4. **VLM supplement** *(optional)* — fills null or low-confidence fields using a local vision-language model.
5. **Merging** — results from multiple frames of the same slate are consolidated. Fields that agree across frames are promoted to high confidence; disagreements are flagged.
6. **Output** — JSON sidecar written next to the source video, with optional CSV export.

---

## Limitations

- **macOS only** — Apple Vision is a macOS framework.
- Burn-in timecode or letterbox text can conflict with slate fields. A `FRAME_CROP` config option to exclude these regions is planned.
- Single-character OCR misreads (e.g. `B→8`, `3→0`) that OCR is confident about cannot be corrected without VLM escalation enabled.
- Digital slates (Denecke, Ambient) have not been formally tested, though Apple Vision should handle them.
