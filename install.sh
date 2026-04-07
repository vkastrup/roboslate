#!/bin/bash
# RoboSlate installer
# Creates a self-contained virtual environment inside the project directory.
# Run once after cloning or unpacking. No admin rights required.
#
# Usage:
#   ./install.sh              — standard install
#   ./install.sh --resolve    — also install the DaVinci Resolve script

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/venv"
INSTALL_RESOLVE=0

for arg in "$@"; do
    if [ "$arg" = "--resolve" ]; then
        INSTALL_RESOLVE=1
    fi
done

echo "================================================"
echo "  RoboSlate installer"
echo "  Project: $SCRIPT_DIR"
echo "================================================"
echo ""

# --- Python check ---
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.9 or later and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "[1/4] Python: $PYTHON_VERSION"

# --- Virtual environment ---
if [ -d "$VENV" ]; then
    echo "[2/4] Virtual environment already exists — updating..."
else
    echo "[2/4] Creating virtual environment at $VENV ..."
    python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip -q
echo "      pip upgraded."

# --- Dependencies ---
echo "[3/4] Installing dependencies..."
"$VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" -q
echo "      anthropic and Pillow installed."

# --- ffmpeg check ---
echo "[4/4] Checking for ffmpeg..."
if command -v ffmpeg &>/dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -1)
    echo "      ffmpeg OK: $FFMPEG_VERSION"
    if command -v ffprobe &>/dev/null; then
        echo "      ffprobe OK."
    else
        echo "      WARNING: ffprobe not found. It is needed for duration detection."
        echo "               Install ffmpeg (includes ffprobe): brew install ffmpeg"
    fi
else
    echo ""
    echo "  WARNING: ffmpeg not found in PATH."
    echo "           RoboSlate requires ffmpeg to extract frames from video."
    echo "           Install it with: brew install ffmpeg"
    echo ""
fi

# --- Config ---
if [ ! -f "$SCRIPT_DIR/config.env" ]; then
    cp "$SCRIPT_DIR/config.env.example" "$SCRIPT_DIR/config.env"
    echo ""
    echo "  Created config.env from template."
    echo "  >>> Add your Anthropic API key to config.env before running. <<<"
else
    echo ""
    echo "  config.env already exists — skipping."
fi

# --- DaVinci Resolve script (optional) ---
if [ "$INSTALL_RESOLVE" = "1" ]; then
    echo ""
    echo "[Resolve] Installing DaVinci Resolve script..."

    RESOLVE_SCRIPTS="$HOME/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility"
    RESOLVE_SCRIPT_SRC="$SCRIPT_DIR/resolve/RoboSlate.py"
    RESOLVE_SCRIPT_DST="$RESOLVE_SCRIPTS/RoboSlate.py"

    if [ ! -f "$RESOLVE_SCRIPT_SRC" ]; then
        echo "      ERROR: resolve/RoboSlate.py not found in project directory."
    else
        mkdir -p "$RESOLVE_SCRIPTS"
        # Inject the correct ROBOSLATE_DIR path into the installed script
        sed "s|ROBOSLATE_DIR = \"/path/to/RoboSlate\"|ROBOSLATE_DIR = \"$SCRIPT_DIR\"|" \
            "$RESOLVE_SCRIPT_SRC" > "$RESOLVE_SCRIPT_DST"
        echo "      Installed to: $RESOLVE_SCRIPT_DST"
        echo "      Restart DaVinci Resolve, then find it under:"
        echo "        Workspace → Scripts → Utility → RoboSlate"
    fi
fi

# --- Done ---
echo ""
echo "================================================"
echo "  Install complete."
echo ""
echo "  Usage:"
echo "    $VENV/bin/python roboslate.py --file /path/to/clip.mov"
echo "    $VENV/bin/python roboslate.py --help"
echo ""
echo "  To install DaVinci Resolve script:"
echo "    ./install.sh --resolve"
echo "================================================"
