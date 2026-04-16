#!/usr/bin/env bash
# RoboSlate-arm setup script
# Run once after cloning to create the venv, install deps, and optionally
# download a VLM model for Pass 3 escalation.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== RoboSlate-arm setup ==="
echo ""

# 1. Virtual environment
if [ ! -d venv ]; then
    python3 -m venv venv
    echo "Created venv."
else
    echo "venv already exists — skipping creation."
fi

# 2. Python dependencies
echo "Installing dependencies …"
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q -r requirements.txt
echo "Dependencies installed."

# 3. Config file
if [ ! -f config.env ]; then
    cp config.env.example config.env
    echo "Created config.env from template — edit it to adjust settings."
else
    echo "config.env already exists — skipping."
fi

echo ""

# 4. Optional: VLM escalation
echo "VLM escalation uses a local vision-language model to fill fields that"
echo "Apple Vision OCR missed (roll sticker, camera, take, DoP, and low-confidence"
echo "scene/slate corrections). Requires mlx-vlm and a model download."
echo ""
read -r -p "Set up VLM escalation? [y/N] " vlm_ans
if [[ "$vlm_ans" =~ ^[Yy]$ ]]; then
    echo "Installing mlx-vlm …"
    venv/bin/pip install -q mlx-vlm
    echo "mlx-vlm installed."
    echo ""
    venv/bin/python - <<'PYEOF'
import os, sys
sys.path.insert(0, ".")
from roboslate_arm.config import load_config
from roboslate_arm import vision_mlx as vlm

cfg, _ = load_config(project_dir=".")
cfg["_PROJECT_DIR"] = os.path.abspath(".")

model_path = cfg.get("VLM_MODEL", "mlx-community/gemma-4-26b-a4b-it-4bit")

if not vlm._is_hf_repo_id(model_path):
    print(f"VLM_MODEL is already a local path: {model_path}")
    print("No download needed — make sure ENABLE_VLM_ESCALATION=true in config.env.")
    sys.exit(0)

models_dir = cfg.get("MLX_MODELS_DIR") or os.path.join(".", "mlx", "models")
models_dir = os.path.abspath(models_dir)
os.makedirs(models_dir, exist_ok=True)

try:
    local = vlm._ensure_model_downloaded(model_path, models_dir)
    print(f"\nModel ready at:\n  {local}")
    print(f"\nTo activate, update config.env:")
    print(f"  ENABLE_VLM_ESCALATION=true")
    print(f"  VLM_MODEL={local}")
except RuntimeError as e:
    print(f"\n{e}")
    sys.exit(1)
PYEOF
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run:"
echo "  venv/bin/python roboslate-arm.py --file /path/to/clip.mov"
echo "  venv/bin/python roboslate-arm.py --batch /path/to/rushes --format csv --csv results.csv"
