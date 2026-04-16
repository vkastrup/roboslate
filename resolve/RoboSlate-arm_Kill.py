"""
RoboSlate-arm — Kill / Cancel running batch.

Place this file alongside RoboSlate-arm.py in Resolve's Utility scripts folder:
  ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/

Run via Workspace → Scripts → Utility → RoboSlate-arm_Kill

What it does:
  1. Creates /tmp/roboslate-arm_cancel  — the main script checks this file before
     dispatching each clip and stops cleanly when it exists.
  2. Sends SIGTERM to all running roboslate-arm.py subprocesses so the current
     clip's OCR is also interrupted immediately.
"""

import os
import subprocess

_CANCEL_FILE = "/tmp/roboslate-arm_cancel"


def main():
    print("RoboSlate-arm — Kill")
    print("=" * 40)

    # Write cancel sentinel — the main script polls this between clips.
    try:
        with open(_CANCEL_FILE, "w") as f:
            f.write("cancel\n")
        print(f"  Cancel file written: {_CANCEL_FILE}")
    except Exception as e:
        print(f"  WARNING: could not write cancel file: {e}")

    # Kill any running roboslate-arm subprocesses.
    result = subprocess.run(
        ["pkill", "-TERM", "-f", "roboslate-arm.py"],
        capture_output=True,
    )
    if result.returncode == 0:
        print("  Sent SIGTERM to roboslate-arm.py processes.")
    else:
        print("  No roboslate-arm.py processes found (may have already finished).")

    print("=" * 40)
    print("  Done. The main script will stop after the current clip completes.")
    print()


main()
