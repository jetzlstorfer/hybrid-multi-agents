"""Download a short clip of the Patientengespräch sample.

The clip is intentionally NOT committed to the repo (redistribution concerns).
Run this once before a live demo:

    python scripts/fetch_sample_audio.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

URL = "https://www.youtube.com/watch?v=bhEmB1NTUpk"
OUT = Path("samples/audio/patient.wav")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "-x",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        "-t 90",  # ≤90s clip
        "-o",
        str(OUT.with_suffix(".%(ext)s")),
        URL,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("yt-dlp not found. Install with: pip install -e .[dev]", file=sys.stderr)
        return 1
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
