"""Pytest configuration: ensure the in-repo models.yaml is used."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HYBRID_DEMO_MODELS_FILE", str(ROOT / "models.yaml"))
os.environ.setdefault("FOUNDRY_PROJECT_ENDPOINT", "http://test.invalid")

# Make `import hybrid_demo` work without `pip install -e .` for fast iteration.
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
