#!/usr/bin/env bash
# One-command launcher for the hybrid multi-agent demo.
#
# What this script does:
#   1. Checks prerequisites (Python venv, Node, foundry-local-sdk).
#   2. Pre-downloads the edge models (whisper-large-v3-turbo, phi-4)
#      via foundry-local-sdk so the first backend request is instant.
#      Models are cached in ~/.foundry-local and are only downloaded once.
#   3. Starts the FastAPI backend (port 8000) and Next.js dev server (port 3000).
#   4. Opens the browser.
#
# The Foundry Local SDK runs inference IN-PROCESS inside the Python backend —
# there is no separate service to start.
set -euo pipefail

cd "$(dirname "$0")/.."

# Use the project venv by default when present, so installs and runtime match.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; NC='\033[0m'
info()    { echo -e "${GREEN}[demo]${NC} $*"; }
warn()    { echo -e "${YELLOW}[demo]${NC} $*"; }
die()     { echo -e "${RED}[demo] ERROR:${NC} $*" >&2; exit 1; }

# ── .env ────────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  cp .env.example .env
  warn "Created .env from .env.example."
  warn "Edit it to set FOUNDRY_PROJECT_ENDPOINT (cloud agents) before running."
fi
set -a && . ./.env && set +a

# ── Python SDK check ─────────────────────────────────────────────────────────
info "Using Python interpreter: $PYTHON_BIN"
if ! "$PYTHON_BIN" -c "import hybrid_demo" 2>/dev/null; then
  die "hybrid_demo package not found in $PYTHON_BIN. Run: $PYTHON_BIN -m pip install -e '.[local,dev]'"
fi
if ! "$PYTHON_BIN" -c "import foundry_local_sdk" 2>/dev/null; then
  die "foundry-local-sdk not installed. Install with: $PYTHON_BIN -m pip install -e '.[local]'"
fi

# ── Foundry Local model pre-download ─────────────────────────────────────────
# The foundry-local-sdk handles download + inference in-process (no separate
# service needed). We pre-warm both models here so the first demo request is
# instant rather than blocking for several minutes on a cold download.
info "Pre-warming edge models via foundry-local-sdk (cached after first run)..."
if ! "$PYTHON_BIN" - <<'PYEOF'
import sys
import yaml
from pathlib import Path

try:
    from foundry_local_sdk import Configuration, FoundryLocalManager
except ImportError:
  print("[demo] foundry-local-sdk not importable.", flush=True)
  sys.exit(1)

cfg = yaml.safe_load(Path("models.yaml").read_text())
whisper = cfg["edge"]["transcription"]["model"]
slm     = cfg["edge"]["slm"]["model"]

if FoundryLocalManager.instance is None:
    FoundryLocalManager.initialize(Configuration(app_name="hybrid_demo"))
manager = FoundryLocalManager.instance
manager.download_and_register_eps()

for model_id in (whisper, slm):
    print(f"[demo]   checking {model_id}...", flush=True)
    model = manager.catalog.get_model(model_id)
    if model is None:
        print(f"[demo]   ! model '{model_id}' not found in local catalog.", flush=True)
        sys.exit(1)
    model.download()   # no-op if already cached
    print(f"[demo]   ✓ {model_id} ready", flush=True)
PYEOF
then
  die "Model pre-warm failed. Fix local runtime/model configuration before starting the demo."
fi

# ── Node check ───────────────────────────────────────────────────────────────
if [ ! -d web/node_modules ]; then
  info "node_modules not found — running npm install..."
  (cd web && npm install --no-audit --no-fund)
fi

# ── Start services ────────────────────────────────────────────────────────────
cleanup() {
  info "Shutting down..."
  [ -n "${BACKEND_PID:-}" ] && kill "${BACKEND_PID}" 2>/dev/null || true
  [ -n "${WEB_PID:-}" ] && kill "${WEB_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

BACKEND_PID=""
if curl -sf http://localhost:8000/healthz &>/dev/null; then
  warn "Backend already running on :8000; reusing existing process."
else
  STALE_PID="$(lsof -t -nP -iTCP:8000 -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
  if [ -n "$STALE_PID" ]; then
    die "Port 8000 is occupied by PID $STALE_PID, but /healthz is not responding. Stop it (kill $STALE_PID) and retry."
  fi
  info "Starting backend (http://localhost:8000)..."
  "$PYTHON_BIN" -m uvicorn hybrid_demo.ag_ui_server:app --host 0.0.0.0 --port 8000 &
  BACKEND_PID=$!
fi

# Wait for backend to be ready before opening browser.
for i in $(seq 1 40); do
  if curl -sf http://localhost:8000/healthz &>/dev/null; then
    break
  fi
  if [ -n "$BACKEND_PID" ] && ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    die "Backend failed to start. Check port 8000 and backend logs."
  fi
  sleep 0.5
done

if ! curl -sf http://localhost:8000/healthz &>/dev/null; then
  die "Backend not reachable at http://localhost:8000/healthz"
fi

info "Starting web UI (http://localhost:3000)..."
(cd web && npm run dev -- --port 3000) &
WEB_PID=$!

info "Opening browser → http://localhost:3000"
open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || true
info "Press Ctrl-C to stop."
wait
