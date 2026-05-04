#!/usr/bin/env bash
# One-command launcher for the hybrid multi-agent demo.
#
# What this script does:
#   1. Checks prerequisites (Python venv, Node, foundry-local-sdk).
#   2. Pre-downloads the SLM (phi-4-mini) via foundry-local-sdk.
#      Whisper is handled by faster-whisper (downloads from Hugging Face on
#      first use; cached in ~/.cache/huggingface/).
#   3. Starts the FastAPI backend (port 8000) and Next.js dev server (port 3000).
#   4. Opens the browser.
#
# The Foundry Local SDK handles phi-4-mini inference IN-PROCESS inside the
# Python backend. Whisper transcription runs via faster-whisper (also
# in-process). Neither requires a separate service.
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

WEB_LOG="${WEB_LOG:-/tmp/hybrid-demo-web.log}"
WEB_PORT="${WEB_PORT:-3000}"
NEXT_DEV_ENGINE="${NEXT_DEV_ENGINE:-webpack}"

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
# Pre-warm only the SLM (phi-4-mini) via foundry-local-sdk.
# Whisper is skipped when TRANSCRIPTION_BACKEND=faster-whisper (the default) —
# faster-whisper downloads its own model from Hugging Face on first use.
TRANSCRIPTION_BACKEND="${TRANSCRIPTION_BACKEND:-faster-whisper}"
if [ "$TRANSCRIPTION_BACKEND" = "faster-whisper" ]; then
  info "Transcription backend: faster-whisper (Foundry whisper model will NOT be pre-warmed)"
else
  info "Transcription backend: foundry"
fi

info "Pre-warming edge SLM via foundry-local-sdk (cached after first run)..."
if ! "$PYTHON_BIN" - <<'PYEOF'
import sys, os
import yaml
from pathlib import Path

try:
    from foundry_local_sdk import Configuration, FoundryLocalManager
except ImportError:
  print("[demo] foundry-local-sdk not importable.", flush=True)
  sys.exit(1)

cfg = yaml.safe_load(Path("models.yaml").read_text())
slm = cfg["edge"]["slm"]["model"]

# Only include whisper when using the foundry backend
backend = os.environ.get("TRANSCRIPTION_BACKEND", "faster-whisper").lower()
models_to_warm = [slm]
if backend != "faster-whisper":
    models_to_warm.insert(0, cfg["edge"]["transcription"]["model"])

if FoundryLocalManager.instance is None:
    FoundryLocalManager.initialize(Configuration(app_name="hybrid_demo"))
manager = FoundryLocalManager.instance
manager.download_and_register_eps()

for model_id in models_to_warm:
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
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  die "Node.js/npm not found. Install Node 20 LTS (recommended) and retry."
fi

NODE_VERSION="$(node -v 2>/dev/null || true)"
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
if [ "${NODE_MAJOR:-0}" -ge 22 ]; then
  warn "Detected Node $NODE_VERSION. This repo's Next.js dev server can exit immediately on Node 22 in some environments."
  warn "If web startup fails below, switch to Node 20 LTS and re-run."
fi

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

WEB_PID=""
if curl -sf "http://localhost:${WEB_PORT}" >/dev/null 2>&1; then
  warn "Web UI already running on :${WEB_PORT}; reusing existing process."
else
  STALE_WEB_PID="$(lsof -t -nP -iTCP:${WEB_PORT} -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
  if [ -n "$STALE_WEB_PID" ]; then
    die "Port ${WEB_PORT} is occupied by PID $STALE_WEB_PID, but the web UI is not responding. Stop it (kill $STALE_WEB_PID) and retry."
  fi

  info "Starting web UI (http://localhost:${WEB_PORT}) via Next.js ${NEXT_DEV_ENGINE} dev server..."
  rm -f "$WEB_LOG"
  if [ "$NEXT_DEV_ENGINE" = "webpack" ]; then
    (cd web && npx next dev --webpack --port "$WEB_PORT") >"$WEB_LOG" 2>&1 &
  else
    (cd web && npx next dev --port "$WEB_PORT") >"$WEB_LOG" 2>&1 &
  fi
  WEB_PID=$!
fi

for i in $(seq 1 40); do
  if curl -sf "http://localhost:${WEB_PORT}" >/dev/null 2>&1; then
    break
  fi
  if [ -n "$WEB_PID" ] && ! kill -0 "$WEB_PID" 2>/dev/null; then
    warn "Web process exited during startup. Last log lines:"
    tail -n 80 "$WEB_LOG" || true
    die "Web UI failed to start. Current Node: ${NODE_VERSION:-unknown}."
  fi
  sleep 0.5
done

if ! curl -sf "http://localhost:${WEB_PORT}" >/dev/null 2>&1; then
  warn "Web process did not become ready on :${WEB_PORT}. Last log lines:"
  tail -n 80 "$WEB_LOG" || true
  die "Web UI not reachable at http://localhost:${WEB_PORT}"
fi

info "Opening browser → http://localhost:${WEB_PORT}"
open "http://localhost:${WEB_PORT}" 2>/dev/null || xdg-open "http://localhost:${WEB_PORT}" 2>/dev/null || true
info "Press Ctrl-C to stop."
if [ -n "$WEB_PID" ]; then
  wait "$WEB_PID"
elif [ -n "$BACKEND_PID" ]; then
  warn "Reusing existing web UI process on :${WEB_PORT}; waiting on backend process."
  wait "$BACKEND_PID"
else
  warn "Reusing existing backend and web processes; launcher exiting without taking ownership."
fi
