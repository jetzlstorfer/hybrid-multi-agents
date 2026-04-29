#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

stage="${1:-all}"
shift || true

case "$stage" in
  transcript)
    targets=(tests/test_transcription_agent.py)
    ;;
  pii)
    targets=(tests/test_pii_agent.py)
    ;;
  redaction)
    targets=(tests/test_redaction.py)
    ;;
  summary)
    targets=(tests/test_summary_agent.py)
    ;;
  rehydration)
    targets=(tests/test_rehydration_agent.py)
    ;;
  cloud)
    targets=(tests/test_cloud_agents.py)
    ;;
  all)
    targets=(
      tests/test_transcription_agent.py
      tests/test_pii_agent.py
      tests/test_redaction.py
      tests/test_summary_agent.py
      tests/test_rehydration_agent.py
      tests/test_cloud_agents.py
    )
    ;;
  *)
    echo "Usage: $0 {transcript|pii|redaction|summary|rehydration|cloud|all} [pytest args...]" >&2
    exit 2
    ;;
esac

exec "$PYTHON_BIN" -m pytest -q "${targets[@]}" "$@"
