#!/usr/bin/env bash
# Rebuild the article corpus from HF vbpl-vn using the (expanded) allowlist.
# Streaming + regex only (no GPU/embedding) — safe to run in background.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="backend/.venv/bin/python"
[ -x "$PY" ] || PY="backend/venv/bin/python"
[ -x "$PY" ] || PY="python3"
echo "Using interpreter: $PY"

"$PY" backend/build_corpus.py --mode allowlist "$@"
