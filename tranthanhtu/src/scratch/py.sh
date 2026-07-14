#!/usr/bin/env bash
# Generic runner: executes a python script with the project venv interpreter.
# Usage: bash scratch/py.sh <script.py> [args...]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="backend/.venv/bin/python"
[ -x "$PY" ] || PY="backend/venv/bin/python"
[ -x "$PY" ] || PY="python3"
exec "$PY" "$@"
