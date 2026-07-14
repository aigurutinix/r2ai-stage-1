#!/usr/bin/env bash
# Rebuild the backend Python virtual environment from scratch.
set -euo pipefail

cd "$(dirname "$0")"

ENV_DIR="venv"

echo ">>> Removing old environment ($ENV_DIR) ..."
rm -rf "$ENV_DIR"

echo ">>> Creating fresh environment with python3.12 ..."
python3.12 -m venv "$ENV_DIR"

echo ">>> Upgrading pip ..."
"./$ENV_DIR/bin/python" -m pip install --upgrade pip

echo ">>> Installing requirements ..."
"./$ENV_DIR/bin/python" -m pip install -r requirements.txt

echo ">>> Verifying protobuf runtime_version ..."
"./$ENV_DIR/bin/python" - <<'PY'
from google.protobuf import runtime_version
print("protobuf runtime_version OK")
PY

echo ">>> DONE"
