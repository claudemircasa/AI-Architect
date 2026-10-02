#!/usr/bin/env bash
# > [!AML-DOC-FILE]
# @file        scripts/setup-backend.sh
# @description Creates the backend virtual environment on a TensorFlow-compatible
#              interpreter and installs the engine in editable mode.
# @module      scripts
# @exports     (executable script)
# @created     2026-09-30
# @context     Enforces ERL E-001: TensorFlow 2.21 ships no wheel for Python 3.14,
#              so the interpreter is pinned to 3.13 and verified before any install.

set -euo pipefail

REQUIRED_MINOR=13
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.13}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/backend/.venv"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "error: interpreter not found: ${PYTHON_BIN}" >&2
  echo "install it with:  brew install python@3.${REQUIRED_MINOR}" >&2
  echo "or point PYTHON_BIN at an existing Python 3.${REQUIRED_MINOR}" >&2
  exit 1
fi

MINOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info[1])')"
if [[ "${MINOR}" != "${REQUIRED_MINOR}" ]]; then
  echo "error: ${PYTHON_BIN} is Python 3.${MINOR}, but TensorFlow 2.21 requires 3.${REQUIRED_MINOR}" >&2
  exit 1
fi

echo "==> creating virtual environment at ${VENV}"
"${PYTHON_BIN}" -m venv "${VENV}"

echo "==> upgrading pip"
"${VENV}/bin/python" -m pip install --quiet --upgrade pip

echo "==> installing the engine and its dependencies (downloads TensorFlow, ~500MB)"
"${VENV}/bin/python" -m pip install --editable "${ROOT}/backend"

echo "==> verifying the installation"
"${VENV}/bin/python" - <<'PY'
import keras, sys, tensorflow as tf
from nnarch.catalog import bootstrap
print(f"    python     {'.'.join(map(str, sys.version_info[:3]))}")
print(f"    tensorflow {tf.__version__}")
print(f"    keras      {keras.__version__}")
print(f"    layers     {len(bootstrap().all())} registered")
PY

echo "==> done. start the engine with:  ./scripts/run-engine.sh"
