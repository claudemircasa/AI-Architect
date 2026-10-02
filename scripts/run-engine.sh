#!/usr/bin/env bash
# > [!AML-DOC-FILE]
# @file        scripts/run-engine.sh
# @description Starts the backend engine on port 8756 for local development.
# @module      scripts
# @exports     (executable script)
# @created     2026-09-30
# @context     During development the engine is started by hand; in the packaged
#              app the Tauri shell spawns it as a sidecar [amm: E.5].

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/backend/.venv"
PORT="${PORT:-8756}"

if [[ ! -x "${VENV}/bin/nnarch-engine" ]]; then
  echo "error: engine not installed. run ./scripts/setup-backend.sh first" >&2
  exit 1
fi

exec "${VENV}/bin/nnarch-engine" --port "${PORT}" "$@"
