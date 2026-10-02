#!/usr/bin/env bash
#
# > [!AML-DOC-FILE]
# @file        scripts/bundle-runtime.sh
# @description Build the self-contained Python runtime that ships inside the desktop
#              app, so an installed copy needs nothing on the machine but itself.
# @module      scripts/bundle-runtime
# @exports     desktop/src-tauri/runtime/python
# @created     2026-10-01
# @context     The development venv cannot be shipped: its `bin/python3` is a symlink
#              to the Homebrew interpreter that created it, so copying the directory
#              copies a pointer to a file the user does not have. This builds on a
#              relocatable CPython instead — python-build-standalone, the same
#              distribution `uv` installs — and installs the engine into it [E-027].
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/desktop/src-tauri/runtime"
CACHE="$ROOT/.cache/runtime"

PBS_TAG="20260929"
PY_VERSION="3.13.15"

case "$(uname -m)" in
  arm64|aarch64) PBS_ARCH="aarch64-apple-darwin" ;;
  x86_64)        PBS_ARCH="x86_64-apple-darwin" ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

TARBALL="cpython-${PY_VERSION}+${PBS_TAG}-${PBS_ARCH}-install_only_stripped.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${TARBALL}"

mkdir -p "$CACHE"

if [ ! -f "$CACHE/$TARBALL" ]; then
  echo "==> downloading a relocatable CPython ${PY_VERSION}"
  curl -fL --progress-bar -o "$CACHE/$TARBALL.part" "$URL"
  mv "$CACHE/$TARBALL.part" "$CACHE/$TARBALL"
else
  echo "==> using the cached CPython ${PY_VERSION}"
fi

echo "==> extracting into $RUNTIME"
rm -rf "$RUNTIME"
mkdir -p "$RUNTIME"
tar -xzf "$CACHE/$TARBALL" -C "$RUNTIME"

PY="$RUNTIME/python/bin/python3.13"
[ -x "$PY" ] || { echo "the extracted runtime has no interpreter at $PY" >&2; exit 1; }

echo "==> installing the engine and TensorFlow into it"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet "$ROOT/backend"

echo "==> pruning what only a build needs"
SITE="$RUNTIME/python/lib/python3.13/site-packages"
# C++ headers for building against TensorFlow; nothing reads them at runtime.
rm -rf "$SITE/tensorflow/include"
# Static libraries and the test suite of the interpreter itself.
rm -rf "$RUNTIME/python/lib/python3.13/test" "$RUNTIME/python/lib/python3.13/config-3.13-darwin"
find "$RUNTIME/python" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$RUNTIME/python" -name '*.pyc' -delete 2>/dev/null || true

echo "==> verifying the runtime stands on its own"
"$PY" - <<'PYCHECK'
import sys, sysconfig
import tensorflow as tf
import keras
import nnarch
from nnarch.catalog.registry import bootstrap

base = sysconfig.get_paths()["purelib"]
assert "/runtime/python/" in base, f"the interpreter is resolving outside the bundle: {base}"
registry = bootstrap()
print(f"    python     {sys.version.split()[0]}")
print(f"    tensorflow {tf.__version__}")
print(f"    keras      {keras.__version__}")
print(f"    layers     {registry.as_json()['count']}")
PYCHECK

SIZE="$(du -sh "$RUNTIME" | cut -f1)"
echo "==> runtime ready: $SIZE at $RUNTIME"
