#!/usr/bin/env bash
#
# > [!AML-DOC-FILE]
# @file        scripts/install-runtime.sh
# @description Install the engine's Python runtime into a directory the app owns,
#              downloading the interpreter and TensorFlow as it goes.
# @module      scripts/install-runtime
# @exports     <target>/python  (a self-contained interpreter with the engine in it)
# @created     2026-10-02
# @context     This is what `bundle-runtime.sh` does, aimed at a directory chosen at
#              run time rather than at the build tree, so the installer does not have
#              to carry a gigabyte of TensorFlow [E-045]. The app runs it on first
#              launch and reports its progress; every line it prints beginning with
#              `STEP ` is meant to be read by a person watching a progress screen.
#
#              It is a script rather than Rust on purpose: downloading, unpacking and
#              running pip are things a shell does well, and this one is the same
#              sequence the build-time script already proves out.
#
set -euo pipefail

TARGET="${1:?usage: install-runtime.sh <target-directory> [engine-source]}"
ENGINE_SRC="${2:-}"

PBS_TAG="20260929"
PY_VERSION="3.13.15"

case "$(uname -m)" in
  arm64|aarch64) PBS_ARCH="aarch64-apple-darwin" ;;
  x86_64)        PBS_ARCH="x86_64-apple-darwin" ;;
  *) echo "FAIL unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

TARBALL="cpython-${PY_VERSION}+${PBS_TAG}-${PBS_ARCH}-install_only_stripped.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${TARBALL}"

say() { printf 'STEP %s\n' "$*"; }

# A half-written runtime is worse than none: everything happens beside the target and
# is moved into place in one step at the end.
STAGE="${TARGET}.installing"
rm -rf "$STAGE"
mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT

say "Downloading Python ${PY_VERSION}"
curl -fL --silent --show-error -o "$STAGE/python.tar.gz" "$URL"

say "Unpacking the interpreter"
tar -xzf "$STAGE/python.tar.gz" -C "$STAGE"
rm -f "$STAGE/python.tar.gz"

PY="$STAGE/python/bin/python3.13"
[ -x "$PY" ] || { echo "FAIL the download produced no interpreter" >&2; exit 1; }

say "Installing TensorFlow and the engine (this is the long part)"
"$PY" -m pip install --quiet --upgrade pip
if [ -n "$ENGINE_SRC" ] && [ -d "$ENGINE_SRC" ]; then
  "$PY" -m pip install --quiet "$ENGINE_SRC"
else
  echo "FAIL no engine source at '${ENGINE_SRC}'" >&2
  exit 1
fi

say "Trimming what only a build needs"
SITE="$STAGE/python/lib/python3.13/site-packages"
rm -rf "$SITE/tensorflow/include"
rm -rf "$STAGE/python/lib/python3.13/test" "$STAGE/python/lib/python3.13/config-3.13-darwin"
find "$STAGE/python" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

say "Checking it stands on its own"
"$PY" - <<'PYCHECK'
import sysconfig
import tensorflow as tf
import keras
from nnarch.catalog.registry import bootstrap

base = sysconfig.get_paths()["purelib"]
assert ".installing/python/" in base, f"the interpreter resolves outside its own directory: {base}"
print(f"STEP Ready: TensorFlow {tf.__version__}, Keras {keras.__version__}, "
      f"{bootstrap().as_json()['count']} layers")
PYCHECK

# Only now does it become the real thing.
rm -rf "$TARGET"
mv "$STAGE" "$TARGET"
trap - EXIT

printf 'DONE %s\n' "$TARGET"
