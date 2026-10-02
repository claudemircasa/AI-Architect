#!/usr/bin/env bash
#
# > [!AML-DOC-FILE]
# @file        scripts/package-app.sh
# @description Build the distributable app and wrap it in a disk image.
# @module      scripts/package-app
# @exports     desktop/src-tauri/target/release/bundle/dmg/AI Architect.dmg
# @created     2026-10-01
# @context     The app ships without Python or TensorFlow and installs them on first
#              launch, which takes the download from 326MB to about 8MB [E-045]. What
#              it does carry is the installer script and the engine's own source,
#              both of which are copied in here rather than by hand, because a copy
#              made once goes stale and ships yesterday's engine [E-037].
#
#              Tauri's own DMG step is not used: it mounts a read-write image and
#              drives the Finder to arrange its icons, and the unmount loses a race
#              against indexing [E-028]. A compressed image made in one pass needs no
#              mount, so there is no race to lose.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAURI="$ROOT/desktop/src-tauri"
BUNDLE="$TAURI/target/release/bundle"
APP="$BUNDLE/macos/AI Architect.app"
OUT="$BUNDLE/dmg/AI Architect.dmg"

echo "==> staging what the app carries"
rm -rf "$TAURI/engine-src"
mkdir -p "$TAURI/engine-src"
rsync -a --exclude '.venv' --exclude '__pycache__' --exclude '*.egg-info' \
      --exclude 'tests' "$ROOT/backend/" "$TAURI/engine-src/"
cp "$ROOT/scripts/install-runtime.sh" "$TAURI/install-runtime.sh"
echo "    engine source: $(du -sh "$TAURI/engine-src" | cut -f1)"

echo "==> building the app"
(cd "$ROOT/desktop" && npm run tauri build)

[ -d "$APP" ] || { echo "the build produced no app at $APP" >&2; exit 1; }

echo "==> checking it can install its own engine"
for required in "Contents/Resources/install-runtime.sh" \
                "Contents/Resources/engine-src/pyproject.toml" \
                "Contents/Resources/engine-src/src/nnarch/__main__.py"; do
  [ -e "$APP/$required" ] || { echo "missing from the app: $required" >&2; exit 1; }
done
echo "    installer and engine source are both present"

echo "==> staging the disk image"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

mkdir -p "$BUNDLE/dmg"
rm -f "$OUT"
hdiutil create \
  -volname "AI Architect" \
  -srcfolder "$STAGE" \
  -fs HFS+ \
  -format UDZO \
  -imagekey zlib-level=6 \
  -quiet \
  "$OUT"

echo "==> done: app $(du -sh "$APP" | cut -f1), dmg $(du -sh "$OUT" | cut -f1)"
echo "    Python and TensorFlow are downloaded on first launch."
