#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH=${1:-careeragent_ai_patch_final.zip}

if [ ! -f "$ZIP_PATH" ]; then
  echo "Zip not found: $ZIP_PATH" >&2
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
RB="_rollback/$TS"
mkdir -p "$RB"

# Backup only the paths we overwrite
for p in src/careeragent app/ui app/main.py; do
  if [ -e "$p" ]; then
    mkdir -p "$RB/$(dirname $p)"
    cp -R "$p" "$RB/$(dirname $p)/" 2>/dev/null || true
  fi
done

echo "Backup created at $RB"

unzip -o "$ZIP_PATH" -d .

echo "Patch applied. To rollback: rm -rf src/careeragent app/ui app/main.py && cp -R $RB/src/careeragent src/ && cp -R $RB/app/ui app/ && cp -R $RB/app/main.py app/"
