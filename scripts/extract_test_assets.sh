#!/usr/bin/env bash
# One-time script to extract STEP test assets from KiCad demo boards.
#
# Prerequisites:
#   pip install zstandard
#   ~/sources/scripts/embed_steps_kicad.py must exist
#
# This script:
#   1. Runs embed_steps_kicad.py -i on each board to embed STEP models
#   2. Extracts all embedded STEP files via extract_steps.py
#   3. Deduplicates by filename into test_assets/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEMO_DIR="$HOME/sources/demo/boards"
EMBED_SCRIPT="$HOME/sources/scripts/embed_steps_kicad.py"
EXTRACT_SCRIPT="$SCRIPT_DIR/extract_steps.py"
OUTPUT_DIR="$REPO_DIR/test_assets"
TMP_DIR=$(mktemp -d)

trap 'rm -rf "$TMP_DIR"' EXIT

PCB_FILES=(
    "$DEMO_DIR/DM0001/layout/layout.kicad_pcb"
    "$DEMO_DIR/DM0002/layout/layout.kicad_pcb"
    "$DEMO_DIR/DM0003/layout/DM0003/layout.kicad_pcb"
)

echo "=== Step 1: Embedding STEP models into KiCad boards ==="
for pcb in "${PCB_FILES[@]}"; do
    if [ ! -f "$pcb" ]; then
        echo "WARN: $pcb not found, skipping"
        continue
    fi
    echo "Embedding models in $pcb..."
    python3 "$EMBED_SCRIPT" -i "$pcb"
done

echo ""
echo "=== Step 2: Extracting embedded STEP files ==="
i=0
for pcb in "${PCB_FILES[@]}"; do
    if [ ! -f "$pcb" ]; then
        continue
    fi
    board_tmp="$TMP_DIR/board_$i"
    i=$((i + 1))
    echo "Extracting from $pcb..."
    python3 "$EXTRACT_SCRIPT" "$pcb" "$board_tmp"
done

echo ""
echo "=== Step 3: Deduplicating into test_assets/ ==="
mkdir -p "$OUTPUT_DIR"

count=0
for f in "$TMP_DIR"/*/*.step "$TMP_DIR"/*/*.stp "$TMP_DIR"/*/*.STEP "$TMP_DIR"/*/*.STP; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    if [ ! -f "$OUTPUT_DIR/$name" ]; then
        cp "$f" "$OUTPUT_DIR/$name"
        echo "  Added: $name"
        count=$((count + 1))
    fi
done

echo ""
echo "Done! $count unique STEP files in $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
