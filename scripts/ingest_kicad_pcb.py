#!/usr/bin/env python3
"""Ingest STEP components from a KiCad .kicad_pcb file into the regression suite.

Extracts embedded STEP/STP 3D models from one or more .kicad_pcb files,
deduplicates them into test_assets/, and regenerates the regression baseline.

The .kicad_pcb files must already have their 3D models embedded (KiCad does
this automatically when footprints reference library models). If your PCB
references external .step files that aren't embedded, open it in KiCad first
so the editor embeds them on save.

Prerequisites:
    pip install zstandard

Usage:
    python3 scripts/ingest_kicad_pcb.py <file.kicad_pcb> [file2.kicad_pcb ...]
"""

import os
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
EXTRACT_SCRIPT = os.path.join(SCRIPT_DIR, "extract_steps.py")
TEST_ASSETS = os.path.join(REPO_DIR, "test_assets")
BASELINE_JSON = os.path.join(TEST_ASSETS, "baseline.json")

STEP_EXTENSIONS = (".step", ".stp")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.kicad_pcb> [...]", file=sys.stderr)
        sys.exit(1)

    pcb_files = sys.argv[1:]
    for pcb in pcb_files:
        if not os.path.exists(pcb):
            print(f"ERROR: {pcb} not found", file=sys.stderr)
            sys.exit(1)

    # --- Step 1: Extract embedded STEP files into a temp dir ---
    print("=== Extracting embedded STEP files ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, pcb in enumerate(pcb_files):
            board_tmp = os.path.join(tmp_dir, f"board_{i}")
            subprocess.run(
                [sys.executable, EXTRACT_SCRIPT, pcb, board_tmp],
                check=True,
            )

        # --- Step 2: Deduplicate into test_assets/ ---
        print()
        print("=== Deduplicating into test_assets/ ===")
        os.makedirs(TEST_ASSETS, exist_ok=True)
        added = 0
        skipped = 0
        for dirpath, _dirs, files in os.walk(tmp_dir):
            for name in sorted(files):
                if not name.lower().endswith(STEP_EXTENSIONS):
                    continue
                dest = os.path.join(TEST_ASSETS, name)
                if os.path.exists(dest):
                    skipped += 1
                    continue
                src = os.path.join(dirpath, name)
                with open(src, "rb") as f_in, open(dest, "wb") as f_out:
                    f_out.write(f_in.read())
                print(f"  Added: {name}")
                added += 1

        print(f"\n{added} new files added, {skipped} duplicates skipped")

    if added == 0:
        print("No new files to ingest — baseline unchanged.")
        return

    # --- Step 3: Regenerate baseline ---
    print()
    print("=== Regenerating regression baseline ===")
    env = {**os.environ, "RUST_LOG": "error"}
    result = subprocess.run(
        [
            "cargo", "run", "--release", "--example", "regression_test",
            "--", "--save-baseline", BASELINE_JSON,
        ],
        cwd=REPO_DIR,
        env=env,
    )
    if result.returncode != 0:
        print("ERROR: baseline generation failed", file=sys.stderr)
        sys.exit(1)

    # --- Step 4: Verify ---
    print()
    print("=== Verifying baseline ===")
    result = subprocess.run(
        [
            "cargo", "run", "--release", "--example", "regression_test",
            "--", "--compare", BASELINE_JSON, "--timing-threshold", "0.20",
        ],
        cwd=REPO_DIR,
        env=env,
    )
    if result.returncode != 0:
        print("WARNING: baseline verification found regressions", file=sys.stderr)
        sys.exit(1)

    print()
    print("Done! Remember to commit test_assets/ and baseline.json.")


if __name__ == "__main__":
    main()
