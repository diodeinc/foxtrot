#!/usr/bin/env python3
"""Extract embedded STEP files from a KiCad .kicad_pcb file.

KiCad embeds 3D models as zstd-compressed, base64-encoded data inside
(embedded_files (file (name "...") (type model) (data |...|))) blocks.

Usage:
    python3 extract_steps.py <input.kicad_pcb> <output_dir>
"""

import base64
import os
import re
import sys

try:
    import zstandard as zstd
except ImportError:
    print("ERROR: zstandard not installed. Run: pip install zstandard", file=sys.stderr)
    sys.exit(1)


def extract_embedded_steps(pcb_path, output_dir):
    """Extract all embedded STEP/STP files from a .kicad_pcb file."""
    with open(pcb_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Match embedded_files blocks at the top level (board-level, not footprint-level)
    # We want ALL embedded files with type model
    # Pattern: (file (name "X") (type model) (data |BASE64|))
    pattern = re.compile(
        r'\(file\s*\n?\s*\(name\s+"([^"]+)"\)\s*\n?\s*\(type\s+model\)\s*\n?\s*\(data\s+\|([^|]*)\|',
        re.DOTALL,
    )

    os.makedirs(output_dir, exist_ok=True)
    extracted = []
    dctx = zstd.ZstdDecompressor()

    for match in pattern.finditer(content):
        name = match.group(1)
        b64_data = match.group(2)

        # Strip whitespace from base64 data
        b64_clean = b64_data.replace("\n", "").replace("\r", "").replace("\t", "").replace(" ", "")

        try:
            compressed = base64.b64decode(b64_clean)
            decompressed = dctx.decompress(compressed)
        except Exception as e:
            print(f"  WARN: Failed to decode {name}: {e}", file=sys.stderr)
            continue

        out_path = os.path.join(output_dir, name)
        with open(out_path, "wb") as f:
            f.write(decompressed)
        extracted.append(name)
        print(f"  Extracted: {name} ({len(decompressed)} bytes)")

    return extracted


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.kicad_pcb> <output_dir>", file=sys.stderr)
        sys.exit(1)

    pcb_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(pcb_path):
        print(f"ERROR: {pcb_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting from {pcb_path}...")
    extracted = extract_embedded_steps(pcb_path, output_dir)
    print(f"Extracted {len(extracted)} files to {output_dir}")


if __name__ == "__main__":
    main()
