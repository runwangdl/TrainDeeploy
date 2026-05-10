#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""
Emit a per-test footprint summary to GITHUB_STEP_SUMMARY.

Walks `DeeployTest/TEST_SIRACUSA/` for generated `TrainingNetwork.c` files
and reports, per fixture: MEMORYARENA_L1/L2/L3 sizes (peak working sets +
L3 storage) and the number of distinct numTiles shapes.

The numbers come from grepping the generated C — they're a build-time
proxy for "how much memory pressure does this configuration put on the
target".  This is the closest stand-in for the cycle comparison the user
wants until the L3-untiled sim OOM is debugged and we can collect real
gvsoc cycle counts.

Used in the siracusa-tiled CI workflow.  Safe to run with no matching
files (just emits an empty summary).
"""

import os
import re
import sys
from pathlib import Path

ARENA_RE = re.compile(r"MEMORYARENA_(L1|L2|L3)\s*=.*\*\s*(\d+)")
TILES_RE = re.compile(r"numTiles\[\d+\]\s*=\s*\{[^}]+\}")


def parse_one(c_path: Path) -> dict:
    arenas = {"L1": 0, "L2": 0, "L3": 0}
    tile_shapes = set()
    for line in c_path.read_text(errors="replace").splitlines():
        m = ARENA_RE.search(line)
        if m:
            arenas[m.group(1)] = max(arenas[m.group(1)], int(m.group(2)))
        for t in TILES_RE.findall(line):
            tile_shapes.add(t)
    return {"arenas": arenas, "tile_shapes": len(tile_shapes)}


def fmt_kb(n: int) -> str:
    if n == 0:
        return "—"
    return f"{n / 1024:.1f} KB"


def main() -> int:
    test_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("DeeployTest/TEST_SIRACUSA")
    if not test_root.is_dir():
        print(f"[footprint-summary] {test_root} not a directory; skipping", file=sys.stderr)
        return 0

    rows = []
    for c_path in sorted(test_root.rglob("TrainingNetwork.c")):
        rel = c_path.relative_to(test_root).parent
        info = parse_one(c_path)
        rows.append((str(rel), info))

    # Pick the pytest marker label (passed by the workflow) for the section title.
    label = os.environ.get("FOOTPRINT_SUMMARY_LABEL", "training")

    out_lines = [
        f"### Build footprint — `{label}`",
        "",
        "| Fixture | L1 working | L2 working | L3 storage | Distinct tile shapes |",
        "|---|--:|--:|--:|--:|",
    ]
    for path, info in rows:
        a = info["arenas"]
        out_lines.append(
            f"| `{path}` | {fmt_kb(a['L1'])} | {fmt_kb(a['L2'])} | {fmt_kb(a['L3'])} | {info['tile_shapes']} |")
    if not rows:
        out_lines.append("| _(no TrainingNetwork.c found)_ | | | | |")
    out_lines.append("")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"[footprint-summary] wrote {len(rows)} rows to {summary_path}", file=sys.stderr)
    else:
        # Local invocation: print to stdout for visibility.
        print("\n".join(out_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
