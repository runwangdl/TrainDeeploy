#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""
Emit a per-test footprint + cycle summary to GITHUB_STEP_SUMMARY.

Two passes:

1. **Build footprint** — walks `DeeployTest/TEST_SIRACUSA/` for generated
   `TrainingNetwork.c` files and reports per fixture: MEMORYARENA_L1/L2/L3
   sizes (peak working sets + L3 storage) and distinct numTiles shapes.
2. **Cycle counts** — parses `DeeployTest/out.txt` (where the test runner
   appends every sim's stdout) for `BENCH train_cycles=… opt_cycles=…
   weight_sram=…` lines, correlating each line to its preceding `Testing
   <test_dir>` banner.  Skipped fixtures contribute no cycle row.

Used in the siracusa-tiled CI workflow.  Safe to run with no matching
files (just emits an empty section).
"""

import os
import re
import sys
from pathlib import Path

ARENA_RE = re.compile(r"MEMORYARENA_(L1|L2|L3)\s*=.*\*\s*(\d+)")
TILES_RE = re.compile(r"numTiles\[\d+\]\s*=\s*\{[^}]+\}")
TESTING_RE = re.compile(r"Testing\s+(\S+)\s+on\s+\S+\s+Platform")
BENCH_RE = re.compile(r"BENCH\s+train_cycles=(\d+)\s+opt_cycles=(\d+)\s+weight_sram=(\d+)")


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


def fmt_cycles(n: int) -> str:
    if n == 0:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def parse_cycles(out_txt: Path) -> dict:
    """Returns {test_dir: {train_cycles, opt_cycles, weight_sram}}.

    Each `Testing <path>` banner in out.txt opens a section; the next
    `BENCH …` line in that section is the cycle row for that fixture.
    Sections without a BENCH line (skipsim, sim crash) get no entry.
    """
    if not out_txt.is_file():
        return {}
    out: dict = {}
    current = None
    for line in out_txt.read_text(errors="replace").splitlines():
        m = TESTING_RE.search(line)
        if m:
            current = m.group(1)
            continue
        m = BENCH_RE.search(line)
        if m and current is not None:
            out[current] = {
                "train_cycles": int(m.group(1)),
                "opt_cycles": int(m.group(2)),
                "weight_sram": int(m.group(3)),
            }
    return out


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

    cycles = parse_cycles(test_root.parent / "out.txt")

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

    # Cycle table — only renders if at least one fixture actually simulated.
    cycle_rows = []
    for path, _info in rows:
        # The `Testing` banner uses the absolute test_dir path; match by basename.
        match_key = next((k for k in cycles if k.endswith(path) or path.endswith(Path(k).name)), None)
        if match_key:
            cycle_rows.append((path, cycles[match_key]))
    out_lines.append(f"### Cycle counts (gvsoc) — `{label}`")
    out_lines.append("")
    out_lines.append("| Fixture | train_cycles | opt_cycles | weight_sram |")
    out_lines.append("|---|--:|--:|--:|")
    if cycle_rows:
        for path, c in cycle_rows:
            out_lines.append(
                f"| `{path}` | {fmt_cycles(c['train_cycles'])} | "
                f"{fmt_cycles(c['opt_cycles'])} | {fmt_kb(c['weight_sram'])} |")
    else:
        out_lines.append(
            "| _(no BENCH lines in out.txt — sim was --skipsim'd or crashed)_ | | | |")
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
