#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Parse a --profileMicrobenchmark run into a per-node utilisation table. -- QW
#
#   python3 parse_microbench.py logs/run_microbench.log results/per_node
#
# Emits <out>.csv (machine-readable) and <out>.md (the table in the README).
# Every counter comes from perf_bench_print() in TargetLibraries/PULPOpen/inc/perf_utils.h,
# and is CORE 0 ONLY -- PULPMicrobenchmark's pre/post templates are guarded by
# `pi_core_id() == 0`.

import csv
import re
import sys

FIELDS = [("cycles", "Cycles"), ("instr", "Instructions"), ("loads", "Loads"), ("stores", "Stores"),
          ("branches", "Branches"), ("taken_branches", "Taken Branches"), ("rvc", "Compressed \\(RVC\\)"),
          ("ld_stall", "Load Stalls"), ("jmp_stall", "Jump Stalls"), ("imiss", "I-cache Misses"),
          ("tcdm_cont", "TCDM Contentions"), ("ld_ext", "External Loads"), ("st_ext", "External Stores")]


def parse(path):
    txt = open(path, errors = "ignore").read()
    # QW (exp6): on GAP9 the fabric controller replays the cluster's records after this marker;
    #     keep only that replay so nothing is counted twice.
    if "[perf] records=" in txt:
        txt = txt.split("[perf] records=", 1)[1]
    rows = []
    for blk in re.split(r"=== Performance Statistics: ", txt)[1:]:
        row = {"node": blk.split(" ===")[0]}
        for key, label in FIELDS:
            m = re.search(rf"{label}:\s+(\d+)", blk)
            row[key] = int(m.group(1)) if m else 0
        row["ipc"] = row["instr"] / row["cycles"] if row["cycles"] else 0.0
        rows.append(row)
    return rows


def main():
    log, out = sys.argv[1], sys.argv[2]
    rows = parse(log)
    keys = ["node", "cycles", "instr", "ipc"] + [k for k, _ in FIELDS[2:]]

    with open(out + ".csv", "w", newline = "") as fh:
        w = csv.DictWriter(fh, fieldnames = keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if k == "ipc" else r[k]) for k in keys})

    tot_c = sum(r["cycles"] for r in rows)
    tot_i = sum(r["instr"] for r in rows)
    with open(out + ".md", "w") as fh:
        fh.write("| node | cycles | share | instr | IPC | loads | stores | branch | ld-stall | imiss | tcdm |\n")
        fh.write("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        for r in rows:
            fh.write(f"| `{r['node']}` | {r['cycles']:,} | {100*r['cycles']/tot_c:.2f}% | {r['instr']:,} | "
                     f"**{r['ipc']:.3f}** | {r['loads']:,} | {r['stores']:,} | {r['branches']:,} | "
                     f"{r['ld_stall']:,} | {r['imiss']:,} | {r['tcdm_cont']} |\n")
        fh.write(f"| **total (core 0)** | **{tot_c:,}** | 100% | **{tot_i:,}** | **{tot_i/tot_c:.3f}** | | | | | | |\n")

    print(f"{len(rows)} nodes -> {out}.csv, {out}.md")
    for r in rows:
        print(f"  {r['node'][:46]:<48} {r['cycles']:>10,} cyc  IPC {r['ipc']:.3f}")


if __name__ == "__main__":
    main()
