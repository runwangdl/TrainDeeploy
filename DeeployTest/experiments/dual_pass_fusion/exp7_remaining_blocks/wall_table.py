#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp7 step 5d: node-level comparison in WALL-CLOCK cycles from --profileTiling runs (FC getCycles around each
node: Pre-Kernel = input DMA, Kernel = the cluster kernel incl. waiting for the slowest core, Post-Kernel = output DMA).
Core-0 perf counters (node_table.py) stop while a core sleeps in a barrier, so they hide load imbalance.

  A (unfused, both passes) = 2 x [perturb weight + perturb bias + weight transpose + conv], each node's Total
  C (fused)                = the fused node's Total
  input transposes: separate; output transpose: not counted (fixture boundary)
Usage: python3 wall_table.py [log_suffix]   reads block<i>_{unfused,fused_cluster}/logs/run_profileTiling<suffix>.log
"""
import glob, re, sys

sfx = sys.argv[1] if len(sys.argv) > 1 else "_5d"
PAT = re.compile(r"^\[(\S+?)_L2\]\[SB\]\[\d+ ops\]\[Tile (\d+)\] (Pre-Kernel|Kernel     |Post-Kernel):\s*(\d+) cycles")


def parse(path):
    d = {}
    for line in open(path, errors = "ignore"):
        m = PAT.match(line)
        if m:
            node, kind, cyc = m.group(1), m.group(3).strip(), int(m.group(4))
            e = d.setdefault(node, {"Pre-Kernel": 0, "Kernel": 0, "Post-Kernel": 0})
            e[kind] += cyc  # summed over tiles
    return d


tot = lambda e: e["Pre-Kernel"] + e["Kernel"] + e["Post-Kernel"]
print(f"{'block':>5} | {'A unfused':>10} {'(perturb':>9} {'w-tr':>6} {'conv)':>8} | {'C fused':>9} {'(kernel':>8} {'DMA)':>6} | "
      f"{'A/C':>5} | {'in-tr A':>7} {'in-tr C':>7}")
SA = SC = 0
for b in (1, 2, 3, 4):
    d = glob.glob(f"exp7?_block{b}")[0]
    U = parse(f"{d}/block{b}_unfused/logs/run_profileTiling{sfx}.log")
    F = parse(f"{d}/block{b}_fused_cluster/logs/run_profileTiling{sfx}.log")
    pert = tot(U[f"rqsp_blocks{b}convweight_int8"]) + tot(U[f"rqsp_blocks{b}convbias_rqsadd"])
    wt = tot(U[f"_MERGE_CONVRQ_PASS_0_blocks{b}convweight_int8_pert_tensor_transpose"])
    conv = tot(U["_MERGE_CONVRQ_PASS_0"])
    A = 2 * (pert + wt + conv)
    f = F[f"dualconv_blocks{b}"]
    C = tot(f)
    tA = 2 * tot(U["_MERGE_CONVRQ_PASS_0_input_0_transpose"])
    tC = tot(F[f"dualconv_blocks{b}_input_0_transpose"]) + tot(F[f"dualconv_blocks{b}_input_3_transpose"])
    SA += A; SC += C
    print(f"{b:>5} | {A:>10,} {2*pert:>9,} {2*wt:>6,} {2*conv:>8,} | {C:>9,} {f['Kernel']:>8,} {f['Pre-Kernel']+f['Post-Kernel']:>6,} | "
          f"{A/C:>5.2f} | {tA:>7,} {tC:>7,}")
print(f"{'1-4':>5} | {SA:>10,} {'':>9} {'':>6} {'':>8} | {SC:>9,} {'':>8} {'':>6} | {SA/SC:>5.2f} |")
print(f"@370 MHz, blocks 1-4 both passes (nodes only): {SA/370e3:.2f} ms -> {SC/370e3:.2f} ms")
