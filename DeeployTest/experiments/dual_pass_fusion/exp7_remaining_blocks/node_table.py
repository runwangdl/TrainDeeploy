#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp7 step 5a: node-level comparison per block, from the profiled runs' per-node CSVs (core 0, no new runs).

  unfused / mid (both passes) = 2 x (perturb weight + perturb bias + weight transpose + conv node)
  fused                       = the fused RequantizedDualConv node
  input transposes            = reported separately (2 x 1 per pass unfused, 2 for fused): one per pass is
                                real in today's full network (ReLU drops the chain to NCHW)
  output transpose, DMA/control outside the nodes: not counted (fixture boundary)
Usage: python3 node_table.py [fused_suffix]   reads exp7?_block<i>/block<i>_{unfused,mid}/results/per_node.csv and
       block<i>_fused_cluster/results/per_node<fused_suffix>.csv (suffix = a later optimisation run, e.g. _5b)
"""
import csv, glob, sys

sfx = sys.argv[1] if len(sys.argv) > 1 else ""


def load(p):
    return {r["node"]: int(r["cycles"]) for r in csv.DictReader(open(p))}


def unfused_nodes(R, b):
    perturb = R[f"rqsp_blocks{b}convweight_int8"] + R[f"rqsp_blocks{b}convbias_rqsadd"]
    wt = R[f"_MERGE_CONVRQ_PASS_0_blocks{b}convweight_int8_pert_tensor_transpose"]
    conv = R["_MERGE_CONVRQ_PASS_0"]
    tin = R["_MERGE_CONVRQ_PASS_0_input_0_transpose"]
    return perturb, wt, conv, tin


rows = []
print(f"{'block':>5} | {'A unfused':>10} {'(perturb':>9} {'w-tr':>6} {'conv)':>8} | {'B Conv4x2':>10} {'(conv)':>8} | "
      f"{'C fused':>9} | {'A/C':>5} {'B/C':>5} {'A/B':>5} | {'in-tr A':>7} {'in-tr C':>7}")
tot = [0, 0, 0]
for b in (1, 2, 3, 4):
    d = glob.glob(f"exp7?_block{b}")[0]
    U = load(f"{d}/block{b}_unfused/results/per_node.csv")
    M = load(f"{d}/block{b}_mid/results/per_node.csv")
    F = load(f"{d}/block{b}_fused_cluster/results/per_node{sfx}.csv")
    pu, wu, cu, tu = unfused_nodes(U, b)
    pm, wm, cm, tm = unfused_nodes(M, b)
    A = 2 * (pu + wu + cu)
    B = 2 * (pm + wm + cm)
    C = F[f"dualconv_blocks{b}"]
    tC = F[f"dualconv_blocks{b}_input_0_transpose"] + F[f"dualconv_blocks{b}_input_3_transpose"]
    tot[0] += A; tot[1] += B; tot[2] += C
    print(f"{b:>5} | {A:>10,} {2*pu:>9,} {2*wu:>6,} {2*cu:>8,} | {B:>10,} {2*cm:>8,} | {C:>9,} | "
          f"{A/C:>5.2f} {B/C:>5.2f} {A/B:>5.2f} | {2*tu:>7,} {tC:>7,}")
print(f"{'1-4':>5} | {tot[0]:>10,} {'':>9} {'':>6} {'':>8} | {tot[1]:>10,} {'':>8} | {tot[2]:>9,} | "
      f"{tot[0]/tot[2]:>5.2f} {tot[1]/tot[2]:>5.2f} {tot[0]/tot[1]:>5.2f} |")
