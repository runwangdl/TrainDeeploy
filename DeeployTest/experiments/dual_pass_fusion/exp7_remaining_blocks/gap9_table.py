#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp7 step 7: GAP9 table per block from run_gap9_block.sh's logs (plain runtime) and per-node tables (master-core wall
time per node). 'nodes' = without the fixture-boundary transposes (inputs NCHW->HWC, output HWC->NCHW).
Usage: python3 gap9_table.py [blocks ...]   -> prints, and writes exp7?_block<i>/block<i>_fused_NE16/results/summary_gap9.txt
"""
import csv, glob, re, sys

VARIANTS = [("A", "2 x pulp-nn, cluster", 2), ("B", "2 x NE16 im2col (exp6 route)", 2),
            ("C_cl", "fused, cluster (DualConv2In)", 1), ("C", "fused NE16, exp6-style (2 chains)", 1),
            ("C_S", "fused NE16 node, sequential", 1), ("C_P", "fused NE16 node, pipelined", 1)]
DROP_IN = ("input_0_transpose", "input_3_transpose", "_xp_hwc", "_xm_hwc")
DROP_OUT = ("output_0_pre_transpose", "out_nchw")


def nodes(d, v):
    return {r["node"]: int(r["cycles"]) for r in csv.DictReader(open(f"{d}/results/gap9_{v}_per_node.csv"))}


def runtime(d, v):
    for l in open(f"{d}/logs/gap9_{v}_plain.log", errors = "ignore"):
        m = re.match(r"Runtime: (\d+)", l)
        if m:
            return int(m.group(1))


def table(b):
    d = glob.glob(f"exp7?_block{b}")[0] + f"/block{b}_fused_NE16"
    L = [f"exp7 block {b} -- GAP9 GVSoC, both passes, exact; --l1 86000 --l2 1500000 --defaultMemLevel L2 --cores 8",
         "nodes = master-core wall time per node, without the fixture-boundary transposes", "",
         f"{'variant':44s} {'fixture':>9s} {'vs A':>6s} {'nodes':>9s} {'vs A':>6s}"]
    res = {}
    for v, name, k in VARIANTS:
        n = nodes(d, v)
        nd = k * sum(c for x, c in n.items() if not any(s in x for s in DROP_IN + DROP_OUT))
        res[v] = (k * runtime(d, v), nd, n)
    for v, name, k in VARIANTS:
        fx, nd, _ = res[v]
        L.append(f"{v:5s}{name:39s} {fx:>9,} {res['A'][0]/fx:>5.2f}x {nd:>9,} {res['A'][1]/nd:>5.2f}x")
    L.append("")
    for v in ("C_cl", "C_P"):
        L.append(f"{v} per node: " + ", ".join(f"{x} {c:,}" for x, c in res[v][2].items()))
    open(f"{d}/results/summary_gap9.txt", "w").write("\n".join(L) + "\n")
    return L, res


if __name__ == "__main__":
    blocks = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4]
    allres = {}
    for b in blocks:
        L, res = table(b)
        allres[b] = res
        print("\n".join(L) + "\n")
    print(f"{'block':>5} | {'A':>9} {'C_cl':>9} {'C_P':>9} | {'C_cl/A':>7} {'C_P/A':>7} | best (fixture)  | nodes: A, C_cl, C_P")
    tot = {"A": [0, 0], "C_cl": [0, 0], "C_P": [0, 0], "best": [0, 0]}
    for b, res in allres.items():
        a, c, p = res["A"], res["C_cl"], res["C_P"]
        best = min(("C_cl", c), ("C_P", p), key = lambda t: t[1][0])
        for kk, val in (("A", a), ("C_cl", c), ("C_P", p), ("best", best[1])):
            tot[kk][0] += val[0]; tot[kk][1] += val[1]
        print(f"{b:>5} | {a[0]:>9,} {c[0]:>9,} {p[0]:>9,} | {a[0]/c[0]:>6.2f}x {a[0]/p[0]:>6.2f}x | {best[0]:5s} {a[0]/best[1][0]:>5.2f}x | "
              f"{a[1]:,}, {c[1]:,}, {p[1]:,}")
    if len(allres) > 1:
        print(f"{'sum':>5} | {tot['A'][0]:>9,} {tot['C_cl'][0]:>9,} {tot['C_P'][0]:>9,} | {tot['A'][0]/tot['C_cl'][0]:>6.2f}x "
              f"{tot['A'][0]/tot['C_P'][0]:>6.2f}x | best per block {tot['A'][0]/tot['best'][0]:>5.2f}x (fixture {tot['best'][0]:,}; "
              f"nodes {tot['A'][1]:,} -> {tot['best'][1]:,} = {tot['A'][1]/tot['best'][1]:.2f}x)")
