#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# exp11: the exp6 block-0 per-node table for each of blocks 1-4 (GAP9). -- QW
#
#   python3 make_tables.py            (reads ../logs/b<i>_<v>_{plain,mb}.log, writes ../results/)
#
# Node cycles: --profileMicrobenchmark (per-node wall time, cluster master core 8). Total: the plain run's Runtime.
# Single-pass variants (A, B, Bp) are run once and doubled (both passes), as in exp6's "A (x2)".
# Columns as in exp6 results/summary.txt: preparation | input + weight transpose | im2col gather | conv |
# im2col + conv | output transpose | outside nodes | total.

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "exp6_NE16_fusion"))
from parse_microbench import parse  # noqa: E402

LOGS = os.path.join(HERE, "..", "logs")
RES = os.path.join(HERE, "..", "results")
VARIANTS = [("A", "2 x pulp-nn (cluster)", 2), ("B", "2 x NE16 im2col", 2), ("Bp", "2 x Conv4x2 (cluster)", 2),
            ("Ccl", "fused, cluster (DualConv2In)", 1), ("C", "fused, NE16 (pipelined)", 1)]
COLS = ["perturb", "prep", "transpose", "im2col", "conv", "im2col_conv", "out_transpose", "outside", "total"]
HEAD = ["perturbation", "NE16 prep (encode, input bias)", "input + weight transpose", "im2col gather", "conv",
        "im2col + conv", "output transpose", "outside nodes", "total"]
# QW: perturbation = rqsp_* nodes (A, B, Bp); phase 1 of the fused cluster node from the probe build (Ccl, subtracted
#     from its conv); dual_wenc + dual_bias for the fused NE16 node (C: perturbation and encode in one node, kept
#     together, shown in the perturbation column).


def classify(node):
    n = node
    if n.startswith("DC2PROBE"):
        return None
    if n.startswith("rqsp_") or n.endswith("_dual_wenc") or n.endswith("_dual_bias"):
        return "perturb"
    if n.endswith("_sbias") or n.endswith("_wenc"):
        return "prep"
    if "input_" in n and n.endswith("_transpose") or n.endswith("_pert_tensor_transpose") or n.endswith("_hwc"):
        return "transpose"
    if n.endswith("_im2col"):
        return "im2col"
    if n.endswith("output_0_pre_transpose") or n.endswith("_out_nchw"):
        return "out_transpose"
    if n.endswith("_ne16_dualconv"):
        return "fused_ne16"  # column build + NE16 jobs, overlapped (pipelined): only im2col + conv is meaningful
    if n.startswith("_MERGE_CONVRQ_PASS") or re.match(r"dualconv_blocks\d+$", n):
        return "conv"
    raise ValueError(f"unclassified node {node!r}")


def runtime(path):
    m = re.search(r"^Runtime:\s+(\d+)", open(path, errors = "ignore").read(), re.M)
    return int(m.group(1)) if m else None


def errors(path):
    m = re.search(r"Errors: (\d+) out of (\d+)", open(path, errors = "ignore").read())
    return f"{m.group(1)}/{m.group(2)}" if m else "n/a"


def row(i, v, mult):
    mb, pl = (os.path.join(LOGS, f"b{i}_{v}_{m}.log") for m in ("mb", "plain"))
    if not (os.path.exists(mb) and os.path.exists(pl)):
        return None
    r = dict.fromkeys(COLS, 0)
    nodes = []
    for rec in parse(mb):
        k = classify(rec["node"])
        if k is None:
            continue
        nodes.append((rec["node"], rec["cycles"] * mult))
        if k == "fused_ne16":
            r["im2col_conv"] += rec["cycles"] * mult
        else:
            r[k] += rec["cycles"] * mult
    if v == "Ccl":  # split the fused node: phase 1 (perturbation) from the probe build, the rest is the conv
        pr = probe(i)
        if pr:
            r["perturb"] = max(pr["p1"])
            r["conv"] -= r["perturb"]
    if r["im2col_conv"] == 0:
        r["im2col_conv"] = r["im2col"] + r["conv"]
    rt = runtime(pl)
    r["total"] = rt * mult if rt is not None else 0
    node_sum = r["perturb"] + r["prep"] + r["transpose"] + r["im2col_conv"] + r["out_transpose"]
    r["outside"] = r["total"] - node_sum
    r["errors"] = errors(pl)
    r["nodes"] = nodes
    return r


def probe(i):
    p = os.path.join(LOGS, f"b{i}_Cpr_plain.log")
    if not os.path.exists(p):
        return None
    recs = [x for x in parse(p) if x["node"].startswith("DC2PROBE")]
    p1 = [x["cycles"] for x in recs if "_p1_" in x["node"]]
    p2 = [x["cycles"] for x in recs if "_p2_" in x["node"]]
    if not p1 or not p2:
        return None
    return {"p1": p1, "p2": p2, "runtime": runtime(p), "errors": errors(p)}


def fmt(x):
    return "—" if x == 0 else f"{x:,}"


def main():
    os.makedirs(RES, exist_ok = True)
    out = []
    for i in (1, 2, 3, 4):
        rows = [(v, label, row(i, v, mult)) for v, label, mult in VARIANTS]
        rows = [x for x in rows if x[2] is not None]
        if not rows:
            continue
        base = next((r for v, _, r in rows if v == "A"), None)
        out.append(f"## Block {i}\n")
        out.append("| variant | " + " | ".join(HEAD) + " | vs A | errors |")
        out.append("|---|" + "--:|" * len(HEAD) + "--:|---|")
        for v, label, r in rows:
            cells = []
            for c in COLS:
                if v == "C" and c in ("im2col", "conv"):
                    cells.append("(in the node)")
                elif v == "C" and c == "perturb":
                    cells.append(f"{r[c]:,} (incl. encode)")
                elif v == "C" and c == "prep":
                    cells.append("(in perturbation)")
                else:
                    cells.append(fmt(r[c]) if c != "outside" else f"{r[c]:,}")
            sp = f"{base['total'] / r['total']:.2f}x" if base and r["total"] else ""
            out.append(f"| {v}: {label} | " + " | ".join(cells) + f" | {sp} | {r['errors']} |")
        pr = probe(i)
        if pr:
            out.append("")
            out.append(f"Inside the fused cluster node (probe build, per-core WORK cycles before each closing barrier; "
                       f"Runtime {pr['runtime']:,}, errors {pr['errors']}):")
            out.append(f"- phase 1 (perturb W+/W-, add+/add-): max {max(pr['p1']):,}, min {min(pr['p1']):,}  "
                       f"per core {pr['p1']}")
            out.append(f"- phase 2 (both convs): max {max(pr['p2']):,}, min {min(pr['p2']):,}  per core {pr['p2']}")
        out.append("\nPer node (cycles, both passes):\n")
        for v, label, r in rows:
            out.append(f"- {v}: " + ", ".join(f"`{n}` {c:,}" for n, c in r["nodes"]))
        out.append("")
    txt = "\n".join(out) + "\n"
    open(os.path.join(RES, "breakdown.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
