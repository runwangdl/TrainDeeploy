#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp21c task 3: per-block cycle accounting of the whole-network runs (one update step = n_accum 4 training-graph runs).

Reads the --profileTiling logs (logs/net_{eco,ne16}_prof.log): every "[node_L2][SB][n ops][Tile t] Pre-Kernel / Kernel /
Post-Kernel: c cycles" line (exp8 metric: wall clock of the node's tiles, cluster master). The training graph runs 4x
(n_accum 4) -- the unfused graph computes ONE pass per run, so the harness runs it 8x (4 windows x {+, -}); the fused
graph computes both passes per run, 4 runs. Update-graph nodes are listed separately. Per-window numbers = sum / 4.
Plain-run BENCH (logs/net_{eco,ne16}_plain.log) gives the true total; BENCH - node sum = harness / closure overhead.

  python3 parse_network.py  -> results/network_per_block.txt   -- QW exp21c
"""
import collections
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "logs")
RES = os.path.join(HERE, "..", "results")
PAT = re.compile(r"^\[(\S+?)_L2\]\[SB\]\[\d+ ops\]\[Tile \d+\] (Pre-Kernel|Kernel     |Post-Kernel):\s*(\d+) cycles", re.M)
N_WIN = 4


def block_of(n):
    """(block, kind) of a node of either graph; block 'in' / 'head' / 'opt' for the rest."""
    m = re.search(r"blocks_?(\d)", n)
    # unfused conv chain
    m2 = re.match(r"_MERGE_CONVRQ_PASS_(\d)$", n)
    if m2:
        return int(m2.group(1)), "conv"
    m2 = re.match(r"_MERGE_CONVRQ_PASS_(\d)_blocks\dconvweight_int8_pert_tensor_transpose", n)
    if m2:
        return int(m2.group(1)), "weight transpose"
    if n.startswith("_MERGE_CONVRQ_PASS_0_blocks_0_conv_input") or n.startswith("QCDQ_blocks_0_conv_input"):
        return "in", "input quant + transpose"
    m2 = re.match(r"rqsp_blocks(\d)conv(weight|bias)", n)
    if m2:
        return int(m2.group(1)), "conv perturbation (int8 w / int32 b)"
    # fused NE16 / fused cluster chain
    m2 = re.match(r"wrappedInnerForwardImpl(?:_(\d))?Conv_(ne16wenc|ne16bias|im2col|ne16)__dual", n)
    if m2:
        b = int(m2.group(1) or 0)
        k = {"ne16wenc": "NE16 prep: perturb + encode (wenc)", "ne16bias": "NE16 prep: bias (re-perturb + row sums)",
             "im2col": "im2col (block 0)", "ne16": "conv (NE16 node)"}[m2.group(2)]
        return b, k
    m2 = re.match(r"wrappedInnerForwardImpl(?:_(\d))?Conv__dual", n)
    if m2:
        return int(m2.group(1) or 0), "conv (fused cluster DualConv2In, incl. perturbation)"
    m2 = re.match(r"dualconv_blocks(\d)", n)
    if m2:
        return int(m2.group(1)), "conv (fused cluster DualConv2In, incl. perturbation)"
    # BN tails / BN perturbation
    m2 = re.match(r"bn(?:_(\d))?BatchNormalization_tail", n)
    if m2:
        return int(m2.group(1) or 0), "BN+pool+ReLU+Quant tail"
    m2 = re.match(r"pert_blocks(\d)bn(weight|bias)", n)
    if m2:
        return int(m2.group(1)), "BN perturbation (fp32)"
    if n.startswith("bn_4BatchNormalization") or n.startswith("relu_4") or n.startswith("global_pool"):
        return 4, "BN/ReLU/pool (block 4, untailed)"
    if "Gemm" in n or "CrossEntropy" in n or n.startswith("pert_fc"):
        return "head", "FC + loss (+ fc perturbation)"
    return "other", n


def parse(path):
    tot = collections.OrderedDict()
    for n, _k, c in PAT.findall(open(path, errors = "ignore").read()):
        tot[n] = tot.get(n, 0) + int(c)
    return tot


def bench(path):
    if not os.path.exists(path):
        return None, None
    m = re.search(r"BENCH train_cycles_hi=(\d+) train_cycles_lo=(\d+) opt_cycles_hi=(\d+) opt_cycles_lo=(\d+)",
                  open(path, errors = "ignore").read())
    if not m:
        return None, None
    return (int(m.group(1)) << 32) + int(m.group(2)), (int(m.group(3)) << 32) + int(m.group(4))


def main():
    out = []
    cols = {}
    for cfg in ("eco", "ne16", "fus"):
        p = os.path.join(LOGS, f"net_{cfg}_prof.log")
        if not os.path.exists(p):
            continue
        cols[cfg] = parse(p)
    opt_names = set()
    # update-graph nodes: a node name that only appears in the optimizer network C file
    for cfg in cols:
        for f in os.listdir(os.path.join(HERE, "..", "codegen")):
            if f.startswith(f"net_{cfg}_prof_") and "Optimizer" in f:
                src = open(os.path.join(HERE, "..", "codegen", f), errors = "ignore").read()
                opt_names |= set(re.findall(r"\[(\S+?)_L2\]\[SB\]", src))
    agg = {}
    for cfg, nodes in cols.items():
        a = collections.OrderedDict()
        for n, c in nodes.items():
            if n in opt_names:
                key = ("opt", "update graph")
            else:
                key = block_of(n)
            a[key] = a.get(key, 0) + c
        agg[cfg] = a
    keys = []
    for cfg in agg:
        for k in agg[cfg]:
            if k not in keys:
                keys.append(k)
    order = {"in": -1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, "head": 5, "other": 6, "opt": 7}
    keys.sort(key = lambda k: (order.get(k[0], 9), str(k[1])))
    names = list(agg)
    out.append("exp21c task 3 -- whole-network node accounting, ONE update step (n_accum 4 windows), --profileTiling wall "
               "clock summed over all tiles and all graph runs; per-window = /4.")
    out.append("columns: " + ", ".join(f"{c} = logs/net_{c}_prof.log" for c in names))
    out.append("")
    out.append(f"{'block':>6} {'node kind':58s}" + "".join(f"{c + ' /window':>18s}" for c in names))
    blk_tot = {c: collections.OrderedDict() for c in names}
    for k in keys:
        out.append(f"{str(k[0]):>6} {k[1][:58]:58s}" + "".join(f"{agg[c].get(k, 0) / N_WIN:>18,.0f}" for c in names))
        for c in names:
            blk_tot[c][k[0]] = blk_tot[c].get(k[0], 0) + agg[c].get(k, 0)
    out.append("")
    out.append("per block (all node kinds of the block), per window:")
    for b in sorted({k[0] for k in keys}, key = lambda x: order.get(x, 9)):
        out.append(f"{str(b):>6} {'':58s}" + "".join(f"{blk_tot[c].get(b, 0) / N_WIN:>18,.0f}" for c in names))
    out.append("")
    for c in names:
        tr = sum(v for k, v in agg[c].items() if k[0] != "opt")
        op = sum(v for k, v in agg[c].items() if k[0] == "opt")
        bt, bo = bench(os.path.join(LOGS, f"net_{c}_plain.log"))
        bpt, bpo = bench(os.path.join(LOGS, f"net_{c}_prof.log"))
        out.append(f"{c}: training node sum {tr:,} (/window {tr / N_WIN:,.0f}); update node sum {op:,}; "
                   f"plain BENCH train {bt if bt is None else f'{bt:,}'} opt {bo if bo is None else f'{bo:,}'}; "
                   f"prof BENCH train {bpt if bpt is None else f'{bpt:,}'} (printf-inflated)")
        if bt:
            out.append(f"    outside the nodes (plain BENCH - node sum): {bt - tr:,} ({(bt - tr) / bt * 100:.1f} %)")
    os.makedirs(RES, exist_ok = True)
    txt = "\n".join(out)
    open(os.path.join(RES, "network_per_block.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
