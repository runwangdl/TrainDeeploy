#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp21c task 3: reconcile the whole-network speedup (unfused 13.27 M -> fused NE16 6.57 M per update step) with the
per-block fixture speedups. Per window (one of the n_accum = 4 training-graph executions of an update step):

  conv chain of block i (exp8 metric) = conv node(s) + its int8 weight / int32 bias perturbation + weight transpose,
                                        and for NE16 the wenc / bias prep nodes and block 0's im2col node
  everything else                     = input quant/transpose, BN perturbation (fp32), BN+pool+ReLU+Quant tails,
                                        block-4 BN/ReLU/pool, FC + loss (+ fc perturbation)
  outside the nodes                   = plain BENCH / 4 - node sum / 4

Fixture comparison: exp11 per-block microbench node sums WITHOUT NCHW boundary transposes ("sum basis"; A = 2 x pulp-nn
incl. its perturbation and weight-transpose nodes), block 0 from guide 05 sec. 6.6. -- QW exp21c
  python3 reconcile.py  -> results/reconcile.txt
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_network import block_of, parse, bench  # noqa: E402

LOGS = os.path.join(HERE, "..", "logs")
RES = os.path.join(HERE, "..", "results")
N_WIN = 4
EXP9 = {"eco": (13266794, 558310), "ne16": (6567942, 550742), "fus": (8564124, 550883)}  # exp9 memsim/summary.txt
# exp11 results/breakdown.md + logs/o3enc (fixture node sums, both passes, no boundary transposes)
FIX = {  # block: (A sum, D node, E nodes (wenc + bias + conv))
    1: (32842 + 3670 + 6296 + 623498, 425022, 31656 + 11980 + 198062),
    2: (32920 + 3646 + 6878 + 176246, 128916, 32208 + 12058 + 61303),
    3: (54962 + 3790 + 9874 + 41768, 50042, 54595 + 18414 + 21911),
    4: (103648 + 3788 + 12504 + 45170, 42673, 105320 + 31558 + 17071),
}
FIX0 = (1209582, 255923)  # guide 05 sec. 6.6, node level fixture, unfused vs fused NE16


def is_conv(kind):
    return any(s in kind for s in ("conv", "weight transpose", "NE16 prep", "im2col"))


def per_window(cfg):
    p = os.path.join(LOGS, f"net_{cfg}_prof.log")
    if not os.path.exists(p) or "BENCH train_cycles" not in open(p, errors = "ignore").read():
        return None  # missing or still running
    opt = set()
    cg = os.path.join(HERE, "..", "codegen", f"net_{cfg}_prof_OptimizerNetwork.c")
    if os.path.exists(cg):
        opt = set(re.findall(r"\[(\S+?)_L2\]\[SB\]", open(cg, errors = "ignore").read()))
    conv = {b: 0 for b in range(5)}
    rest = {}
    for n, c in parse(p).items():
        if n in opt:
            continue
        b, k = block_of(n)
        if isinstance(b, int) and is_conv(k):
            conv[b] += c / N_WIN
        else:
            rest[k] = rest.get(k, 0) + c / N_WIN
    return conv, rest


def main():
    L = []
    data = {c: per_window(c) for c in ("eco", "ne16", "fus")}
    data = {c: v for c, v in data.items() if v}
    tot = {}
    for c in data:
        bt, bo = bench(os.path.join(LOGS, f"net_{c}_plain.log"))
        tot[c] = (bt or EXP9[c][0], "plain BENCH (this run)" if bt else "exp9 memsim BENCH")
    L.append("exp21c task 3 -- whole network vs per-block fixtures, per window (training graph; /4 of one update step)")
    L.append("")
    L.append("1) conv chain per block (wall clock, --profileTiling, logs/net_<cfg>_prof.log)")
    hdr = f"{'block':>5} {'unfused (eco)':>14} {'fused NE16':>12} {'speedup':>8}"
    if "fus" in data:
        hdr += f" {'fused cluster':>14} {'speedup':>8}"
    hdr += f" | {'fixture A':>10} {'fixture D':>10} {'fixture E':>10} {'A/D':>6} {'A/E':>6} {'used':>8}"
    L.append(hdr)
    sums = {c: 0 for c in data}
    for b in range(5):
        e, n = data["eco"][0][b], data["ne16"][0][b]
        for c in data:
            sums[c] += data[c][0][b]
        row = f"{b:>5} {e:>14,.0f} {n:>12,.0f} {e / n:>7.2f}x"
        if "fus" in data:
            f_ = data["fus"][0][b]
            row += f" {f_:>14,.0f} {e / f_:>7.2f}x"
        if b == 0:
            row += f" | {FIX0[0]:>10,} {'':>10} {FIX0[1]:>10,} {'':>6} {FIX0[0] / FIX0[1]:>5.2f}x {'NE16':>8}"
        else:
            a, d, ee = FIX[b]
            row += f" | {a:>10,} {d:>10,} {ee:>10,} {a / d:>5.2f}x {a / ee:>5.2f}x {'NE16' if b < 3 else 'cluster':>8}"
        L.append(row)
    row = f"{'0-4':>5} {sums['eco']:>14,.0f} {sums['ne16']:>12,.0f} {sums['eco'] / sums['ne16']:>7.2f}x"
    if "fus" in data:
        row += f" {sums['fus']:>14,.0f} {sums['eco'] / sums['fus']:>7.2f}x"
    L.append(row)
    L.append("")
    L.append("2) everything else, per window")
    keys = sorted({k for c in data for k in data[c][1]})
    L.append(f"{'':48s}" + "".join(f"{c:>14s}" for c in data))
    for k in keys:
        L.append(f"{k[:48]:48s}" + "".join(f"{data[c][1].get(k, 0):>14,.0f}" for c in data))
    rs = {c: sum(data[c][1].values()) for c in data}
    L.append(f"{'sum non-conv':48s}" + "".join(f"{rs[c]:>14,.0f}" for c in data))
    L.append("")
    L.append("3) totals per window")
    for c in data:
        ns = sums[c] + rs[c]
        bt = tot[c][0] / N_WIN
        L.append(f"{c:5s} conv {sums[c]:>11,.0f} ({sums[c] / bt * 100:4.1f} %)  non-conv {rs[c]:>11,.0f} ({rs[c] / bt * 100:4.1f} %)  "
                 f"outside nodes {bt - ns:>9,.0f} ({(bt - ns) / bt * 100:4.1f} %)  BENCH/4 {bt:>11,.0f}  [{tot[c][1]}: {tot[c][0]:,}]")
    e, n = tot["eco"][0], tot["ne16"][0]
    L.append("")
    L.append(f"whole step: {e:,} -> {n:,} = {e / n:.3f}x (exp9: 13,266,794 -> 6,567,942 = {13266794 / 6567942:.3f}x)")
    ce, cn = sums["eco"], sums["ne16"]
    re_, rn = rs["eco"], rs["ne16"]
    oe, on = e / N_WIN - ce - re_, n / N_WIN - cn - rn
    L.append(f"Amdahl: conv {ce:,.0f} -> {cn:,.0f} ({ce / cn:.2f}x), non-conv {re_:,.0f} -> {rn:,.0f} ({re_ / rn:.3f}x), "
             f"outside {oe:,.0f} -> {on:,.0f}")
    L.append(f"  if ONLY the convs changed: {(ce + re_ + oe):,.0f} / {(cn + re_ + oe):,.0f} = {(ce + re_ + oe) / (cn + re_ + oe):.3f}x")
    L.append(f"  conv share of the unfused window: {ce / (e / N_WIN) * 100:.1f} %; of the fused window: {cn / (n / N_WIN) * 100:.1f} %")
    w = {b: data["eco"][0][b] / ce for b in range(5)}
    L.append("  unfused conv-cycle weight per block: " + ", ".join(f"b{b} {w[b] * 100:.1f} %" for b in range(5)))
    os.makedirs(RES, exist_ok = True)
    open(os.path.join(RES, "reconcile.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
