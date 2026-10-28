#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp8: wall-clock per node category from a --profileTiling log of the MeZO runner (training graph executions + update
graph). Usage: python3 categorize.py <log> [<log> ...]   (one column per log)"""
import collections, re, sys

PAT = re.compile(r"^\[(\S+?)_L2\]\[SB\]\[\d+ ops\]\[Tile \d+\] (Pre-Kernel|Kernel     |Post-Kernel):\s*(\d+) cycles", re.M)


def cat(n):
    if "dual" in n and "Conv" in n and "transpose" not in n: return "conv (fused dual)"
    if n.startswith("rqsp_") and "conv" in n: return "conv perturbation"
    if "_mode" in n and n.startswith("pert_"): return "fp32 perturbation"
    if n.startswith("pert_"): return "fp32 perturbation"
    if "transpose" in n.lower() and ("CONVRQ" in n or "Conv" in n): return "conv transposes"
    if "MERGE_CONVRQ" in n: return "conv (pulp-nn)"
    if "BatchNorm" in n or n.startswith("bn"): return "BN"
    if "Quant" in n or "Dequant" in n: return "Quant/Dequant"
    if "pool" in n.lower(): return "pool"
    if "relu" in n.lower(): return "ReLU"
    if "transpose" in n.lower(): return "other transposes"
    if "Gemm" in n or "CrossEntropy" in n or "Reshape" in n: return "fc + loss"
    return "other"


def agg(path):
    a = collections.OrderedDict()
    for n, k, c in PAT.findall(open(path, errors = "ignore").read()):
        a[cat(n)] = a.get(cat(n), 0) + int(c)
    return a


if __name__ == "__main__":
    cols = [agg(p) for p in sys.argv[1:]]
    keys = sorted({k for c in cols for k in c}, key = lambda k: -max(c.get(k, 0) for c in cols))
    print(f"{'category':22s}" + "".join(f"{p.split('/')[-1][:24]:>26s}" for p in sys.argv[1:]))
    for k in keys:
        print(f"{k:22s}" + "".join(f"{c.get(k, 0):>26,}" for c in cols))
    print(f"{'sum':22s}" + "".join(f"{sum(c.values()):>26,}" for c in cols))
