#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp8: compare two MeZO-runner logs built with -D DUMP_WEIGHTS: [WDUMP] tensors after every step and the raw loss bits.
Usage: python3 compare_dumps.py <unfused.log> <fused.log>"""
import re, sys


def dumps(p):
    d = {}
    for l in open(p, errors = "ignore"):
        m = re.search(r"\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)", l)
        if m:
            d[(int(m.group(1)), int(m.group(2)))] = m.group(4).split()
    return d


u, f = dumps(sys.argv[1]), dumps(sys.argv[2])
bad = [(k, sum(a != b for a, b in zip(u[k], f.get(k, [])))) for k in sorted(u) if u[k] != f.get(k)]
words = sum(len(v) for v in u.values())
lu = re.findall(r"(?:lp|lm)_bits=0x[0-9a-f]+", open(sys.argv[1], errors = "ignore").read())
lf = re.findall(r"(?:lp|lm)_bits=0x[0-9a-f]+", open(sys.argv[2], errors = "ignore").read())
print(f"tensors: {len(u)} vs {len(f)} dumps, {words} words; mismatching (step, tensor): {bad if bad else 'none'}")
print(f"loss bits: {'identical' if lu == lf else 'DIFFER'} ({len(lu)} values)")
if lu != lf:
    print(" unfused", lu, "\n fused  ", lf)
