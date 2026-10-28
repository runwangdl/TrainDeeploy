#!/usr/bin/env python3
# For one trace window: every conditional branch PC -> executions, taken count, source line. -- QW
#   python3 branch_classify.py <window.log> <objdump -d -l output of the kernel>
import re, sys
from collections import defaultdict
trace, dis = sys.argv[1], sys.argv[2]
pc2line, cur = {}, "?"
for l in open(dis, errors = "ignore"):
    m = re.match(r"^; .*/([^/:]+):(\d+)", l)
    if m:
        cur = f"{m.group(1)}:{m.group(2)}"; continue
    m = re.match(r"^\s*([0-9a-f]+):", l)
    if m:
        pc2line[int(m.group(1), 16)] = cur
seq = []
for l in open(trace):
    if "pe0/insn" not in l: continue
    m = re.search(r" M ([0-9a-f]{8}) (\S+)", l)
    if m: seq.append((int(m.group(1), 16), m.group(2)))
BR = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz", "bnez"}
br = defaultdict(lambda: [0, 0, ""])
for i, (pc, mn) in enumerate(seq[:-1]):
    if mn in BR:
        taken = seq[i + 1][0] not in (pc + 2, pc + 4)
        br[pc][0] += 1; br[pc][1] += taken; br[pc][2] = mn
print(f"{'PC':>10} {'insn':6} {'exec':>5} {'taken':>5}  source line")
for pc, (n, t, mn) in sorted(br.items()):
    print(f"{pc:>10x} {mn:6} {n:5d} {t:5d}  {pc2line.get(pc, '?')}")
print("total branches", sum(v[0] for v in br.values()), "taken", sum(v[1] for v in br.values()),
      "| instructions", len(seq))
