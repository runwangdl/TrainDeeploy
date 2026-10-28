#!/usr/bin/env python3
# Tally one pixel pair of the pe0 trace by function: lines touched vs lines refilled. -- QW
#   python3 tally_pair.py <QZO_Conv.s> <pairNNN.log>
import re, sys, bisect
from collections import Counter, defaultdict
dis, pair = sys.argv[1], sys.argv[2]
funcs = []
for l in open(dis, errors = "ignore"):
    m = re.match(r"^([0-9a-f]{8}) <([^>]+)>:", l)
    if m:
        funcs.append((int(m.group(1), 16), m.group(2)))
funcs.sort()
starts = [f[0] for f in funcs]
def fn(addr):
    i = bisect.bisect_right(starts, addr) - 1
    return funcs[i][1] if i >= 0 else "?"
pcs, refills = [], []
for l in open(pair):
    if "pe0/insn" in l:
        m = re.search(r" M ([0-9a-f]{8}) ", l)
        if m: pcs.append(int(m.group(1), 16))
    m = re.search(r"Refilling line \(addr: 0x([0-9a-f]+)", l)
    if m: refills.append(int(m.group(1), 16))
touched, executed, refill_lines = defaultdict(set), Counter(), defaultdict(Counter)
for pc in pcs:
    touched[fn(pc)].add(pc >> 4); executed[fn(pc)] += 1
for a in refills:
    refill_lines[fn(a)][a >> 4] += 1
print(f"{len(pcs)} instructions, {len(set(p >> 4 for p in pcs))} distinct lines touched, "
      f"{len(refills)} refills on {len(set(refills))} distinct lines\n")
print(f"{'function':34s} {'instr':>6s} {'lines touched':>14s} {'lines refilled':>15s} {'refills':>8s}")
for f in sorted(touched, key = lambda f: -len(touched[f])):
    rl = refill_lines.get(f, Counter())
    print(f"{f:34s} {executed[f]:6d} {len(touched[f]):14d} {len(rl):15d} {sum(rl.values()):8d}")
print("\nlines refilled more than once within the pair:")
for f, rl in refill_lines.items():
    for line, c in sorted(rl.items()):
        if c > 1: print(f"  {f:34s} 0x{line << 4:08x}  x{c}")
