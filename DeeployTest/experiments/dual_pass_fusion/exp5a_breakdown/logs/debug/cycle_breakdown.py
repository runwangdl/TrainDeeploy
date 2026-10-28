#!/usr/bin/env python3
# Attribute the cycles of one trace window to code categories, via the compiler's line info. -- QW
#   python3 cycle_breakdown.py <window.log> <objdump -d -l output> <FlatConv|DualConv> [pixels]
# Cost of an instruction = cycle stamp of the next instruction - its own stamp (stalls charged to the
# instruction that precedes the gap). The last instruction of the window is charged 1.
import re, sys
from collections import defaultdict
trace, dis, kind = sys.argv[1], sys.argv[2], sys.argv[3]
pixels = int(sys.argv[4]) if len(sys.argv) > 4 else 1

# category by source line range (inclusive); anything else in the kernel file -> "kernel:other"
CATS = {
    "FlatConv": [("setup / per-tile", 44, 63), ("pixel-loop control", 64, 65), ("gather", 66, 82),
                 ("channel-loop control", 84, 87), ("dot product", 88, 91), ("requant + store", 93, 94),
                 ("pixel-loop control", 96, 99), ("barrier", 101, 101)],
    "DualConv": [("setup / per-tile", 76, 103), ("setup / per-tile", 118, 128), ("pixel-loop control", 130, 131),
                 ("gather", 132, 148), ("channel-loop control", 150, 154), ("dot product", 155, 159),
                 ("requant + store", 161, 164), ("pixel-loop control", 166, 169), ("barrier", 171, 171)],
}[kind]
srcfile = kind + ".c"

pc2line, cur = {}, None
for l in open(dis, errors = "ignore"):
    m = re.match(r"^; .*/([^/:]+):(\d+)", l)
    if m:
        cur = (m.group(1), int(m.group(2))); continue
    m = re.match(r"^\s*([0-9a-f]+):", l)
    if m:
        pc2line[int(m.group(1), 16)] = cur

def category(pc):
    fl = pc2line.get(pc)
    if fl is None: return "outside kernel"
    f, ln = fl
    if f != srcfile: return "inlined from " + f
    for name, a, b in CATS:
        if a <= ln <= b: return name
    return f"kernel:other (line {ln})"

seq = []
for l in open(trace):
    if "pe0/insn" not in l: continue
    m = re.match(r"^\d+: (\d+): .* M ([0-9a-f]{8}) (\S+)", l)
    if m: seq.append((int(m.group(1)), int(m.group(2), 16), m.group(3)))

cyc, cnt = defaultdict(int), defaultdict(int)
for i, (t, pc, mn) in enumerate(seq):
    cost = seq[i + 1][0] - t if i + 1 < len(seq) else 1
    c = category(pc); cyc[c] += cost; cnt[c] += 1
tot_c, tot_i = sum(cyc.values()), sum(cnt.values())
print(f"{kind}: {tot_i} instructions, {tot_c} cycles in the window ({pixels} pixel(s))\n")
print(f"{'category':28s} {'instr':>7s} {'cycles':>7s} {'stall':>6s} {'%cyc':>6s} {'cyc/px':>7s}")
order = ["setup / per-tile", "pixel-loop control", "gather", "channel-loop control", "dot product", "requant + store", "barrier"]
for c in order + sorted(k for k in cyc if k not in order):
    if c not in cyc: continue
    print(f"{c:28s} {cnt[c]:7d} {cyc[c]:7d} {cyc[c]-cnt[c]:6d} {100*cyc[c]/tot_c:6.1f} {cyc[c]/pixels:7.1f}")
pay = cyc["dot product"] + cyc["requant + store"]
print(f"\npayload (dot + requant/store) = {pay} cycles = {100*pay/tot_c:.1f} % of the window")
