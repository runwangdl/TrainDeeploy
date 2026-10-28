#!/usr/bin/env python3
# Core-0 timeline of the whole fixture, segmented at symbol transitions (kernels, forks). -- QW
#   python3 timeline_segments.py <trace_pe0.log> <ELF .s> <conv kernel symbol>
# Why by symbol transitions and not by Network.c line: the input/weight transposes and the DMA/tile
# code are INLINED into RunNetwork on core 0 (the *_cluster_fork symbols are only what the worker
# cores jump to), so for core 0 they are indistinguishable by symbol and appear as "RunNetwork"
# segments between the kernels. Their identity follows from their position in the node order.
import re, sys, bisect
trace, dis, conv = sys.argv[1], sys.argv[2], sys.argv[3]
funcs = []
for l in open(dis, errors = "ignore"):
    m = re.match(r"^([0-9a-f]{8}) <([^>]+)>:", l)
    if m: funcs.append((int(m.group(1), 16), m.group(2)))
funcs.sort(); starts = [f[0] for f in funcs]
def sym(pc):
    i = bisect.bisect_right(starts, pc) - 1
    return funcs[i][1] if i >= 0 else "?"
seq = []
for l in open(trace):
    if "pe0/insn" not in l: continue
    m = re.match(r"^\d+: (\d+): .* M ([0-9a-f]{8}) ", l)
    if m: seq.append((int(m.group(1)), int(m.group(2), 16)))
first = next(i for i, (t, pc) in enumerate(seq) if sym(pc) == "RunNetwork")
NET = lambda s: s == "RunNetwork" or s.endswith("_cluster_fork") or s.startswith(("PULP_", "pulp_nn_", "ApplyPerturb", "Xorshift"))
last = max(i for i, (t, pc) in enumerate(seq) if NET(sym(pc)))
def group(s):
    if s.startswith(("PULP_", "pulp_nn_")): return "conv kernel"
    if "output_0_pre_transpose" in s: return "output transpose (fork)"
    if s.startswith(("ApplyPerturb", "Xorshift")): return "perturb kernel"
    if s == "RunNetwork": return "RunNetwork (inline: transposes-in, DMA, tiling, fork/join)"
    return s
runs = []; cur = None
for i in range(first, last + 1):
    t, pc = seq[i]; g = group(sym(pc)); cost = seq[i + 1][0] - t if i + 1 < len(seq) else 1
    if cur and cur[0] == g: cur[1] += 1; cur[2] += cost
    else: cur = [g, 1, cost]; runs.append(cur)
tot = seq[last][0] - seq[first][0]
print(f"core 0 timeline inside RunNetwork: {tot} cycles. Segments in order (group, instructions, cycles):")
for g, n, c in runs:
    if c >= 200: print(f"  {g:62s} {n:7d} {c:8d}")
from collections import defaultdict
agg = defaultdict(int)
for g, n, c in runs: agg[g] += c
print("\nby group:")
for g in sorted(agg, key = lambda g: -agg[g]): print(f"  {g:62s} {agg[g]:8d}  {100*agg[g]/tot:5.1f} %")
