#!/usr/bin/env python3
# exp21a: byte-exact decomposition of the BP single-step GAP9 memory (exp9 memsim/bp) -- QW
#   python3 analyze_bp_mem.py [evidence_dir]        (default: ./evidence)
# Reads TrainingNetwork.c / OptimizerNetwork.c (every *_malloc) and deeployStates/memory_alloc.html (tiler memory map:
# per-tensor address range + lifetime in schedule steps) and prints markdown tables + writes steps.csv for the figure.
#
# Layout facts used (all verified below by asserts):
#   L2 plot y-axis = [0, 99,752) promoted graph inputs | [99,752, 787,240) PROMOTED_POOL_L2 | [787,240, ...) MEMORYARENA_L2
#   L3 plot y-axis = 880 B of constants (832 B BN running stats = separate cl_ram_malloc, 48 B folded consts that are not
#   in the C) + 20 B tileIdxPtr (C stack locals, no memory) + MEMORYARENA_L3 at plot offset 900.
import csv, json, os, re, sys
from collections import defaultdict

EV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence")
TN = open(os.path.join(EV, "TrainingNetwork.c")).read()
ON = open(os.path.join(EV, "OptimizerNetwork.c")).read()
H = open(os.path.join(EV, "memory_alloc.html")).read()
SZ = {"int8_t": 1, "uint8_t": 1, "float32_t": 4, "int64_t": 8, "int32_t": 4}
NW = 22  # trainable tensors: input_2..input_23 = weights, input_24..input_45 = their grad accumulators, input_46 = flag


def line_of(src, needle):
    return src[:src.find(needle)].count("\n") + 1


# ---------------------------------------------------------------- 1. mallocs in the generated C
mall = []
for m in re.finditer(r"(\w+) = (?:\([^)]*\))?\s*(pi_l2_malloc|cl_ram_malloc|pi_l1_malloc)\((?:\(void \*\)0, )?"
                     r"(?:sizeof\((\w+)\) \* (\d+)|(\d+))\)", TN):
    name, fn, ty, n, raw = m.groups()
    b = SZ[ty] * int(n) if ty else int(raw)
    mall.append((name.replace("DeeployNetwork_", ""), fn, b, TN[:m.start()].count("\n") + 1))
assert not re.search(r"(pi_l2_malloc|cl_ram_malloc)\(", ON), "OptimizerNetwork.c is expected to allocate nothing"


def param(k):  # graph-input index -> meaning (order verified against the tile lifetimes in the plot)
    if k == 0: return "data window (input_0)"
    if k == 1: return "label (input_1)"
    if 2 <= k <= 21:
        blk, r = divmod(k - 2, 4)
        return f"block{blk} " + ["conv W", "conv b", "BN gamma", "BN beta"][r] + f" (input_{k})"
    if k == 22: return "fc W (input_22)"
    if k == 23: return "fc b (input_23)"
    if 24 <= k <= 45: return "grad-acc of " + param(k - NW).split(" (")[0] + f" (input_{k})"
    if k == 46: return "lazy-reset flag (input_46)"


L2 = [(n, b, ln) for n, fn, b, ln in mall if fn == "pi_l2_malloc"]
L3 = [(n, b, ln) for n, fn, b, ln in mall if fn == "cl_ram_malloc"]
L1 = [(n, b, ln) for n, fn, b, ln in mall if fn == "pi_l1_malloc"]
print("## 1. Static allocations in TrainingNetwork.c (OptimizerNetwork.c: none, it aliases these)\n")
print("| level | buffer | bytes | TrainingNetwork.c line | what it is |\n|---|---|--:|--:|---|")
what = {"MEMORYARENA_L2": "tiler arena: L2 staging (double-buffered tiles) of L3-resident tensors",
        "PROMOTED_POOL_L2": "promoted activations/gradients, whole tensors, packed by lifetime",
        "MEMORYARENA_L3": "L3 home of every non-promoted tensor (params, grad-acc, BN-side activations)",
        "MEMORYARENA_L1": "L1 tile arena (single-buffered)"}
for lvl, lst in (("L2", L2), ("L3", L3), ("L1", L1)):
    for n, b, ln in lst:
        w = what.get(n)
        if w is None and n.startswith("input_"): w = "promoted graph input: " + param(int(n[6:]))
        if w is None and "running_" in n: w = "BN running stat (frozen BN, hex-loaded)"
        print(f"| {lvl} | {n} | {b:,} | {ln} | {w} |")
sL2, sL3 = sum(b for _, b, _ in L2), sum(b for _, b, _ in L3)
print(f"| **L2 sum** | | **{sL2:,}** | | |\n| **L3 sum** | | **{sL3:,}** | | |\n")
assert sL2 == 1024552 and sL3 == 1006480, (sL2, sL3)
ARENA_L2 = dict((n, b) for n, b, _ in L2)["MEMORYARENA_L2"]
POOL = dict((n, b) for n, b, _ in L2)["PROMOTED_POOL_L2"]
ARENA_L3 = dict((n, b) for n, b, _ in L3)["MEMORYARENA_L3"]


# ---------------------------------------------------------------- 2. memory map from the plot
def fig(level):
    i = H.find(f'id="plot-{level}"'); j = H.find("var fig = ", i); k = H.find(";\n", j)
    out = []
    for t in json.loads(H[j + 10:k])["data"]:
        if not t.get("x") or (t.get("line") or {}).get("color") == "red": continue
        out.append(dict(name = t["name"], lo = min(t["y"]), hi = max(t["y"]), s0 = min(t["x"]), s1 = max(t["x"]),
                        color = t.get("fillcolor")))
    return out


def union_bytes(iv):
    tot, cur = 0, None
    for a, b in sorted(iv):
        if cur is None or a > cur[1]:
            tot += (cur[1] - cur[0]) if cur else 0; cur = [a, b]
        else: cur[1] = max(cur[1], b)
    return tot + ((cur[1] - cur[0]) if cur else 0)


alive = lambda t, s: t["s0"] <= s <= t["s1"]
l2, l3 = fig("L2"), fig("L3")
CONST_END = 99752
POOL_END = CONST_END + POOL
const = [t for t in l2 if t["color"] == "lightblue"]
pool = [t for t in l2 if t["color"] == "orange"]
arena = [t for t in l2 if t["color"] is None]
assert all(t["hi"] <= CONST_END for t in const) and sum(t["hi"] - t["lo"] for t in const) == CONST_END
assert all(CONST_END <= t["lo"] and t["hi"] <= POOL_END for t in pool)
assert all(t["lo"] >= POOL_END for t in arena) and max(t["hi"] for t in arena) == POOL_END + ARENA_L2
A3 = 900  # plot offset of MEMORYARENA_L3
l3a = [t for t in l3 if t["color"] is None]
assert min(t["lo"] for t in l3a) == A3 and max(t["hi"] for t in l3a) == A3 + ARENA_L3
steps = range(0, 106)


def live(ts, s): return union_bytes([(t["lo"], t["hi"]) for t in ts if alive(t, s)])


rows = [(s, live(pool, s), live(arena, s), live(l3a, s)) for s in steps]
with open(os.path.join(os.path.dirname(EV), "steps.csv"), "w", newline = "") as f:
    w = csv.writer(f); w.writerow(["step", "l2_promoted_inputs", "l2_promoted_pool_live", "l2_tile_arena_live",
                                   "l3_arena_live"])
    for s, p, a, l in rows: w.writerow([s, CONST_END, p, a, l])


def comp(ts, s, base):
    seen, out = set(), []
    for t in sorted((t for t in ts if alive(t, s)), key = lambda t: t["lo"]):
        r = (t["lo"], t["hi"])
        if r in seen: continue
        seen.add(r); out.append((r[0] - base, r[1] - r[0], t["name"]))
    return out


def short(n):
    n = re.sub(r"^node_\d+_", "", n)
    n = n.replace("blocks_", "b").replace("BatchNormalization", "BN").replace("_tensor", "")
    return n[:110]


# pool
ps = max(steps, key = lambda s: live(pool, s))
print(f"## 2. PROMOTED_POOL_L2 ({POOL:,} B) = its peak, schedule step {ps}\n")
print("| pool offset | bytes | tensor |\n|--:|--:|---|")
c = comp(pool, ps, CONST_END)
for o, b, n in c: print(f"| {o:,} | {b:,} | {short(n)} |")
print(f"| | **{sum(b for _, b, _ in c):,}** | |\n")
assert sum(b for _, b, _ in c) == POOL
# arena
as_ = max(steps, key = lambda s: live(arena, s))
print(f"## 3. MEMORYARENA_L2 = tile arena ({ARENA_L2:,} B) = its peak, schedule step {as_}\n")
print("| arena offset | bytes | L2 staging buffer (2 x tile, double-buffered) for |\n|--:|--:|---|")
c = comp(arena, as_, POOL_END)
for o, b, n in c: print(f"| {o:,} | {b:,} | {short(n)} |")
print(f"| | **{sum(b for _, b, _ in c):,}** | |\n")
assert sum(b for _, b, _ in c) == ARENA_L2
# L2 time-peak (what mem_plot_summary.py reported)
tot = [(s, CONST_END + p + a) for s, p, a, _ in rows]
pk = max(tot, key = lambda x: x[1])
print(f"## 4. L2 live-bytes peak over time = {pk[1]:,} B at step {pk[0]} "
      f"(inputs {CONST_END:,} + pool live {rows[pk[0]][1]:,} + tile-arena live {rows[pk[0]][2]:,})\n")
print("| step | promoted inputs | pool live | tile-arena live | L2 live | L3-arena live |\n|--:|--:|--:|--:|--:|--:|")
for s in sorted({pk[0], ps, as_, max(steps, key = lambda s: rows[s][3])}):
    _, p, a, l = rows[s]
    print(f"| {s} | {CONST_END:,} | {p:,} | {a:,} | {CONST_END + p + a:,} | {l:,} |")
print()

# ---------------------------------------------------------------- 3. L3 arena
print(f"## 5. MEMORYARENA_L3 ({ARENA_L3:,} B)\n")
cat = defaultdict(int); persist = [t for t in l3a if t["s0"] < 0]
seen = set()
for t in persist:
    r = (t["lo"], t["hi"])
    if r in seen: continue
    seen.add(r)
    k = int(re.sub(r"\D", "", t["name"])) if t["name"].startswith(("input_", "output_")) else None
    if t["name"].startswith("output_"): continue  # outputs alias the grad-acc inputs (same range)
    p = param(k)
    key = ("gradient accumulators (22, all trainable tensors; 1-user graph I/O -> not promotable)" if 24 <= k <= 45
           else "lazy-reset flag (< 8 B -> not promotable)" if k == 46
           else "params kept in L3 (conv0 W, 5 conv biases, fc b: 1 user; BN gamma/beta: BN neighbour)")
    cat[key] += r[1] - r[0]
pers_end = max(t["hi"] for t in persist) - A3
s3 = max(steps, key = lambda s: rows[s][3])
print(f"Persistent region [0, {pers_end:,}) (alive the whole schedule):\n")
print("| category | bytes |\n|---|--:|")
for k, v in cat.items(): print(f"| {k} | {v:,} |")
print(f"| **persistent sum** | **{sum(cat.values()):,}** |\n")
assert sum(cat.values()) == pers_end
print(f"Activation region [{pers_end:,}, {ARENA_L3:,}) = its peak, step {s3} (block-0 backward):\n")
print("| arena offset | bytes | tensor |\n|--:|--:|---|")
c = [x for x in comp(l3a, s3, A3) if x[0] >= pers_end]
for o, b, n in c: print(f"| {o:,} | {b:,} | {short(n)} |")
gap = ARENA_L3 - pers_end - sum(b for _, b, _ in c)
others = [x for x in comp([t for t in l3a if t["s0"] >= 0], 0, A3)]
print(f"| | {gap:,} | other ranges not live at step {s3} (output_0 loss 4 B, blocks_0_1_weight_grad 32 B,...) |")
print(f"| | **{ARENA_L3 - pers_end:,}** | |\n")
# exact static layout of the activation region tail
tail = sorted({(t['lo'] - A3, t['hi'] - A3, t['name']) for t in l3a if t['hi'] - A3 > 1005000})
print("Tail ranges:", [(a, b, short(n)) for a, b, n in tail], "\n")

# ---------------------------------------------------------------- 4. redundancy / single-level bound
names2 = {t["name"] for t in pool + const}
names3 = {t["name"] for t in l3a}
both = sorted(names2 & names3)
print(f"## 6. Same tensor name homed in both L2 (pool/inputs) and L3 arena: {both or 'none'}\n")
comb = max(steps, key = lambda s: CONST_END + rows[s][1] + rows[s][3])
print(f"## 7. Single-level lower bound: max_step(promoted inputs + pool live + L3 live) = "
      f"{CONST_END + rows[comb][1] + rows[comb][3]:,} B at step {comb} (+ L3 BN stats 832, + staging)\n")
