#!/usr/bin/env python3
# Level-1 breakdown of the conv node: attribute every cycle core 0 spent inside the kernel to a
# code category, by ADDRESS RANGE (hand-derived from the disassembly) and, as a cross-check, by
# C SOURCE LINE (compiler debug info). -- QW, exp5a_conv_breakdown
#
#   python3 conv_breakdown.py <trace window> <line-annotated disassembly> <kernel: FlatConv|DualConv> [pixels]
#
# Cost of an instruction = cycle stamp of the NEXT instruction - its own stamp. Stalls (load-use,
# branch flush, fetch miss) are therefore charged to the instruction that precedes the gap, and the
# category sums add up to the window's span exactly.
import re, sys
from collections import defaultdict

trace, dis, kind = sys.argv[1], sys.argv[2], sys.argv[3]
pixels = int(sys.argv[4]) if len(sys.argv) > 4 else 1
# optional: shift every address-range / EXIT / ENTRY constant by this many bytes (same code at another address) -- QW
import os
OFF = int(os.environ.get("ADDR_OFFSET", "0"), 0)

# --- categories by address range (inclusive), read off bin/QZO_Conv.s / QZO_DualConv.s ------------
ADDR = {
    "FlatConv": [
        (0x1c00f1b0, 0x1c00f2fc, "setup (per tile)"),
        (0x1c00f300, 0x1c00f314, "pixel-loop control"),
        (0x1c00f4d4, 0x1c00f4e8, "pixel-loop control"),
        (0x1c00f318, 0x1c00f31c, "gather"),
        (0x1c00f398, 0x1c00f4ac, "gather"),
        (0x1c00f320, 0x1c00f344, "channel-loop control"),
        (0x1c00f388, 0x1c00f394, "channel-loop control"),
        (0x1c00f348, 0x1c00f35c, "dot product"),
        (0x1c00f360, 0x1c00f384, "requant + store"),
        (0x1c00f4b0, 0x1c00f4d0, "channel loop (empty-K copy, dead)"),
        (0x1c00f4ec, 0x1c00f52c, "epilogue (barrier + restore)"),
    ],
    # DualConv (build_dual/bin/QZO_DualConv.s, kernel 1c00f0e8-1c00f808). Phase 1 = the perturbation,
    # once per tile; everything from the partition on is the pixel loop (phase 2).
    "DualConv": [
        (0x1c00f0e8, 0x1c00f208, "setup (per tile)"),                 # prologue, scratch map
        (0x1c00f20c, 0x1c00f380, "phase 1: perturb W (per tile)"),    # dc_chunk + dc_perturb_i8
        (0x1c00f384, 0x1c00f4ac, "phase 1: perturb b (per tile)"),    # dc_chunk + dc_perturb_i32
        (0x1c00f4b0, 0x1c00f4e4, "setup (per tile)"),                 # patch tail zero
        (0x1c00f4e8, 0x1c00f4ec, "phase 1: barrier (per tile)"),
        (0x1c00f4f0, 0x1c00f5a8, "setup (per tile)"),                 # partition p0/p1, oy/ox, hoisted invariants
        (0x1c00f5ac, 0x1c00f5c4, "pixel-loop control"),
        (0x1c00f690, 0x1c00f6a8, "pixel-loop control"),
        (0x1c00f5c8, 0x1c00f5cc, "gather"),
        (0x1c00f6ac, 0x1c00f7c4, "gather"),
        (0x1c00f5d0, 0x1c00f5ec, "channel-loop control"),
        (0x1c00f640, 0x1c00f664, "channel-loop control"),
        (0x1c00f5f0, 0x1c00f5f4, "channel loop (empty-K copy, dead)"),
        (0x1c00f668, 0x1c00f68c, "dot product"),
        (0x1c00f5f8, 0x1c00f63c, "requant + store"),
        (0x1c00f7c8, 0x1c00f808, "epilogue (barrier + restore)"),
    ],
}
# --- categories by C source line (inclusive) --------------------------------------------------------
LINES = {
    "FlatConv": [("setup (per tile)", 44, 63), ("pixel-loop control", 64, 65), ("gather", 66, 82),
                 ("channel-loop control", 84, 87), ("dot product", 88, 91), ("requant + store", 93, 94),
                 ("pixel-loop control", 96, 99), ("epilogue (barrier + restore)", 101, 102)],
    "DualConv": [("setup (per tile)", 76, 128), ("pixel-loop control", 130, 131), ("gather", 132, 148),
                 ("channel-loop control", 150, 154), ("dot product", 155, 159), ("requant + store", 161, 164),
                 ("pixel-loop control", 166, 169), ("epilogue (barrier + restore)", 171, 172)],
}
srcfile = kind + ".c"

pc2line, cur = {}, None
for l in open(dis, errors = "ignore"):
    m = re.match(r"^; .*/([^/:]+):(\d+)", l)
    if m:
        cur = (m.group(1), int(m.group(2))); continue
    m = re.match(r"^\s*([0-9a-f]+):", l)
    if m:
        pc2line[int(m.group(1), 16)] = cur

def cat_by_addr(pc):
    for a, b, name in ADDR.get(kind, []):
        if a + OFF <= pc <= b + OFF: return name
    return "outside kernel"

def cat_by_line(pc):
    fl = pc2line.get(pc)
    if fl is None: return "outside kernel"
    f, ln = fl
    if f != srcfile: return "inlined from " + f
    for name, a, b in LINES[kind]:
        if a <= ln <= b: return name
    return f"kernel:other (line {ln})"

seq = []
for l in open(trace):
    if "pe0/insn" not in l: continue
    m = re.match(r"^\d+: (\d+): .* M ([0-9a-f]{8}) ", l)
    if m: seq.append((int(m.group(1)), int(m.group(2), 16)))

# Two gaps that belong to neither category they fall on by address:
#  * the epilogue's `lui` -> the barrier-register `lw`: the core sleeps in the end-of-tile barrier
#    until the slowest core arrives (GVSoC stamps the blocking `lw` when it completes, so the wait
#    appears as the gap after the `lui`), so that gap is "end-of-tile barrier wait";
#  * `ret` of tile k -> kernel entry of tile k+1: the inter-tile DMA / sequencing, outside the kernel.
EXIT  = {"FlatConv": (0x1c00f4ec, 0x1c00f4f0), "DualConv": (0x1c00f7c8, 0x1c00f7cc)}  # (epilogue lui, barrier lw)
ENTRY = {"FlatConv": (0x1c00f52c, 0x1c00f1b0), "DualConv": (0x1c00f808, 0x1c00f0e8)}  # (ret, kernel start)

def tally(catfn):
    cyc, cnt = defaultdict(int), defaultdict(int)
    for i, (t, pc) in enumerate(seq):
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        cost = nxt[0] - t if nxt else 1
        c = catfn(pc)
        if nxt and (pc - OFF, nxt[1] - OFF) == EXIT.get(kind) and cost > 1:
            cyc[c] += 1; cnt[c] += 1; cyc["end-of-tile barrier wait"] += cost - 1; continue
        if nxt and (pc - OFF, nxt[1] - OFF) == ENTRY.get(kind) and cost > 1:
            cyc[c] += 1; cnt[c] += 1; cyc["inter-tile gap (outside kernel)"] += cost - 1; continue
        cyc[c] += cost; cnt[c] += 1
    return cyc, cnt

order = ["setup (per tile)", "phase 1: perturb W (per tile)", "phase 1: perturb b (per tile)", "phase 1: barrier (per tile)", "pixel-loop control", "gather", "channel-loop control", "dot product",
         "requant + store", "epilogue (barrier + restore)", "end-of-tile barrier wait", "inter-tile gap (outside kernel)"]
span = seq[-1][0] - seq[0][0] + 1
print(f"{kind}: {len(seq)} instructions, span {span} cycles ({pixels} pixel(s) on core 0)\n")
for title, fn in (("BY ADDRESS RANGE", cat_by_addr), ("BY C SOURCE LINE (cross-check)", cat_by_line)):
    if kind not in ADDR and fn is cat_by_addr: continue
    cyc, cnt = tally(fn); tot = sum(cyc.values())
    print(f"--- {title} ---")
    print(f"{'category':36s} {'instr':>8s} {'cycles':>8s} {'stall':>7s} {'%cyc':>6s} {'cyc/px':>8s}")
    for c in order + sorted(k for k in cyc if k not in order):
        if c not in cyc: continue
        print(f"{c:36s} {cnt[c]:8d} {cyc[c]:8d} {cyc[c]-cnt[c]:7d} {100*cyc[c]/tot:6.1f} {cyc[c]/pixels:8.1f}")
    pay = cyc["dot product"] + cyc["requant + store"]
    print(f"{'TOTAL':36s} {sum(cnt.values()):8d} {tot:8d} {tot-sum(cnt.values()):7d}")
    print(f"payload (dot + requant/store) = {pay} cycles = {100*pay/tot:.1f} %\n")
