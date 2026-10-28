#!/usr/bin/env python3
"""exp21c impl: conv chain per block and update step (training graph, 4 windows) from the --profileTiling logs, with
the block classifier of ../scripts/parse_network.py. Engine choice per block: NE16 (prep + transposes + conv node) vs
fused cluster DualConv2In.   python3 conv_chain.py > ../results/conv_chain.txt   -- QW exp21c"""
import collections, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import parse_network as P  # noqa: E402
L = os.path.join(HERE, "..", "logs"); OLD = os.path.join(HERE, "..", "..", "logs")
runs = [("base ne16=012 (exp21c log)", os.path.join(OLD, "net_ne16_prof.log")),
        ("fused cluster only (exp21c log)", os.path.join(OLD, "net_fus_prof.log"))]
runs += [(f"{c} ne16={b}", os.path.join(L, f"net_{c}_prof_ne16_{b}.log")) for c, b in
         (("BH", "012"), ("BHS", "012"), ("BHS", "0123"), ("BHS", "01234"))]
CONVK = ("NE16 prep", "im2col", "conv", "weight transpose")
print("conv chain (prep + im2col + conv node(s), incl. the x_hwc/out transposes classified to the block) per UPDATE STEP")
print("(sum of 4 windows), cycles; block 'other' rows are the transposes P.block_of does not classify")
for name, p in runs:
    if not os.path.exists(p):
        continue
    tot = P.parse(p)
    blk = collections.defaultdict(int); other = collections.defaultdict(int)
    for n, c in tot.items():
        b, k = P.block_of(n)
        if any(k.startswith(x) for x in CONVK):
            blk[b] += c
        elif b == "other":
            other[k] += c
    print(f"\n{name}:  " + "  ".join(f"b{b} {blk[b]:,}" for b in sorted(x for x in blk if isinstance(x, int))))
    hw = {k: v for k, v in other.items() if "hwc" in k or "nchw" in k}
    for k in sorted(hw):
        print(f"    {k}: {hw[k]:,}")
