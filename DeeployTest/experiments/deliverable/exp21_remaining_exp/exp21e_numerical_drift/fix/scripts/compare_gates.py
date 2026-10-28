#!/usr/bin/env python3
"""exp21e/fix: Siracusa gates, baseline tree (bb7deadb) vs fixed branch: errors, loss lines, trained weights
([WDUMP] words, bit for bit), cycles. usage: python3 compare_gates.py <logdir> [gate ...]          -- QW exp21e"""
import hashlib, os, re, sys

d = sys.argv[1]
gates = sys.argv[2:] or ["zo_single", "qzo_single", "qzo_unfused", "qzo_fused", "b0_unfused", "b0_fused", "bp_single"]


def parse(p):
    if not os.path.exists(p):
        return None
    t = open(p, errors="ignore").read()
    err = re.findall(r"Errors: (\d+) out of (\d+)", t)
    losses = re.findall(r"\[(?:loss|L\+|L-|lp|lm)[^\]]*\] computed=([-\d.]+)", t)
    wd = re.findall(r"\[WDUMP s=\d+ wi=\d+ n=\d+\][^\r\n]*", t)
    cyc = re.findall(r"BENCH [^\r\n]*|Runtime: \d+ cycles", t)
    return dict(err="/".join(err[-1]) if err else None, exit=re.findall(r"^EXIT (\d+)", t, re.M),
                losses=losses, wsum=hashlib.md5("\n".join(wd).encode()).hexdigest()[:12] if wd else None,
                nw=sum(int(re.search(r"n=(\d+)", w).group(1)) for w in wd), cyc=cyc[-1] if cyc else None)


print(f"{'gate':12s} {'base errors':12s} {'fixed errors':12s} {'losses equal':13s} {'weights (md5 of WDUMP)':34s} cycles base -> fixed")
for g in gates:
    b, f = parse(f"{d}/gate_base_{g}.log"), parse(f"{d}/gate_fixed_{g}.log")
    if not b or not f:
        print(f"{g:12s} missing"); continue
    leq = "yes" if b["losses"] == f["losses"] else "NO"
    w = "-" if not b["wsum"] else (f"identical {b['wsum']} ({b['nw']} words)" if b["wsum"] == f["wsum"] else
                                    f"DIFFER {b['wsum']} vs {f['wsum']}")
    print(f"{g:12s} {str(b['err']):12s} {str(f['err']):12s} {leq:13s} {w:34s} {b['cyc']} -> {f['cyc']}")
