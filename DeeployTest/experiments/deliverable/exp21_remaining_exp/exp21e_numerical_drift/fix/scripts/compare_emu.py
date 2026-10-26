#!/usr/bin/env python3
"""exp21e/fix: device BP log vs the C-order emulator (bp_emu.py) -- loss bits ([losshex i], DUMP_LOSS_HEX) and the
trained weights after the last step ([WDUMP s=S-1], DUMP_WEIGHTS), per tensor: #bit-equal / max ulp.
usage: python3 compare_emu.py <device log> <emu npz from bp_emu.py --out>                        -- QW exp21e"""
import re, sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from bp_emu import NAMES  # noqa: E402

log, npz = sys.argv[1], sys.argv[2]
e = np.load(npz)
txt = open(log, errors="ignore").read()
dev = {int(m.group(1)): int(m.group(2), 16) for m in re.finditer(r"\[losshex (\d+)\] ([0-9a-f]{8}) ([0-9a-f]{8})", txt)}
dev.update({int(m.group(1)): int(m.group(2), 16)                                  # GAP9 harness format
            for m in re.finditer(r"LOSSDUMP (\d+) computed_hex=([0-9a-f]{8})", txt)})
el = e["loss"].view(np.uint32)
n = min(len(dev), len(el))
eq = sum(dev[i] == el[i] for i in range(n))
print(f"losses: {eq}/{n} bit-equal device == emulator")
for i in range(n):
    if dev[i] != el[i]:
        d = np.uint32(dev[i]).view(np.float32); m = el[i].view(np.float32)
        print(f"  first loss mismatch fwd {i}: dev {dev[i]:08x} ({d:.7g})  emu {el[i]:08x} ({m:.7g})  ulp {int(dev[i]) - int(el[i])}")
        break
steps = [int(s) for s in re.findall(r"\[WDUMP s=(\d+) wi=0 ", txt)]
if steps:
    S = max(steps)
    d = {}
    for m in re.finditer(r"\[WDUMP s=%d wi=(\d+) n=(\d+)\]([^\r\n]*)" % S, txt):
        d.setdefault(int(m.group(1)), []).extend(int(h, 16) for h in m.group(3).split())
    tot = tot_eq = 0
    for i, k in enumerate(NAMES):
        dv = np.array(d.get(i, []), np.uint32); ev = e[k].ravel().view(np.uint32)
        if dv.size != ev.size:
            print(f"  {k}: size mismatch {dv.size} vs {ev.size}"); continue
        ne = int((dv != ev).sum()); tot += dv.size; tot_eq += dv.size - ne
        ulp = np.abs(dv.astype(np.int64) - ev.astype(np.int64)).max()
        if ne:
            print(f"  {k:20s} {dv.size - ne}/{dv.size} bit-equal, max ulp {ulp}")
    print(f"weights after step {S}: {tot_eq}/{tot} bit-equal")
