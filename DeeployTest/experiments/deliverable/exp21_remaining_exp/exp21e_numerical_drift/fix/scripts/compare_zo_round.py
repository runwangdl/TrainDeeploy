#!/usr/bin/env python3
"""exp21e/fix: complete ZO round on the device vs the out-of-place reference fixture: raw L+/L- bits printed by the
MeZO harness ([PHASE] +eps/-eps loss read OK: lp_bits/lm_bits) vs outputs.npz loss_plus/loss_minus, per-epoch
growth, and the final weights ([WDUMP]) vs the fixture's. Also the same device losses vs the OLD (in-place, exp18)
reference, for contrast.
usage: python3 compare_zo_round.py <device log> <new fixture dir> [<old fixture dir>]          -- QW exp21e"""
import re, sys
import numpy as np

log, fx = sys.argv[1], sys.argv[2]
old = sys.argv[3] if len(sys.argv) > 3 else None
t = open(log, errors="ignore").read()
lp = np.array([int(h, 16) for h in re.findall(r"lp_bits=0x([0-9a-f]{8})", t)], np.uint32).view(np.float32)
lm = np.array([int(h, 16) for h in re.findall(r"lm_bits=0x([0-9a-f]{8})", t)], np.uint32).view(np.float32)
err = re.findall(r"Errors: (\d+) out of (\d+)", t)
print(f"harness: Errors: {err[-1][0]} out of {err[-1][1]}" if err else "harness: no Errors line")


def cmp(tag, refdir):
    o = np.load(f"{refdir}/outputs.npz")
    n = min(len(lp), len(o["loss_plus"]))
    rp, rm = o["loss_plus"][:n], o["loss_minus"][:n]
    d = np.concatenate([np.abs(lp[:n].astype(np.float64) - rp), np.abs(lm[:n].astype(np.float64) - rm)])
    eq = np.mean(np.concatenate([lp[:n] == rp, lm[:n] == rm]))
    per_ep = np.abs(lm[:n].astype(np.float64) - rm).reshape(-1, 54)[:, :].mean(axis=1) if n % 54 == 0 else None
    print(f"[{tag}] {2 * n} losses: max |d| {d.max():.3e}, median {np.median(d):.3e}, p99 {np.percentile(d, 99):.3e}, "
          f"#>1e-3 {(d > 1e-3).sum()}, bit-equal {eq * 100:.1f} %")
    if per_ep is not None:
        print(f"[{tag}] mean |dL-| per epoch: first {per_ep[0]:.2e}, epoch 60 {per_ep[59]:.2e}, last {per_ep[-1]:.2e}")
    names = [k for k in o.files if k not in ("log_prob", "loss_plus", "loss_minus")]
    return o, names


o, names = cmp("new out-of-place reference", fx)
if old:
    cmp("old in-place reference (exp18 fixture, first 1620 steps)", old)
S = max([int(s) for s in re.findall(r"\[WDUMP s=(\d+) wi=0 ", t)] or [-1])
if S >= 0:
    d = {}
    for m in re.finditer(r"\[WDUMP s=%d wi=(\d+) n=(\d+)\]([^\r\n]*)" % S, t):
        d.setdefault(int(m.group(1)), []).extend(int(h, 16) for h in m.group(3).split())
    mx = 0.0; ulps = []
    for i, k in enumerate(names):
        dv = np.array(d.get(i, []), np.uint32)
        rv = np.asarray(o[k], np.float32).ravel()
        if dv.size != rv.size:
            print(f"  WARNING wi={i} {k}: size {dv.size} vs {rv.size}"); continue
        mx = max(mx, float(np.abs(dv.view(np.float32) - rv).max()))
        ulps.append(np.abs(dv.astype(np.int64) - rv.view(np.uint32).astype(np.int64)))
    u = np.concatenate(ulps)
    print(f"final weights (WDUMP s={S}) vs reference: max |dw| {mx:.3e}, median {np.median(u):.0f} ulp, max {u.max()} ulp "
          f"over {u.size} words")
