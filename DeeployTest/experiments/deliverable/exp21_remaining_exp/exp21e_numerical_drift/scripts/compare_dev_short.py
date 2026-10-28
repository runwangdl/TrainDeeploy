"""exp21e: 2-step device ZO runs (default -ffast-math vs strict fp32) vs exp18 device + host references, in ulps."""
import re, numpy as np
B = "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift"
def bits(log):
    L = open(log, errors="ignore").read().splitlines()
    lp = [int(m.group(1), 16) for l in L for m in [re.search(r"\] \+eps loss read OK: lp_bits=0x([0-9a-f]{8})", l)] if m]
    lm = [int(m.group(1), 16) for l in L for m in [re.search(r"\] -eps loss read OK: lm_bits=0x([0-9a-f]{8})", l)] if m]
    return np.array(lp, np.uint32).view(np.float32), np.array(lm, np.uint32).view(np.float32)
def ulp(a, b):
    return np.abs(np.asarray(a, np.float32).view(np.int32).astype(np.int64) - np.asarray(b, np.float32).view(np.int32).astype(np.int64))
t = np.load(B + "/results/trace_zo.npz")
oop = np.load(B + "/results/zo_host_oop_2700.npz"); f64 = np.load(B + "/results/zo_host_f64_2700.npz")
arms = {"default": bits(B + "/logs/dev_zo_default_2steps.log"), "strict": bits(B + "/logs/dev_zo_strict_2steps.log")}
out = open(B + "/results/dev_short_zo.txt", "w")
def P(s): print(s); out.write(s + "\n")
for nm, (lp, lm) in arms.items():
    n = len(lp)
    if n == 0:
        P(f"{nm}: no losses parsed"); continue
    dv = np.concatenate([lp, lm]); e18 = np.concatenate([t["dev_plus"][:n], t["dev_minus"][:n]])
    ho = np.float32(np.concatenate([oop["Lp"][:n], oop["Lm"][:n]])); hf = np.concatenate([f64["Lp"][:n], f64["Lm"][:n]])
    rel = lambda a: np.abs(dv.astype(np.float64) - a) / np.abs(a)
    P(f"{nm:8s} n={2*n}: ==exp18 device bits {np.sum(ulp(dv, e18) == 0)}/{2*n} | ulp vs host-oop fp32: median {np.median(ulp(dv, ho)):.0f} max {ulp(dv, ho).max()} "
      f"| rel err vs f64: median {np.median(rel(hf)):.1e} max {rel(hf).max():.1e}")
hr = np.abs(np.float32(np.concatenate([oop["Lp"][:8], oop["Lm"][:8]])).astype(np.float64) - np.concatenate([f64["Lp"][:8], f64["Lm"][:8]])) / np.abs(np.concatenate([f64["Lp"][:8], f64["Lm"][:8]]))
P(f"host PyTorch fp32 (oop) rel err vs f64 on the same 16 losses: median {np.median(hr):.1e} max {hr.max():.1e}")
if all(len(a[0]) for a in arms.values()):
    d, s = arms["default"], arms["strict"]
    u = ulp(np.concatenate(d), np.concatenate(s)); P(f"default vs strict device: ulp median {np.median(u):.0f} max {u.max()}  identical {np.sum(u==0)}/{len(u)}")
