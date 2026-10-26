"""exp21e BP: device [AMSIG mb] (argmax-mask signature, DEEPLOY_ARGMAX_MASK_SIG) vs host PyTorch argmax signatures.
sig = sum over all pooling windows of (within-window argmax offset + 1), sig2 = sum of squares (uint32)."""
import re, numpy as np
B = "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift"
L = open(B + "/logs/dev_bp_argmaxsig_165steps.log", errors="ignore").read().splitlines()
dev = {}
for l in L:
    m = re.match(r"\[AMSIG (\d+)\] ([0-9a-f]{8}) ([0-9a-f]{8})", l.strip())
    if m:
        dev[int(m.group(1))] = (int(m.group(2), 16), int(m.group(3), 16))
n = len(dev)
out = open(B + "/results/amsig_bp.txt", "w")
def P(s): print(s); out.write(s + "\n")
P(f"device AMSIG lines: {n}")
for v in ("orig", "ulp1", "ulp2", "f64"):
    am = np.load(f"{B}/results/bp_host_{v}.npz")["argmax"].astype(np.int64) + 1
    s1 = am.sum(1); s2 = (am * am).sum(1)
    mism = [mb for mb in range(n) if (s1[mb], s2[mb]) != dev[mb]]
    P(f"host {v:5s}: device sig == host sig on {n - len(mism)}/{n} forwards; mismatching fwd: {mism[:30]}")
    for mb in mism[:6]:
        P(f"    fwd {mb}: device ({dev[mb][0]},{dev[mb][1]})  host ({s1[mb]},{s2[mb]})  d_sig={dev[mb][0]-s1[mb]:+d} d_sig2={dev[mb][1]-s2[mb]:+d}")
# loss trace of this short device run vs the fixture reference
loss = {}
for l in L:
    m = re.match(r"\[loss (\d+)\] computed=(\S+)\s+ref=(\S+)", l.strip())
    if m:
        loss[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
if loss:
    d = np.array([abs(loss[i][0] - loss[i][1]) for i in sorted(loss)])
    on = np.where(d >= 1e-5)[0]
    P(f"short-run loss: {len(d)} fwd, first |d|>=1e-5 at fwd {on[0] if len(on) else -1}, max {d.max():.2e}")
