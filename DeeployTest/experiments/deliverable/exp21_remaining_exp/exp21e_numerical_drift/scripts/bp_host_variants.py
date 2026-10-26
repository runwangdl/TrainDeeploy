"""exp21e BP -- re-run the exp17 host reference (PyTorch frozen-BN, SUM-accum SGD) with controlled changes.

variants
  orig      : replica of Onnx4Deeploy speechnet_exporter._frozen_pytorch_reference (must be bit-equal to the
              fixture's 2160 losses). Also records MaxPool near-tie statistics on every forward.
  f64       : same loop in float64 ("exact-arithmetic" trajectory)
  ulp<k>    : orig with every trainable initial weight moved by +-1 ulp (random sign, seed k): a pure
              ulp-level perturbation with NO change of arithmetic -> measures the trajectory's intrinsic
              sensitivity (Lyapunov growth) without any device involvement
  thr<n>    : orig with torch.set_num_threads(n) (different oneDNN/ATen reduction split, same math)
  ulp<k>F   : COUNTERFACTUAL -- ulp<k>, but every MaxPool backward routes its gradient to the argmax that
              'orig' chose for that forward (forward values untouched). If the drift onsets vanish, the
              MaxPool argmax flips are causal.

usage: python3 bp_host_variants.py <variant>
"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, "/app/TrainDeeploy/DeeployTest/experiments/deliverable/exp21_remaining_exp/exp21e_numerical_drift/scripts")
from common import *  # noqa

variant = sys.argv[1]
n_steps, n_accum, lr = 540, 4, 3e-4
dt = torch.float64 if variant == "f64" else torch.float32
if variant.startswith("thr"):
    torch.set_num_threads(int(variant[3:]))
X, Y = load_windows("speechnet_train_fullfrozen_b1_fold3_ref"); N = len(X)
init = bp_init_map()
FORCED = variant.endswith("F")
if FORCED:
    AM_REF = np.load(f"{B}/results/bp_host_orig.npz")["argmax"]
    variant_seed = variant[3:-1]
else:
    variant_seed = variant[3:]
if variant.startswith("ulp"):
    rng = np.random.default_rng(int(variant_seed))
    for k in list(init):
        if "running" in k:
            continue
        a = np.asarray(init[k], np.float32)
        sgn = rng.choice([-1, 1], size=a.shape)
        init[k] = np.where(sgn > 0, np.nextafter(a, np.float32(np.inf)), np.nextafter(a, np.float32(-np.inf))).astype(np.float32)
model = build_model(init, dt)
for p in model.parameters():
    p.requires_grad_(True)
params = list(model.parameters())
assert len(params) == 22
opt = torch.optim.SGD(params, lr=lr)
crit = torch.nn.CrossEntropyLoss()

# MaxPool near-tie recorder (orig only): for every pooling window with a positive max, relative gap top1-top2
ties = []   # per forward: [n_exact_ties, n_rel<1e-6, n_rel<1e-5, min_rel_gap] per pool layer
cur = []
am = []     # per forward: concatenated within-window argmax offsets of the 3 pools
AM = []
def hook(mod, inp, out):
    x = inp[0].detach()
    k = mod.kernel_size[1]
    C, H, W = x.shape[1:]
    Wo = W // k
    v = x[0, :, :, :Wo * k].reshape(C, H, Wo, k)
    am.append(torch.argmax(v, dim=-1).to(torch.uint8).numpy().ravel())   # first-occurrence argmax (= device strict '>')
    s, _ = torch.sort(v, dim=-1, descending=True)
    t1, t2 = s[..., 0], s[..., 1]
    pos = t1 > 0
    rel = ((t1 - t2) / torch.where(pos, t1, torch.ones_like(t1)))[pos]
    cur.append([int((rel == 0).sum()), int((rel < 1e-6).sum()), int((rel < 1e-5).sum()),
                float(rel[rel > 0].min()) if (rel > 0).any() else 1.0])
if True:   # all variants record argmax + tie stats (hooks do not change arithmetic)
    for m in model.modules():
        if isinstance(m, torch.nn.MaxPool2d):
            m.register_forward_hook(hook)

class ForcedPool(torch.autograd.Function):
    """forward = true max-pool value; backward = scatter to a GIVEN within-window index."""
    @staticmethod
    def forward(ctx, x, k, idx):
        C, H, W = x.shape[1:]; Wo = W // k
        v = x[0, :, :, :Wo * k].reshape(C, H, Wo, k)
        ctx.save_for_backward(idx); ctx.shp = (x.shape, k, C, H, Wo)
        return v.max(dim=-1).values.unsqueeze(0)
    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors; xs, k, C, H, Wo = ctx.shp
        gv = torch.zeros(C, H, Wo, k, dtype=g.dtype)
        gv.scatter_(-1, idx.reshape(C, H, Wo, 1), g[0].unsqueeze(-1))
        gx = torch.zeros(xs, dtype=g.dtype); gx[0, :, :, :Wo * k] = gv.reshape(C, H, Wo * k)
        return gx, None, None
FCTX = {"mb": 0, "off": 0}
import os
FR = tuple(int(v) for v in os.environ.get("FORCE_RANGE", "0:100000").split(":"))   # forward window to force
class ForcedPoolMod(torch.nn.Module):
    def __init__(self, k): super().__init__(); self.k = k; self.kernel_size = (1, k)
    def forward(self, x):
        C, H, W = x.shape[1:]; n = C * H * (W // self.k)
        idx = torch.from_numpy(AM_REF[FCTX["mb"], FCTX["off"]:FCTX["off"] + n].astype(np.int64)); FCTX["off"] += n
        if not (FR[0] <= FCTX["mb"] < FR[1]):   # outside the forcing window: natural (own) argmax
            Wo = x.shape[3] // self.k
            idx = torch.argmax(x.detach()[0, :, :, :Wo * self.k].reshape(x.shape[1], x.shape[2], Wo, self.k), -1).reshape(-1)
        hook(self, (x,), None)
        return ForcedPool.apply(x, self.k, idx)
if FORCED:
    for blk in model.blocks:
        if isinstance(blk[3], torch.nn.MaxPool2d):
            blk[3] = ForcedPoolMod(blk[3].kernel_size[1])

losses = []
t0 = time.time()
for s in range(n_steps):
    opt.zero_grad()
    for a in range(n_accum):
        mb = s * n_accum + a
        x = torch.from_numpy(X[mb % N]).to(dt); y = torch.from_numpy(Y[mb % N])
        cur = []; am = []; FCTX["mb"] = mb; FCTX["off"] = 0
        loss = crit(model(x), y)
        loss.backward()
        losses.append(float(loss.detach().item()))
        ties.append(cur); AM.append(np.concatenate(am))
    opt.step()
    if s % 100 == 0:
        print(f"{variant} step {s} {time.time()-t0:.0f}s", flush=True)
W = {t.replace(".", "_"): p.detach().double().numpy().ravel() for t, p in model.named_parameters()}
out = dict(loss=np.array(losses), **{"w_" + k: v for k, v in W.items()})
out["ties"] = np.array(ties, np.float64)   # (2160, 3 pools, 4)
out["argmax"] = np.stack(AM)               # (2160, n_windows) uint8
np.savez(f"{B}/results/bp_host_{variant}{os.environ.get('TAG', '')}.npz", **out)
print("done", variant, f"{time.time()-t0:.0f}s")
