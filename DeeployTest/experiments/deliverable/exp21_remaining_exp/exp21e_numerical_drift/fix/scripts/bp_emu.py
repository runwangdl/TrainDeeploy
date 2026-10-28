#!/usr/bin/env python3
"""exp21e/fix -- C-order host reference for SpeechNet BP (frozen BN, argmax-mask MaxPool, SGD, SUM over n_accum).

An independent re-implementation of the model's forward/backward/SGD maths (it does NOT run device code), in which
every fp32 reduction is evaluated in the SAME order as the Deeploy PULPOpen kernels, one rounding per operation (the
device side is built with -D DEEPLOY_STRICT_FP32=ON -D DEEPLOY_STRICT_FP32_ALL=ON: no fmadd, no reassociation), and
expf/logf are the device's own picolibc routines (libm_emu.c, transliterated from the disassembly). Vectorised over
independent outputs with numpy float32 (one IEEE rounding per ufunc), sequential over every reduction index.

Per op, the order it reproduces (source of truth: TargetLibraries/PULPOpen/src, generated TrainingNetwork.c):
  Conv fwd (Convolution_fp32.c, im2col HWC / 1-ch direct): sum_k from +0, k = (p, q, c), then + bias
  BatchNormInternal frozen (BatchNorm.c): inv_std = 1/sqrt(var+eps); y = ((x-mean)*inv_std)*g + b
  Relu (Relu.c): x > 0 ? x : 0              MaxPool (MaxPool.c): fmax scan from -inf, argmax strict '>' first index
  GlobalAveragePool: sum over (h,w) row-major from +0, * (1/HW)
  Gemm fc (Gemm.c, !transA transB, O blocks of 6, K blocks of 6): sum += (((((a0b0+a1b1)+a2b2)+a3b3)+a4b4)+a5b5)
  SCE (generated): m = max; s = sum_j expf(x_j-m); l = logf(s); log_prob = (x-m)-l; loss = -((x_y-m)-l)
  SCE grad: expf(log_prob) (-1 at label) * 1 * 1      fc dW: dy_i*x_j       fc dx (Gemm !transA !transB): sum_k
  GAP grad: dy * (1/HW)       ReluGrad: x>0 ? dy : 0       MaxPoolGradMask: 0 + dy at the argmax
  BatchNormGrad frozen: dbeta = sum dy, dgamma = sum dy*((x-mean)*inv_std) (row-major), dX = (g*inv_std)*dy
  ConvGradX scatter (ConvGrad.c): dX += dy*w over (co, oy, ox) lexicographic, from +0
  ConvGradW (pulp-trainlib mm_add): per spatial tile T: dW += (sum over T's pixels, ho-major, from +0); tile
            boundaries parsed from the generated code (CoutHW tiling; CinSlice tiles are full-K)
  ConvGradB: sum over (h,w) row-major from +0
  InPlaceAccumulatorV2: acc = g (first window of a step), acc + g after;  SGD: w - (lr*g)
usage: python3 bp_emu.py --fixture <dir> --gen <TrainingNetwork.c> --steps N [--out npz]       -- QW exp21e
"""
import argparse, ctypes, os, re, subprocess, sys, time
import numpy as np

F32 = np.float32
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- libm (device picolibc expf/logf)
def _load_libm():
    so = os.path.join(os.environ.get("EMU_BUILD", "/tmp"), "x21e_libm_emu.so")
    src = os.path.join(HERE, "libm_emu.c")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.check_call(["gcc", "-O2", "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC", "-o", so, src,
                               "-lm"])
    lib = ctypes.CDLL(so)
    for fn in ("emu_expf_v", "emu_logf_v", "emu_logf_gap9_v"):
        getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    return lib


LIBM = _load_libm()


def expf(a):
    a = np.ascontiguousarray(a, F32); o = np.empty_like(a)
    LIBM.emu_expf_v(a.ctypes.data, o.ctypes.data, a.size); return o


TARGET = os.environ.get("EMU_TARGET", "siracusa")   # siracusa: picolibc (LLVM) | gap9: newlib (gap9-sdk gcc)


def logf(a):
    a = np.ascontiguousarray(a, F32); o = np.empty_like(a)
    (LIBM.emu_logf_gap9_v if TARGET == "gap9" else LIBM.emu_logf_v)(a.ctypes.data, o.ctypes.data, a.size); return o


# ----------------------------------------------------------------------------- fixture + tiling
NAMES = ["blocks_0_0_weight", "blocks_0_0_bias", "blocks_0_1_weight", "blocks_0_1_bias",
         "blocks_1_0_weight", "blocks_1_0_bias", "blocks_1_1_weight", "blocks_1_1_bias",
         "blocks_2_0_weight", "blocks_2_0_bias", "blocks_2_1_weight", "blocks_2_1_bias",
         "blocks_3_0_weight", "blocks_3_0_bias", "blocks_3_1_weight", "blocks_3_1_bias",
         "blocks_4_0_weight", "blocks_4_0_bias", "blocks_4_1_weight", "blocks_4_1_bias", "fc_weight", "fc_bias"]
# conv geometry (from the training graph): pads (top, left), pool kernel along W (None = no pool)
BLOCKS = [dict(pt=0, pl=2, pool=8), dict(pt=0, pl=8, pool=4), dict(pt=0, pl=4, pool=4),
          dict(pt=0, pl=0, pool=None), dict(pt=0, pl=0, pool=None)]
CONV_NODE = {0: "node_1_blocks_0", 1: "node_5_blocks_1", 2: "node_9_blocks_2", 3: "node_13_blocks_3",
             4: "node_16_blocks_4"}


def load_fixture(d):
    i = np.load(os.path.join(d, "inputs.npz")); o = np.load(os.path.join(d, "outputs.npz"))
    W = {n: np.asarray(i["arr_%04d" % (k + 2)], F32).copy() for k, n in enumerate(NAMES)}
    n = int(i["meta_data_size"][0]) if "meta_data_size" in i else 54
    X = [np.asarray(i["arr_0000"], F32)] + [np.asarray(i["mb%d_arr_0000" % k], F32) for k in range(1, n)]
    Y = [int(np.asarray(i["arr_0001"]).reshape(-1)[0])] + \
        [int(np.asarray(i["mb%d_arr_0001" % k]).reshape(-1)[0]) for k in range(1, n)]
    RS = {b: (np.asarray(o["blocks_%d_1_running_mean" % b], F32), np.asarray(o["blocks_%d_1_running_var" % b], F32))
          for b in range(5)}
    return W, X, Y, RS, o


def convgradw_wtiles(gen_c, b, Wo):
    """W-tile widths of block b's ConvGradW, read from the generated code: the kernel call's (H_out, W_out) arguments
    (a literal = uniform tiles of that width, '*..._dim_im_out_y_ref' = per-tile table), Cout slabs from dw_reset.
    H tiling is not modelled (asserted). CinSlice / untiled -> [Wo] (every tile has the full GEMM K)."""
    if gen_c is None:
        return [Wo]
    node = CONV_NODE[b]
    call = re.search(r"PULP_ConvGradW2d_fp32_fp32_fp32_CHW_Im2Col\(\s*[^,]*%s[^,]*,\s*([^,]+),\s*([^,]+)," % node, gen_c)
    assert call, node
    h_arg, w_arg = call.group(1).strip(), call.group(2).strip()
    assert not h_arg.startswith("*"), "H-tiled ConvGradW not modelled"
    reset = re.search(r"%s_[A-Za-z0-9_]*ConvGradW_dw_reset\[(\d+)\]\s*=\s*\{([^}]*)\}" % node, gen_c)
    nslab = sum(int(v) for v in reset.group(2).replace("\n", " ").split(",") if v.strip()) if reset else 1
    if w_arg.startswith("*"):
        m = re.search(r"%s_[A-Za-z0-9_]*ConvGradW_dim_im_out_y\[(\d+)\]\s*=\s*\{([^}]*)\}" % node, gen_c)
        ws = [int(v) for v in m.group(2).replace("\n", " ").split(",") if v.strip()]
        ws = ws[:len(ws) // nslab]
    else:
        w = int(w_arg)
        assert Wo % w == 0, (b, w, Wo)
        ws = [w] * (Wo // w)
    assert sum(ws) == Wo, (b, ws, Wo)
    return ws


# ----------------------------------------------------------------------------- ops (x: [C,H,W] float32)
def conv_fwd(x, w, bias, pt, pl):
    """Convolution_fp32.c: per output sum_k col[k]*w[k] from +0, k = (p, q, c); then + bias. stride 1."""
    C, H, Wd = x.shape; F, _, P, Q = w.shape
    Ho, Wo = H + 2 * pt - P + 1, Wd + 2 * pl - Q + 1
    xp = np.zeros((C, H + 2 * pt, Wd + 2 * pl), F32); xp[:, pt:pt + H, pl:pl + Wd] = x
    acc = np.zeros((F, Ho, Wo), F32)
    for p in range(P):
        for q in range(Q):
            for c in range(C):
                acc = acc + xp[c, p:p + Ho, q:q + Wo][None] * w[:, c, p, q][:, None, None]
    return acc + bias[:, None, None]


def bn_fwd(x, g, b, mean, var, eps):
    inv = F32(1.0) / np.sqrt(var + eps)                 # fsqrt.s / fdiv.s: correctly rounded
    return ((x - mean[:, None, None]) * inv[:, None, None]) * g[:, None, None] + b[:, None, None], inv


def maxpool_w(x, k):
    """MaxPool (1,k) stride (1,k), no pad: value (fmax from -inf) + within-window argmax (strict >, first)."""
    C, H, Wd = x.shape; Wo = (Wd - k) // k + 1
    xs = x[:, :, :Wo * k].reshape(C, H, Wo, k)
    m = np.full((C, H, Wo), -np.inf, F32); am = np.zeros((C, H, Wo), np.int64)
    for q in range(k):
        v = xs[..., q]; upd = v > m
        am = np.where(upd, q, am); m = np.where(upd, v, m)   # max value == fmax value (no NaN)
    return m, am


def maxpool_grad(dy, am, k, Wd):
    C, H, Wo = dy.shape
    dx = np.zeros((C, H, Wd), F32)
    hh, ww = np.meshgrid(np.arange(H), np.arange(Wo), indexing="ij")
    for c in range(C):
        dx[c, hh, ww * k + am[c]] = F32(0.0) + dy[c]
    return dx


def gemm_fc(a, Wfc, bfc):
    """Gemm.c (!transA, transB), M=1, N=32, O=9: K in blocks of 6 summed inside (), O in blocks of 6."""
    N = a.shape[0]; O = Wfc.shape[0]; Nu = N - N % 6
    y = np.empty(O, F32)
    for j in range(O):
        s = F32(0.0)
        for k in range(0, Nu, 6):
            t = a[k] * Wfc[j, k]
            for u in range(1, 6):
                t = t + a[k + u] * Wfc[j, k + u]
            s = s + t
        for k in range(Nu, N):
            s = s + a[k] * Wfc[j, k]
        y[j] = s + bfc[j]
    return y


def sce(logits, y):
    m = logits[0]
    for j in range(1, logits.size):
        if logits[j] > m:
            m = logits[j]
    e = expf(logits - m)
    s = F32(0.0)
    for j in range(logits.size):
        s = F32(s + e[j])
    l = logf(np.array([s], F32))[0]
    lp = (logits - m) - l
    loss = -((logits[y] - m) - l)
    loss = F32(F32(F32(0.0) + loss) / F32(1.0))
    return loss, lp


def bn_grad(dy, x, g, mean, inv):
    C = dy.shape[0]
    xh = (x - mean[:, None, None]) * inv[:, None, None]
    dyf = dy.reshape(C, -1); xhf = xh.reshape(C, -1)
    sb = np.zeros(C, F32); sg = np.zeros(C, F32)
    for i in range(dyf.shape[1]):
        sb = sb + dyf[:, i]; sg = sg + dyf[:, i] * xhf[:, i]
    s = g * inv
    return s[:, None, None] * dy, sg, sb


def conv_grad_x(dy, w, pt, pl, H, Wd):
    """ConvGrad.c scatter: dX[ci,ih,iw] += dy[co,oy,ox]*w[co,ci,ky,kx], (co, oy, ox) lexicographic, from +0."""
    F, Ho, Wo = dy.shape; _, C, P, Q = w.shape
    acc = np.zeros((C, H + 2 * pt, Wd + 2 * pl), F32)       # padded frame; pad region discarded
    for co in range(F):
        for ky in reversed(range(P)):                         # oy = ih + pt - ky ascending <=> ky descending
            for kx in reversed(range(Q)):
                acc[:, ky:ky + Ho, kx:kx + Wo] = acc[:, ky:ky + Ho, kx:kx + Wo] + \
                    dy[co][None] * w[co, :, ky, kx][:, None, None]
    return acc[:, pt:pt + H, pl:pl + Wd]


def conv_grad_w(dy, x, P, Q, pt, pl, wtiles):
    """pulp-trainlib im2row + mm_add: per W-tile, temp = sum over its pixels (ho-major) from +0; dW += temp."""
    F, Ho, Wo = dy.shape; C, H, Wd = x.shape
    xp = np.zeros((C, H + 2 * pt, Wd + 2 * pl), F32); xp[:, pt:pt + H, pl:pl + Wd] = x
    # col[ho, wo] -> [C, P, Q] window of the padded input
    dW = np.zeros((F, C, P, Q), F32)
    w0 = 0
    for tw in wtiles:
        tmp = np.zeros((F, C, P, Q), F32)
        for ho in range(Ho):
            for wo in range(w0, w0 + tw):
                col = xp[:, ho:ho + P, wo:wo + Q]
                tmp = tmp + dy[:, ho, wo][:, None, None, None] * col[None]
        dW = dW + tmp
        w0 += tw
    return dW


def sum_rowmajor(a):
    """sequential sum over the last axes (row-major) per leading index, from +0."""
    C = a.shape[0]; af = a.reshape(C, -1); s = np.zeros(C, F32)
    for i in range(af.shape[1]):
        s = s + af[:, i]
    return s


# ----------------------------------------------------------------------------- forward + backward
def fwd_bwd(W, x, y, RS, eps, wt):
    cache = []
    h = x.reshape(1, 14, 700).astype(F32)
    for b, cfg in enumerate(BLOCKS):
        wc, bc = W["blocks_%d_0_weight" % b], W["blocks_%d_0_bias" % b]
        z = conv_fwd(h, wc, bc, cfg["pt"], cfg["pl"])
        mean, var = RS[b]
        u, inv = bn_fwd(z, W["blocks_%d_1_weight" % b], W["blocks_%d_1_bias" % b], mean, var, eps)
        r = np.where(u > 0, u, F32(0.0)).astype(F32)
        c = dict(hin=h, z=z, u=u, inv=inv)
        if cfg["pool"]:
            h, am = maxpool_w(r, cfg["pool"]); c["am"] = am; c["Wr"] = r.shape[2]
        else:
            h = r
        cache.append(c)
    C, Hh, Ww = h.shape
    inv_hw = F32(1.0) / F32(Hh * Ww)
    gap = sum_rowmajor(h) * inv_hw
    logits = gemm_fc(gap, W["fc_weight"], W["fc_bias"])
    loss, lp = sce(logits, y)
    # backward
    G = {}
    p = expf(lp)
    dlog = p.copy(); dlog[y] = p[y] - F32(1.0)
    dlog = (dlog * F32(1.0)) * F32(1.0)
    G["fc_bias"] = F32(0.0) + dlog
    G["fc_weight"] = (dlog[:, None] * gap[None, :]).astype(F32)
    dgap = np.zeros(32, F32)
    for k in range(9):
        dgap = dgap + dlog[k] * W["fc_weight"][k]
    dh = np.broadcast_to((dgap * inv_hw)[:, None, None], (C, Hh, Ww)).astype(F32)
    for b in reversed(range(5)):
        cfg, c = BLOCKS[b], cache[b]
        if cfg["pool"]:
            dh = maxpool_grad(dh, c["am"], cfg["pool"], c["Wr"])
        du = np.where(c["u"] > 0, dh, F32(0.0)).astype(F32)
        mean, _ = RS[b]
        dz, dg, db = bn_grad(du, c["z"], W["blocks_%d_1_weight" % b], mean, c["inv"])
        G["blocks_%d_1_weight" % b], G["blocks_%d_1_bias" % b] = dg, db
        wc = W["blocks_%d_0_weight" % b]
        _, Cin, P, Q = wc.shape
        Hin, Win = c["hin"].shape[1:]
        if b > 0:
            dh_next = conv_grad_x(dz, wc, cfg["pt"], cfg["pl"], Hin, Win)
        G["blocks_%d_0_weight" % b] = conv_grad_w(dz, c["hin"], P, Q, cfg["pt"], cfg["pl"], wt[b])
        G["blocks_%d_0_bias" % b] = sum_rowmajor(dz)
        if b > 0:
            dh = dh_next
    return loss, lp, G


def train(W, X, Y, RS, eps, lr, n_steps, n_accum, wt, log_every=0):
    W = {k: v.copy() for k, v in W.items()}
    losses = []
    t0 = time.time()
    for s in range(n_steps):
        acc = None
        for a in range(n_accum):
            mb = s * n_accum + a
            loss, _, G = fwd_bwd(W, X[mb % len(X)], Y[mb % len(X)], RS, eps, wt)
            losses.append(loss)
            if acc is None:
                acc = {k: G[k].reshape(W[k].shape).astype(F32).copy() for k in NAMES}
            else:
                for k in NAMES:
                    acc[k] = acc[k] + G[k].reshape(W[k].shape)
        for k in NAMES:
            W[k] = W[k] - lr * acc[k]
        if log_every and (s + 1) % log_every == 0:
            print(f"step {s + 1}/{n_steps}  loss {losses[-1]:.6f}  {time.time() - t0:.0f}s", flush=True)
    return np.array(losses, F32), W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--gen", default=None, help="generated TrainingNetwork.c (ConvGradW tiling)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eps", type=float, default=9.999999747378752e-06)
    ap.add_argument("--out", default=None)
    ap.add_argument("--log-every", type=int, default=0)
    a = ap.parse_args()
    W, X, Y, RS, _ = load_fixture(a.fixture)
    gen_c = open(a.gen).read() if a.gen else None
    Wo = {0: 701, 1: 88, 2: 23, 3: 5, 4: 5}
    wt = {b: convgradw_wtiles(gen_c, b, Wo[b]) for b in range(5)}
    print("ConvGradW W-tiles:", wt, flush=True)
    losses, Wf = train(W, X, Y, RS, F32(a.eps), F32(a.lr), a.steps, a.accum, wt, a.log_every)
    for i, l in enumerate(losses[:16]):
        print(f"[emu loss {i}] {l:.6f} {np.float32(l).view(np.uint32):08x}")
    if a.out:
        np.savez(a.out, loss=losses, **{k: v for k, v in Wf.items()})
        print("saved", a.out)


if __name__ == "__main__":
    main()
