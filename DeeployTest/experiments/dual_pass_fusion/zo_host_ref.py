# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Host mirror of the Siracusa FP32 ZO datapath -- QW (dual-pass-fusion prototype).
#
# Every constant here is a transcription of a device source file; the citation is on the
# function that mirrors it.  Validated bit-exact against the shipped
# Tests/Kernels/FP32/ZOPerturb_QW/PerturbRademacher fixture (see selftest at the bottom).

import numpy as np

M32 = 0xFFFFFFFF

# Siracusa cluster.  NUM_CORES is a compile-time define in the generated C.
NUM_CORES = 8


def xorshift32(state: int) -> int:
    """TargetLibraries/PULPOpen/src/RandomNoise.c :: Xorshift32 (<<13, >>17, <<5)."""
    state ^= (state << 13) & M32
    state ^= state >> 17
    state ^= (state << 5) & M32
    return state & M32


def apply_rademacher_perturbation(src: np.ndarray, seed: int, direction: int,
                                  eps: float) -> np.ndarray:
    """TargetLibraries/PULPOpen/src/RandomNoise.c :: ApplyRademacherPerturbation.

    One xorshift32 draw supplies 32 sign bits, consumed LSB-first; the tail (< 32
    elements) takes one more draw.  `direction == 0` flips eps.  Note this is the
    packed-bit variant: USE_SEQUENTIAL_RADEMACHER is not defined by any build in the
    tree, so the `#else` branch of FloatPerturbRademacherTemplate is what compiles.
    """
    size = src.size
    out = np.empty_like(src)
    state = ((seed * 1664525) + 1013904223) & M32
    e = np.float32(eps if direction != 0 else -eps)
    i = 0
    for _ in range(size // 32):
        state = xorshift32(state)
        bits = state
        for _b in range(32):
            out[i] = src[i] + (e if (bits & 1) else -e)
            bits >>= 1
            i += 1
    leftover = size % 32
    if leftover:
        state = xorshift32(state)
        bits = state
        for _b in range(leftover):
            out[i] = src[i] + (e if (bits & 1) else -e)
            bits >>= 1
            i += 1
    return out


def perturb_rademacher(x: np.ndarray, seed: int, node_idx: int, eps: float,
                       direction: int = 1, num_cores: int = NUM_CORES,
                       seed_base: int = 0, tile_seed_offset: int = 0) -> np.ndarray:
    """Deeploy/Targets/PULPOpen/Templates/FloatPerturbRademacherTemplate.py.

    The template splits the FLAT tensor into NUM_CORES contiguous chunks and gives each
    core its own stream:
        chunk_seed = ((seed + perturb_seed_base) + NUM_CORES*node_id + core_id)
                     ^ (tile_seed_offset * 0x9E3779B1)
    `node_id` is the parser's `nodeIdx`, which is the ONNX node's `idx` ATTRIBUTE
    (Deeploy/Targets/Generic/Parsers.py:3100) -- not its position in the schedule.  So a
    node extracted into a smaller graph keeps the full network's z-stream as long as its
    `idx` attribute is preserved.

    Runtime defaults come from TargetLibraries/PULPOpen/src/ZORuntime.c:
    perturbation_sign = 1 (+eps), perturb_seed_base = 0, eps override off.  ZORuntime.c is
    linked into EVERY binary, so this op also runs correctly in a plain inference test.
    """
    flat = x.reshape(-1).astype(np.float32)
    size = flat.size
    log2core = int(np.log2(num_cores))
    chunk = (size >> log2core) + (1 if (size & (num_cores - 1)) else 0)
    out = flat.copy()
    for core in range(num_cores):
        start = min(chunk * core, size)
        stop = min(start + chunk, size)
        if stop <= start:
            continue
        chunk_seed = (((seed + seed_base) + num_cores * node_idx + core)
                      ^ ((tile_seed_offset * 0x9E3779B1) & M32)) & M32
        out[start:stop] = apply_rademacher_perturbation(flat[start:stop], chunk_seed,
                                                        direction, eps)
    return out.reshape(x.shape)


def conv2d_im2col_hwc(x_nchw: np.ndarray, w_fchw: np.ndarray, b: np.ndarray,
                      pads, strides) -> np.ndarray:
    """TargetLibraries/PULPOpen/src/Convolution_fp32.c :: PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC.

    The device runs NHWC after Deeploy inserts the layout transposes, and accumulates
    `sum = 0; for k in 0..P*Q*C-1: sum += im2col[k] * weight[k]; out = sum + bias[f]`
    with the im2col window laid out (p, q, c).  Reproduced here in that exact order and in
    float32, so the only residual difference against the device is whatever clang's
    -ffast-math contraction does to the same expression (well inside the 1e-4 OUTPUT_TOL).
    """
    pad_top, pad_left, pad_bottom, pad_right = pads   # ONNX order: x1_begin, x2_begin, x1_end, x2_end
    sp, sq = strides
    n, c_in, h, w_in_dim = x_nchw.shape
    f_total, c_w, p, q = w_fchw.shape
    assert n == 1 and c_w == c_in
    h_out = (h + pad_top + pad_bottom - p) // sp + 1
    w_out = (w_in_dim + pad_left + pad_right - q) // sq + 1

    x = x_nchw.astype(np.float32)
    w = w_fchw.astype(np.float32)
    out = np.empty((1, f_total, h_out, w_out), dtype=np.float32)

    # Zero-padded copy, so the im2col window is a plain slice.
    xp = np.zeros((c_in, h + pad_top + pad_bottom, w_in_dim + pad_left + pad_right),
                  dtype=np.float32)
    xp[:, pad_top:pad_top + h, pad_left:pad_left + w_in_dim] = x[0]

    for f in range(f_total):
        acc = np.zeros((h_out, w_out), dtype=np.float32)
        # k iterates (p, q, c) -- the im2col buffer's own layout, hence the device's
        # accumulation order.
        for kp in range(p):
            for kq in range(q):
                for kc in range(c_in):
                    patch = xp[kc,
                               kp:kp + h_out * sp:sp,
                               kq:kq + w_out * sq:sq]
                    acc = (acc + patch * w[f, kc, kp, kq]).astype(np.float32)
        out[0, f] = (acc + np.float32(b[f])).astype(np.float32)
    return out


def _selftest(repo_root: str = "..") -> None:
    """Bit-exactness check against the shipped PerturbRademacher kernel fixture."""
    import os
    base = os.path.join(repo_root, "Tests/Kernels/FP32/ZOPerturb_QW/PerturbRademacher")
    x = np.load(os.path.join(base, "inputs.npz"))["input"]
    y = np.load(os.path.join(base, "outputs.npz"))["output"]
    got = perturb_rademacher(x, seed=42, node_idx=0, eps=0.01)
    bad = int((got != y).sum())
    print(f"[selftest] PerturbRademacher vs shipped fixture: {bad}/{y.size} mismatches")
    assert bad == 0


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else "..")


# --------------------------------------------------------------------------------------
# Block-0 forward chain -- needed only to produce the ACTIVATION that feeds block 1's conv,
# so that an extracted block-1 fixture is fed real data rather than noise. -- QW
# --------------------------------------------------------------------------------------

def batchnorm_frozen(x, gamma, beta, running_mean, running_var, epsilon):
    """TargetLibraries/PULPOpen/src/BatchNorm.c :: PULP_BatchNormInternal_fp32, g_bn_frozen_stats.

    y = (x - running_mean) * 1/sqrt(running_var + eps) * gamma + beta, per channel, NCHW.
    The ZO runner always sets `-D BN_FROZEN_STATS=ON`, so this is the branch that runs; the
    batch-statistics branch is never taken in our flow.
    """
    x = x.astype(np.float32)
    inv_std = (np.float32(1.0) / np.sqrt(running_var.astype(np.float32) + np.float32(epsilon)))
    shape = (1, -1) + (1,) * (x.ndim - 2)
    return ((x - running_mean.astype(np.float32).reshape(shape)) * inv_std.reshape(shape) *
            gamma.astype(np.float32).reshape(shape) + beta.astype(np.float32).reshape(shape)).astype(np.float32)


def relu(x):
    return np.maximum(x, np.float32(0.0)).astype(np.float32)


def maxpool2d(x, kernel_shape, strides):
    """Plain NCHW max pool, no padding, ceil_mode=0 -- the block-0/1 MaxPool attributes."""
    kh, kw = kernel_shape
    sh, sw = strides
    n, c, h, w = x.shape
    h_out = (h - kh) // sh + 1
    w_out = (w - kw) // sw + 1
    out = np.empty((n, c, h_out, w_out), dtype = np.float32)
    for i in range(h_out):
        for j in range(w_out):
            out[:, :, i, j] = x[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw].max(axis = (2, 3))
    return out
