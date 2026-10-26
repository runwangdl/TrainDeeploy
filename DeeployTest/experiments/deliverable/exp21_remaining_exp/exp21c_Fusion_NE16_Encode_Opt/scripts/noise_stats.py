#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp21c task 1: what the int8 weight perturbation looks like on blocks 1-4 (fixture weights / multipliers), to judge the
linearity and bit-plane ideas. Per output channel c the noise is one of two constants (DualPerturb.h:172-173):
up_c = (mul_c + r) >> S, dn_c = (r - mul_c) >> S, r = 2^(S-1), S = log2D_w = 15; z = +1 -> (W+, W-) = (W+up, W+dn),
z = -1 -> (W+dn, W+up); then clamp to [-127, 127]. Prints |noise| range, whether up = -dn, and the expected number of
CLAMPED elements per pass (where conv(x, W+eps z) != conv(x, W) + conv(x, eps z)).  -- QW exp21c"""
import os
import numpy as np
import onnx
HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "..", "fixtures")
out = []
for b in (1, 2, 3, 4):
    d = os.path.join(F, f"QZO_DualConv2in_NE16P_B{b}")
    m = onnx.load(os.path.join(d, "network.onnx"))
    I = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
    I.update(dict(np.load(os.path.join(d, "inputs.npz"))))
    w = I[f"blocks.{b}.conv.weight_int8"].astype(np.int64)
    mul = I[f"blocks.{b}.conv.weight_int8_pmul"].astype(np.int64)
    S = 15; r = 1 << (S - 1)
    up = (mul + r) >> S; dn = (r - mul) >> S
    W = w.reshape(w.shape[0], -1)
    cu = np.abs(W + up[:, None]) > 127; cd = np.abs(W + dn[:, None]) > 127
    e_cl = (cu.mean() + cd.mean()) / 2
    out.append(f"block {b}: W {tuple(w.shape)} ({W.size} elems)  up {up.min()}..{up.max()}  dn {dn.min()}..{dn.max()}  "
               f"channels with up != -dn: {(up + dn != 0).sum()}/{len(up)}  |W|=127 elems: {(np.abs(W) == 127).sum()}  "
               f"expected clamped elems per pass: {e_cl * W.size:.1f} ({e_cl * 100:.2f} %), channels at risk: "
               f"{(cu | cd).any(1).sum()}/{len(up)}")
os.makedirs(os.path.join(HERE, "..", "results"), exist_ok = True)
open(os.path.join(HERE, "..", "results", "noise_stats.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
