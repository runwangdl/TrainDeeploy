"""Show that NE16's bit-serial dataflow reproduces an ordinary dot product.
Mirrors gvsoc ne16_matrixvec.cpp:96-99 (__BinConvBlock) + :337 (scale=1<<q) + :357 (qw loop)."""
import numpy as np, sys
sys.path.insert(0, "/app/Deeploy")
from Deeploy.Targets.NE16.TopologyOptimizationPasses.Passes import _weightEncode

# ---- toy: 4 input channels, 2-bit weights, so it fits on one screen -------------
w = np.array([3, 1, 0, 2], dtype=np.uint8)     # weights   (one output channel)
x = np.array([10, 20, 30, 40], dtype=np.int64) # activations
print("w =", list(w), " x =", list(x))
print("ordinary dot product  =", int((w.astype(np.int64) * x).sum()))
print()
total = 0
for q in range(2):                              # qw = 2 bitplanes
    plane = (w >> q) & 1                        # bit q of every channel
    masked = int((plane * x).sum())             # __BinConvBlock: sum of x where bit==1
    scale = 1 << q                              # matrixvec_cycle: scale = 1 << mv_qw_iter
    total += masked * scale
    print(f"  pass q={q}: bitplane {list(plane)} -> masked sum {masked:3d} x 2^{q} = {masked*scale}")
print("  bit-serial total      =", total)
print()

# ---- the real thing: 16 channels, qw=8, using Deeploy's actual encoder ----------
rng = np.random.default_rng(0)
W = rng.integers(0, 256, size=(1, 16, 1, 1), dtype=np.uint8)
X = rng.integers(0, 128, size=16).astype(np.int64)
enc = _weightEncode(W, bits=8).reshape(-1)      # 16 bytes = 8 bitplanes x 2 bytes
acc = 0
for q in range(8):
    lo, hi = int(enc[2*q]), int(enc[2*q+1])
    plane = np.array([(lo >> b) & 1 for b in range(8)] + [(hi >> b) & 1 for b in range(8)])
    acc += int((plane * X).sum()) * (1 << q)    # exactly the two lines above
print("16ch/8bit: bit-serial from the ENCODED bytes =", acc)
print("16ch/8bit: ordinary dot product              =", int((W.reshape(-1).astype(np.int64) * X).sum()))
print("match:", acc == int((W.reshape(-1).astype(np.int64) * X).sum()))
