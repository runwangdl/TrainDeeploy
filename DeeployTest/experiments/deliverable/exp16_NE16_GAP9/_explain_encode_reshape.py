import numpy as np, sys
sys.path.insert(0, "/app/Deeploy")
from Deeploy.Targets.NE16.TopologyOptimizationPasses.Passes import (
    _weightEncode, _bestReshapeOption, _nSubtiles, _findAllReshapeOptions)

print("="*72); print("Q1: what _weightEncode actually produces")
print("="*72)
# one output channel, 16 input channels, 1x1 kernel -> the smallest complete unit
w = np.array([[1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255]], dtype=np.uint8).reshape(1,16,1,1)
print("input weights (uint8, cout=1, cin=16, 1x1):")
print("  ", list(w.reshape(-1)))
print("  as bits (LSB first):")
for ci in range(16):
    v = int(w.reshape(-1)[ci])
    print(f"    cin{ci:2d} = {v:3d} = bits[q0..q7] {[ (v>>q)&1 for q in range(8) ]}")
enc = _weightEncode(w, bits=8)
print(f"\nencoded shape {enc.shape}  (cout, cinMajor, bits*H*W*cinMinorBytes)")
e = enc.reshape(-1)
print("encoded bytes:", list(e))
print("\nread back as 8 bitplanes x 2 bytes (=16 cin bits):")
for q in range(8):
    lo, hi = int(e[2*q]), int(e[2*q+1])
    bits = [(lo >> b) & 1 for b in range(8)] + [(hi >> b) & 1 for b in range(8)]
    print(f"  bitplane q={q} (weight of 2^{q}): bytes=({lo:3d},{hi:3d}) -> cin bits {bits}")

print()
print("="*72); print("Q3: what the reshape actually buys, for our 14x87")
print("="*72)
H, W = 14, 87
print(f"original spatial ({H},{W}) -> subtiles = ceil({H}/3)*ceil({W}/3) = {_nSubtiles((H,W))}")
opts = sorted({tuple(sorted(o)) for o in _findAllReshapeOptions(H*W)})
scored = sorted(((_nSubtiles(o), o) for o in opts))
print("all factorisations considered (best 6):")
for n, o in scored[:6]:
    print(f"    {str(o):12s} -> {n} subtiles")
best = _bestReshapeOption(H*W)
print(f"\n_bestReshapeOption({H*W}) = {best} -> {_nSubtiles(best)} subtiles")
print(f"gain vs original: {_nSubtiles((H,W))} -> {_nSubtiles(best)} "
      f"= {100*(1-_nSubtiles(best)/_nSubtiles((H,W))):.1f}% fewer")
