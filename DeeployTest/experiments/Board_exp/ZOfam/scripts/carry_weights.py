#!/usr/bin/env python3
# Copied unchanged from silentwear-zo/rounds/carry_weights.py (Run) for Board_exp. -- QW
"""Replace the 22 initial trainable weights (arr_0002..) of a fixture inputs.npz by the last [WDUMP] of a device log.
   carry_weights.py <uart.log> <inputs.npz> <train network.onnx>"""
import re, sys
import numpy as np, onnx
log, npz, net = sys.argv[1:4]
ELEM = {1: '<f4', 3: 'i1', 6: '<i4'}
ins = onnx.load(net).graph.input[2:]
last = {}
for l in open(log, errors='ignore'):
    m = re.search(r"\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)", l)
    if m: last[int(m.group(2))] = np.array([int(h, 16) for h in m.group(4).split()], '<u4')
assert len(last) == len(ins) == 22, (len(last), len(ins))
d = dict(np.load(npz))
for wi, i in enumerate(ins):
    key = f'arr_{wi + 2:04d}'; old = d[key]
    dt = np.dtype(ELEM[i.type.tensor_type.elem_type])
    v = last[wi].view(np.uint8).view(np.int8) if dt == np.int8 else last[wi].view(dt)
    v = v[:old.size].reshape(old.shape).astype(old.dtype)
    assert v.dtype == old.dtype and v.shape == old.shape, (key, i.name)
    d[key] = v
np.savez(npz, **d)
print(f"carried 22 tensors from {log} into {npz}")
