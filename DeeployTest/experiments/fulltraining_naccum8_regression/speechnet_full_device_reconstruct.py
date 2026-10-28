#!/usr/bin/env python3
# Reconstruct the FULL-TRAINING device final weights from [WDUMP] lines, inject into the
# infer graph, eval on batch-2. Unlike head-only (1 weight), full training dumps up to 32 weight
# tensors. We map each device wi -> infer-graph initializer ROBUSTLY: by element count, then (when
# a size is ambiguous, e.g. several length-C BN tensors) by nearest match to the ORT reference
# (outputs.npz) value, which the device only perturbs slightly (drift). running_mean/var are NOT
# trained on-device (BatchNormInternal never updates them) -> kept = infer-graph frozen values.
#
# Usage: python3 speechnet_full_device_reconstruct.py <run.log> <train_fixture_dir> <out_infer_dir>
import re, struct, sys, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper

LOG, TRAIN_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
B2="/app/Onnx4Deeploy/onnx/model/speechnet_infer_batch2"

# ---- 1. parse WDUMP (use the LAST step's dump = final weights) ----
pat=re.compile(r'\[WDUMP s=(\d+) wi=(\d+) n=(\d+)\]\s*([0-9a-fA-F ]+)')
dumps={}  # (step,wi) -> np.array
for line in open(LOG):
    m=pat.search(line)
    if not m: continue
    step,wi,n=int(m.group(1)),int(m.group(2)),int(m.group(3))
    words=m.group(4).split()
    arr=np.array([struct.unpack('<f',struct.pack('<I',int(w,16)))[0] for w in words],dtype=np.float32)
    assert len(arr)==n, f"wi={wi} expected {n} got {len(arr)}"
    dumps[(step,wi)]=arr
laststep=max(s for s,_ in dumps)
dev={wi:dumps[(laststep,wi)] for (s,wi) in dumps if s==laststep}
print(f"parsed {len(dev)} device weight tensors at final step {laststep}")

# ---- 2. infer graph initializers + ORT reference (outputs.npz) ----
gi=onnx.load(TRAIN_DIR+"/network_infer.onnx")
inits={init.name: numpy_helper.to_array(init).copy() for init in gi.graph.initializer}
try: ref={k:np.asarray(v) for k,v in np.load(TRAIN_DIR+"/outputs.npz").items() if k!='loss'}
except Exception: ref={}
def refname(k): return k  # outputs.npz keys already match init names for full training
RUNSTAT=lambda n: n.endswith('running_mean') or n.endswith('running_var')

# ---- 3. map device wi -> init name ----
# Group init names by element count; assign device tensors of that count to the names,
# disambiguating by nearest-value to ORT ref (falls back to graph order).
from collections import defaultdict
names_by_size=defaultdict(list)
for nm,arr in inits.items():
    if RUNSTAT(nm): continue          # frozen: never injected
    names_by_size[arr.size].append(nm)
dev_by_size=defaultdict(list)
for wi,arr in sorted(dev.items()):
    dev_by_size[arr.size].append((wi,arr))

assigned={}  # init name -> device array
for sz,names in names_by_size.items():
    cand=dev_by_size.get(sz,[])
    if len(cand)<len(names):
        print(f"  WARN size {sz}: {len(names)} names but only {len(cand)} device tensors")
    used=set()
    for nm in names:
        r=ref.get(refname(nm))
        best=None;bd=None
        for j,(wi,arr) in enumerate(cand):
            if j in used: continue
            d=float(np.abs(arr-r.ravel()).sum()) if r is not None and r.size==arr.size else float(wi)
            if bd is None or d<bd: bd=d;best=j
        if best is not None:
            used.add(best); assigned[nm]=cand[best][1]
print(f"assigned {len(assigned)} / {len([n for n in inits if not RUNSTAT(n)])} trainable initializers from device dump")

# ---- 4. inject + write infer graph ----
newinits=[]
for init in gi.graph.initializer:
    nm=init.name
    if nm in assigned:
        a=assigned[nm].reshape(inits[nm].shape).astype(inits[nm].dtype)
        newinits.append(numpy_helper.from_array(a,nm))
    else:
        newinits.append(init)
del gi.graph.initializer[:]; gi.graph.initializer.extend(newinits)
import os; os.makedirs(OUT,exist_ok=True)
onnx.save(gi, OUT+"/network.onnx")

# ---- 5. eval balanced acc on batch-2 ----
X=np.load(B2+"/inputs.npz")["input"].astype(np.float32); y=np.load(B2+"/inputs.npz")["label"].astype(np.int64)
sess=ort.InferenceSession(OUT+"/network.onnx",providers=['CPUExecutionProvider'])
inm=sess.get_inputs()[0].name
pred=np.array([int(np.argmax(sess.run(None,{inm:X[i:i+1]})[0])) for i in range(len(y))])
cls=sorted(set(y.tolist())); bacc=float(np.mean([(pred[y==k]==k).mean() for k in cls]))
np.savez(OUT+"/inputs.npz",**dict(np.load(B2+"/inputs.npz")))
print(f"DEVICE full-training reconstructed -> batch2 balanced acc = {100*bacc:.2f}%")
print(f"(infer graph written to {OUT}/network.onnx)")
