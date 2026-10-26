#!/usr/bin/env python3
"""exp21e/fix: build the BP round-1 fixture whose REFERENCE is the C-order emulator (bp_emu.py) instead of PyTorch.

Inputs, graphs and optimizer graph are copied unchanged from the exp17 fixture (same data, same initial weights,
same recipe: 540 steps, n_accum 4 SUM, SGD lr 3e-4, frozen BN, argmax-mask MaxPool); only outputs.npz changes:
  loss       : the emulator's 2160 per-forward losses
  <trainable>: the emulator's weights after the last step;  running stats: unchanged (frozen)
The ConvGradW W-tiling the emulator reproduces is parsed from the generated TrainingNetwork.c given with --gen.
usage: python3 make_corder_fixture.py --src <exp17 train fixture> --dst <new train fixture> --gen <TrainingNetwork.c>
       (the _optimizer sibling is copied next to --dst)                                        -- QW exp21e
"""
import argparse, hashlib, json, os, shutil, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bp_emu  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--gen", required=True)
ap.add_argument("--steps", type=int, default=540); ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
a = ap.parse_args()

W, X, Y, RS, o = bp_emu.load_fixture(a.src)
gen = open(a.gen).read()
Wo = {0: 701, 1: 88, 2: 23, 3: 5, 4: 5}
wt = {b: bp_emu.convgradw_wtiles(gen, b, Wo[b]) for b in range(5)}
t0 = time.time()
losses, Wf = bp_emu.train(W, X, Y, RS, np.float32(9.999999747378752e-06), np.float32(a.lr), a.steps, a.accum, wt,
                          log_every=60)
print(f"emulated {a.steps} steps in {time.time() - t0:.0f}s; ConvGradW W-tiles {wt}")

os.makedirs(a.dst, exist_ok=True)
for f in os.listdir(a.src):
    if f != "outputs.npz":
        shutil.copy(os.path.join(a.src, f), os.path.join(a.dst, f))
out = {k: np.asarray(o[k]) for k in o.files}
for k in bp_emu.NAMES:
    out[k] = Wf[k].reshape(out[k].shape).astype(np.float32)
out["loss"] = losses.astype(np.float32)
np.savez(os.path.join(a.dst, "outputs.npz"), **out)
src_opt = a.src.replace("_train", "_optimizer"); dst_opt = a.dst.replace("_train", "_optimizer")
shutil.copytree(src_opt, dst_opt, dirs_exist_ok=True)
pyt = np.asarray(o["loss"], np.float64)
prov = dict(reference="C-order emulator exp21e/fix/scripts/bp_emu.py + libm_emu.c (device picolibc expf/logf)",
            src_fixture=os.path.abspath(a.src), gen_c=os.path.abspath(a.gen),
            gen_c_sha256=hashlib.sha256(gen.encode()).hexdigest(), convgradw_wtiles=wt,
            steps=a.steps, n_accum=a.accum, lr=a.lr,
            vs_pytorch_ref=dict(max_abs=float(np.abs(losses - pyt).max()),
                                n_gt_1e3=int((np.abs(losses - pyt) > 1e-3).sum())))
json.dump(prov, open(os.path.join(a.dst, "corder_provenance.json"), "w"), indent=1)
print(json.dumps(prov, indent=1))
