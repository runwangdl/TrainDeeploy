"""Cross-check the SpeechNet_Test fixtures against the fixtures the experiments ran on.

Run inside `agitated_hugle` (needs onnx + numpy):
    python3 /app/TrainDeeploy/SpeechNet_Test/scripts/crosscheck.py [pair ...] > logs/crosscheck.txt

Two kinds of check:
  * identity  -- a re-export of an existing fixture must be identical: every .onnx graph equal
                 (serialized model, or node-by-node + initializer values when metadata differs) and
                 every .npz array bit-identical.
  * prefix    -- a single-step fixture must equal round 1's first step: same graph, same initial
                 weights, same first data window, same first reference loss(es).
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

APP = Path("/app/TrainDeeploy")
NEW = APP / "SpeechNet_Test"
OLD = APP / "DeeployTest/Tests/Models/Training/SpeechNet"
DLV = APP / "DeeployTest/experiments/deliverable"
KER = APP / "DeeployTest/Tests/Kernels/Integer"


def graph_signature(path):
    m = onnx.load(str(path))
    g = m.graph
    nodes = [(n.op_type, tuple(n.input), tuple(n.output), tuple(sorted(
        (a.name, onnx.helper.get_attribute_value(a).__repr__()) for a in n.attribute))) for n in g.node]
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    ios = ([i.name for i in g.input], [o.name for o in g.output])
    return m, nodes, inits, ios


def cmp_onnx(a, b):
    ma, na, ia, ioa = graph_signature(a)
    mb, nb, ib, iob = graph_signature(b)
    if ma.SerializeToString() == mb.SerializeToString():
        return "identical (bytes)"
    probs = []
    if ioa != iob:
        probs.append("graph inputs/outputs differ")
    if na != nb:
        probs.append(f"nodes differ ({len(na)} vs {len(nb)})")
    if set(ia) != set(ib):
        probs.append(f"initializer names differ ({len(ia)} vs {len(ib)})")
    else:
        bad = [k for k in ia if ia[k].shape != ib[k].shape or not np.array_equal(ia[k], ib[k])]
        if bad:
            probs.append(f"{len(bad)} initializers differ in value, e.g. {bad[:3]}")
    return "identical (graph + initializers; metadata differs)" if not probs else "DIFFERENT: " + "; ".join(probs)


def cmp_npz(a, b):
    A, B = np.load(a), np.load(b)
    ka, kb = set(A.files), set(B.files)
    probs = []
    if ka != kb:
        probs.append(f"keys differ: only-new {sorted(ka - kb)[:5]} only-old {sorted(kb - ka)[:5]}")
    bad = [k for k in sorted(ka & kb) if A[k].shape != B[k].shape or not np.array_equal(A[k], B[k])]
    if bad:
        probs.append(f"{len(bad)}/{len(ka & kb)} arrays differ, e.g. {bad[:4]}")
    return f"identical ({len(ka)} arrays)" if not probs else "DIFFERENT: " + "; ".join(probs)


def identity(name, new, old):
    print(f"\n== identity: {name}\n   new {new}\n   old {old}")
    files = sorted({p.name for p in new.iterdir() if p.suffix in (".onnx", ".npz")} &
                   {p.name for p in old.iterdir() if p.suffix in (".onnx", ".npz")})
    only_new = sorted({p.name for p in new.iterdir()} - {p.name for p in old.iterdir()})
    only_old = sorted({p.name for p in old.iterdir()} - {p.name for p in new.iterdir()})
    for f in files:
        fn = cmp_onnx if f.endswith(".onnx") else cmp_npz
        print(f"   {f:28s} {fn(new / f, old / f)}")
    if only_new or only_old:
        print(f"   files only in new: {only_new}   only in old: {only_old}")


def prefix(name, single, round1, loss_keys):
    """single-step fixture == first step of round 1."""
    print(f"\n== prefix: {name}\n   single {single}\n   round1 {round1}")
    print(f"   network.onnx                 {cmp_onnx(single / 'network.onnx', round1 / 'network.onnx')}")
    S, R = np.load(single / "inputs.npz"), np.load(round1 / "inputs.npz")
    base = sorted(k for k in S.files if k.startswith("arr_"))
    same = [k for k in base if k in R.files and np.array_equal(S[k], R[k])]
    print(f"   inputs.npz arr_* (window 0 + initial weights): {len(same)}/{len(base)} equal to round 1")
    meta = {k: S[k].tolist() for k in S.files if k.startswith("meta_")}
    extra = [k for k in S.files if k.startswith("mb")]
    print(f"   single-step meta {meta}; extra windows {len(extra) // 2}")
    So, Ro = np.load(single / "outputs.npz"), np.load(round1 / "outputs.npz")
    for k in loss_keys:
        if k in So.files and k in Ro.files:
            s, r = np.ravel(So[k]), np.ravel(Ro[k])
            n = min(len(s), len(r))
            print(f"   outputs {k:12s} single {s[:n][:2]}  round1 first {r[:n][:2]}  equal={np.array_equal(s[:n], r[:n])}")
        else:
            print(f"   outputs {k}: missing (single has {So.files[:8]}...)")


def calib_diff(name, new, old):
    """Same structure / weights / data; report which constants differ (expected: calibration only)."""
    print(f"\n== calibration-only difference: {name}\n   new {new}\n   old {old}")
    _, na, ia, ioa = graph_signature(new / "network.onnx")
    _, nb, ib, iob = graph_signature(old / "network.onnx")
    print(f"   graph inputs/outputs equal: {ioa == iob}; node count {len(na)} vs {len(nb)}; "
          f"op sequence equal: {[n[0] for n in na] == [n[0] for n in nb]}")
    diff = sorted(k for k in ia if k in ib and not np.array_equal(ia[k], ib[k]))
    print(f"   initializers differing: {len(diff)}/{len(ia)} -> {diff}")
    S, R = np.load(new / "inputs.npz"), np.load(old / "inputs.npz")
    base = sorted(k for k in S.files if k.startswith("arr_"))
    same = [k for k in base if k in R.files and np.array_equal(S[k], R[k])]
    print(f"   inputs.npz arr_* (window 0 + initial weights): {len(same)}/{len(base)} equal")


if __name__ == "__main__":
    want = set(sys.argv[1:])
    run = lambda k: not want or k in want
    if run("bp"):
        identity("BP round 1 train", NEW / "speechnet_train_BP_round1", OLD / "speechnet_train_fullfrozen_b1_fold3_ref")
        identity("BP round 1 optimizer", NEW / "speechnet_optimizer_BP_round1", OLD / "speechnet_optimizer_fullfrozen_b1_fold3_ref")
        identity("BP round 1 infer (batch 2, exp17 device weights)", NEW / "speechnet_infer_BP_round1", OLD / "speechnet_infer_fullfrozen_b1_fold3_ref")
        prefix("BP single step vs round 1", NEW / "speechnet_train_BP_single_step", NEW / "speechnet_train_BP_round1", ["loss"])
    if run("zo"):
        identity("ZO round 1 train", NEW / "speechnet_train_ZO_round1", OLD / "speechnet_zo_train_b1_fold3_ref")
        identity("ZO round 1 update", NEW / "speechnet_update_ZO_round1", OLD / "speechnet_zo_update_b1_fold3_ref")
        identity("ZO round 1 infer (batch 2, exp18 device weights)", NEW / "speechnet_infer_ZO_round1", OLD / "speechnet_zo_infer_b1_fold3_ref")
        prefix("ZO single step vs round 1", NEW / "speechnet_train_ZO_single_step", NEW / "speechnet_train_ZO_round1", ["loss_plus", "loss_minus"])
    if run("qzo"):
        # QZO round 1 is not regenerated (dropped 2026-10-12). The single step uses the 99.9 calibration,
        # exp19 used 99.99: same graph structure, same weights and window, only the calibration-dependent
        # constants (Quant/Dequant scales, requant mul/add) may differ.
        calib_diff("QZO single step (pooled@99.9) vs exp19 round 1 (pooled@99.99)",
                   NEW / "speechnet_train_QZO_single_step", OLD / "speechnet_qzo19_train")
        calib_diff("QZO single-step update graph vs exp19", NEW / "speechnet_update_QZO_single_step", OLD / "speechnet_qzo19_update")
    if run("kernels"):
        for new, old in (("speechnet_unfused_b0_cluster", "QZO_Conv"), ("speechnet_fused_b0_cluster", "QZO_DualConv"),
                         ("speechnet_unfused_b0_NE16", "QZO_Conv_NE16"), ("speechnet_fused_b0_NE16", "QZO_DualConv_NE16")):
            identity(f"kernel {new}", NEW / new, KER / old)
