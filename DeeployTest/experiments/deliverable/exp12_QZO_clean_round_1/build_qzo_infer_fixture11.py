# SPDX-License-Identifier: MIT
"""exp11 — quantized inference fixture builder with the CORRECT RQS-add rounding-offset
handling (fixes the exp9/10 eval's +0.5 LSB systematic double-offset).

Background (qzo_weight_integerize.py:366 "exp10 ROUNDING FIX"): for the TRAIN graph the RQS
`add` is a VARIABLE, so bif bakes the `+div/2` rounding constant into its values (device kernel
and host executor both truncate). The device WDUMP therefore contains the offset. In an
INFERENCE fixture the add is a CONSTANT initializer and the shipped Conv+RequantShift merge
pass bakes `+div/2` AGAIN on device — so injecting the dump unmodified double-counts it.

Correct construction (this script):
  * reference graph  = adds WITH the offset (as built / as dumped)  → run_onnx_graph truncates,
                       trunc(x + div/2) == round(x)
  * device network.onnx = same graph with `div/2` SUBTRACTED from every RequantShift add
                       initializer → the merge pass re-adds it on device == round(x)
  Both sides then compute identical rounding semantics, and reference == device.

Usage (inside agitated_hugle, from this directory):
  python3 build_qzo_infer_fixture11.py --fixture-dir /app/Onnx4Deeploy/QZO_exp/exp11_round1_fix \
      [--dump-npz results/dumped_weights.npz] --out-dir qinfer_round1 [--windows N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, "/app/Onnx4Deeploy")
EVAL_CACHE = "/app/Onnx4Deeploy/QZO_exp/exp_calibration/data_cache_incr.npz"  # evX1/evY1 = whole batch 2


def integerize(fixture_dir: str) -> onnx.ModelProto:
    from onnx4deeploy.transform.qzo_weight_integerize import build_int8_forward
    base = onnx.load(str(Path(fixture_dir) / "network.onnx"))
    model, _ = build_int8_forward(base)
    return model


def rqs_add_names_and_div(model: onnx.ModelProto):
    """{add_initializer_name: div} for every RequantShift whose add input is an initializer."""
    initset = {i.name for i in model.graph.initializer}
    out = {}
    for n in model.graph.node:
        if n.op_type != "RequantShift" or len(n.input) < 3:
            continue
        add_name = n.input[2]
        if add_name not in initset:
            continue
        div = 1 << 16
        for a in n.attribute:
            if a.name == "div":
                v = onnx.helper.get_attribute_value(a)
                div = int(numpy_helper.to_array(v)) if hasattr(v, "dims") else int(v)
        out[add_name] = div
    return out


def inject(model: onnx.ModelProto, dump_npz: str) -> int:
    dump = np.load(dump_npz)
    initmap = {i.name: k for k, i in enumerate(model.graph.initializer)}
    n = 0
    for name in dump.files:
        assert name in initmap, f"dump tensor {name!r} not an initializer of the infer graph"
        k = initmap[name]
        old = numpy_helper.to_array(model.graph.initializer[k])
        new = np.asarray(dump[name])
        assert old.dtype == new.dtype, f"{name}: {old.dtype} vs dump {new.dtype}"
        model.graph.initializer[k].CopyFrom(
            numpy_helper.from_array(new.reshape(old.shape), name=name))
        n += 1
    return n


def strip_offsets(model: onnx.ModelProto, adds: dict) -> int:
    """Subtract div/2 from every RQS add initializer (device-consumed graph only)."""
    initmap = {i.name: k for k, i in enumerate(model.graph.initializer)}
    n = 0
    for name, div in adds.items():
        k = initmap[name]
        v = numpy_helper.to_array(model.graph.initializer[k]).astype(np.int64) - div // 2
        model.graph.initializer[k].CopyFrom(
            numpy_helper.from_array(v.astype(np.int32), name=name))
        n += 1
    return n


def reference_logits(net_path: str, X: np.ndarray) -> np.ndarray:
    from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph
    outs = []
    for i in range(X.shape[0]):
        r = run_onnx_graph(net_path, {"input": X[i:i + 1]})
        outs.append(np.asarray(r[0] if isinstance(r, (list, tuple)) else r, np.float32).reshape(-1))
        if (i + 1) % 30 == 0:
            print(f"  reference {i+1}/{X.shape[0]}", flush=True)
    return np.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-dir", required=True)
    ap.add_argument("--dump-npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--windows", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    model = integerize(a.fixture_dir)                      # adds WITH offset (bif bakes div/2)
    if a.dump_npz:
        n = inject(model, a.dump_npz)                       # dump also carries the offset — ok
        print(f"[fixture] injected {n} dumped tensors")
    adds = rqs_add_names_and_div(model)
    print(f"[fixture] RQS add initializers: {list(adds)[:2]}… ({len(adds)}), div={set(adds.values())}")

    # exp11 FIX (2026-09-08): generate the host reference on the EXACT graph the device
    #   compiles (network.onnx, offset-stripped) — NOT the offset-included variant. The
    #   assumed executor equivalence trunc(x+div/2)==round(x) between the two graphs is FALSE
    #   (empirically they differ by up to 1.5 in logits); the device matches run_onnx_graph on
    #   network.onnx to ~0.087. Keep the offset-included graph only as a debug artifact. -- QW
    import copy
    dev = copy.deepcopy(model)
    ns = strip_offsets(dev, adds)                           # device graph (offset OUT; merge re-adds)
    dev_path = out / "network.onnx"
    onnx.save(dev, str(dev_path))
    onnx.save(model, str(out / "network_ref_offset.onnx"))  # debug-only (offset IN)
    print(f"[fixture] stripped div/2 from {ns} adds for the device graph")

    d = np.load(EVAL_CACHE)
    X, Y = d["evX1"].astype(np.float32), d["evY1"].astype(np.int64)
    if a.windows:
        X, Y = X[:a.windows], Y[:a.windows]
    ref = reference_logits(str(dev_path), X)               # reference on the DEVICE graph -- QW fix
    np.savez(out / "inputs.npz", input=X, label=Y)
    np.savez(out / "outputs.npz", output=ref)
    bal = float(np.mean([(ref.argmax(1) == Y)[Y == c].mean() for c in np.unique(Y)]))
    print(f"[fixture] {X.shape[0]} windows; host-reference balanced accuracy = {100*bal:.2f}%")


if __name__ == "__main__":
    main()
