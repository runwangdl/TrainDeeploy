# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""exp8 debug tool: an inference fixture of the dual-pass graph TRUNCATED at one tensor, with the reference from the
Onnx4Deeploy oracle run on the oracle-expanded dual graph (oracle_check_dual.py's expansion). Running it on the device
tells whether everything up to that tensor is lowered correctly.

  docker exec agitated_hugle bash -lc 'cd /app && PYTHONPATH=/app/Onnx4Deeploy:/app/TrainDeeploy python3 \
    TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/exp8_whole_network/bisect_dual.py <tensor> <tag>'
writes exp8_whole_network/bisect/<tag>/{network.onnx, inputs.npz, outputs.npz}
"""
import os
import sys
import tempfile

import numpy as np
import onnx
import onnx_graphsurgeon as gs

sys.path.insert(0, "/app/TrainDeeploy")
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.ZODualPassFusion import zoDualPassFusion
from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FX = "/app/TrainDeeploy/SpeechNet_Test/speechnet_train_QZO_single_step"


def expand_for_oracle(g: gs.Graph) -> gs.Graph:
    """RequantizedDualConv / PerturbRademacherFixed -> ops the oracle implements (per-sign chains + Concat)."""
    k = [0]

    def nm(b):
        k[0] += 1
        return f"{b}__x{k[0]}"

    new, rm = [], []
    for n in list(g.nodes):
        if n.op == "PerturbRademacherFixed":
            mode = int(n.attrs["mode"])
            outs = []
            for sgn in ((1, -1) if mode == 0 else ((1,) if mode == 1 else (-1,))):
                o = gs.Variable(nm(n.name), dtype = np.float32)
                new.append(gs.Node("PerturbRademacher", nm(n.name), inputs = [n.inputs[0]], outputs = [o],
                                   attrs = {"seed": n.attrs["seed"], "idx": n.attrs["idx"],
                                            "eps": float(n.attrs["eps"]) * sgn}))
                outs.append(o)
            out = n.outputs[0]
            n.outputs = []
            rm.append(n)
            pad = int(n.attrs.get("pad_cols", 0))
            if pad:
                off = int(n.attrs.get("col_offset", 0))
                rows, cols = n.inputs[0].shape
                parts = []
                if off:
                    parts.append(gs.Constant(nm("zl"), np.zeros((rows, off), np.float32)))
                parts.append(outs[0])
                if pad - off - cols:
                    parts.append(gs.Constant(nm("zr"), np.zeros((rows, pad - off - cols), np.float32)))
                new.append(gs.Node("Concat", nm("pad"), inputs = parts, outputs = [out], attrs = {"axis": 1}))
            else:
                new.append(gs.Node("Identity" if len(outs) == 1 else "Concat", nm("j"), inputs = outs, outputs = [out],
                                   attrs = {} if len(outs) == 1 else {"axis": 0}))
        elif n.op == "DualHalf":
            C = int(n.outputs[0].shape[1]); h = int(n.attrs["half"])
            out = n.outputs[0]; n.outputs = []; rm.append(n)
            new.append(gs.Node("Slice", nm("half"), outputs = [out], inputs = [
                n.inputs[0], gs.Constant(nm("st"), np.array([h * C])), gs.Constant(nm("en"), np.array([(h + 1) * C])),
                gs.Constant(nm("ax"), np.array([1]))]))
        elif n.op == "RequantizedDualConv":
            x, W, mul, add, wmul, bmul = n.inputs[:6]
            stacked = int(n.attrs.get("stacked_input", 0))
            Ci = W.shape[1]
            outs = []
            for s in (0, 1):
                xs = x
                if stacked:
                    xs = gs.Variable(nm("xs"), dtype = np.int8)
                    new.append(gs.Node("Slice", nm("sl"), outputs = [xs], inputs = [
                        x, gs.Constant(nm("st"), np.array([s * Ci])), gs.Constant(nm("en"), np.array([(s + 1) * Ci])),
                        gs.Constant(nm("ax"), np.array([1]))]))
                wm_s = wmul if s == 0 else gs.Constant(nm("nwm"), -np.asarray(wmul.values))
                bm_s = bmul if s == 0 else gs.Constant(nm("nbm"), -np.asarray(bmul.values))
                wp, bp = gs.Variable(nm("wp"), dtype = np.int8), gs.Variable(nm("bp"), dtype = np.int32)
                new.append(gs.Node("RQSPerturbRademacher", nm("pw"), domain = "mezo", inputs = [W, wm_s], outputs = [wp],
                                   attrs = {"seed": n.attrs["seed_w"], "idx": n.attrs["idx_w"], "div": n.attrs["div_w"],
                                            "n_levels": 256, "signed": 1}))
                new.append(gs.Node("RQSPerturbRademacher", nm("pb"), domain = "mezo", inputs = [add, bm_s], outputs = [bp],
                                   attrs = {"seed": n.attrs["seed_b"], "idx": n.attrs["idx_b"], "div": n.attrs["div_b"],
                                            "n_levels": 4294967296, "signed": 1}))
                co, y = gs.Variable(nm("co"), dtype = np.int32), gs.Variable(nm("y"), dtype = np.int8)
                new.append(gs.Node("Conv", nm("conv"), inputs = [xs, wp], outputs = [co],
                                   attrs = {a: n.attrs[a] for a in ("kernel_shape", "pads", "strides", "dilations", "group")}))
                new.append(gs.Node("RequantShift", nm("rqs"), domain = "ai.onnx.contrib", inputs = [co, mul, bp], outputs = [y],
                                   attrs = {a: n.attrs[a] for a in ("div", "n_levels", "signed")}))
                outs.append(y)
            out = n.outputs[0]
            n.outputs = []
            rm.append(n)
            new.append(gs.Node("Concat", nm("cat"), inputs = outs, outputs = [out], attrs = {"axis": 1}))
    for n in rm:
        g.nodes.remove(n)
    g.nodes.extend(new)
    g.cleanup().toposort()
    return g


def main():
    tensor, tag = sys.argv[1], sys.argv[2]
    model = onnx.load(f"{FX}/network.onnx")
    npz = np.load(f"{FX}/inputs.npz")
    base = [npz[k] for k in sorted(k for k in npz.files if k.startswith("arr_"))]
    feed = {i.name: np.asarray(v) for i, v in zip(model.graph.input, base)}

    # truncated dual graph (what the device runs)
    g = zoDualPassFusion(gs.import_onnx(model))
    t = g.tensors()[tensor]
    g.outputs = [t]
    g.cleanup().toposort()
    g.inputs = [i for i in g.inputs if len(i.outputs)]  # graphsurgeon's cleanup keeps unused graph inputs
    used = [i.name for i in g.inputs]
    dmodel = gs.export_onnx(g)
    dmodel.opset_import.extend([o for o in model.opset_import if o.domain not in [x.domain for x in dmodel.opset_import]])

    # reference: oracle on the expanded dual graph
    ge = expand_for_oracle(zoDualPassFusion(gs.import_onnx(model)))
    ge.outputs = [ge.tensors()[tensor]]
    ge.cleanup().toposort()
    em = gs.export_onnx(ge)
    em.opset_import.extend([o for o in model.opset_import if o.domain not in [x.domain for x in em.opset_import]])
    tmp = tempfile.mktemp(suffix = ".onnx")
    onnx.save(em, tmp)
    ref = np.asarray(run_onnx_graph(tmp, {k: feed[k] for k in [i.name for i in ge.inputs]}, output_names = [tensor])[0])
    os.unlink(tmp)

    dst = os.path.join(HERE, "bisect", tag)
    os.makedirs(dst, exist_ok = True)
    onnx.save(dmodel, f"{dst}/network.onnx")
    np.savez(f"{dst}/inputs.npz", **{k: feed[k] for k in used})
    np.savez(f"{dst}/outputs.npz", **{tensor: ref.astype(np.float32 if ref.dtype.kind == 'f' else ref.dtype)})
    print(f"[bisect] {tag}: {tensor} {ref.shape} {ref.dtype} range [{ref.min()}, {ref.max()}]; inputs {used}")


if __name__ == "__main__":
    main()
