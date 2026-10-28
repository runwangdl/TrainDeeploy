# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the QZO_Conv prototype fixture -- QW (dual-pass-fusion, exp3).
#
# SpeechNet block 0's QUANTIZED conv lifted verbatim out of the QZO training graph, with the two
# RQSPerturbRademacher nodes that feed it and the RequantShift that Deeploy fuses onto it:
#
#   blocks.0.conv.weight_int8 [8,1,1,4] i8 --RQSPerturbRademacher(idx=0, div=2^15)--> weight_pert --.
#   blocks.0.conv.bias_rqsadd [8]      i32 --RQSPerturbRademacher(idx=1, div=2^16)--> bias_pert ----+
#   <quantized window> [1,1,14,700] i8 ------------------------------------------------------------+--> Conv --> i32
#                                                                    RequantShift(mul_pc, bias_pert) --> i8 [1,8,14,701]
#
# On the device PULPConvRequantMergePass fuses Conv + RequantShift into ONE RequantizedConv node
# (pulp_nn_conv), exactly as in the full QZO network; the fixture keeps the two ONNX nodes, the
# pass does the fusion -- no fixture surgery. Node names, attribute protos (RequantShift's `div` /
# `n_levels` / `signed` are TENSOR-typed and must stay so), domains (`mezo`, `ai.onnx.contrib`) and
# the three int32 initializers are copied from the source graph.
#
# The conv's data input is the Quant node's OUTPUT in the full graph (int8); it is promoted to a
# graph input here and its value produced by Onnx4Deeploy's own op implementations
# (`run_onnx_graph`), i.e. the same oracle that validated the QZO flow bit-exact on device.
#
# Both perturb nodes stay RUNTIME nodes. In particular the requant bias is NOT baked in as a
# constant: PULPConvRequantMergePass bakes +div/2 rounding only into a CONSTANT add, so a
# constant-bias fixture would exercise the wrong branch (exp16c, --variable-bias lesson).
#
# Run in agitated_hugle (needs onnx4deeploy):
#   docker exec agitated_hugle bash -lc 'cd /app && PYTHONPATH=/app/Onnx4Deeploy:/app/TrainDeeploy \
#     python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_qzo_conv_fixture.py [--block i]'

import argparse
import os

import numpy as np
import onnx
from onnx import helper

from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph

HERE = os.path.dirname(os.path.abspath(__file__))
DEEPLOY_TEST = os.path.abspath(os.path.join(HERE, "..", ".."))
# exp19: the QZO fixture built on the artifacts_reference checkpoint (correct weights).
DEFAULT_SRC = os.path.join(DEEPLOY_TEST, "Tests/Models/Training/SpeechNet/speechnet_qzo19_train")
DEFAULT_DST = os.path.join(DEEPLOY_TEST, "Tests/Kernels/Integer/QZO_Conv")


# QW (dual_pass_fusion exp7): any block, not only block 0. Block i's nodes carry index i; the conv node of
#     block 0 has no suffix (`/wrappedInnerForwardImpl/Conv`), blocks 1-4 have `_<i>`. --block 0 still
#     writes the exp3 fixture byte for byte.
def block_names(block):
    conv = "/wrappedInnerForwardImpl/Conv" if block == 0 else f"/wrappedInnerForwardImpl_{block}/Conv"
    act = f"/blocks_{block}_conv_input_quant_1/Clip_output_0"          # Quant output = conv data input
    w_raw, b_raw = f"blocks.{block}.conv.weight_int8", f"blocks.{block}.conv.bias_rqsadd"
    nodes = [f"rqsp_blocks.{block}.conv.weight_int8", f"rqsp_blocks.{block}.conv.bias_rqsadd", conv,
             f"QSTANDALONE_QCDQ_/blocks_{block}_conv_output_quant_1/Clip_Quant"]
    out = f"/blocks_{block}_conv_output_quant_1/Clip_output_0"
    quant = f"QCDQ_/blocks_{block}_conv_input_quant_1/Clip_Quant"
    return act, w_raw, b_raw, nodes, out, quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default = DEFAULT_SRC)
    ap.add_argument("--dst", default = None,
                    help = "default: Tests/Kernels/Integer/QZO_Conv (block 0), QZO_Conv_B<i> (block i)")
    ap.add_argument("--block", type = int, default = 0, choices = range(5))
    args = ap.parse_args()
    if args.dst is None:
        args.dst = DEFAULT_DST if args.block == 0 else f"{DEFAULT_DST}_B{args.block}"
    ACT, W_RAW, B_RAW, NODES, OUT, QUANT = block_names(args.block)

    src = os.path.join(args.src, "network.onnx")
    model = onnx.load(src)
    g = model.graph
    npz = np.load(os.path.join(args.src, "inputs.npz"))
    base = [npz[k] for k in sorted(k for k in npz.files if k.startswith("arr_"))]
    assert len(base) == len(g.input), (len(base), len(g.input))
    feed = {i.name: np.asarray(v) for i, v in zip(g.input, base)}

    nodes_by_name = {n.name: n for n in g.node}
    nodes = [nodes_by_name[n] for n in NODES]
    inputs = {i.name: i for i in g.input}
    vinfo = {v.name: v for v in g.value_info}
    inits = {t.name: t for t in g.initializer}

    # --- the quantized window: run the real Quant node through Onnx4Deeploy's implementation ----
    act = np.asarray(run_onnx_graph(src, feed, output_names = [ACT])[0]).astype(np.int8)
    w_raw = feed[W_RAW].astype(np.int8)
    b_raw = feed[B_RAW].astype(np.int32)

    # --- graph interface --------------------------------------------------------------------
    graph_inputs = [vinfo[ACT], inputs[W_RAW], inputs[B_RAW]]          # ACT promoted from value_info
    needed_inits = sorted({i for n in nodes for i in n.input if i in inits})
    graph_inits = [inits[i] for i in needed_inits]
    internal = [vinfo[n.output[0]] for n in nodes[:3]]                  # weight_pert, bias_pert, Conv_output_0
    out_vi = vinfo[OUT]
    graph = helper.make_graph(nodes, "QZO_Conv", graph_inputs, [out_vi],
                              initializer = graph_inits, value_info = internal)
    m = helper.make_model(graph, opset_imports = list(model.opset_import))
    m.ir_version = model.ir_version
    m.producer_name = "build_qzo_conv_fixture.py"
    os.makedirs(args.dst, exist_ok = True)
    fx = os.path.join(args.dst, "network.onnx")
    onnx.save(m, fx)

    # --- reference: the fixture graph itself through the oracle, cross-checked vs the full graph --
    feed_fx = {ACT: act, W_RAW: w_raw, B_RAW: b_raw}
    y = np.asarray(run_onnx_graph(fx, feed_fx, output_names = [OUT])[0]).astype(np.int8)
    y_full = np.asarray(run_onnx_graph(src, feed, output_names = [OUT])[0]).astype(np.int8)
    assert y.shape == y_full.shape and np.array_equal(y, y_full), \
        f"fixture output != full-graph output at the same tensor ({(y != y_full).sum()} differ)"
    wpert = np.asarray(run_onnx_graph(fx, feed_fx, output_names = [nodes[0].output[0]])[0]).astype(np.int8)

    # generateNetwork.py / testMVP.py consume inputs.npz POSITIONALLY -> order == graph inputs.
    np.savez(os.path.join(args.dst, "inputs.npz"), **{ACT: act, W_RAW: w_raw, B_RAW: b_raw})
    np.savez(os.path.join(args.dst, "outputs.npz"), **{OUT: y})

    print(f"[QZO_Conv block {args.block}] wrote {args.dst}")
    print(f"  nodes        : {[f'{n.op_type}:{n.name}' for n in nodes]}")
    print(f"  initializers : {needed_inits}")
    print(f"  value_info   : {[v.name for v in internal]}")
    print(f"  act (i8)     : {act.shape} range [{act.min()},{act.max()}]  (Quant scale {[a for a in nodes_by_name[QUANT].attribute if a.name=='scale'][0].f})")
    print(f"  weight (i8)  : {w_raw.shape} range [{w_raw.min()},{w_raw.max()}]  perturbed on device -> [{wpert.min()},{wpert.max()}], "
          f"{int((wpert != w_raw).sum())}/{w_raw.size} elements moved")
    print(f"  bias (i32)   : {b_raw.tolist()}")
    print(f"  output (i8)  : {OUT} {y.shape} range [{y.min()},{y.max()}]  == full-graph tensor: True")
    print(f"  MACs         : {int(np.prod(y.shape)) * int(np.prod(w_raw.shape[1:])):,}")


if __name__ == "__main__":
    main()
