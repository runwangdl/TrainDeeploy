#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp16 STEP 2a — build a single-layer SpeechNet **block-1** fixture that NE16 can claim.

Why this fixture and not a toy model
------------------------------------
NE16 implements only 1x1, 3x3 and 3x3-depthwise (CONFIG0 [6:5]); SpeechNet's convs are
1x4/1x16/1x8/7x1/7x1, so `NE16Engine.canExecute` accepts 0/5 (see _ne16_canexecute_probe.py).
STEP 2a's goal is to prove the *plumbing* — engine coloring, `_weightEncode`, the tile
constraints, the `ne16_task_t` codegen and the pulp-nnx link — with **zero new Deeploy logic**,
before the 1xk decomposition (STEP 2b) is written.

So we take block 1 of the real exp12 **inference** fixture, which
  * carries the real int8 weights as gs.Constant   (so BLOCKER 2 is absent), and
  * is fed by MaxPool<-Relu, so its activations are >= 0
    (so BLOCKER 3, the phantom signed-input bit 26, is absent),
and reduce only the kernel 1x16 -> 1x1 so that BLOCKER 1 is absent too. Everything else stays
real: Cin=8, Cout=16, the 14x87 spatial extent, the per-channel requantisation, and real
activations taken from the real evaluation windows.

Block 1 is the right layer to pick: 2,523,136 of SpeechNet's 3,711,680 conv MACs (68 %).

Source of truth for every tensor:
  TrainDeeploy/DeeployTest/experiments/deliverable/exp12_QZO_clean_round_1/qinfer/network.onnx
  TrainDeeploy/DeeployTest/experiments/deliverable/exp12_QZO_clean_round_1/qinfer/inputs.npz

Run in **agitated_hugle** (it is the container that has Onnx4Deeploy on the path):
  docker exec agitated_hugle bash -lc 'cd /app && PYTHONPATH=/app/Onnx4Deeploy python3 \
    TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/build_block1_pw_fixture.py'
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import helper, numpy_helper

from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph

# --- names in the exp12 qinfer graph (verified by _inspect; see Findings.md) -------------------
SRC_ACT = "/blocks_1_conv_input_quant_1/Clip_output_0"  # int8 activation feeding block-1 Conv
SRC_CONV = "/wrappedInnerForwardImpl_1/Conv"  # block-1 Conv node
SRC_WEIGHT = "blocks.1.conv.weight_int8"  # [16, 8, 1, 16] int8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qinfer", default = "TrainDeeploy/DeeployTest/experiments/deliverable/"
                    "exp12_QZO_clean_round_1/qinfer")
    ap.add_argument("--out", default = "TrainDeeploy/DeeployTest/Tests/Models/NE16/speechnet_b1_pw_rq")
    ap.add_argument("--tap", type = int, default = 0, help = "which of the 16 taps to keep as the 1x1 kernel")
    ap.add_argument("--window", type = int, default = 0, help = "which eval window supplies the activation")
    args = ap.parse_args()

    src_path = os.path.join(args.qinfer, "network.onnx")
    model = onnx.load(src_path)
    g = gs.import_onnx(model)

    # ---- 1. real activation feeding block-1's Conv, from a real evaluation window -------------
    npz = np.load(os.path.join(args.qinfer, "inputs.npz"))
    window = npz["input"][args.window:args.window + 1].astype(np.float32)
    act = run_onnx_graph(src_path, {"input": window}, output_names = [SRC_ACT])
    act = act[0] if isinstance(act, (list, tuple)) else act
    act = np.asarray(act)
    print(f"[act ] {SRC_ACT}\n       shape={act.shape} dtype={act.dtype} "
          f"min={act.min()} max={act.max()}")
    assert act.min() >= 0, ("block-1 activations are expected to be >= 0 (fed by MaxPool<-Relu); "
                            f"got min={act.min()}. BLOCKER 3 would then apply here too.")

    # ---- 2. real weight, reduced 1x16 -> 1x1 by keeping one tap ------------------------------
    conv = next(n for n in g.nodes if n.name == SRC_CONV)
    w_full = np.asarray(next(t for t in conv.inputs if isinstance(t, gs.Constant)
                             and t.name == SRC_WEIGHT).values)
    w_pw = w_full[:, :, :, args.tap:args.tap + 1].copy()  # [16, 8, 1, 1]
    print(f"[w   ] {SRC_WEIGHT} {list(w_full.shape)} -> tap {args.tap} -> {list(w_pw.shape)} "
          f"dtype={w_pw.dtype} range=[{w_pw.min()},{w_pw.max()}]")

    # ---- 3. the RequantShift that consumes this Conv ------------------------------------------
    rqs = next(n for n in g.nodes if n.op == "RequantShift" and n.inputs[0] is conv.outputs[0])
    mul = np.asarray(rqs.inputs[1].values).astype(np.int32)
    add = np.asarray(rqs.inputs[2].values).astype(np.int32)
    # QW: copy the RAW AttributeProtos, do NOT round-trip through gs and re-make them. Deeploy's
    #     PULPConvRequantMergePass does `int(np.log2(rqs.attrs['div'].values))`, i.e. it requires
    #     `div` to be a TENSOR-valued attribute (the QZO exporter emits it via
    #     numpy_helper.from_array — see Onnx4Deeploy/.../qzo_weight_integerize.py). Rebuilding it
    #     as a plain int gives "AttributeError: 'int' object has no attribute 'values'" during
    #     lowering. Copying the protos preserves every attribute's original type exactly. -- QW
    src_rqs_proto = next(n for n in model.graph.node if n.name == rqs.name)
    rqs_attr_protos = list(src_rqs_proto.attribute)
    print(f"[rqs ] {rqs.name}\n       mul{list(mul.shape)} add{list(add.shape)} "
          f"attrs={[(a.name, onnx.AttributeProto.AttributeType.Name(a.type)) for a in rqs_attr_protos]}")

    # ---- 4. assemble the single-layer graph ---------------------------------------------------
    N, Cin, H, W = act.shape
    Cout = w_pw.shape[0]
    inp = helper.make_tensor_value_info("input", onnx.TensorProto.INT8, [N, Cin, H, W])
    out = helper.make_tensor_value_info("output", onnx.TensorProto.INT8, [N, Cout, H, W])

    init = [
        numpy_helper.from_array(w_pw.astype(np.int8), "b1_weight_int8"),
        numpy_helper.from_array(mul, "b1_rqs_mul"),
        numpy_helper.from_array(add, "b1_rqs_add"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "b1_weight_int8"], ["conv_out"], name = "b1_conv",
                         kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1],
                         dilations = [1, 1], group = 1),
        helper.make_node("RequantShift", ["conv_out", "b1_rqs_mul", "b1_rqs_add"], ["output"],
                         name = "b1_rqs", domain = src_rqs_proto.domain),
    ]
    nodes[1].attribute.extend(rqs_attr_protos)  # -- QW: verbatim attribute protos (see above)
    graph = helper.make_graph(nodes, "speechnet_b1_pw_rq", [inp], [out], initializer = init)
    m = helper.make_model(graph, opset_imports = list(model.opset_import))
    m.ir_version = model.ir_version

    os.makedirs(args.out, exist_ok = True)
    onnx_path = os.path.join(args.out, "network.onnx")
    onnx.save(m, onnx_path)

    # ---- 5. host reference, through the SAME executor the QZO pipeline uses -------------------
    ref = run_onnx_graph(onnx_path, {"input": act.astype(np.int8)}, output_names = ["output"])
    ref = np.asarray(ref[0] if isinstance(ref, (list, tuple)) else ref)
    print(f"[ref ] output shape={ref.shape} dtype={ref.dtype} range=[{ref.min()},{ref.max()}]")

    np.savez(os.path.join(args.out, "inputs.npz"), input = act.astype(np.int8))
    np.savez(os.path.join(args.out, "outputs.npz"), output = ref.astype(np.int8))
    print(f"\nwrote {args.out}/{{network.onnx,inputs.npz,outputs.npz}}")
    print(f"  conv: [1,{Cin},{H},{W}] x [{Cout},{Cin},1,1] -> [1,{Cout},{H},{W}]  "
          f"({Cout * Cin * H * W:,} MAC)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
