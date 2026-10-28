# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the QZO_DualConv fixture -- QW (dual_pass_fusion, exp5).
#
# The FUSED counterpart of exp3's QZO_Conv: the same block-0 quantized conv, but the two
# RQSPerturbRademacher nodes, the Conv and the RequantShift collapse into ONE node that takes the
# UNPERTURBED weight and requant-add plus the two perturb multipliers, generates both signs of the
# noise internally, and emits A+ and A- stacked on the channel axis:
#
#   act [1,1,14,700] i8, W [8,1,1,4] i8, mul_pc [8] i32, b [8] i32, wmul [8] i32, bmul [8] i32
#       --RequantizedDualConv(seed_w=42, idx_w=0, div_w=2^15, seed_b=42, idx_b=1, div_b=2^16, ...)-->
#   out [1,16,14,701] i8    channels 0..7 = A+ (theta+eps*z), 8..15 = A- (theta-eps*z)
#
# Reference: the ORIGINAL exp3 graph run through Onnx4Deeploy's `run_onnx_graph` -- the oracle that
# validated the QZO flow bit-exact on device -- twice: as-is for A+, and with the two perturb
# multipliers NEGATED for A-. That is how the host computes the L- pass everywhere in this project
# (negating M == negating the Rademacher sign, which is what the device's `perturbation_sign=0`
# does), so A- here is exactly the tensor the second RunTrainingNetwork() would produce.
# The oracle has no implementation of the new op; the fused kernel is validated AGAINST these two
# single-pass runs, not against itself.
#
# Run in agitated_hugle (needs onnx4deeploy):
#   docker exec agitated_hugle bash -lc 'cd /app && PYTHONPATH=/app/Onnx4Deeploy:/app/TrainDeeploy \
#     python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_dual_conv_fixture.py'

import argparse
import os
import tempfile

import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper as nh

from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph

HERE = os.path.dirname(os.path.abspath(__file__))
DEEPLOY_TEST = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_SRC = os.path.join(DEEPLOY_TEST, "Tests/Kernels/Integer/QZO_Conv")       # exp3's fixture
DEFAULT_DST = os.path.join(DEEPLOY_TEST, "Tests/Kernels/Integer/QZO_DualConv")

ACT = "/blocks_0_conv_input_quant_1/Clip_output_0"
W_RAW, B_RAW = "blocks.0.conv.weight_int8", "blocks.0.conv.bias_rqsadd"
WMUL, BMUL = "blocks.0.conv.weight_int8_pmul", "blocks.0.conv.bias_rqsadd_pmul"
MUL_PC = "QSTANDALONE_QCDQ_/blocks_0_conv_output_quant_1/Clip_Quant_mul_pc"
OUT_SINGLE = "/blocks_0_conv_output_quant_1/Clip_output_0"
OUT_DUAL = "/blocks_0_conv_dual_output_0"


def _attr(node, name):
    a = next(x for x in node.attribute if x.name == name)
    if a.type == onnx.AttributeProto.TENSOR:
        return int(nh.to_array(a.t).flat[0])
    if a.type == onnx.AttributeProto.INTS:
        return [int(v) for v in a.ints]
    return int(a.i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default = DEFAULT_SRC)
    ap.add_argument("--dst", default = DEFAULT_DST)
    args = ap.parse_args()

    src = os.path.join(args.src, "network.onnx")
    model = onnx.load(src)
    g = model.graph
    by_name = {n.name: n for n in g.node}
    inits = {t.name: t for t in g.initializer}
    vinfo = {v.name: v for v in g.value_info}
    npz = np.load(os.path.join(args.src, "inputs.npz"))
    feed = {k: npz[k] for k in npz.files}                       # keyed by graph-input name
    y_single = np.load(os.path.join(args.src, "outputs.npz"))[OUT_SINGLE]

    pw, pb = by_name["rqsp_blocks.0.conv.weight_int8"], by_name["rqsp_blocks.0.conv.bias_rqsadd"]
    conv, rqs = by_name["/wrappedInnerForwardImpl/Conv"], by_name["QSTANDALONE_QCDQ_/blocks_0_conv_output_quant_1/Clip_Quant"]

    # --- the fused node: every attribute is copied from the four nodes it replaces ----------------
    div = _attr(rqs, "div")
    node = helper.make_node(
        "RequantizedDualConv", [ACT, W_RAW, MUL_PC, B_RAW, WMUL, BMUL], [OUT_DUAL],
        name = "dualconv_blocks.0", domain = "mezo",
        kernel_shape = _attr(conv, "kernel_shape"), pads = _attr(conv, "pads"),
        strides = _attr(conv, "strides"), dilations = _attr(conv, "dilations"), group = _attr(conv, "group"),
        shift = int(np.log2(div)),
        seed_w = _attr(pw, "seed"), idx_w = _attr(pw, "idx"), div_w = _attr(pw, "div"),
        seed_b = _attr(pb, "seed"), idx_b = _attr(pb, "idx"), div_b = _attr(pb, "div"))

    # `div` / `n_levels` / `signed` must stay TENSOR-typed: Deeploy's RQSParserInterface unpacks them
    # with NodeParser._unpack_const, which rejects plain ints. Copy the RequantShift's protos verbatim.
    for a in rqs.attribute:
        if a.name in ("div", "n_levels", "signed"):
            node.attribute.append(a)

    graph_inputs = [next(i for i in g.input if i.name == n) for n in (ACT, W_RAW, B_RAW)]
    out_vi = helper.make_tensor_value_info(OUT_DUAL, onnx.TensorProto.INT8, [1, 16, 14, 701])
    graph = helper.make_graph([node], "QZO_DualConv", graph_inputs, [out_vi],
                              initializer = [inits[MUL_PC], inits[WMUL], inits[BMUL]])
    m = helper.make_model(graph, opset_imports = list(model.opset_import))
    m.ir_version = model.ir_version
    m.producer_name = "build_dual_conv_fixture.py"
    os.makedirs(args.dst, exist_ok = True)
    onnx.save(m, os.path.join(args.dst, "network.onnx"))

    # --- reference: exp3's graph, + pass as-is, - pass with NEGATED perturb multipliers -----------
    a_plus = np.asarray(run_onnx_graph(src, feed, output_names = [OUT_SINGLE])[0]).astype(np.int8)
    assert np.array_equal(a_plus, y_single), "oracle no longer reproduces exp3's recorded output"
    neg = onnx.load(src)
    for t in neg.graph.initializer:
        if t.name in (WMUL, BMUL):
            t.CopyFrom(nh.from_array(-nh.to_array(t), t.name))
    with tempfile.NamedTemporaryFile(suffix = ".onnx", delete = False) as tmp:
        onnx.save(neg, tmp.name)
        a_minus = np.asarray(run_onnx_graph(tmp.name, feed, output_names = [OUT_SINGLE])[0]).astype(np.int8)
    os.unlink(tmp.name)
    stacked = np.concatenate([a_plus, a_minus], axis = 1)          # [1,16,14,701]
    assert stacked.shape == (1, 16, 14, 701)

    np.savez(os.path.join(args.dst, "inputs.npz"), **{ACT: feed[ACT], W_RAW: feed[W_RAW], B_RAW: feed[B_RAW]})
    np.savez(os.path.join(args.dst, "outputs.npz"), **{OUT_DUAL: stacked})
    np.savez(os.path.join(args.dst, "reference_plus_minus.npz"), a_plus = a_plus, a_minus = a_minus)

    print(f"[QZO_DualConv] wrote {args.dst}")
    print(f"  node        : RequantizedDualConv  inputs={node.input}")
    print(f"  attrs       : {{'seed_w': {_attr(pw,'seed')}, 'idx_w': {_attr(pw,'idx')}, 'div_w': {_attr(pw,'div')}, "
          f"'seed_b': {_attr(pb,'seed')}, 'idx_b': {_attr(pb,'idx')}, 'div_b': {_attr(pb,'div')}, 'div': {div}}}")
    print(f"  A+ == exp3  : True   A+ range [{a_plus.min()},{a_plus.max()}]   A- range [{a_minus.min()},{a_minus.max()}]")
    print(f"  A+ vs A-    : {int((a_plus != a_minus).sum())}/{a_plus.size} elements differ")
    print(f"  output      : {OUT_DUAL} {stacked.shape} int8 (ch 0-7 = A+, 8-15 = A-)")


if __name__ == "__main__":
    main()
