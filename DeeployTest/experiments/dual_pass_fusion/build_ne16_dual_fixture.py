# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# QW (dual_pass_fusion exp6, task 2): block 0's FUSED dual-pass QZO conv on NE16, as a Deeploy fixture.
#
#   ACT   [1,1,14,700] i8 ──NE16Im2ColU8(1x4, pads 0/2/0/2, +128)──► col [1,14,701,4] u8   (once, both passes)
#   W_RAW [8,1,1,4] i8 + wmul ──NE16DualPerturbEncode(seed_w, idx_w, div_w)──► weight_enc [16,1,16] u8
#                                     rows 0..7 = W+ encoded, rows 8..15 = W- encoded (one 1x1 weight, Cout = 16)
#   W_RAW, wmul, B_RAW, bmul, mul ──NE16DualBias(seeds...)──► add2 [16] i32  = [add+' ; add-']  (signed-input corrected)
#   Conv(col, weight_enc; 1x1, Cout 16, ne16_weight_preencoded, weight_offset=-128, channels_first=0) ──► i32 [1,14,701,16]
#   RequantShift(·, mul2 = [mul; mul], add2; exp5's div/n_levels/signed attrs, channels_first=0) ──► i8  } merged on NE16
#   Transpose(0,3,1,2) ──► OUT [1,16,14,701] i8   == exp5's outputs.npz (A+ in channels 0..7, A- in 8..15), byte for byte
#
# Inputs (ACT, W_RAW, B_RAW), the three initializers (mul, wmul, bmul), the seed/idx/div attributes and the
# expected output are all COPIED from exp5's fixture (Tests/Kernels/Integer/QZO_DualConv), whose reference was
# the exp3 four-node graph run twice through the Onnx4Deeploy oracle. So "bit-exact" here means: identical to
# what the cluster's fused kernel produced, which was identical to two pulp-nn passes.
#
# Run (needs onnx + numpy only):
#   python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_ne16_dual_fixture.py

import argparse
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
DEEPLOY_TEST = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_SRC = os.path.join(DEEPLOY_TEST, "Tests/Models/Training/SpeechNet/speechnet_qzo19_train")
EXP5_FIXTURE = os.path.join(DEEPLOY_TEST, "Tests/Kernels/Integer/QZO_DualConv")
DEFAULT_DST = os.path.join(DEEPLOY_TEST, "Tests/Kernels/Integer/QZO_DualConv_NE16")

RQS = "QSTANDALONE_QCDQ_/blocks_0_conv_output_quant_1/Clip_Quant"  # exp3's RequantShift: domain + tensor attrs
WEIGHT_OFFSET, INPUT_OFFSET, NE16_CIN_SUBTILE = -128, 128, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default = DEFAULT_SRC)
    ap.add_argument("--exp5", default = EXP5_FIXTURE)
    ap.add_argument("--dst", default = DEFAULT_DST)
    args = ap.parse_args()

    src_model = onnx.load(os.path.join(args.src, "network.onnx"))
    rqs = next(n for n in src_model.graph.node if n.name == RQS)
    rqs_tensor_attrs = [a for a in rqs.attribute if a.name in ("div", "n_levels", "signed")]

    m5 = onnx.load(os.path.join(args.exp5, "network.onnx"))
    node5 = next(n for n in m5.graph.node if n.op_type == "RequantizedDualConv")
    ACT, W_RAW, MUL, B_RAW, WMUL, BMUL = list(node5.input)
    OUT = node5.output[0]
    a5 = {a.name: a for a in node5.attribute}
    seeds = {k: int(a5[k].i) for k in ("seed_w", "idx_w", "div_w", "seed_b", "idx_b", "div_b")}
    kh, kw = (int(v) for v in a5["kernel_shape"].ints)
    pads = [int(v) for v in a5["pads"].ints]
    strides = [int(v) for v in a5["strides"].ints]
    inits5 = {t.name: t for t in m5.graph.initializer}

    x_in = np.load(os.path.join(args.exp5, "inputs.npz"))
    y_ref = np.load(os.path.join(args.exp5, "outputs.npz"))[OUT].astype(np.int8)
    act, w_raw, b_raw = x_in[ACT].astype(np.int8), x_in[W_RAW].astype(np.int8), x_in[B_RAW].astype(np.int32)
    Nb, Cin, H, W = act.shape
    Cout = w_raw.shape[0]
    K = Cin * kh * kw
    Ho, Wo = H + pads[0] + pads[2] - kh + 1, W + pads[1] + pads[3] - kw + 1
    assert y_ref.shape == (Nb, 2 * Cout, Ho, Wo), y_ref.shape
    cinMajor = (K + NE16_CIN_SUBTILE - 1) // NE16_CIN_SUBTILE
    mul = numpy_helper.to_array(inits5[MUL]).astype(np.int32)
    mul2 = np.concatenate([mul, mul]).astype(np.int32)

    im2col = helper.make_node("NE16Im2ColU8", [ACT], ["col"], name = "b0_im2col",
                              kernel_shape = [kh, kw], pads = pads, strides = strides)
    im2col.attribute.append(helper.make_attribute("ne16_input_offset", INPUT_OFFSET))

    wenc = helper.make_node("NE16DualPerturbEncode", [W_RAW, WMUL], ["weight_enc"], name = "b0_dual_wenc")
    for k in ("seed_w", "idx_w", "div_w"):
        wenc.attribute.append(helper.make_attribute(k, seeds[k]))

    dbias = helper.make_node("NE16DualBias", [W_RAW, WMUL, B_RAW, BMUL, MUL], ["add2"], name = "b0_dual_bias")
    for k, v in seeds.items():
        dbias.attribute.append(helper.make_attribute(k, v))
    dbias.attribute.append(helper.make_attribute("ne16_input_offset", INPUT_OFFSET))

    conv_n = helper.make_node("Conv", ["col", "weight_enc"], ["conv_out"], name = "b0_ne16conv",
                              kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1], dilations = [1, 1],
                              group = 1)
    for k, v in (("ne16_weight_preencoded", 1), ("weight_offset", WEIGHT_OFFSET), ("channels_first", 0)):
        conv_n.attribute.append(helper.make_attribute(k, v))

    rqs_n = helper.make_node("RequantShift", ["conv_out", "mul2", "add2"], ["y_nhwc"], name = "b0_rqs",
                             domain = rqs.domain)
    rqs_n.attribute.extend(rqs_tensor_attrs)  # verbatim: div / n_levels / signed stay TENSOR-typed
    rqs_n.attribute.append(helper.make_attribute("channels_first", 0))

    tr = helper.make_node("Transpose", ["y_nhwc"], [OUT], name = "b0_out_nchw", perm = [0, 3, 1, 2])
    node_list = [wenc, dbias, im2col, conv_n, rqs_n, tr]

    graph_inputs = [
        helper.make_tensor_value_info(ACT, onnx.TensorProto.INT8, [Nb, Cin, H, W]),
        helper.make_tensor_value_info(W_RAW, onnx.TensorProto.INT8, list(w_raw.shape)),
        helper.make_tensor_value_info(B_RAW, onnx.TensorProto.INT32, list(b_raw.shape)),
    ]
    inits = [inits5[WMUL], inits5[BMUL], inits5[MUL], numpy_helper.from_array(mul2, "mul2")]
    vi = [
        helper.make_tensor_value_info("weight_enc", onnx.TensorProto.UINT8, [2 * Cout, cinMajor, 16]),
        helper.make_tensor_value_info("add2", onnx.TensorProto.INT32, [2 * Cout]),
        helper.make_tensor_value_info("col", onnx.TensorProto.UINT8, [Nb, Ho, Wo, K]),
        helper.make_tensor_value_info("conv_out", onnx.TensorProto.INT32, [Nb, Ho, Wo, 2 * Cout]),
        helper.make_tensor_value_info("y_nhwc", onnx.TensorProto.INT8, [Nb, Ho, Wo, 2 * Cout]),
    ]
    out_vi = helper.make_tensor_value_info(OUT, onnx.TensorProto.INT8, [Nb, 2 * Cout, Ho, Wo])
    graph = helper.make_graph(node_list, "QZO_DualConv_NE16", graph_inputs, [out_vi], initializer = inits,
                              value_info = vi)
    m = helper.make_model(graph, opset_imports = list(m5.opset_import))
    m.ir_version = m5.ir_version
    m.producer_name = "build_ne16_dual_fixture.py"
    os.makedirs(args.dst, exist_ok = True)
    onnx.save(m, os.path.join(args.dst, "network.onnx"))
    np.savez(os.path.join(args.dst, "inputs.npz"), **{ACT: act, W_RAW: w_raw, B_RAW: b_raw})
    np.savez(os.path.join(args.dst, "outputs.npz"), **{OUT: y_ref})
    print(f"[QZO_DualConv_NE16] wrote {args.dst}")
    print(f"  nodes : {[f'{n.op_type}:{n.name}' for n in node_list]}")
    print(f"  seeds : {seeds}")
    print(f"  NE16 1x1: Cin*K = {K} lanes of 16, Cout = {2*Cout} of 32 (W+ rows then W- rows); ref {OUT} {y_ref.shape}")


if __name__ == "__main__":
    main()
