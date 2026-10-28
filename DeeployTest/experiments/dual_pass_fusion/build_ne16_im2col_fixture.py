# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# QW (dual_pass_fusion exp6, task 1): block 0's QZO conv in the GAP9 SDK's NE16 form -- im2col on the
# cluster + ONE NE16 1x1 job over the folded `Cin*K` channels -- as a Deeploy fixture.
#
#   ACT   [1,1,14,700] i8 ──NE16Im2ColU8(1x4, pads 0/2/0/2, +128)──► col [1,14,701,4] u8   (NHWC-native)
#   W_RAW [8,1,1,4]  i8 ──RQSPerturbRademacher(verbatim)──► weight_pert [8,1,1,4]
#                          ├─ NE16WeightEncode(ne16_taps=1, ne16_fold=1) ──► weight_enc [8,1,16] u8
#                          └─ NE16SignedInputBias(weight_pert, mul, bias_pert) ──► add_corr [8] i32
#   B_RAW [8]        i32 ──RQSPerturbRademacher(verbatim)──► bias_pert [8] i32
#   Conv(col, weight_enc; 1x1, pads 0, ne16_weight_preencoded, weight_offset=-128, channels_first=0) ──► i32
#   RequantShift(·, mul, add_corr; verbatim attrs, channels_first=0)  ──► i8 [1,14,701,8]   } merged on NE16
#   Transpose(0,3,1,2) ──► OUT [1,8,14,701] i8   == exp3's outputs.npz, byte for byte
#
# Inputs and the expected output are COPIED from exp3's fixture (Tests/Kernels/Integer/QZO_Conv), so
# the device result is compared against exactly the tensor pulp-nn produced on Siracusa in exp3 and
# the flat kernel reproduced in exp5a. No host oracle is needed (and none exists for the NE16 ops).
#
# The perturb nodes and the RequantShift attributes are copied VERBATIM from the source QZO graph
# (seed/idx/div protos, `div`/`n_levels`/`signed` TENSOR attrs), as exp3's builder does, so the
# device draws the same noise stream. weight_pert keeps its verbatim shape [8,1,1,4] (the perturb
# tile constraint pairs input/output dims by index, so the two must match); the ENCODER folds the
# taps into the input channels (`ne16_fold=1`): same bytes, read as a 1x1 weight with cin = 4.
#
# Run (needs onnx + numpy only):
#   python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_ne16_im2col_fixture.py [--block i]

import argparse
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
DEEPLOY_TEST = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_SRC = os.path.join(DEEPLOY_TEST, "Tests/Models/Training/SpeechNet/speechnet_qzo19_train")
EXP3_FIXTURE = os.path.join(DEEPLOY_TEST, "Tests/Kernels/Integer/QZO_Conv")
DEFAULT_DST = os.path.join(DEEPLOY_TEST, "Tests/Kernels/Integer/QZO_Conv_NE16")

# QW (exp7 step 7): any block (--block i). Block i's tensors and expected output come from exp7's unfused
#     fixture QZO_Conv_B<i> instead of exp3's; --block 0 writes exactly the exp6 fixture.
def block_names(b):
    conv = "/wrappedInnerForwardImpl/Conv" if b == 0 else f"/wrappedInnerForwardImpl_{b}/Conv"
    return (f"/blocks_{b}_conv_input_quant_1/Clip_output_0", f"blocks.{b}.conv.weight_int8", f"blocks.{b}.conv.bias_rqsadd",
            f"rqsp_blocks.{b}.conv.weight_int8", f"rqsp_blocks.{b}.conv.bias_rqsadd", conv,
            f"QSTANDALONE_QCDQ_/blocks_{b}_conv_output_quant_1/Clip_Quant", f"/blocks_{b}_conv_output_quant_1/Clip_output_0")

WEIGHT_OFFSET = -128
INPUT_OFFSET = 128
NE16_CIN_SUBTILE = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default = DEFAULT_SRC)
    ap.add_argument("--exp3", default = None, help = "unfused fixture: QZO_Conv (block 0) / QZO_Conv_B<i>")
    ap.add_argument("--dst", default = None, help = "QZO_Conv_NE16 (block 0) / QZO_Conv_NE16_B<i>")
    ap.add_argument("--block", type = int, default = 0, choices = range(5))
    args = ap.parse_args()
    b = args.block
    args.exp3 = args.exp3 or (EXP3_FIXTURE if b == 0 else f"{EXP3_FIXTURE}_B{b}")
    args.dst = args.dst or (DEFAULT_DST if b == 0 else f"{DEFAULT_DST}_B{b}")
    ACT, W_RAW, B_RAW, PERT_W, PERT_B, CONV, RQS, OUT = block_names(b)
    pfx = f"b{b}"

    model = onnx.load(os.path.join(args.src, "network.onnx"))
    g = model.graph
    nodes = {n.name: n for n in g.node}
    inits = {t.name: t for t in g.initializer}
    conv, rqs = nodes[CONV], nodes[RQS]
    pert_w, pert_b = nodes[PERT_W], nodes[PERT_B]

    kh, kw = (int(v) for v in next(a for a in conv.attribute if a.name == "kernel_shape").ints)
    pads = [int(v) for v in next(a for a in conv.attribute if a.name == "pads").ints]
    strides = [int(v) for v in next(a for a in conv.attribute if a.name == "strides").ints]
    assert strides == [1, 1], strides
    if b == 0:
        assert (kh, kw) == (1, 4) and pads == [0, 2, 0, 2], (kh, kw, pads)

    # exp3's tensors: same inputs, same expected output
    x_in = np.load(os.path.join(args.exp3, "inputs.npz"))
    y_ref = np.load(os.path.join(args.exp3, "outputs.npz"))[OUT].astype(np.int8)
    act, w_raw, b_raw = x_in[ACT].astype(np.int8), x_in[W_RAW].astype(np.int8), x_in[B_RAW].astype(np.int32)
    Nb, Cin, H, W = act.shape
    Cout = w_raw.shape[0]
    K = Cin * kh * kw
    Ho, Wo = H + pads[0] + pads[2] - kh + 1, W + pads[1] + pads[3] - kw + 1
    assert y_ref.shape == (Nb, Cout, Ho, Wo), y_ref.shape
    mul_name = [i for i in rqs.input if i in inits][0]
    cinMajor = (K + NE16_CIN_SUBTILE - 1) // NE16_CIN_SUBTILE

    wp_name, bp_name = pert_w.output[0], pert_b.output[0]

    # ---- nodes ---------------------------------------------------------------------------------
    im2col = helper.make_node("NE16Im2ColU8", [ACT], ["col"], name = f"{pfx}_im2col",
                              kernel_shape = [kh, kw], pads = pads, strides = strides)
    im2col.attribute.append(helper.make_attribute("ne16_input_offset", INPUT_OFFSET))

    enc = helper.make_node("NE16WeightEncode", [wp_name], ["weight_enc"], name = f"{pfx}_wenc")
    enc.attribute.append(helper.make_attribute("ne16_taps", 1))
    enc.attribute.append(helper.make_attribute("ne16_bits", 8))
    enc.attribute.append(helper.make_attribute("ne16_fold", 1))  # taps folded into cin: [Co,Ci,kh,kw] -> 1x1 over Ci*kh*kw channels

    sbias = helper.make_node("NE16SignedInputBias", [wp_name, mul_name, bp_name], ["add_corr"], name = f"{pfx}_sbias")
    sbias.attribute.append(helper.make_attribute("ne16_input_offset", INPUT_OFFSET))

    conv_n = helper.make_node("Conv", ["col", "weight_enc"], ["conv_out"], name = f"{pfx}_ne16conv",
                              kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1], dilations = [1, 1],
                              group = 1)
    for k, v in (("ne16_weight_preencoded", 1), ("weight_offset", WEIGHT_OFFSET), ("channels_first", 0)):
        conv_n.attribute.append(helper.make_attribute(k, v))

    rqs_n = helper.make_node("RequantShift", ["conv_out", mul_name, "add_corr"], ["y_nhwc"], name = f"{pfx}_rqs",
                             domain = rqs.domain)
    rqs_n.attribute.extend(list(rqs.attribute))  # verbatim: div / n_levels / signed stay TENSOR-typed
    rqs_n.attribute.append(helper.make_attribute("channels_first", 0))

    tr = helper.make_node("Transpose", ["y_nhwc"], [OUT], name = f"{pfx}_out_nchw", perm = [0, 3, 1, 2])

    node_list = [pert_w, pert_b, sbias, enc, im2col, conv_n, rqs_n, tr]

    # ---- interface ---------------------------------------------------------------------------
    graph_inputs = [
        helper.make_tensor_value_info(ACT, onnx.TensorProto.INT8, [Nb, Cin, H, W]),
        helper.make_tensor_value_info(W_RAW, onnx.TensorProto.INT8, list(w_raw.shape)),
        helper.make_tensor_value_info(B_RAW, onnx.TensorProto.INT32, list(b_raw.shape)),
    ]
    needed_inits = sorted({i for n in node_list for i in n.input if i in inits})
    vi = [
        helper.make_tensor_value_info(wp_name, onnx.TensorProto.INT8, list(w_raw.shape)),  # verbatim shape; the encoder folds
        helper.make_tensor_value_info(bp_name, onnx.TensorProto.INT32, [Cout]),
        helper.make_tensor_value_info("add_corr", onnx.TensorProto.INT32, [Cout]),
        helper.make_tensor_value_info("weight_enc", onnx.TensorProto.UINT8, [Cout, cinMajor, 16]),
        helper.make_tensor_value_info("col", onnx.TensorProto.UINT8, [Nb, Ho, Wo, K]),
        helper.make_tensor_value_info("conv_out", onnx.TensorProto.INT32, [Nb, Ho, Wo, Cout]),
        helper.make_tensor_value_info("y_nhwc", onnx.TensorProto.INT8, [Nb, Ho, Wo, Cout]),
    ]
    out_vi = helper.make_tensor_value_info(OUT, onnx.TensorProto.INT8, [Nb, Cout, Ho, Wo])
    graph = helper.make_graph(node_list, "QZO_Conv_NE16" if b == 0 else f"QZO_Conv_NE16_B{b}", graph_inputs, [out_vi],
                              initializer = [inits[i] for i in needed_inits], value_info = vi)
    m = helper.make_model(graph, opset_imports = list(model.opset_import))
    m.ir_version = model.ir_version
    m.producer_name = "build_ne16_im2col_fixture.py"
    os.makedirs(args.dst, exist_ok = True)
    onnx.save(m, os.path.join(args.dst, "network.onnx"))  # no shape inference: every shape is explicit
    np.savez(os.path.join(args.dst, "inputs.npz"), **{ACT: act, W_RAW: w_raw, B_RAW: b_raw})
    np.savez(os.path.join(args.dst, "outputs.npz"), **{OUT: y_ref})

    # host-side sanity of the im2col contract (what the cluster kernel must produce)
    col = np.full((Ho, Wo, K), INPUT_OFFSET, dtype = np.uint8)
    xp = np.pad(act[0], ((0, 0), (pads[0], pads[2]), (pads[1], pads[3])), constant_values = 0)
    for ci in range(Cin):
        for ky in range(kh):
            for kx in range(kw):
                col[:, :, (ci * kh + ky) * kw + kx] = (xp[ci, ky:ky + Ho, kx:kx + Wo].astype(np.int32) +
                                                     INPUT_OFFSET).astype(np.uint8)
    np.savez(os.path.join(args.dst, "col_golden.npz"), col = col[None])

    print(f"[QZO_Conv_NE16] wrote {args.dst}")
    print(f"  nodes : {[f'{n.op_type}:{n.name}' for n in node_list]}")
    print(f"  inits : {needed_inits}")
    print(f"  col   : [{Nb},{Ho},{Wo},{K}] u8 (= {Nb*Ho*Wo*K:,} B); weight_enc [{Cout},{cinMajor},16]; "
          f"NE16 1x1 over Cin*K = {K} lanes of 16, Cout = {Cout} of 32")
    print(f"  ref   : exp3's {OUT} {y_ref.shape} (exact compare)")


if __name__ == "__main__":
    main()
