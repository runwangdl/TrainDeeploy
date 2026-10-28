# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# QW (dual_pass_fusion exp7, step 7): the FUSED dual-pass conv of blocks 1-4 on NE16 (GAP9), i.e. exp6's task 2
# for the case x+ != x-. Block 0 fused both passes into ONE NE16 job (Cout = 2*Co over the shared column); here the
# two passes have different input columns, so they are two NE16 jobs. What the passes share is the preparation:
# W+ / W- and add+ / add- come from the same Rademacher draw (exp5/exp6 seeds), computed by the exp6 prep nodes
# with ne16_sign_sel = 1 (+) / 2 (-).
#
#   x+ ──NE16Im2ColU8──► col+ ──┐
#   W, wmul ──NE16DualPerturbEncode(sel 1)──► wenc+ ──┤
#   W, wmul, b, bmul, mul ──NE16DualBias(sel 1)──► add+' ┴─► Conv(1x1, NE16) + RequantShift ──► Transpose ──► A+ [1,Co,Ho,Wo]
#   x- ──NE16Im2ColU8──► col- ──┐
#   ... (sel 2) ...                                      ┴─► Conv(1x1, NE16) + RequantShift ──► Transpose ──► A- [1,Co,Ho,Wo]
#
# Two graph outputs (A+, A-) instead of block 0's channel-stacked one: the two NE16 jobs write separate tensors.
# Inputs and the expected A+ / A- are COPIED from exp7's cluster fixture QZO_DualConv2in_B<i> (build_dual_conv2in_
# fixture.py), whose reference is the unfused single-block graph through the Onnx4Deeploy oracle.
#
# Run (needs onnx + numpy only):
#   python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_ne16_dual2in_fixture.py --block 1

import argparse
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
DEEPLOY_TEST = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_SRC = os.path.join(DEEPLOY_TEST, "Tests/Models/Training/SpeechNet/speechnet_qzo19_train")
WEIGHT_OFFSET, INPUT_OFFSET, NE16_CIN_SUBTILE = -128, 128, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type = int, default = 1, choices = range(1, 5))
    ap.add_argument("--src", default = DEFAULT_SRC, help = "full QZO graph (RequantShift attribute protos)")
    ap.add_argument("--cluster", default = None, help = "default Tests/Kernels/Integer/QZO_DualConv2in_B<i>")
    ap.add_argument("--dst", default = None,
                    help = "default Tests/Kernels/Integer/QZO_DualConv2in_NE16_B<i> (or _NE16P_B<i> / _NE16S_B<i>)")
    ap.add_argument("--single-node", type = int, choices = (0, 1), default = None,
                    help = "one NE16DualConv2in node (word-wise HWC column, SDK pipeline); value = ne16_pipeline "
                           "(1 double buffered -> _NE16P_, 0 sequential -> _NE16S_)")
    args = ap.parse_args()
    b = args.block
    args.cluster = args.cluster or os.path.join(DEEPLOY_TEST, f"Tests/Kernels/Integer/QZO_DualConv2in_B{b}")
    tag = "NE16" if args.single_node is None else ("NE16P" if args.single_node else "NE16S")
    args.dst = args.dst or os.path.join(DEEPLOY_TEST, f"Tests/Kernels/Integer/QZO_DualConv2in_{tag}_B{b}")

    RQS = f"QSTANDALONE_QCDQ_/blocks_{b}_conv_output_quant_1/Clip_Quant"
    src_model = onnx.load(os.path.join(args.src, "network.onnx"))
    rqs = next(n for n in src_model.graph.node if n.name == RQS)
    rqs_tensor_attrs = [a for a in rqs.attribute if a.name in ("div", "n_levels", "signed")]

    mc = onnx.load(os.path.join(args.cluster, "network.onnx"))
    nodec = next(n for n in mc.graph.node if n.op_type == "RequantizedDualConv")
    ACT, W_RAW, MUL, B_RAW, WMUL, BMUL, ACT_M = list(nodec.input)
    ac = {a.name: a for a in nodec.attribute}
    seeds = {k: int(ac[k].i) for k in ("seed_w", "idx_w", "div_w", "seed_b", "idx_b", "div_b")}
    kh, kw = (int(v) for v in ac["kernel_shape"].ints)
    pads = [int(v) for v in ac["pads"].ints]
    strides = [int(v) for v in ac["strides"].ints]
    assert strides == [1, 1], strides
    initsc = {t.name: t for t in mc.graph.initializer}

    x_in = np.load(os.path.join(args.cluster, "inputs.npz"))
    ref = np.load(os.path.join(args.cluster, "reference_plus_minus.npz"))
    a_plus, a_minus = ref["a_plus"].astype(np.int8), ref["a_minus"].astype(np.int8)
    act, act_m = x_in[ACT].astype(np.int8), x_in[ACT_M].astype(np.int8)
    w_raw, b_raw = x_in[W_RAW].astype(np.int8), x_in[B_RAW].astype(np.int32)
    Nb, Cin, H, W = act.shape
    Cout = w_raw.shape[0]
    K = Cin * kh * kw
    Ho, Wo = H + pads[0] + pads[2] - kh + 1, W + pads[1] + pads[3] - kw + 1
    assert a_plus.shape == (Nb, Cout, Ho, Wo) == a_minus.shape, a_plus.shape
    cinMajor = (K + NE16_CIN_SUBTILE - 1) // NE16_CIN_SUBTILE
    OUT_P, OUT_M = f"/blocks_{b}_conv_ne16_A_plus", f"/blocks_{b}_conv_ne16_A_minus"

    if args.single_node is not None:
        return single_node(args, b, rqs_tensor_attrs, mc, initsc, seeds, (kh, kw), pads, strides, ACT, W_RAW, MUL,
                           B_RAW, WMUL, BMUL, ACT_M, act, act_m, w_raw, b_raw, a_plus, a_minus, Nb, Cin, H, W, Cout, K,
                           Ho, Wo, cinMajor, nodec)

    nodes, vi = [], []
    for sel, sgn, x_name, out_name in ((1, "p", ACT, OUT_P), (2, "m", ACT_M, OUT_M)):
        pfx = f"b{b}{sgn}"
        im2col = helper.make_node("NE16Im2ColU8", [x_name], [f"col_{sgn}"], name = f"{pfx}_im2col",
                                  kernel_shape = [kh, kw], pads = pads, strides = strides)
        im2col.attribute.append(helper.make_attribute("ne16_input_offset", INPUT_OFFSET))
        wenc = helper.make_node("NE16DualPerturbEncode", [W_RAW, WMUL], [f"wenc_{sgn}"], name = f"{pfx}_dual_wenc")
        for k in ("seed_w", "idx_w", "div_w"):
            wenc.attribute.append(helper.make_attribute(k, seeds[k]))
        wenc.attribute.append(helper.make_attribute("ne16_sign_sel", sel))
        dbias = helper.make_node("NE16DualBias", [W_RAW, WMUL, B_RAW, BMUL, MUL], [f"add_{sgn}"],
                                 name = f"{pfx}_dual_bias")
        for k, v in seeds.items():
            dbias.attribute.append(helper.make_attribute(k, v))
        dbias.attribute.append(helper.make_attribute("ne16_input_offset", INPUT_OFFSET))
        dbias.attribute.append(helper.make_attribute("ne16_sign_sel", sel))
        conv_n = helper.make_node("Conv", [f"col_{sgn}", f"wenc_{sgn}"], [f"conv_out_{sgn}"], name = f"{pfx}_ne16conv",
                                  kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1], dilations = [1, 1],
                                  group = 1)
        for k, v in (("ne16_weight_preencoded", 1), ("weight_offset", WEIGHT_OFFSET), ("channels_first", 0)):
            conv_n.attribute.append(helper.make_attribute(k, v))
        rqs_n = helper.make_node("RequantShift", [f"conv_out_{sgn}", MUL, f"add_{sgn}"], [f"y_nhwc_{sgn}"],
                                 name = f"{pfx}_rqs", domain = rqs.domain)
        rqs_n.attribute.extend(rqs_tensor_attrs)  # verbatim: div / n_levels / signed stay TENSOR-typed
        rqs_n.attribute.append(helper.make_attribute("channels_first", 0))
        tr = helper.make_node("Transpose", [f"y_nhwc_{sgn}"], [out_name], name = f"{pfx}_out_nchw", perm = [0, 3, 1, 2])
        nodes += [wenc, dbias, im2col, conv_n, rqs_n, tr]
        vi += [
            helper.make_tensor_value_info(f"wenc_{sgn}", onnx.TensorProto.UINT8, [Cout, cinMajor, 16]),
            helper.make_tensor_value_info(f"add_{sgn}", onnx.TensorProto.INT32, [Cout]),
            helper.make_tensor_value_info(f"col_{sgn}", onnx.TensorProto.UINT8, [Nb, Ho, Wo, K]),
            helper.make_tensor_value_info(f"conv_out_{sgn}", onnx.TensorProto.INT32, [Nb, Ho, Wo, Cout]),
            helper.make_tensor_value_info(f"y_nhwc_{sgn}", onnx.TensorProto.INT8, [Nb, Ho, Wo, Cout]),
        ]

    graph_inputs = [
        helper.make_tensor_value_info(ACT, onnx.TensorProto.INT8, [Nb, Cin, H, W]),
        helper.make_tensor_value_info(W_RAW, onnx.TensorProto.INT8, list(w_raw.shape)),
        helper.make_tensor_value_info(B_RAW, onnx.TensorProto.INT32, list(b_raw.shape)),
        helper.make_tensor_value_info(ACT_M, onnx.TensorProto.INT8, [Nb, Cin, H, W]),
    ]
    outs = [helper.make_tensor_value_info(n, onnx.TensorProto.INT8, [Nb, Cout, Ho, Wo]) for n in (OUT_P, OUT_M)]
    graph = helper.make_graph(nodes, f"QZO_DualConv2in_NE16_B{b}", graph_inputs, outs,
                              initializer = [initsc[WMUL], initsc[BMUL], initsc[MUL]], value_info = vi)
    m = helper.make_model(graph, opset_imports = list(mc.opset_import))
    m.ir_version = mc.ir_version
    m.producer_name = "build_ne16_dual2in_fixture.py"
    os.makedirs(args.dst, exist_ok = True)
    onnx.save(m, os.path.join(args.dst, "network.onnx"))
    # generateNetwork.py / testMVP.py consume inputs.npz POSITIONALLY -> order == graph inputs.
    np.savez(os.path.join(args.dst, "inputs.npz"), **{ACT: act, W_RAW: w_raw, B_RAW: b_raw, ACT_M: act_m})
    np.savez(os.path.join(args.dst, "outputs.npz"), **{OUT_P: a_plus, OUT_M: a_minus})
    print(f"[QZO_DualConv2in_NE16 block {b}] wrote {args.dst}")
    print(f"  nodes : {[f'{n.op_type}:{n.name}' for n in nodes]}")
    print(f"  seeds : {seeds}")
    print(f"  NE16 1x1 per pass: Cin*K = {K} ({cinMajor} x 16 lanes), Cout = {Cout} of 32; columns 2 x "
          f"[{Nb},{Ho},{Wo},{K}] u8 = 2 x {Nb*Ho*Wo*K:,} B; refs A+ / A- {a_plus.shape} from {os.path.basename(args.cluster)}")


def single_node(args, b, rqs_tensor_attrs, mc, initsc, seeds, ks, pads, strides, ACT, W_RAW, MUL, B_RAW, WMUL, BMUL,
                ACT_M, act, act_m, w_raw, b_raw, a_plus, a_minus, Nb, Cin, H, W, Cout, K, Ho, Wo, cinMajor, nodec):
    """QW (exp7 step 7): the pipelined form -- x+/x- transposed to HWC, ONE stacked prep (one draw, rows in (ky,kx,ci)
    order), ONE NE16DualConv2in node, stacked output [A+ | A-] (same tensor as the cluster fixture's outputs.npz).

        x+ ─Transpose─► x+_hwc ─┐
        x- ─Transpose─► x-_hwc ─┤
        W, wmul ──NE16DualPerturbEncode(sel 0, hwc 1)──► wenc2 [2Co, cinMajor, 16] ─┤
        W, wmul, b, bmul, mul ──NE16DualBias(sel 0)──► add2 [2Co] ──────────────────┴─► NE16DualConv2in ─► y [1,Ho,Wo,2Co]
                                                                                       ─Transpose─► OUT [1,2Co,Ho,Wo]
    """
    OUT = nodec.output[0]
    div = next(int(numpy_helper.to_array(a.t).flat[0]) for a in rqs_tensor_attrs if a.name == "div")
    tp = helper.make_node("Transpose", [ACT], ["xp_hwc"], name = f"b{b}_xp_hwc", perm = [0, 2, 3, 1])
    tm = helper.make_node("Transpose", [ACT_M], ["xm_hwc"], name = f"b{b}_xm_hwc", perm = [0, 2, 3, 1])
    wenc = helper.make_node("NE16DualPerturbEncode", [W_RAW, WMUL], ["wenc2"], name = f"b{b}_dual_wenc")
    for k in ("seed_w", "idx_w", "div_w"):
        wenc.attribute.append(helper.make_attribute(k, seeds[k]))
    wenc.attribute.append(helper.make_attribute("ne16_sign_sel", 0))
    wenc.attribute.append(helper.make_attribute("ne16_hwc", 1))
    dbias = helper.make_node("NE16DualBias", [W_RAW, WMUL, B_RAW, BMUL, MUL], ["add2"], name = f"b{b}_dual_bias")
    for k, v in seeds.items():
        dbias.attribute.append(helper.make_attribute(k, v))
    dbias.attribute.append(helper.make_attribute("ne16_input_offset", INPUT_OFFSET))
    dbias.attribute.append(helper.make_attribute("ne16_sign_sel", 0))
    conv = helper.make_node("NE16DualConv2in", ["xp_hwc", "xm_hwc", "wenc2", MUL, "add2"], ["y_hwc"],
                            name = f"b{b}_ne16_dualconv", kernel_shape = list(ks), pads = pads, strides = strides,
                            shift = int(np.log2(div)), weight_offset = WEIGHT_OFFSET,
                            ne16_pipeline = int(args.single_node))
    tr = helper.make_node("Transpose", ["y_hwc"], [OUT], name = f"b{b}_out_nchw", perm = [0, 3, 1, 2])
    nodes = [tp, tm, wenc, dbias, conv, tr]
    vi = [
        helper.make_tensor_value_info("xp_hwc", onnx.TensorProto.INT8, [Nb, H, W, Cin]),
        helper.make_tensor_value_info("xm_hwc", onnx.TensorProto.INT8, [Nb, H, W, Cin]),
        helper.make_tensor_value_info("wenc2", onnx.TensorProto.UINT8, [2 * Cout, cinMajor, 16]),
        helper.make_tensor_value_info("add2", onnx.TensorProto.INT32, [2 * Cout]),
        helper.make_tensor_value_info("y_hwc", onnx.TensorProto.INT8, [Nb, Ho, Wo, 2 * Cout]),
    ]
    graph_inputs = [
        helper.make_tensor_value_info(ACT, onnx.TensorProto.INT8, [Nb, Cin, H, W]),
        helper.make_tensor_value_info(W_RAW, onnx.TensorProto.INT8, list(w_raw.shape)),
        helper.make_tensor_value_info(B_RAW, onnx.TensorProto.INT32, list(b_raw.shape)),
        helper.make_tensor_value_info(ACT_M, onnx.TensorProto.INT8, [Nb, Cin, H, W]),
    ]
    stacked = np.concatenate([a_plus, a_minus], axis = 1)
    outs = [helper.make_tensor_value_info(OUT, onnx.TensorProto.INT8, list(stacked.shape))]
    tag = "NE16P" if args.single_node else "NE16S"
    graph = helper.make_graph(nodes, f"QZO_DualConv2in_{tag}_B{b}", graph_inputs, outs,
                              initializer = [initsc[WMUL], initsc[BMUL], initsc[MUL]], value_info = vi)
    m = helper.make_model(graph, opset_imports = list(mc.opset_import))
    m.ir_version = mc.ir_version
    m.producer_name = "build_ne16_dual2in_fixture.py --single-node"
    os.makedirs(args.dst, exist_ok = True)
    onnx.save(m, os.path.join(args.dst, "network.onnx"))
    np.savez(os.path.join(args.dst, "inputs.npz"), **{ACT: act, W_RAW: w_raw, B_RAW: b_raw, ACT_M: act_m})
    np.savez(os.path.join(args.dst, "outputs.npz"), **{OUT: stacked})
    print(f"[QZO_DualConv2in_{tag} block {b}] wrote {args.dst}")
    print(f"  nodes : {[f'{n.op_type}:{n.name}' for n in nodes]}")
    job_w = max(1, min(11264 // K, Ho * Wo) // 3)  # as NE16DualConv2inParser
    print(f"  NE16 per job: 3 x {job_w} output px, K = {K} ({cinMajor} x 16 lanes), Cout = {Cout}; {Ho * Wo} px per pass; "
          f"column buffers 2 x {3 * job_w * K:,} B; ne16_pipeline = {args.single_node}; ref {OUT} {stacked.shape}")


if __name__ == "__main__":
    main()
