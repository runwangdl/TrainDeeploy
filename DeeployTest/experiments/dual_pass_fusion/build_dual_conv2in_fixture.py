# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the fused two-input fixture QZO_DualConv2in_B<i> -- QW (dual_pass_fusion, exp7 step 4).
#
# The FUSED counterpart of exp7's unfused QZO_Conv_B<i> (build_qzo_conv_fixture.py --block i) for blocks 1-4,
# where the + and - passes have DIFFERENT input activations:
#
#   x+ [1,Ci,H,W] i8, W [Co,Ci,Kh,Kw] i8 (raw CHW, unperturbed), mul_pc [Co] i32, b [Co] i32, wmul [Co] i32,
#   bmul [Co] i32, x- [1,Ci,H,W] i8
#       --RequantizedDualConv(weight_chw=1, seed/idx/div of both perturb nodes, ...)-->
#   out [1, 2*Co, Ho, Wo] i8    channels 0..Co-1 = A+ (theta+eps*z), Co..2Co-1 = A- (theta-eps*z)
#
# Inputs: x+ / x- are block i's real input tensors of the + and - forward passes, both computed by Onnx4Deeploy's
# oracle (`run_onnx_graph`) on the full QZO training graph -- the - pass with every perturbation negated (the
# RQSPerturbRademacher multipliers *_pmul and the PerturbRademacher `eps`), which is how the host computes L-.
# With --same-input, x- := x+ (debug variant: isolates the weight order from the two-input path).
#
# Reference: the UNFUSED single-block graph QZO_Conv_B<i> (perturb nodes + Conv + RequantShift) through the same
# oracle, on x+ as-is for A+, and on x- with the two perturb multipliers NEGATED for A- (exp5's convention).
# A+ is also checked against the full graph's block-i output tensor.
#
# Run in agitated_hugle (needs onnx4deeploy); build QZO_Conv_B<i> first:
#   docker exec agitated_hugle bash -lc 'cd /app && PYTHONPATH=/app/Onnx4Deeploy:/app/TrainDeeploy \
#     python3 TrainDeeploy/DeeployTest/experiments/dual_pass_fusion/build_dual_conv2in_fixture.py --block 1'

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
DEFAULT_FULL = os.path.join(DEEPLOY_TEST, "Tests/Models/Training/SpeechNet/speechnet_qzo19_train")


def _attr(node, name):
    a = next(x for x in node.attribute if x.name == name)
    if a.type == onnx.AttributeProto.TENSOR:
        return int(nh.to_array(a.t).flat[0])
    if a.type == onnx.AttributeProto.INTS:
        return [int(v) for v in a.ints]
    return int(a.i)


def _negated(model, pmul_names = None):
    """Copy of `model` computing the - pass: perturb multipliers negated (all *_pmul, or only `pmul_names`)
    and every PerturbRademacher eps negated."""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    for t in m.graph.initializer:
        if (pmul_names is None and t.name.endswith("_pmul")) or (pmul_names is not None and t.name in pmul_names):
            t.CopyFrom(nh.from_array(-nh.to_array(t), t.name))
    for n in m.graph.node:
        if n.op_type == "PerturbRademacher":
            for a in n.attribute:
                if a.name == "eps":
                    a.f = -a.f
    return m


def _run(model, feed, outs):
    with tempfile.NamedTemporaryFile(suffix = ".onnx", delete = False) as tmp:
        onnx.save(model, tmp.name)
        try:
            return [np.asarray(v) for v in run_onnx_graph(tmp.name, feed, output_names = outs)]
        finally:
            os.unlink(tmp.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type = int, default = 1, choices = range(1, 5))
    ap.add_argument("--full", default = DEFAULT_FULL, help = "full QZO training graph (x+ / x- source)")
    ap.add_argument("--src", default = None, help = "unfused fixture, default Tests/Kernels/Integer/QZO_Conv_B<i>")
    ap.add_argument("--dst", default = None, help = "default Tests/Kernels/Integer/QZO_DualConv2in_B<i>[_same]")
    ap.add_argument("--same-input", action = "store_true", help = "x- := x+ (debug variant)")
    args = ap.parse_args()
    i = args.block
    args.src = args.src or os.path.join(DEEPLOY_TEST, f"Tests/Kernels/Integer/QZO_Conv_B{i}")
    args.dst = args.dst or os.path.join(DEEPLOY_TEST,
                                        f"Tests/Kernels/Integer/QZO_DualConv2in_B{i}" + ("_same" if args.same_input else ""))

    ACT = f"/blocks_{i}_conv_input_quant_1/Clip_output_0"
    ACT_M = f"/blocks_{i}_conv_input_quant_1/Clip_output_0_minus"
    W_RAW, B_RAW = f"blocks.{i}.conv.weight_int8", f"blocks.{i}.conv.bias_rqsadd"
    WMUL, BMUL = f"{W_RAW}_pmul", f"{B_RAW}_pmul"
    MUL_PC = f"QSTANDALONE_QCDQ_/blocks_{i}_conv_output_quant_1/Clip_Quant_mul_pc"
    OUT_SINGLE = f"/blocks_{i}_conv_output_quant_1/Clip_output_0"
    OUT_DUAL = f"/blocks_{i}_conv_dual_output_0"

    # --- x+ / x- from the full graph's two forward passes ---------------------------------------------
    full = onnx.load(os.path.join(args.full, "network.onnx"))
    fnpz = np.load(os.path.join(args.full, "inputs.npz"))
    base = [fnpz[k] for k in sorted(k for k in fnpz.files if k.startswith("arr_"))]
    ffeed = {inp.name: np.asarray(v) for inp, v in zip(full.graph.input, base)}
    x_plus, y_full_plus = _run(full, ffeed, [ACT, OUT_SINGLE])
    x_plus = x_plus.astype(np.int8)
    if args.same_input:
        x_minus = x_plus.copy()
    else:
        x_minus = _run(_negated(full), ffeed, [ACT])[0].astype(np.int8)

    # --- the unfused single-block fixture: its nodes give the attributes, its graph the reference -------
    src = onnx.load(os.path.join(args.src, "network.onnx"))
    g = src.graph
    by_name = {n.name: n for n in g.node}
    inits = {t.name: t for t in g.initializer}
    snpz = np.load(os.path.join(args.src, "inputs.npz"))
    sfeed = {k: snpz[k] for k in snpz.files}
    assert np.array_equal(sfeed[ACT], x_plus), "QZO_Conv_B<i> was built from a different x+; rebuild it"
    pw, pb = by_name[f"rqsp_blocks.{i}.conv.weight_int8"], by_name[f"rqsp_blocks.{i}.conv.bias_rqsadd"]
    conv = by_name[f"/wrappedInnerForwardImpl_{i}/Conv"]
    rqs = by_name[f"QSTANDALONE_QCDQ_/blocks_{i}_conv_output_quant_1/Clip_Quant"]

    a_plus = _run(src, sfeed, [OUT_SINGLE])[0].astype(np.int8)
    assert np.array_equal(a_plus, y_full_plus.astype(np.int8)), "A+ != full graph's block output"
    a_minus = _run(_negated(src, {WMUL, BMUL}), {**sfeed, ACT: x_minus}, [OUT_SINGLE])[0].astype(np.int8)
    stacked = np.concatenate([a_plus, a_minus], axis = 1)
    co = a_plus.shape[1]

    # --- the fused node -----------------------------------------------------------------------------------
    div = _attr(rqs, "div")
    node = helper.make_node(
        "RequantizedDualConv", [ACT, W_RAW, MUL_PC, B_RAW, WMUL, BMUL, ACT_M], [OUT_DUAL],
        name = f"dualconv_blocks.{i}", domain = "mezo",
        kernel_shape = _attr(conv, "kernel_shape"), pads = _attr(conv, "pads"),
        strides = _attr(conv, "strides"), dilations = _attr(conv, "dilations"), group = _attr(conv, "group"),
        shift = int(np.log2(div)), weight_chw = 1,
        seed_w = _attr(pw, "seed"), idx_w = _attr(pw, "idx"), div_w = _attr(pw, "div"),
        seed_b = _attr(pb, "seed"), idx_b = _attr(pb, "idx"), div_b = _attr(pb, "div"))
    for a in rqs.attribute:  # div / n_levels / signed stay TENSOR-typed (RQSParserInterface unpacks them)
        if a.name in ("div", "n_levels", "signed"):
            node.attribute.append(a)

    act_vi = next(v for v in list(g.input) if v.name == ACT)
    act_m_vi = onnx.ValueInfoProto()
    act_m_vi.CopyFrom(act_vi)
    act_m_vi.name = ACT_M
    graph_inputs = [act_vi, next(v for v in g.input if v.name == W_RAW), next(v for v in g.input if v.name == B_RAW),
                    act_m_vi]
    out_vi = helper.make_tensor_value_info(OUT_DUAL, onnx.TensorProto.INT8, list(stacked.shape))
    graph = helper.make_graph([node], f"QZO_DualConv2in_B{i}", graph_inputs, [out_vi],
                              initializer = [inits[MUL_PC], inits[WMUL], inits[BMUL]])
    m = helper.make_model(graph, opset_imports = list(src.opset_import))
    m.ir_version = src.ir_version
    m.producer_name = "build_dual_conv2in_fixture.py"
    os.makedirs(args.dst, exist_ok = True)
    onnx.save(m, os.path.join(args.dst, "network.onnx"))

    # generateNetwork.py / testMVP.py consume inputs.npz POSITIONALLY -> order == graph inputs.
    np.savez(os.path.join(args.dst, "inputs.npz"),
             **{ACT: x_plus, W_RAW: sfeed[W_RAW], B_RAW: sfeed[B_RAW], ACT_M: x_minus})
    np.savez(os.path.join(args.dst, "outputs.npz"), **{OUT_DUAL: stacked})
    np.savez(os.path.join(args.dst, "reference_plus_minus.npz"), a_plus = a_plus, a_minus = a_minus)

    print(f"[QZO_DualConv2in block {i}] wrote {args.dst}")
    print(f"  node      : RequantizedDualConv inputs={list(node.input)}")
    print(f"  attrs     : seed_w {_attr(pw,'seed')} idx_w {_attr(pw,'idx')} div_w {_attr(pw,'div')}  "
          f"seed_b {_attr(pb,'seed')} idx_b {_attr(pb,'idx')} div_b {_attr(pb,'div')}  div {div}  weight_chw 1")
    print(f"  x+        : {x_plus.shape} range [{x_plus.min()},{x_plus.max()}]")
    print(f"  x-        : {x_minus.shape} range [{x_minus.min()},{x_minus.max()}]  "
          f"{'(= x+, --same-input)' if args.same_input else f'{int((x_plus != x_minus).sum())}/{x_plus.size} differ from x+'}")
    print(f"  A+ == full graph block output: True;  A+ vs A-: {int((a_plus != a_minus).sum())}/{a_plus.size} differ")
    print(f"  output    : {OUT_DUAL} {stacked.shape} int8 (ch 0-{co-1} = A+, {co}-{2*co-1} = A-)")


if __name__ == "__main__":
    main()
