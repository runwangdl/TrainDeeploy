#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""exp16a — build the SpeechNet block-1 fixtures for the all-pointwise NE16 decomposition.

Produces three fixtures, all fed the SAME real data (see Plan.md §4.2):

  b1_ref_1x16      the original 1x16 Conv + RequantShift        -> cluster reference
  b1_pw_plain      16 pointwise taps, PLAIN int8 weights        -> host-verifiable + cluster A/B
  b1_pw_ne16       16 pointwise taps, NE16 PRE-ENCODED weights  -> the NE16 run

Everything real, taken from the TRAINING graph
`DeeployTest/Tests/Models/Training/SpeechNet/speechnet_qzo12_train/network.onnx`:
  * the int8 activation feeding block-1's Conv (eval window 0, via the host executor)
  * `blocks.1.conv.weight_int8` PERTURBED by the real RQSPerturbRademacher (seed 42, idx 2,
    pmul from the graph) -- i.e. exactly the runtime tensor the device would hand the conv
  * the real RequantShift mul / add / div

Blocker 1a is solved by delivering the weight as a GRAPH INPUT already in NE16 bit-serial
layout, with a FIXED `weight_offset = -128`: every int8 weight satisfies w+128 in [0,255], so
the offset never has to be recomputed when the weight changes. Encoding uses Deeploy's own
`_weightEncode`, imported -- never reimplemented.

Run in agitated_hugle (the container with Onnx4Deeploy importable):
  docker exec agitated_hugle bash -lc 'cd /app && \
    PYTHONPATH=/app/Onnx4Deeploy:/app/TrainDeeploy python3 \
    TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16a_PW_single_layer/build_fixtures.py'
"""

import argparse
import os
import sys

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import helper, numpy_helper

from onnx4deeploy.utils.onnx_node_implementations import run_onnx_graph

# QW: `agitated_hugle` (the only container with Onnx4Deeploy importable) has no `mako`, so
#     `import Deeploy...` fails on DeeployTypes. Rather than REIMPLEMENT the encoder -- which
#     would risk the fixture and the compiler disagreeing -- lift Deeploy's own `_weightEncode`
#     source out of the file by AST and exec it. It is pure numpy, so it needs nothing else.
#     Provably the same bytes as the compile-time encoder. -- QW
def _load_deeploy_weight_encode(
        path = "TrainDeeploy/Deeploy/Targets/NE16/TopologyOptimizationPasses/Passes.py"):
    import ast
    tree = ast.parse(open(path).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_weightEncode")
    import numpy.typing as npt  # the function is annotated npt.NDArray[...]
    ns = {"np": np, "npt": npt}
    mod = ast.Module(body = [fn], type_ignores = [])
    ast.fix_missing_locations(mod)
    exec(compile(mod, path, "exec"), ns)
    return ns["_weightEncode"]


_weightEncode = _load_deeploy_weight_encode()

TRAIN = "TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_qzo12_train"
ACT = "/blocks_1_conv_input_quant_1/Clip_output_0"  # int8 activation into block-1 Conv
WPERT = "blocks.1.conv.weight_int8_pert"  # the runtime (perturbed) weight
CONV = "/wrappedInnerForwardImpl_1/Conv"

WEIGHT_OFFSET = -128  # fixed, valid for every int8 weight -- see Plan.md 3.Blocker1



def _save_with_shapes(m, path):
    """QW: run ONNX shape inference before saving.

    helper.make_graph only annotates graph inputs/outputs, so every INTERMEDIATE tensor
    (xpad, x00, c00, a01, ...) comes back from onnx_graphsurgeon with shape=None. Deeploy's
    PULPNCHWtoNHWCPass then dies with `TypeError: object of type 'NoneType' has no len()` at
    LoweringOptimizationPasses.py:276. Deeploy also asserts every tensor has a shape
    (DeeployTypes.py `_assertTensorsHaveShape`). strict_mode=False so the custom-domain
    RequantShift does not abort inference. -- QW
    """
    import onnx.shape_inference
    m = onnx.shape_inference.infer_shapes(m, strict_mode = False)
    onnx.save(m, path)
    return m

def _rqs_node(name, inp, mul_name, add_name, out, attr_protos, domain = ""):
    # QW: the DOMAIN matters -- RequantShift is a Deeploy custom op and both run_onnx_graph and
    #     Deeploy's parsers dispatch on it. Carry the source node's domain verbatim. -- QW
    n = helper.make_node("RequantShift", [inp, mul_name, add_name], [out], name = name,
                         domain = domain)
    n.attribute.extend(attr_protos)  # verbatim: `div` MUST stay TENSOR-typed (see exp16 Findings)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default = TRAIN)
    ap.add_argument("--out", default = "TrainDeeploy/DeeployTest/Tests/Models/NE16")
    ap.add_argument("--window", type = int, default = 0)
    ap.add_argument("--taps", type = int, default = 0,
                    help = "truncate the 1xK kernel to the first K taps (0 = keep all 16). "
                           "Used to measure how Deeploy's compile time scales with tap count: "
                           "binding type-check does TWO ctxt.copy() per candidate "
                           "(DeeployTypes.py:1606,1768), so cost grows with nodes x buffers.")
    ap.add_argument("--suffix", default = "", help = "appended to fixture dir names")
    ap.add_argument("--crop-h", type = int, default = 0, dest = "crop_h",
                    help = "crop the activation to this many rows (0 = full 14). exp16a pins the "
                           "conv to a SINGLE tile, so the int32 output (14*88*16*4 = 78,848 B) plus "
                           "the two layout transposes overflow GAP9's ~110 KB L1 -> "
                           "'Allocation failed for allocator 2'. Cropping keeps every value real "
                           "while making the single-tile policy fit.")
    ap.add_argument("--crop-w", type = int, default = 0, dest = "crop_w",
                    help = "crop the activation to this many columns (0 = full 87)")
    ap.add_argument("--dense3x3", action = "store_true", default = False,
                    help = "exp16b: emit the 1xK conv for the DENSE 3x3 chunk decomposition "
                           "(ceil(K/3) dispatches, each a 3x3 kernel with only the middle row "
                           "populated) instead of the all-pointwise one. Implies --nhwc.")
    ap.add_argument("--perturb", action = "store_true", default = False,
                    help = "exp16c phase 2 / blocker 1b: put the REAL RQSPerturbRademacher node in "
                           "front of the encoder, fed the UNPERTURBED int8 weight, so the device "
                           "runs the full QZO weight path perturb -> encode -> conv. Implies "
                           "--device-encode. --neg-pmul negates the per-channel multiplier, which "
                           "is exactly the L- pass (see RandomNoiseQuant.c: negating the Rademacher "
                           "sign == negating M), so both ZO passes are testable under the plain "
                           "runner without touching perturbation_sign.")
    ap.add_argument("--neg-pmul", action = "store_true", default = False, dest = "neg_pmul",
                    help = "negate the perturbation multiplier -> the L- pass (see --perturb)")
    ap.add_argument("--device-encode", action = "store_true", default = False, dest = "device_encode",
                    help = "exp16c / blocker 1b: deliver the weight UNENCODED as an int8 graph "
                           "input and insert an NE16WeightEncode node, so the bit-serial encoding "
                           "happens ON DEVICE. This is what the QZO loop needs -- there the weight "
                           "is an RQSPerturbRademacher output, not a gs.Constant, so the host "
                           "cannot pre-encode it. Implies --nhwc.")
    ap.add_argument("--nhwc", action = "store_true", default = False,
                    help = "emit the 1xK fixture NHWC-native (input, output and golden already "
                           "channels-last, conv+RQS tagged channels_first=0). Removes BOTH layout "
                           "Transposes: without it the int32 conv output is materialised in two "
                           "layouts (2 x 78,848 B at full extent) and the L1 arena request hits "
                           "108,416 B against ~98 KB usable -> 'Allocation failed for allocator 2'.")
    args = ap.parse_args()

    src = os.path.join(args.train, "network.onnx")
    model = onnx.load(src)
    g = gs.import_onnx(model)
    conv = next(n for n in g.nodes if n.name == CONV)
    rqs = next(n for n in g.nodes if n.op == "RequantShift" and n.inputs[0] is conv.outputs[0])
    src_rqs_proto = next(n for n in model.graph.node if n.name == rqs.name)
    rqs_protos = list(src_rqs_proto.attribute)
    rqs_domain = src_rqs_proto.domain

    pads = [int(v) for v in conv.attrs["pads"]]  # [h_beg, w_beg, h_end, w_end] = [0,8,0,8]
    ksh = [int(v) for v in conv.attrs["kernel_shape"]]  # [1, 16]
    assert ksh[0] == 1, f"exp16a handles 1xk only; got {ksh}"
    K = ksh[1]
    if args.taps and args.taps < K:   # -- QW: scaled-down variant, see --taps help
        K = args.taps
        pads = [0, K // 2, 0, K // 2]
        ksh = [1, K]
        print(f"[taps ] TRUNCATED to first {K} taps; pads -> {pads}")

    # ---- 1. real inputs, straight out of the training graph -----------------------------------
    npz = np.load(os.path.join(args.train, "inputs.npz"))
    base = [npz[k] for k in sorted(k for k in npz.files if k.startswith("arr_"))]
    feed = {i.name: np.asarray(v) for i, v in zip(model.graph.input, base)}
    act, wpert = (np.asarray(t) for t in run_onnx_graph(src, feed, output_names = [ACT, WPERT]))
    # QW (exp16c phase 2): the RQSPerturbRademacher node that PRODUCES wpert, plus its two inputs.
    #     Copying the node's AttributeProtos verbatim keeps `idx` (which feeds the device kernel's
    #     chunk_seed as node_id) and `seed` identical to the real graph, so the device RNG stream
    #     matches the host reference exactly. -- QW
    pert_node = next((n for n in g.nodes if n.op == "RQSPerturbRademacher"
                      and n.outputs[0].name == WPERT), None)
    if args.perturb:
        assert pert_node is not None, f"no RQSPerturbRademacher producing {WPERT}"
        src_pert_proto = next(n for n in model.graph.node if n.name == pert_node.name)
        pert_protos, pert_domain = list(src_pert_proto.attribute), src_pert_proto.domain
        w_raw = np.asarray(feed[pert_node.inputs[0].name]).astype(np.int8)
        w_pmul = np.asarray(pert_node.inputs[1].values).astype(np.int32)
        print(f"[pert ] {pert_node.name} domain={pert_domain!r} attrs="
              f"{ {a.name: a.i for a in pert_protos if a.type == onnx.AttributeProto.INT} } "
              f"w_raw{list(w_raw.shape)} pmul{list(w_pmul.shape)} range [{w_pmul.min()},{w_pmul.max()}]")
        if args.neg_pmul:
            # QW: the L- pass. RandomNoiseQuant.c:31 -- "negating r is exactly equivalent to
            #     negating the per-channel multiplier M" -- so running the host reference with -M
            #     gives the L- golden, and the device reproduces it with the NEUTRAL
            #     perturbation_sign = +1 default. That keeps both ZO passes testable under the
            #     plain inference runner, with no ZO-runtime globals to drive. -- QW
            pg = helper.make_graph(
                [helper.make_node("RQSPerturbRademacher", ["w", "pmul"], ["wp"],
                                  name = "p", domain = pert_domain)],
                "perturb_only",
                [helper.make_tensor_value_info("w", onnx.TensorProto.INT8, list(w_raw.shape))],
                [helper.make_tensor_value_info("wp", onnx.TensorProto.INT8, list(w_raw.shape))],
                initializer = [numpy_helper.from_array((-w_pmul).astype(np.int32), "pmul")])
            pg.node[0].attribute.extend(pert_protos)
            pm = helper.make_model(pg, opset_imports = list(model.opset_import))
            pm.ir_version = model.ir_version
            ppath = "/tmp/_exp16c_perturb_neg.onnx"
            onnx.save(pm, ppath)
            wpert = np.asarray(run_onnx_graph(ppath, {"w": w_raw}, output_names = ["wp"])[0])
            print(f"[pert-] L- weight recomputed with -pmul: range [{wpert.min()},{wpert.max()}] "
                  f"(differs from L+ in {int((wpert != np.asarray(run_onnx_graph(src, feed, output_names=[WPERT])[0])).sum())} of {wpert.size} elements)")
    if args.crop_h or args.crop_w:   # -- QW: see --crop-h help
        act = act[:, :, :args.crop_h or act.shape[2], :args.crop_w or act.shape[3]].copy()
        print(f"[crop ] activation cropped to {list(act.shape)}")
    print(f"[act  ] {act.shape} {act.dtype}  range [{act.min()},{act.max()}]")
    print(f"[wpert] {wpert.shape} {wpert.dtype}  range [{wpert.min()},{wpert.max()}]   (RQSPerturbRademacher output)")
    assert act.min() >= 0, "block-1 activations expected >= 0 (MaxPool<-Relu); BLOCKER 3 would apply"

    mul = np.asarray(rqs.inputs[1].values).astype(np.int32)
    add_t = rqs.inputs[2]
    add = np.asarray(add_t.values).astype(np.int32) if isinstance(add_t, gs.Constant) else \
        np.asarray(run_onnx_graph(src, feed, output_names = [add_t.name])[0]).astype(np.int32)
    print(f"[rqs  ] mul{list(mul.shape)} add{list(add.shape)} attrs="
          f"{[(a.name, onnx.AttributeProto.AttributeType.Name(a.type)) for a in rqs_protos]}")

    N, Cin, H, W = act.shape
    wpert = wpert[:, :, :, :K]          # -- QW: honour --taps
    Cout = wpert.shape[0]
    Wout = W + pads[1] + pads[3] - K + 1
    print(f"[shape] in [{N},{Cin},{H},{W}] * [{Cout},{Cin},1,{K}] pads={pads} -> out [{N},{Cout},{H},{Wout}]")

    opset = list(model.opset_import)

    # ---- 2. reference fixture: the original 1x16 ----------------------------------------------
    ref_dir = os.path.join(args.out, f"b1_ref_1x{K}{args.suffix}")
    ref = helper.make_graph(
        [helper.make_node("Conv", ["input", "weight"], ["conv_out"], name = "b1_conv",
                          kernel_shape = ksh, pads = pads, strides = [1, 1], dilations = [1, 1], group = 1),
         _rqs_node("b1_rqs", "conv_out", "rqs_mul", "rqs_add", "output", rqs_protos, rqs_domain)],
        f"b1_ref_1x{K}{args.suffix}",
        [helper.make_tensor_value_info("input", onnx.TensorProto.INT8, [N, Cin, H, W]),
         helper.make_tensor_value_info("weight", onnx.TensorProto.INT8, list(wpert.shape))],
        [helper.make_tensor_value_info("output", onnx.TensorProto.INT8, [N, Cout, H, Wout])],
        initializer = [numpy_helper.from_array(mul, "rqs_mul"), numpy_helper.from_array(add, "rqs_add")])
    m_ref = helper.make_model(ref, opset_imports = opset)
    m_ref.ir_version = model.ir_version
    os.makedirs(ref_dir, exist_ok = True)
    _save_with_shapes(m_ref, os.path.join(ref_dir, "network.onnx"))
    feed_ref = {"input": act.astype(np.int8), "weight": wpert.astype(np.int8)}
    ref_out = np.asarray(run_onnx_graph(os.path.join(ref_dir, "network.onnx"), feed_ref, output_names = ["output"])[0])
    np.savez(os.path.join(ref_dir, "inputs.npz"), **feed_ref)
    np.savez(os.path.join(ref_dir, "outputs.npz"), output = ref_out.astype(np.int8))
    print(f"[ref  ] output {ref_out.shape} range [{ref_out.min()},{ref_out.max()}] -> {ref_dir}")

    # ---- 2b. the 1xK fixture for the TEMPLATE-level decomposition (STEP 2b / option A) --------
    #     ONE Conv(1xK) node, claimed by NE16Engine.is1xKConv, decomposed into K pointwise
    #     dispatches INSIDE the template: tap j is an infeat_addr offset, and the K partial sums
    #     accumulate in NE16 via streamin. No Slice, no Pad, no Add chain -> the graph stays at
    #     2 nodes and Deeploy's context stays small.
    #
    #     Weight layout: per-tap NE16 encoding stacked as (K, cout, cinMajor, bits*cinMinorBytes),
    #     so tap j's block is CONTIGUOUS at offset j*cout*cinMajor*16 and is byte-identical to
    #     what a standalone 1x1 conv of that tap would use. weight_offset = -128 fixed. -- QW
    if args.perturb:
        args.device_encode = True              # -- QW: exp16c phase 2 builds on phase 1
    if args.dense3x3 or args.device_encode:
        args.nhwc = True                       # -- QW: exp16b/exp16c always run NHWC-native
    kdir = os.path.join(args.out, (f"b1_1x{K}_dense3x3{args.suffix}" if args.dense3x3
                                   else f"b1_1x{K}_pertenc{args.suffix}" if args.perturb
                                   else f"b1_1x{K}_devenc{args.suffix}" if args.device_encode
                                   else f"b1_1x{K}_ne16{args.suffix}"))
    os.makedirs(kdir, exist_ok = True)
    # QW: RANK 3, taps stacked along dim0 -> (K*cout, cinMajor, encBytes).
    #     A rank-4 weight is TRANSPOSED by PULPNCHWtoNHWCPass (it treats any 4-D conv input as an
    #     activation: [2,16,1,16] -> [2,1,16,16]), which destroys the bit-serial encoding. The
    #     existing PW path is rank 3 for exactly this reason and is left untouched. Tap j's block
    #     stays contiguous at byte offset j*cout*cinMajor*encBytes. -- QW
    if args.dense3x3:
        # exp16b: chunk c holds taps 3c,3c+1,3c+2 in the MIDDLE ROW of a 3x3 kernel; rows 0 and 2
        # are zero, which is what makes the native H padding 1/1 harmless. Taps >= K are zero.
        CHUNK, = (3,)
        n_chunks = (K + CHUNK - 1) // CHUNK
        blocks = []
        for c in range(n_chunks):
            k33 = np.zeros((Cout, Cin, 3, 3), dtype = np.int32)
            for d in range(CHUNK):
                j = c * CHUNK + d
                if j < K:
                    k33[:, :, 1, d] = wpert[:, :, 0, j]
            blocks.append(_weightEncode((k33 - WEIGHT_OFFSET).astype(np.uint8), bits = 8))
        enc_taps = np.concatenate(blocks, axis = 0)   # (chunks*cout, cinMajor, bits, 9*cinMinorBytes)
        # QW: the input must be EXACTLY Wout + halo wide, where halo = 3*(chunks-1)+2 (chunk c
        #     starts at offset 3c and its 3x3 window reads 2 past its own output). Deriving it
        #     from the halo rather than hardcoding +2 keeps the fixture consistent with
        #     NE161xKConv2DTileConstraint for every K, not just K=16.
        halo33 = 3 * (n_chunks - 1) + 2
        extra_w = (Wout + halo33) - (W + pads[1] + pads[3])
        assert extra_w >= 0, f"negative extra_w={extra_w}: Wout={Wout} halo={halo33}"

        node_pads = [1, 0, 1, 0]                      # H 1/1 native (legal in 3x3 mode), W 0/0
        print(f"[dense3x3] {n_chunks} chunks of 3 taps; weight {enc_taps.shape} "
              f"({enc_taps.nbytes} B) vs pointwise {K * Cout * 1 * 16} B")
    else:
        n_chunks, extra_w, node_pads = 0, 0, [0, 0, 0, 0]
        enc_taps = np.concatenate([_weightEncode((wpert[:, :, :, j:j + 1].astype(np.int32) - WEIGHT_OFFSET).astype(np.uint8),
                                                 bits = 8) for j in range(K)], axis = 0)   # (K*cout, cinMajor, 16)
    kconv = helper.make_node("Conv", ["input", "weight_enc"], ["conv_out"], name = "b1_conv1xk",
                             kernel_shape = ksh, pads = node_pads, strides = [1, 1],
                             dilations = [1, 1], group = 1)
    kconv.attribute.append(helper.make_attribute("ne16_weight_preencoded", 1))
    kconv.attribute.append(helper.make_attribute("weight_offset", WEIGHT_OFFSET))
    kconv.attribute.append(helper.make_attribute("ne16_taps", K))
    if args.dense3x3:
        kconv.attribute.append(helper.make_attribute("ne16_chunks", n_chunks))  # -- QW (exp16b)
    if args.nhwc:  # -- QW: NHWC-native, no layout Transposes at all (see --nhwc help)
        kconv.attribute.append(helper.make_attribute("channels_first", 0))
    krqs = _rqs_node("b1_rqs", "conv_out", "rqs_mul", "rqs_add", "output", rqs_protos, rqs_domain)
    # QW: state channels_first=1 EXPLICITLY. Because the 1xK conv is NOT merged with its
    #     RequantShift (streamin needs int32 out), the RQS ends up OUTSIDE the NHWC region the
    #     layout pass wraps around the conv -- it consumes the NCHW [1,Cout,H,W] tensor produced
    #     by the conv's output Transpose. The deployer's default_channels_first is False for
    #     GAP9/NE16, so without this attribute RequantShiftTileConstraint reads the channel
    #     offset from cube.offset[-1] (the W axis) and slices mul/add out of bounds:
    #     "Rectangle offset should be zero ... HyperRectangle(offset=(0,16), dims=(1,16))
    #     and reference shape (1,16)". DeeployTypes.py:1198 lets a node attribute override the
    #     default, and NCHW is simply the truth for this node. -- QW
    #     With --nhwc the RQS stays channels-last, which is the deployer default anyway, so the
    #     attribute below is only needed in the NCHW-native variant.
    krqs.attribute.append(helper.make_attribute("channels_first", 0 if args.nhwc else 1))
    # QW (exp16c / blocker 1b): with --device-encode the conv's weight input is produced by an
    #     NE16WeightEncode NODE fed the raw int8 weight, instead of arriving pre-encoded. That is
    #     the only structural difference from exp16a -- the conv, its attributes and the golden are
    #     untouched, so a pass here isolates "the device encoder reproduces _weightEncode exactly".
    #     In the real QZO graph the int8 weight comes from RQSPerturbRademacher rather than from a
    #     graph input; phase 2 swaps that producer in. -- QW
    kNodes = [kconv, krqs]
    kIns = [helper.make_tensor_value_info("input", onnx.TensorProto.INT8,
                                          [N, H, W + pads[1] + pads[3] + extra_w, Cin] if args.nhwc
                                          else [N, Cin, H, W + pads[1] + pads[3] + extra_w])]
    kVinfo = [helper.make_tensor_value_info("conv_out", onnx.TensorProto.INT32,
                                            [N, H, Wout, Cout] if args.nhwc else [N, Cout, H, Wout])]
    kInits = [numpy_helper.from_array(mul, "rqs_mul"), numpy_helper.from_array(add, "rqs_add")]
    if args.device_encode:  # -- QW
        # QW (exp16c phase 2): with --perturb the encoder is fed by the REAL RQSPerturbRademacher
        #     instead of by a graph input -- perturb -> encode -> conv, the full QZO weight path on
        #     device. The node's AttributeProtos are copied verbatim so `idx` (the device kernel's
        #     chunk_seed node_id) and `seed` match the training graph, and the device RNG stream
        #     therefore reproduces the host's wpert bit for bit. -- QW
        encSrc = "weight"
        if args.perturb:
            encSrc = "weight_pert"
            pn = helper.make_node("RQSPerturbRademacher", ["weight", "w_pmul"], ["weight_pert"],
                                  name = "b1_wpert", domain = pert_domain)
            pn.attribute.extend(pert_protos)   # verbatim -- `div` stays TENSOR-typed
            kNodes = [pn] + kNodes
            kInits.append(numpy_helper.from_array(
                (-w_pmul if args.neg_pmul else w_pmul).astype(np.int32), "w_pmul"))
            kVinfo.append(helper.make_tensor_value_info("weight_pert", onnx.TensorProto.INT8,
                                                        list(wpert.shape)))
        enc_node = helper.make_node("NE16WeightEncode", [encSrc], ["weight_enc"],
                                    name = "b1_wenc")
        enc_node.attribute.append(helper.make_attribute("ne16_taps", K))
        enc_node.attribute.append(helper.make_attribute("ne16_bits", 8))
        kNodes = [enc_node] + kNodes if not args.perturb else kNodes[:1] + [enc_node] + kNodes[1:]
        kIns.append(helper.make_tensor_value_info("weight", onnx.TensorProto.INT8, list(wpert.shape)))
        kVinfo.append(helper.make_tensor_value_info("weight_enc", onnx.TensorProto.UINT8,
                                                    list(enc_taps.shape)))
    else:
        kIns.append(helper.make_tensor_value_info("weight_enc", onnx.TensorProto.UINT8,
                                                  list(enc_taps.shape)))
    kg = helper.make_graph(
        kNodes,
        f"b1_1x{K}_ne16{args.suffix}",
        kIns,
        [helper.make_tensor_value_info("output", onnx.TensorProto.INT8,
                                       [N, H, Wout, Cout] if args.nhwc else [N, Cout, H, Wout])],
        initializer = kInits,
        value_info = kVinfo)
    mk = helper.make_model(kg, opset_imports = opset)
    mk.ir_version = model.ir_version
    onnx.save(mk, os.path.join(kdir, "network.onnx"))     # no shape inference: shapes are explicit
    act_pad_k = np.pad(act, ((0, 0), (0, 0), (0, 0), (pads[1], pads[3] + extra_w)),
                       mode = "constant", constant_values = 0).astype(np.int8)
    k_in = act_pad_k.transpose(0, 2, 3, 1).copy() if args.nhwc else act_pad_k
    # The golden always comes from the NCHW reference graph (ONNX Conv is NCHW-by-definition, so
    # run_onnx_graph must see NCHW); it is simply permuted for the NHWC-native device fixture.
    k_out = ref_out.transpose(0, 2, 3, 1).copy() if args.nhwc else ref_out
    if args.perturb:  # -- QW (exp16c phase 2): the device gets the UNPERTURBED weight and does both steps
        np.savez(os.path.join(kdir, "inputs.npz"), input = k_in, weight = w_raw.astype(np.int8))
    elif args.device_encode:  # -- QW (exp16c phase 1): the device gets the RAW (already perturbed) int8 weight
        np.savez(os.path.join(kdir, "inputs.npz"), input = k_in, weight = wpert.astype(np.int8))
    else:
        np.savez(os.path.join(kdir, "inputs.npz"), input = k_in, weight_enc = enc_taps)
    np.savez(os.path.join(kdir, "outputs.npz"), output = k_out.astype(np.int8))
    # QW (exp16c): keep the host-encoded bytes next to the fixture as the golden the DEVICE
    #     encoder must reproduce, so a mismatch can be localised to the encoder rather than the conv.
    if args.device_encode:
        np.savez(os.path.join(kdir, "weight_enc_golden.npz"), weight_enc = enc_taps)
    print(f"[b1_1x{K}_ne16] ONE Conv(1x{K}) node, weight_enc {enc_taps.shape} uint8 "
          f"({enc_taps.nbytes} B){' ENCODED ON DEVICE' if args.device_encode else ''}, "
          f"input {list(k_in.shape)} "
          f"{'NHWC-native' if args.nhwc else 'NCHW'} -> {kdir}")

    # ---- 3. decomposed fixtures: 16 pointwise taps ---------------------------------------------
    # conv_{1xK}(W,X)[w] = sum_j conv_{1x1}(W[:,:,0,j], Xpad)[w+j]  -> tap j slices Xpad[j : j+Wout]
    for variant, encode in ((f"b1_pw{K}_plain{args.suffix}", False), (f"b1_pw{K}_ne16{args.suffix}", True)):
        d = os.path.join(args.out, variant)
        os.makedirs(d, exist_ok = True)
        nodes, inits, gins, feed_d, vinfo = [], [], [], {}, []

        # QW **TEMPORARY**: the activation is PRE-PADDED in the fixture data, so the graph has
        #     no Pad node. Reason: GAP9 offers only Pad1DParser / Pad2DParser and neither binds a
        #     4-D constant-value int8 Pad -> "PARSING FAILED - Backtracking exhausted at root!,
        #     Deepest successful exploration: Layer 3 'b1_pad'". This isolates whether Slice
        #     binds. The PROPER fix is a GAP9 Pad binding (or native NE16 padding, which exists
        #     only in 3x3 mode) -- see Findings.md. -- QW
        Wp = W + pads[1] + pads[3]
        act_pad = np.pad(act, ((0, 0), (0, 0), (0, 0), (pads[1], pads[3])),
                         mode = "constant", constant_values = 0).astype(np.int8)
        gins.append(helper.make_tensor_value_info("input", onnx.TensorProto.INT8, [N, Cin, H, Wp]))
        feed_d["input"] = act_pad

        acc = None
        for j in range(K):
            wj = wpert[:, :, :, j:j + 1]  # [Cout, Cin, 1, 1] int8
            wname = f"w{j:02d}"
            if encode:
                # BLOCKER 1a: weight arrives as a GRAPH INPUT, already NE16 bit-serial encoded,
                # with a FIXED weight_offset = -128 (valid for any int8 weight, never recomputed).
                enc = _weightEncode((wj.astype(np.int32) - WEIGHT_OFFSET).astype(np.uint8), bits = 8)
                gins.append(helper.make_tensor_value_info(wname, onnx.TensorProto.UINT8, list(enc.shape)))
                feed_d[wname] = enc
            else:
                gins.append(helper.make_tensor_value_info(wname, onnx.TensorProto.INT8, list(wj.shape)))
                feed_d[wname] = wj.astype(np.int8)

            # QW: Slice must have FIVE inputs (data, starts, ends, axes, steps) and, for int8
            #     data, only PULPDMASliceBindings applies -- PULPSliceBindings takes
            #     FloatDataTypes for data_in. That binding types starts/ends/axes/steps as
            #     **uint8_t** (Bindings.py:154-164), so the index tensors must be uint8, not the
            #     int64 the ONNX spec suggests. Omitting `steps` or using int64 gives
            #     "PARSING FAILED ... Deepest successful exploration: Layer 0 'b1_slice00'".
            #     All our values fit: starts<=15, ends<=103, axes=3, steps=1. -- QW
            inits += [numpy_helper.from_array(np.array([j], dtype = np.uint8), f"s{j}_beg"),
                      numpy_helper.from_array(np.array([j + Wout], dtype = np.uint8), f"s{j}_end"),
                      numpy_helper.from_array(np.array([3], dtype = np.uint8), f"s{j}_ax"),
                      numpy_helper.from_array(np.array([1], dtype = np.uint8), f"s{j}_st")]
            nodes.append(helper.make_node("Slice",
                                          ["input", f"s{j}_beg", f"s{j}_end", f"s{j}_ax", f"s{j}_st"],
                                          [f"x{j:02d}"], name = f"b1_slice{j:02d}"))
            # QW: declare intermediate shapes EXPLICITLY. onnx.shape_inference cannot infer
            #     through a Slice with uint8 index tensors (non-standard per the ONNX spec, but
            #     required by PULPDMASliceBindings), so every downstream tensor would come back
            #     shape=None and PULPNCHWtoNHWCPass dies at LoweringOptimizationPasses.py:276.
            #     We know all of these analytically. -- QW
            vinfo.append(helper.make_tensor_value_info(f"x{j:02d}", onnx.TensorProto.INT8,
                                                       [N, Cin, H, Wout]))
            vinfo.append(helper.make_tensor_value_info(f"c{j:02d}", onnx.TensorProto.INT32,
                                                       [N, Cout, H, Wout]))
            cnode = helper.make_node("Conv", [f"x{j:02d}", wname], [f"c{j:02d}"], name = f"b1_conv{j:02d}",
                                     kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1],
                                     dilations = [1, 1], group = 1)
            if encode:
                # consumed by NE16Engine.isPWConv (C1) and skipped by the weight-layout pass (C2)
                cnode.attribute.append(helper.make_attribute("ne16_weight_preencoded", 1))
                cnode.attribute.append(helper.make_attribute("weight_offset", WEIGHT_OFFSET))
            nodes.append(cnode)
            if acc is None:
                acc = f"c{j:02d}"
            else:
                nodes.append(helper.make_node("Add", [acc, f"c{j:02d}"], [f"a{j:02d}"], name = f"b1_add{j:02d}"))
                vinfo.append(helper.make_tensor_value_info(f"a{j:02d}", onnx.TensorProto.INT32,
                                                           [N, Cout, H, Wout]))
                acc = f"a{j:02d}"

        nodes.append(_rqs_node("b1_rqs", acc, "rqs_mul", "rqs_add", "output", rqs_protos, rqs_domain))
        inits += [numpy_helper.from_array(mul, "rqs_mul"), numpy_helper.from_array(add, "rqs_add")]
        gg = helper.make_graph(nodes, variant, gins,
                               [helper.make_tensor_value_info("output", onnx.TensorProto.INT8,
                                                              [N, Cout, H, Wout])], initializer = inits,
                               value_info = vinfo)
        mm = helper.make_model(gg, opset_imports = opset)
        mm.ir_version = model.ir_version
        _save_with_shapes(mm, os.path.join(d, "network.onnx"))
        np.savez(os.path.join(d, "inputs.npz"), **feed_d)
        np.savez(os.path.join(d, "outputs.npz"), output = ref_out.astype(np.int8))  # SAME golden

        tag = "NE16 pre-encoded" if encode else "plain int8"
        if not encode:
            # ACCEPTANCE CRITERION 1: the decomposition must match the reference bit-exactly,
            # on the host, before any device run.
            got = np.asarray(run_onnx_graph(os.path.join(d, "network.onnx"), feed_d, output_names = ["output"])[0])
            nbad = int((got != ref_out).sum())
            print(f"[{variant}] {tag}: host check vs 1x16 reference -> "
                  f"{'BIT-EXACT' if nbad == 0 else f'{nbad} MISMATCHES'} ({got.size} elems)")
            if nbad:
                return 1
        else:
            print(f"[{variant}] {tag}: weight inputs {K} x {feed_d['w00'].shape} uint8, "
                  f"weight_offset={WEIGHT_OFFSET} (fixed)")
        print(f"           -> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
