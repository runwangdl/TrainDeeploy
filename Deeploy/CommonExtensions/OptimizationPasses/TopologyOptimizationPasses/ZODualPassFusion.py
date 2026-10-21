# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp8): turn a MeZO / QZO training graph (ONE forward pass, the ε-sign a runtime flag) into the
DUAL-PASS graph that computes both passes in one execution -- L+ and L- as outputs 0 and 1.

Layout: the two passes travel CHANNEL-STACKED between the convs ([+ | -], 2*C channels), so every non-conv node runs
once on 2*C channels. The rule: the graph input is pass-independent, every perturbation output is pass-dependent,
pass-dependence propagates downstream. Per node type:

    Conv + RequantShift, weight / add from RQSPerturbRademacher
        input pass-independent (block 0)  -> RequantizedDualConv, 6 inputs [x, W, mul, add, wmul, bmul] (exp5)
        input stacked (blocks 1-4)        -> RequantizedDualConv, 6 inputs, stacked_input = 1, weight_chw = 1 (exp7/8)
        output: [A+ | A-] (2*Co channels)
    Quant, Dequant, Relu, MaxPool, GlobalAveragePool   -> unchanged, shapes 2*C
    BatchNormInternal  -> gamma / beta from PerturbRademacherFixed (mode 0, [+ ; -]); running mean / var duplicated
    Reshape            -> target shape 2*C
    Gemm (fc)          -> DualHalf (copy one pass's half) + two Gemms with PerturbRademacherFixed weights / biases (mode 1 / 2)
    SoftmaxCrossEntropyLoss -> one per pass; graph outputs [loss (+), loss_minus]

Every perturbation keeps its seed / idx / eps / div, so each half equals what the original node produces in that pass
(the update graph, which regenerates z, is untouched). Opt-in: testMVPTraining.py --zo-fused-dual (runner flag of the
same name), with the harness built -D ZO_FUSED_DUAL=ON (one training-graph execution per window).
"""

import math
from typing import Dict, List, Optional, Set

import numpy as np
import onnx_graphsurgeon as gs

_ELEMENTWISE = ("Quant", "Dequant", "Relu", "MaxPool", "GlobalAveragePool")


def _producer(t: gs.Tensor) -> Optional[gs.Node]:
    return t.inputs[0] if len(t.inputs) else None


def _scalar(v) -> int:
    if isinstance(v, gs.Constant):
        return int(np.asarray(v.values).flat[0])
    return int(np.asarray(v).flat[0])


def _double_channels(t: gs.Variable) -> None:
    if t.shape is not None and len(t.shape) >= 2:
        shape = list(t.shape)
        shape[1] = 2 * int(shape[1])
        t.shape = shape
    elif t.shape is not None and len(t.shape) == 1:
        t.shape = [2 * int(t.shape[0])]


def zoDualPassFusion(graph: gs.Graph, ne16_convs: Optional[Set[int]] = None) -> gs.Graph:
    """ne16_convs: indices (0 = first conv in graph order) of the convs to emit in NE16 form (GAP9_w_NE16):
    block 0 (pass-independent input) as exp6's im2col + stacked encode / bias + NE16 1x1 conv + RequantShift,
    stacked-input convs as one pipelined NE16DualConv2in (exp7 step 7); both followed by a Transpose back to the
    NCHW stacked tensor, so the rest of the graph is unchanged. Others: RequantizedDualConv (cluster)."""
    ne16_convs = set(ne16_convs or ())
    conv_idx = [0]
    graph.toposort()
    dual: Set[str] = set()  # names of channel-stacked tensors
    new_nodes: List[gs.Node] = []
    remove: List[gs.Node] = []
    uid = [0]

    def name(base: str) -> str:
        uid[0] += 1
        return f"{base}__dual{uid[0]}"

    def fixed_perturb(p: gs.Node, mode: int, out_shape: List[int]) -> gs.Variable:
        assert p.op == "PerturbRademacher", p.op
        out = gs.Variable(name(p.outputs[0].name + ("_pm" if mode == 0 else "_p" if mode == 1 else "_m")),
                          dtype = p.outputs[0].dtype, shape = out_shape)
        attrs = {"seed": p.attrs["seed"], "eps": p.attrs["eps"], "idx": p.attrs["idx"], "mode": mode}
        new_nodes.append(
            gs.Node(op = "PerturbRademacherFixed", name = name(p.name + f"_mode{mode}"), inputs = [p.inputs[0]],
                    outputs = [out], attrs = attrs))
        return out

    for node in list(graph.nodes):
        if node.op == "Conv":
            pw = _producer(node.inputs[1])
            rqs = node.outputs[0].outputs[0] if len(node.outputs[0].outputs) == 1 else None
            if pw is None or pw.op != "RQSPerturbRademacher" or rqs is None or rqs.op != "RequantShift":
                continue
            pb = _producer(rqs.inputs[2])
            if pb is None or pb.op != "RQSPerturbRademacher":
                continue
            x = node.inputs[0]
            stacked = x.name in dual
            div = _scalar(rqs.attrs["div"])
            attrs = {k: node.attrs[k] for k in ("kernel_shape", "pads", "strides", "dilations", "group")}
            attrs.update(shift = int(math.log2(div)), seed_w = pw.attrs["seed"], idx_w = pw.attrs["idx"],
                         div_w = pw.attrs["div"], seed_b = pb.attrs["seed"], idx_b = pb.attrs["idx"],
                         div_b = pb.attrs["div"])
            for k in ("div", "n_levels", "signed"):  # RequantShift's protos, verbatim
                attrs[k] = rqs.attrs[k]
            if stacked:
                attrs.update(weight_chw = 1, stacked_input = 1)
            out = rqs.outputs[0]
            _double_channels(out)
            dual.add(out.name)
            ci = conv_idx[0]
            conv_idx[0] += 1
            if ci in ne16_convs:
                new_nodes.extend(_ne16_dual_conv(name, node, rqs, pw, pb, x, out, stacked))
            else:
                new_nodes.append(
                    gs.Node(op = "RequantizedDualConv", name = name(node.name), domain = "mezo",
                            inputs = [x, pw.inputs[0], rqs.inputs[1], pb.inputs[0], pw.inputs[1], pb.inputs[1]],
                            outputs = [out], attrs = attrs))
            node.outputs = []
            rqs.outputs = []
            remove += [node, rqs]
            continue

        if not any(i.name in dual for i in node.inputs):
            continue

        if node.op in _ELEMENTWISE:
            _double_channels(node.outputs[0])
            dual.add(node.outputs[0].name)
            continue

        if node.op == "BatchNormInternal":
            x, g, b, rm, rv = node.inputs[:5]
            pg, pbeta = _producer(g), _producer(b)
            C = int(node.inputs[0].shape[1]) // 2
            node.inputs[1] = fixed_perturb(pg, 0, [2 * C])
            node.inputs[2] = fixed_perturb(pbeta, 0, [2 * C])
            for idx in (3, 4):
                t = node.inputs[idx]
                assert isinstance(t, gs.Constant), f"BN running stat {t.name} is not a constant"
                node.inputs[idx] = gs.Constant(name(t.name), np.concatenate([t.values, t.values]))
            for o in node.outputs:
                _double_channels(o)
            dual.add(node.outputs[0].name)
            continue

        if node.op == "Reshape":
            src = node.inputs[0]
            out = node.outputs[0]
            _double_channels(out)
            node.inputs[1] = gs.Constant(name(node.inputs[1].name), np.asarray(out.shape, dtype = np.int64))
            dual.add(out.name)
            continue

        if node.op == "Gemm":
            # Each pass's Gemm gets exactly the unfused [1, C] input through a DualHalf copy. Not Slice (Deeploy's tiled
            # Slice mis-strides), not a zero-padded weight (changes PULP_Gemm's fp32 summation grouping -- GAP9 step-1 L-).
            a = node.inputs[0]
            C = int(a.shape[1]) // 2
            pw, pbias = _producer(node.inputs[1]), _producer(node.inputs[2])
            out_p = node.outputs[0]
            sce = out_p.outputs[0] if len(out_p.outputs) == 1 else None
            assert sce is not None and sce.op == "SoftmaxCrossEntropyLoss", "dual pass: Gemm must feed the loss"
            w_shape, b_shape = list(node.inputs[1].shape), list(node.inputs[2].shape)
            halves = []
            for h in (0, 1):
                t = gs.Variable(name(a.name + ("_p" if h == 0 else "_m")), dtype = a.dtype, shape = [int(a.shape[0]), C])
                new_nodes.append(gs.Node(op = "DualHalf", name = name(node.name + f"_half{h}"), inputs = [a],
                                         outputs = [t], attrs = {"half": h}))
                halves.append(t)
            # + pass: the original Gemm, re-wired
            node.inputs = [halves[0], fixed_perturb(pw, 1, w_shape), fixed_perturb(pbias, 1, b_shape)]
            # - pass: a second Gemm and loss
            out_m = gs.Variable(name(out_p.name + "_minus"), dtype = out_p.dtype, shape = out_p.shape)
            new_nodes.append(
                gs.Node(op = "Gemm", name = name(node.name + "_minus"), attrs = dict(node.attrs), outputs = [out_m],
                        inputs = [halves[1], fixed_perturb(pw, 2, w_shape), fixed_perturb(pbias, 2, b_shape)]))
            loss_m = gs.Variable("loss_minus", dtype = sce.outputs[0].dtype, shape = sce.outputs[0].shape)
            lp_m = gs.Variable(name("log_prob_minus"), dtype = sce.outputs[1].dtype, shape = sce.outputs[1].shape)
            new_nodes.append(
                gs.Node(op = "SoftmaxCrossEntropyLoss", name = name(sce.name + "_minus"), attrs = dict(sce.attrs),
                        inputs = [out_m] + list(sce.inputs[1:]), outputs = [loss_m, lp_m]))
            graph.outputs = [sce.outputs[0], loss_m]
            continue

        raise NotImplementedError(f"zoDualPassFusion: no dual-pass rewrite for {node.op} ({node.name})")

    # for n in remove:
    #     graph.nodes.remove(n)
    # QW exp9: also disconnect the inputs. Removing a node from graph.nodes leaves it in its input tensors' consumer
    # lists (gs keeps tensor.outputs in sync with node.inputs only), so the replaced Conv / RequantShift stayed a
    # phantom consumer of x through lowering -- invisible to cleanup(), but enough to fail single-consumer matches.
    for n in remove:
        n.inputs.clear()
        n.outputs.clear()
        graph.nodes.remove(n)
    graph.nodes.extend(new_nodes)
    graph.cleanup().toposort()
    return graph



def _nonnegative(t: gs.Tensor) -> bool:
    """RW 2026-10-15: True if t is provably >= 0 -- a Relu reached upstream through ops that keep the sign (Quant with
    zero point 0, MaxPool, Transpose, Reshape, the dual-pass stacking). SpeechNet blocks 1-4: Relu -> MaxPool -> Quant."""
    seen = 0
    while seen < 16:
        p = _producer(t)
        if p is None:
            return False
        if p.op == "Relu":
            return True
        if p.op not in ("Quant", "MaxPool", "Transpose", "Reshape"):
            return False
        if p.op == "Quant" and float(p.attrs.get("zero_point", 0.0)) != 0.0:
            return False
        t = p.inputs[0]
        seen += 1
    return False

_NE16_INPUT_OFFSET, _NE16_WEIGHT_OFFSET, _NE16_CIN_SUBTILE = 128, -128, 16


def _ne16_dual_conv(name, conv: gs.Node, rqs: gs.Node, pw: gs.Node, pb: gs.Node, x: gs.Variable, out: gs.Variable,
                    stacked: bool) -> List[gs.Node]:
    """The NE16 form of one dual conv (GAP9_w_NE16), output = `out` (NCHW [1, 2Co, Ho, Wo], already resized)."""
    W, wmul, add, bmul, mul = pw.inputs[0], pw.inputs[1], pb.inputs[0], pb.inputs[1], rqs.inputs[1]
    Co, Ci, kh, kw = (int(v) for v in W.shape)
    K = Ci * kh * kw
    cinMajor = (K + _NE16_CIN_SUBTILE - 1) // _NE16_CIN_SUBTILE
    _, C2, Ho, Wo = (int(v) for v in out.shape)
    seeds = {"seed_w": pw.attrs["seed"], "idx_w": pw.attrs["idx"], "div_w": pw.attrs["div"],
             "seed_b": pb.attrs["seed"], "idx_b": pb.attrs["idx"], "div_b": pb.attrs["div"]}
    pads, strides = list(conv.attrs["pads"]), list(conv.attrs["strides"])
    # RW 2026-10-15: NE16 reads unsigned input. Block 0's input is signed (+128 offset, corrected in the bias). A
    # stacked input behind a Relu is already >= 0: no offset. With the offset, NE16DualConv2in was off by +1 LSB on
    # ~0.4 % of the outputs on the GAP9 silicon (exact in GVSoC; the int32 bias correction 128*sum(w)*mul wraps).
    in_off = 0 if (stacked and _nonnegative(x)) else _NE16_INPUT_OFFSET
    nodes = []
    wenc = gs.Variable(name(W.name + "_ne16enc"), dtype = np.uint8, shape = [2 * Co, cinMajor, 16])
    add2 = gs.Variable(name(add.name + "_ne16add2"), dtype = np.int32, shape = [2 * Co])
    nodes.append(gs.Node(op = "NE16DualPerturbEncode", name = name(conv.name + "_ne16wenc"), inputs = [W, wmul],
                         outputs = [wenc], attrs = {"seed_w": seeds["seed_w"], "idx_w": seeds["idx_w"],
                                                    "div_w": seeds["div_w"], "ne16_sign_sel": 0,
                                                    "ne16_hwc": 1 if stacked else 0}))
    nodes.append(gs.Node(op = "NE16DualBias", name = name(conv.name + "_ne16bias"), inputs = [W, wmul, add, bmul, mul],
                         outputs = [add2], attrs = dict(seeds, ne16_input_offset = in_off, ne16_sign_sel = 0)))
    y_hwc = gs.Variable(name(out.name + "_hwc"), dtype = np.int8, shape = [1, Ho, Wo, C2])
    if stacked:
        _, C2in, H, Win = (int(v) for v in x.shape)
        x_hwc = gs.Variable(name(x.name + "_hwc"), dtype = np.int8, shape = [1, H, Win, C2in])
        nodes.append(gs.Node(op = "Transpose", name = name(conv.name + "_x_hwc"), inputs = [x], outputs = [x_hwc],
                             attrs = {"perm": [0, 2, 3, 1]}))
        nodes.append(gs.Node(op = "NE16DualConv2in", name = name(conv.name + "_ne16"), inputs = [x_hwc, wenc, mul, add2],
                             outputs = [y_hwc], attrs = {"kernel_shape": [kh, kw], "pads": pads, "strides": strides,
                                                         "shift": int(math.log2(_scalar(rqs.attrs["div"]))),
                                                         "weight_offset": _NE16_WEIGHT_OFFSET, "ne16_pipeline": 1,
                                                         "stacked_input": 1, "ne16_input_offset": in_off}))
    else:  # exp6 form: one column shared by both passes, one NE16 job with 2*Co outputs
        col = gs.Variable(name(x.name + "_col"), dtype = np.uint8, shape = [1, Ho, Wo, K])
        nodes.append(gs.Node(op = "NE16Im2ColU8", name = name(conv.name + "_im2col"), inputs = [x], outputs = [col],
                             attrs = {"kernel_shape": [kh, kw], "pads": pads, "strides": strides,
                                      "ne16_input_offset": _NE16_INPUT_OFFSET}))
        mul2 = gs.Constant(name(mul.name + "_x2"), np.concatenate([mul.values, mul.values]).astype(np.int32))
        acc = gs.Variable(name(out.name + "_acc"), dtype = np.int32, shape = [1, Ho, Wo, C2])
        nodes.append(gs.Node(op = "Conv", name = name(conv.name + "_ne16conv"), inputs = [col, wenc], outputs = [acc],
                             attrs = {"kernel_shape": [1, 1], "pads": [0, 0, 0, 0], "strides": [1, 1],
                                      "dilations": [1, 1], "group": 1, "ne16_weight_preencoded": 1,
                                      "weight_offset": _NE16_WEIGHT_OFFSET, "channels_first": 0}))
        rq_attrs = {k: rqs.attrs[k] for k in ("div", "n_levels", "signed")}
        rq_attrs["channels_first"] = 0
        nodes.append(gs.Node(op = "RequantShift", name = name(conv.name + "_ne16rqs"), domain = rqs.domain,
                             inputs = [acc, mul2, add2], outputs = [y_hwc], attrs = rq_attrs))
    nodes.append(gs.Node(op = "Transpose", name = name(conv.name + "_out_nchw"), inputs = [y_hwc], outputs = [out],
                         attrs = {"perm": [0, 3, 1, 2]}))
    return nodes
