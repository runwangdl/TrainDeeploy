# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# exp21m: the fusion-only control network. Copy of the QZO-ECO training graph with, for blocks 1-4, the four nodes
#   RQSPerturbRademacher(W) + RQSPerturbRademacher(b) + Conv + RequantShift
# replaced by ONE `RequantizedDualConv` with `single_pass` = 1 (PULP_PerturbConv4x2_i8_i8_i8_HWC: perturbation inside
# the kernel, one pass, sign = runtime perturbation_sign). Same rewrite as dual_pass_fusion/build_perturb_conv_fixture.py
# (exp11), applied to the whole network. Block 0 is left as is: its RequantizedConv is bound to FlatConv at codegen
# (DEEPLOY_RQCONV_FLAT=1), and it is then the only RequantizedConv left in the graph. inputs/outputs.npz unchanged.
#   /opt/anaconda3/bin/python3.11 build_ctrl_fixture.py
import math
import os
import shutil

import onnx
from onnx import helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "fixtures", "speechnet_train_QZOECO_full_fused_round1")
DST = os.path.join(HERE, "..", "fixtures", "speechnet_train_QZOECO_ctrl_round1")


def _attr(node, name):
    return next(a for a in node.attribute if a.name == name)


def main():
    model = onnx.load(os.path.join(SRC, "network.onnx"))
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    ints = lambda n, k: list(helper.get_attribute_value(_attr(n, k)))
    new, drop = {}, set()
    for conv in [n for n in g.node if n.op_type == "Conv"]:
        pw = prod[conv.input[1]]
        blk = pw.input[0].split(".")[1]
        if blk == "0":
            continue
        (rqs,) = cons[conv.output[0]]
        pb = prod[rqs.input[2]]
        assert rqs.op_type == "RequantShift" and pw.op_type == pb.op_type == "RQSPerturbRademacher"
        assert len(cons[pw.output[0]]) == 1 and len(cons[pb.output[0]]) == 1
        div = int(numpy_helper.to_array(_attr(rqs, "div").t).flat[0])
        node = helper.make_node(
            "RequantizedDualConv", inputs = [conv.input[0], pw.input[0], rqs.input[1], pb.input[0], pw.input[1],
                                             pb.input[1]],
            outputs = list(rqs.output), name = f"perturbconv_blocks{blk}", domain = "mezo",
            kernel_shape = ints(conv, "kernel_shape"), pads = ints(conv, "pads"), strides = ints(conv, "strides"),
            dilations = ints(conv, "dilations"), group = helper.get_attribute_value(_attr(conv, "group")),
            shift = int(math.log2(div)), seed_w = helper.get_attribute_value(_attr(pw, "seed")),
            idx_w = helper.get_attribute_value(_attr(pw, "idx")), div_w = helper.get_attribute_value(_attr(pw, "div")),
            seed_b = helper.get_attribute_value(_attr(pb, "seed")), idx_b = helper.get_attribute_value(_attr(pb, "idx")),
            div_b = helper.get_attribute_value(_attr(pb, "div")), weight_chw = 1, single_pass = 1)
        for k in ("div", "n_levels", "signed"):  # RequantShift's TENSOR-typed protos, verbatim
            node.attribute.append(_attr(rqs, k))
        new[id(rqs)] = node
        drop |= {id(pw), id(pb), id(conv)}
        print(f"block {blk}: {pw.name} + {pb.name} + {conv.name} + {rqs.name} -> {node.name}")
    assert len(new) == 4
    nodes = [new.get(id(n), n) for n in g.node if id(n) not in drop]   # keep topological order (node at rqs's slot)
    del g.node[:]
    g.node.extend(nodes)
    if not any(o.domain == "mezo" for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid("mezo", 1))
    os.makedirs(DST, exist_ok = True)
    onnx.save(model, os.path.join(DST, "network.onnx"))
    for f in ("inputs.npz", "outputs.npz"):
        shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))
    print("ops:", sorted({n.op_type for n in g.node}), "->", DST)


if __name__ == "__main__":
    main()
