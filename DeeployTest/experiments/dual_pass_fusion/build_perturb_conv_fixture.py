# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Build QZO_PerturbConv_B<i> -- QW (dual_pass_fusion exp11): the unfused single-block fixture QZO_Conv_B<i> with its
# four nodes  RQSPerturbRademacher(W) + RQSPerturbRademacher(b) + Conv + RequantShift  replaced by ONE
# `RequantizedDualConv` with `single_pass` = 1 (PULP_PerturbConv4x2_i8_i8_i8_HWC: perturbation in the kernel, one pass).
# Same graph inputs, same initializers, same inputs.npz / outputs.npz -- the node must compute exactly what the four
# nodes compute (the + pass: perturbation_sign = 1 in the kernel harness). Attributes as ZODualPassFusion builds them.
#
#   python3 build_perturb_conv_fixture.py --block 1        (plain onnx + numpy; no oracle needed)
import argparse
import math
import os
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.abspath(os.path.join(HERE, "..", "..", "Tests", "Kernels", "Integer"))


def _attr(node, name):
    return next(a for a in node.attribute if a.name == name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type = int, required = True)
    ap.add_argument("--minus", action = "store_true",
                    help = "the - pass: input x- and reference A- from QZO_DualConv2in_B<i>, attr force_sign = 0")
    args = ap.parse_args()
    src = os.path.join(KERNELS, f"QZO_Conv_B{args.block}")
    dst = os.path.join(KERNELS, f"QZO_PerturbConv{'M' if args.minus else ''}_B{args.block}")
    model = onnx.load(os.path.join(src, "network.onnx"))
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    conv = next(n for n in g.node if n.op_type == "Conv")
    rqs = next(n for n in g.node if n.op_type == "RequantShift")
    pw, pb = prod[conv.input[1]], prod[rqs.input[2]]
    assert pw.op_type == pb.op_type == "RQSPerturbRademacher"
    div = int(numpy_helper.to_array(_attr(rqs, "div").t).flat[0])
    ints = lambda n, k: list(helper.get_attribute_value(_attr(n, k)))
    node = helper.make_node(
        "RequantizedDualConv", inputs = [conv.input[0], pw.input[0], rqs.input[1], pb.input[0], pw.input[1], pb.input[1]],
        outputs = list(rqs.output), name = conv.name + "_perturbconv", domain = "mezo",
        kernel_shape = ints(conv, "kernel_shape"), pads = ints(conv, "pads"), strides = ints(conv, "strides"),
        dilations = ints(conv, "dilations"), group = helper.get_attribute_value(_attr(conv, "group")),
        shift = int(math.log2(div)), seed_w = helper.get_attribute_value(_attr(pw, "seed")),
        idx_w = helper.get_attribute_value(_attr(pw, "idx")), div_w = helper.get_attribute_value(_attr(pw, "div")),
        seed_b = helper.get_attribute_value(_attr(pb, "seed")), idx_b = helper.get_attribute_value(_attr(pb, "idx")),
        div_b = helper.get_attribute_value(_attr(pb, "div")), weight_chw = 1, single_pass = 1,
        **({"force_sign": 0} if args.minus else {}))
    for k in ("div", "n_levels", "signed"):  # RequantShift's TENSOR-typed protos, verbatim
        node.attribute.append(_attr(rqs, k))
    keep = [n for n in g.node if n not in (pw, pb, conv, rqs)]
    del g.node[:]
    g.node.extend(keep + [node])
    used = {i for n in g.node for i in n.input}
    for vi in [v for v in g.value_info if v.name not in used and v.name not in {o.name for o in g.output}]:
        g.value_info.remove(vi)
    os.makedirs(dst, exist_ok = True)
    onnx.save(model, os.path.join(dst, "network.onnx"))
    if not args.minus:
        for f in ("inputs.npz", "outputs.npz"):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    else:  # x- and A- (channels Co..2Co-1 of the fused reference [A+ | A-]) under the unfused fixture's names
        dual = os.path.join(KERNELS, f"QZO_DualConv2in_B{args.block}")
        din, dout = np.load(os.path.join(dual, "inputs.npz")), np.load(os.path.join(dual, "outputs.npz"))
        uin, uout = np.load(os.path.join(src, "inputs.npz")), np.load(os.path.join(src, "outputs.npz"))
        xname = conv.input[0]
        ins = {k: (din[xname + "_minus"] if k == xname else uin[k]) for k in uin.keys()}
        (oname, oref), = uout.items()
        (_, dref), = dout.items()
        co = oref.shape[1]
        assert np.array_equal(dref[:, :co], oref), "fused A+ != unfused reference (fixtures out of sync)"
        np.savez(os.path.join(dst, "inputs.npz"), **ins)
        np.savez(os.path.join(dst, "outputs.npz"), **{oname: dref[:, co:]})
    print(f"{dst}: {[n.op_type for n in g.node]}")


if __name__ == "__main__":
    main()
