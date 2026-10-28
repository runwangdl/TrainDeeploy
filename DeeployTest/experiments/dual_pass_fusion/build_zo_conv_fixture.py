# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the ZO_Conv* prototype fixtures -- QW (dual-pass-fusion).
#
# Each fixture is one SpeechNet block's convolution lifted VERBATIM out of the float-ZO
# training graph, together with the two PerturbRademacher nodes that feed it:
#
#     <block>_weight --PerturbRademacher(idx=W)--> pert --.
#     <block>_bias   --PerturbRademacher(idx=B)--> pert ---+--> Conv --> out
#     <activation> --------------------------------------'
#
# Nothing is re-authored: node names, `idx`/`seed`/`eps` attributes, the Conv attributes and
# the perturb outputs' ValueInfo are copied from the source graph, so Deeploy parses, binds
# and tiles these three nodes exactly as it does inside the full ZO network -- including the
# z-stream, which is keyed on the `idx` ATTRIBUTE rather than on schedule position.
#
# Single step, one window, n_accum = 1: the npz carries exactly one input and no mb<N>_ variants.
#
#   block 0 -> Tests/Kernels/FP32/ZO_Conv          data input = the raw EMG window (a graph input)
#   block 1 -> Tests/Kernels/FP32/ZO_Conv_Block1   data input = block 0's MaxPool output, which is
#                                                  an INTERNAL tensor in the full graph, so it is
#                                                  promoted to a graph input and its value is
#                                                  produced here by running block 0 on the host
#                                                  (conv -> frozen BN -> ReLU -> MaxPool, +eps).
#
# Usage (from DeeployTest/):
#     python3 experiments/dual_pass_fusion/build_zo_conv_fixture.py --block 0
#     python3 experiments/dual_pass_fusion/build_zo_conv_fixture.py --block 1

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zo_host_ref import batchnorm_frozen, conv2d_im2col_hwc, maxpool2d, perturb_rademacher, relu  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEEPLOY_TEST = os.path.abspath(os.path.join(HERE, "..", ".."))

# artifacts_reference checkpoint (the paper-matching one). The non-_ref fixture carries the
# superseded artifacts/ weights -- see memory note "SilentWear reference checkpoints".
DEFAULT_SRC = os.path.join(DEEPLOY_TEST,
                           "Tests/Models/Training/SpeechNet/speechnet_zo_train_b1_fold3_ref")

# Per-block description of what to lift. `data_in` is the Conv's first input in the full graph;
# `data_in_is_graph_input` says whether it is already a graph input there.
BLOCKS = {
    0: dict(
        name = "ZO_Conv",
        w_perturb = "perturbrademacher_0_blocks_0_0_weight",
        b_perturb = "perturbrademacher_1_blocks_0_0_bias",
        conv = "node_1_blocks_0_blocks_0_0_Conv",
        weight = "blocks_0_0_weight",
        bias = "blocks_0_0_bias",
        data_in_is_graph_input = True,
    ),
    1: dict(
        name = "ZO_Conv_Block1",
        w_perturb = "perturbrademacher_6_blocks_1_0_weight",
        b_perturb = "perturbrademacher_7_blocks_1_0_bias",
        conv = "node_5_blocks_1_blocks_1_0_Conv",
        weight = "blocks_1_0_weight",
        bias = "blocks_1_0_bias",
        data_in_is_graph_input = False,
    ),
}

SRC_WINDOW = "arr_0000"   # arr_0001 is the label (dropped: these fixtures have no loss)


def _attrs(node):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _conv_out_shape(x_shape, w_shape, pads, strides):
    h = (x_shape[2] + pads[0] + pads[2] - w_shape[2]) // strides[0] + 1
    w = (x_shape[3] + pads[1] + pads[3] - w_shape[3]) // strides[1] + 1
    return [x_shape[0], w_shape[0], h, w]


def block0_activation(src_graph, nodes_by_name, npz, verbose = True):
    """Run block 0's forward on the host to get the tensor that feeds block 1's conv.

    conv -> BatchNormInternal(frozen) -> ReLU -> MaxPool, all at +eps, i.e. exactly what the
    device computes in the +eps pass of a ZO step.

    NOTE this only has to be *plausible-real* data, not bit-exact: whatever value lands in
    inputs.npz is what BOTH the device and the reference consume, so the fixture is self
    consistent either way. Running the real chain just means the kernel sees realistic
    activations (dynamic range, sparsity after ReLU) rather than noise.
    """
    inits = {t.name: onnx.numpy_helper.to_array(t) for t in src_graph.initializer}

    w = npz["arr_0002"]                                   # blocks_0_0_weight
    b = npz["arr_0003"]                                   # blocks_0_0_bias
    gamma = npz["arr_0004"]                               # blocks_0_1_weight (BN gamma)
    beta = npz["arr_0005"]                                # blocks_0_1_bias   (BN beta)

    pw = _attrs(nodes_by_name["perturbrademacher_0_blocks_0_0_weight"])
    pb = _attrs(nodes_by_name["perturbrademacher_1_blocks_0_0_bias"])
    pg = _attrs(nodes_by_name["perturbrademacher_2_blocks_0_1_weight"])
    pbe = _attrs(nodes_by_name["perturbrademacher_3_blocks_0_1_bias"])
    conv_a = _attrs(nodes_by_name["node_1_blocks_0_blocks_0_0_Conv"])
    bn_a = _attrs(nodes_by_name["node_2_blocks_0_blocks_0_1_BatchNormalization"])
    pool_a = _attrs(nodes_by_name["node_4_blocks_0_blocks_0_3_MaxPool"])

    x = npz[SRC_WINDOW].astype(np.float32)
    y = conv2d_im2col_hwc(x,
                          perturb_rademacher(w, pw["seed"], pw["idx"], pw["eps"], direction = 1),
                          perturb_rademacher(b, pb["seed"], pb["idx"], pb["eps"], direction = 1),
                          conv_a["pads"], conv_a["strides"])
    y = batchnorm_frozen(y,
                         perturb_rademacher(gamma, pg["seed"], pg["idx"], pg["eps"], direction = 1),
                         perturb_rademacher(beta, pbe["seed"], pbe["idx"], pbe["eps"], direction = 1),
                         inits["blocks_0_1_running_mean"], inits["blocks_0_1_running_var"],
                         bn_a["epsilon"])
    y = relu(y)
    y = maxpool2d(y, pool_a["kernel_shape"], pool_a["strides"])
    if verbose:
        nz = 100.0 * (y != 0).mean()
        print(f"  block-0 activation: {y.shape} range [{y.min():.4f}, {y.max():.4f}] "
              f"nonzero {nz:.1f}% (post-ReLU)")
    return y


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", type = int, default = 0, choices = sorted(BLOCKS),
                    help = "which SpeechNet block's conv to lift")
    ap.add_argument("--src", default = DEFAULT_SRC, help = "source ZO training fixture dir")
    ap.add_argument("--dst", default = None, help = "output fixture dir (default: Tests/Kernels/FP32/<name>)")
    ap.add_argument("--window", type = int, default = 0,
                    help = "which mini-batch window to bake in (0 = the base arr_* entry)")
    args = ap.parse_args()

    cfg = BLOCKS[args.block]
    dst = args.dst or os.path.join(DEEPLOY_TEST, "Tests/Kernels/FP32", cfg["name"])

    src_model = onnx.load(os.path.join(args.src, "network.onnx"))
    src_graph = src_model.graph
    src_npz = np.load(os.path.join(args.src, "inputs.npz"))

    nodes_by_name = {n.name: n for n in src_graph.node}
    for key in ("w_perturb", "b_perturb", "conv"):
        assert cfg[key] in nodes_by_name, f"{cfg[key]} not in {args.src}/network.onnx"
    nodes = [nodes_by_name[cfg[k]] for k in ("w_perturb", "b_perturb", "conv")]
    conv_node = nodes[2]
    conv_attrs = _attrs(conv_node)

    src_inputs = {vi.name: vi for vi in src_graph.input}
    src_value_info = {vi.name: vi for vi in src_graph.value_info}

    # --- the Conv's data input --------------------------------------------------------
    data_in_name = conv_node.input[0]
    if cfg["data_in_is_graph_input"]:
        data_in_vi = src_inputs[data_in_name]
        x = src_npz[SRC_WINDOW if args.window == 0 else f"mb{args.window}_{SRC_WINDOW}"].astype(np.float32)
    else:
        # Internal in the full graph -> promote its ValueInfo to a graph input and compute it.
        data_in_vi = src_value_info[data_in_name]
        x = block0_activation(src_graph, nodes_by_name, src_npz)

    x_shape = [d.dim_value for d in data_in_vi.type.tensor_type.shape.dim]
    assert list(x.shape) == x_shape, f"{x.shape} != declared {x_shape}"

    # --- graph interface --------------------------------------------------------------
    graph_inputs = [data_in_vi, src_inputs[cfg["weight"]], src_inputs[cfg["bias"]]]
    w_shape = [d.dim_value for d in src_inputs[cfg["weight"]].type.tensor_type.shape.dim]
    out_shape = _conv_out_shape(x_shape, w_shape, conv_attrs["pads"], conv_attrs["strides"])
    conv_out_name = conv_node.output[0]
    graph_output = helper.make_tensor_value_info(conv_out_name, onnx.TensorProto.FLOAT, out_shape)

    # The perturb outputs must keep their ValueInfo: ZO puts the conv weight on a graph EDGE
    # rather than in an initializer, and Deeploy's NCHW->NHWC lowering reads
    # `node.inputs[1].shape` to count spatial dims
    # (Deeploy/.../LoweringOptimizationPasses.py:267). Drop the value_info and that pass crashes
    # on a shapeless Variable.
    value_info = [src_value_info[n.output[0]] for n in nodes[:2]]

    graph = helper.make_graph(nodes, cfg["name"], graph_inputs, [graph_output],
                              initializer = [], value_info = value_info)
    model = helper.make_model(graph, opset_imports = list(src_model.opset_import))
    model.ir_version = src_model.ir_version
    model.producer_name = "build_zo_conv_fixture.py"

    # --- host reference ---------------------------------------------------------------
    # Defaults of ZORuntime.c: perturbation_sign = 1, i.e. the +eps pass of the ZO step.
    w = src_npz[{0: "arr_0002", 1: "arr_0006"}[args.block]].astype(np.float32)
    b = src_npz[{0: "arr_0003", 1: "arr_0007"}[args.block]].astype(np.float32)
    assert list(w.shape) == w_shape
    pw, pb = _attrs(nodes[0]), _attrs(nodes[1])
    w_pert = perturb_rademacher(w, pw["seed"], pw["idx"], pw["eps"], direction = 1)
    b_pert = perturb_rademacher(b, pb["seed"], pb["idx"], pb["eps"], direction = 1)
    y = conv2d_im2col_hwc(x, w_pert, b_pert, conv_attrs["pads"], conv_attrs["strides"])
    assert list(y.shape) == out_shape

    # --- write ------------------------------------------------------------------------
    # generateNetwork.py / testMVP.py consume inputs.npz/outputs.npz POSITIONALLY
    # (`inputs.files` order zipped onto the graph's inputs), so the write order below is
    # load-bearing; the key names are documentation only.
    os.makedirs(dst, exist_ok = True)
    onnx.save(model, os.path.join(dst, "network.onnx"))
    np.savez(os.path.join(dst, "inputs.npz"),
             **{data_in_name: x, cfg["weight"]: w, cfg["bias"]: b})
    np.savez(os.path.join(dst, "outputs.npz"), **{conv_out_name: y})

    macs = int(np.prod(out_shape)) * w_shape[1] * w_shape[2] * w_shape[3]
    print(f"[{cfg['name']}] wrote {dst}")
    print(f"  nodes       : {[f'{n.op_type}:{n.name}' for n in nodes]}")
    print(f"  value_info  : {[vi.name for vi in value_info]}")
    print(f"  data_in     : {data_in_name} {tuple(x.shape)}"
          f"{'' if cfg['data_in_is_graph_input'] else '  (promoted from an internal tensor)'}")
    print(f"  weight/bias : {tuple(w.shape)} / {tuple(b.shape)}  "
          f"(perturbed on-device, idx {pw['idx']} / {pb['idx']}, seed {pw['seed']}, eps {pw['eps']})")
    print(f"  conv attrs  : kernel {conv_attrs['kernel_shape']} pads {conv_attrs['pads']} "
          f"strides {conv_attrs['strides']} group {conv_attrs['group']}")
    print(f"  output      : {conv_out_name} {tuple(y.shape)}  range [{y.min():.6f}, {y.max():.6f}]")
    print(f"  MACs        : {macs:,}")
    print(f"  |w_pert-w|  : {np.abs(w_pert - w).max():.6f} (expect eps)")


if __name__ == "__main__":
    main()
