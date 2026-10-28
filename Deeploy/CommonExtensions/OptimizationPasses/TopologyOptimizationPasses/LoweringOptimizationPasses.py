# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import List, Optional, Sequence, Tuple, TypeVar, Union

import os
import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import Pass, ReplaceSequentialPatternPass, \
    SequentialPass, contextagnostic
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle


def _singleNodePattern(op: str) -> gs.Graph:
    tensorIn = gs.Variable("input")
    tensorOut = gs.Variable("output")
    node = gs.Node(op = op, name = "node", inputs = [tensorIn], outputs = [tensorOut])
    graph = gs.Graph([node], inputs = [tensorIn], outputs = [tensorOut])
    return graph


def _isDepthwise(node: gs.Node) -> bool:
    if node.op not in ["Conv", "RequantizedConv"]:
        return False

    channels_first = node.attrs.get("channels_first", True)
    spatialDims = len(node.inputs[1].shape) - 2
    shapeIn = node.inputs[0].shape
    chIn = shapeIn[-spatialDims - 1] if channels_first else shapeIn[-1]
    return chIn != 1 and node.attrs.get("group", 1) == chIn


def _createReshape(tensorIn: gs.Tensor,
                   name: str,
                   newShape: Sequence[Union[int, str]],
                   tensorOut: Optional[gs.Tensor] = None) -> Tuple[gs.Node, gs.Tensor]:
    newShapeConst = gs.Constant(name = name + tensorIn.name + "_NewShape", values = np.array(newShape))

    if tensorOut is None:
        tensorOut = gs.Variable(name = name + tensorIn.name + "_Reshaped", dtype = np.float32, shape = newShape)
    else:
        assert newShape == tensorOut.shape

    reshapeNode = gs.Node(name = name + tensorIn.name + "_Reshape",
                          op = "Reshape",
                          inputs = [tensorIn, newShapeConst],
                          outputs = [tensorOut])

    return reshapeNode, tensorOut


def _appendExpandDims(tensor: gs.Tensor, name: str, axis: Union[int, Sequence[int]]) -> Tuple[gs.Node, gs.Tensor]:
    if isinstance(axis, int):
        axes = [axis]
    elif isinstance(axis, tuple):
        axes = list(axis)
    elif isinstance(axis, list):
        axes = axis
    else:
        assert False, f"axis should be of type int or tuple. Got {type(axis)}"

    axes = [(len(tensor.shape) + len(axes) + axis if axis < 0 else axis) for axis in axes]
    assert all(axis >= 0 for axis in axes)

    assert isinstance(tensor.shape, Sequence) and len(tensor.shape) > 0 and isinstance(tensor.shape[0], int)
    assert all(axis < len(tensor.shape) + len(axes) for axis in axes), f"axis out of bounds. axis: {axes}"

    newShape = np.zeros(shape = (len(tensor.shape) + len(axes),), dtype = np.int_)
    for axis in axes:
        newShape[axis] = 1
    newShape[newShape == 0] = tensor.shape

    return _createReshape(tensor, name, newShape.tolist())


def _prependSqueezeDims(tensor: gs.Tensor, name: str, axis: Union[int, Sequence[int]]) -> Tuple[gs.Node, gs.Tensor]:
    if isinstance(axis, int):
        axes = [axis]
    elif isinstance(axis, tuple):
        axes = list(axis)
    elif isinstance(axis, list):
        axes = axis
    else:
        assert False, f"axis should be of type int or tuple. Got {type(axis)}"

    axes = [(len(tensor.shape) + axis if axis < 0 else axis) for axis in axes]
    assert all(axis >= 0 for axis in axes)

    assert isinstance(tensor.shape, Sequence) and len(tensor.shape) > 0 and isinstance(tensor.shape[0], int)
    assert all(axis < len(tensor.shape) + len(axes) for axis in axes), f"axis out of bounds. axis: {axes}"

    oldShape = np.zeros(shape = (len(tensor.shape) + len(axes),), dtype = np.int_)
    for axis in axes:
        oldShape[axis] = 1
    oldShape[oldShape == 0] = tensor.shape

    inputTensor = gs.Variable(name = name + tensor.name + "_Expanded", dtype = np.float32, shape = oldShape.tolist())

    reshapeNode, _ = _createReshape(inputTensor, name, tensor.shape, tensor)

    return reshapeNode, inputTensor


# Permute (0,1,2,3,...,N-2,N-1) -> (0,1,2,3,...,N-1,N-2)
def _swapLastTwoDimsPermutation(N: int) -> List[int]:
    assert N >= 2, "N needs to be larger then 2"
    return [*range(N - 2), N - 1, N - 2]


# Permute channels first <-> channels last:
#   (*<batch dims>, ch, *<spatial dims>) <-> (*<batch dims>, *<spatial dims>, ch)
def _transformLayoutPermutation(dims: int, spatialDims: int, targetChannelsFirst: bool) -> List[int]:
    batchDims = dims - spatialDims - 1
    if targetChannelsFirst:
        ch = dims - 1
        nonBatchPerm = [ch, *range(batchDims, ch)]
    else:
        ch = batchDims
        nonBatchPerm = [*range(ch + 1, dims), ch]
    return list(range(batchDims)) + nonBatchPerm


# Calculate permutation q = p^(-1) s.t. q(p(i)) = i
def _invertPermutation(permutation: Sequence[int]) -> List[int]:
    return [permutation.index(i) for i in range(len(permutation))]


T = TypeVar('T')


def _permute(_list: Sequence[T], permutation: Sequence[int]) -> List[T]:
    assert len(_list) == len(permutation), "Permuted list and permutation must have equal length!"
    return [_list[i] for i in permutation]


def _permuteHyperRectangle(rect: HyperRectangle, permutation: List[int]) -> HyperRectangle:
    assert len(rect.dims) == len(permutation), "Permutation list and HyperRectangle must have equal dimensionality!"
    return HyperRectangle(tuple(_permute(rect.offset, permutation)), tuple(_permute(rect.dims, permutation)))


def _prependTranspose(tensor: gs.Variable, prevNode: gs.Node, perm: List[int]) -> gs.Node:
    prevNodeTensorIdx = prevNode.outputs.index(tensor)
    preTransposeTensor = gs.Variable(f"{prevNode.name}_{tensor.name}_pre_transposed", tensor.dtype,
                                     _permute(tensor.shape, _invertPermutation(perm)))
    transposeNode = gs.Node(op = "Transpose",
                            name = f"{prevNode.name}_{tensor.name}_pre_transpose",
                            attrs = {"perm": perm},
                            inputs = [preTransposeTensor],
                            outputs = [tensor])
    prevNode.outputs[prevNodeTensorIdx] = preTransposeTensor
    return transposeNode


def _appendTranspose(tensor: gs.Variable, nextNode: gs.Node, perm: List[int]) -> gs.Node:
    nextNodeTensorIdx = nextNode.inputs.index(tensor)
    transposedTensor = gs.Variable(f"{nextNode.name}_{tensor.name}_transposed", tensor.dtype,
                                   _permute(tensor.shape, perm))
    transposeNode = gs.Node(op = "Transpose",
                            name = f"{nextNode.name}_{tensor.name}_transpose",
                            attrs = {"perm": perm},
                            inputs = [tensor],
                            outputs = [transposedTensor])
    nextNode.inputs[nextNodeTensorIdx] = transposedTensor
    return transposeNode


def _transformLayoutConst(const: gs.Constant, spatialDims: int, targetChannelsFirst: bool) -> None:
    assert isinstance(const, gs.Constant)
    if len(const.shape) < 2:
        return
    perm = _transformLayoutPermutation(len(const.shape), spatialDims, targetChannelsFirst)
    const.values = const.values.transpose(perm)


def _transformLayoutDwWeightConst(const: gs.Constant, targetChannelsFirst: bool) -> None:
    assert not targetChannelsFirst, "Target layout should be channels_last!"
    assert isinstance(const, gs.Constant)
    dims = len(const.shape)
    perm = [*range(1, dims), 0]
    const.values = const.values.transpose(perm)


def _foldLayoutIntoPerturb(weightVar: gs.Variable, spatialDims: int, targetChannelsFirst: bool) -> bool:  # -- QW
    """ZO (MeZO) weights are produced at runtime by a Perturb node, so the NCHW->NHWC layout change
    can't be folded into a constant. Rather than emit a runtime Transpose (which materialises a SECOND
    full-size weight buffer next to the Perturb output — the source of the ~100x slowdown), push the
    permutation into the Perturb node's CONSTANT base weight at compile time and relabel the Perturb
    output's shape. Layout-preserving: perturb is elementwise (out shape tracks in shape) and the conv
    permutation keeps axis 0 (out channels) fixed. Ported from Deeploy zo-support. Returns True iff
    folded; callers fall back to a runtime Transpose otherwise. -- QW
    """  # -- QW
    if weightVar.shape is None or len(weightVar.shape) < 2:  # -- QW
        return False  # -- QW
    if len(weightVar.inputs) != 1:  # -- QW
        return False  # -- QW
    perturbNode = weightVar.inputs[0]  # -- QW
    if "Perturb" not in perturbNode.op:  # -- QW
        return False  # -- QW
    baseWeight = perturbNode.inputs[0]  # -- QW  data_in: the constant base weight
    if not isinstance(baseWeight, gs.Constant) or list(baseWeight.shape) != list(weightVar.shape):  # -- QW
        return False  # -- QW
    perm = _transformLayoutPermutation(len(weightVar.shape), spatialDims, targetChannelsFirst)  # -- QW
    baseWeight.values = baseWeight.values.transpose(perm)  # -- QW
    weightVar.shape = _permute(weightVar.shape, perm)  # -- QW
    return True  # -- QW


def _transposeMatMulInputs_fun(graph: gs.Graph, match: Match, name: str):
    node = next(iter((match.nodes_map.values())))

    node.attrs['transA'] = node.attrs.get('transA', 0)
    node.attrs['transB'] = node.attrs.get('transB', 0)
    node.attrs['alpha'] = node.attrs.get('alpha', 1.0)
    node.attrs['beta'] = node.attrs.get('beta', 1.0)

    # Prepend transpose on A if it's transposed
    if node.attrs['transA'] == 1:
        tensorA = node.inputs[0]
        perm = _swapLastTwoDimsPermutation(len(tensorA.shape))
        graph.nodes.append(_appendTranspose(tensorA, node, perm))
        node.attrs['transA'] = False

    # Prepend transpose on B if it's not transposed
    if node.attrs['transB'] == 0:
        tensorB = node.inputs[1]
        perm = _swapLastTwoDimsPermutation(len(tensorB.shape))
        graph.nodes.append(_appendTranspose(tensorB, node, perm))
        node.attrs['transB'] = True

    return graph


# SCHEREMO:
# Implements generation of tranpose nodes such that each GEMM/MatMul node implements attributes transA = 0 transB = 1
@contextagnostic
class TransposeMatmulInputsPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = _singleNodePattern("RequantizedGemm")
        name = "_TRANSPOSE_MATMUL_INPUTS_PASS"
        super().__init__(graph, _transposeMatMulInputs_fun, name)


def _NCHWtoNHWC_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):
    node = next(iter((match.nodes_map.values())))

    # RW: nodes tagged keep_channels_first (by PULPConvKeepCHWPass for the GAP9
    # channels-first conv path) must NOT be transposed — they run channels-first
    # natively. The tag is only ever set by that GAP9 pass, so this is inert for
    # every other target/network.
    if node.attrs.get("keep_channels_first", False):
        return graph

    channels_first = node.attrs.get("channels_first", True)
    if (channels_first != default_channels_first):
        tensorIn = node.inputs[0]
        tensorOut = node.outputs[0]

        if node.op in ["RequantizedConv", "Conv", "RequantizedDualConv"]:  # QW exp5: +RequantizedDualConv -- QW
            # QW (exp16a): this infers the SPATIAL rank from the WEIGHT's rank, which assumes the
            #     weight is still the logical NCHW [cout, cin, *spatial] tensor. That holds for
            #     every ordinary conv, and for NE16 convs whose weight is encoded LATER by
            #     NE16AdjustWeightMemoryLayoutPass. It does NOT hold when the weight arrives
            #     ALREADY bit-serial-encoded (rank 3) as a runtime graph input: spatialDims then
            #     comes out as 1 and this pass builds a 1-D permutation [0,1,3,2] for a 4-D
            #     activation, silently swapping H and W. Take the spatial rank from kernel_shape
            #     in that case -- for any ordinary conv the two agree, so nothing else changes.
            if "ne16_weight_preencoded" in node.attrs and "kernel_shape" in node.attrs:  # -- QW
                spatialDims = len(node.attrs["kernel_shape"])  # -- QW
            else:
                spatialDims = len(node.inputs[1].shape) - 2
        elif node.op in [
                "MaxPool", "MaxPoolGrad", "MaxPoolArgmax", "MaxPoolGradMask", "AveragePool", "AveragePoolGrad"
        ]:  # QW: +MaxPoolArgmax/MaxPoolGradMask (Part-4) -- QW
            spatialDims = len(node.attrs["kernel_shape"])
        elif node.op in ["Quant", "Dequant"]:
            # QW: Quant/Dequant are ELEMENTWISE — NHWC is a pure layout permutation of the activation.
            #     Making them NHWC (like their MaxPool/RequantizedConv neighbours) lets the transpose-merge
            #     passes CANCEL the round-trip transposes (NHWC→NCHW→Quant→NCHW→NHWC) that otherwise
            #     scramble C>1 inter-block quant on device (block-0 C=1 hid it; block-1 C=8 exposed it). -- QW
            spatialDims = len(tensorIn.shape) - 2
        elif node.op == "BatchNormInternal":  # RW 2026-10-13: frozen-stat BN, elementwise per channel
            spatialDims = len(tensorIn.shape) - 2
            node.attrs["bn_nhwc"] = 1
        elif node.op == "Pad":
            spatialDims = 2  # Hack based on current status
        else:
            raise ValueError(f"Cannot determine spatialDims for node {node.name} with operator {node.op}")

        # Insert Transpose nodes around the op's activation input/output to convert the data layout.
        permuteIn = _transformLayoutPermutation(len(tensorIn.shape), spatialDims, default_channels_first)
        graph.nodes.append(_appendTranspose(tensorIn, node, permuteIn))

        permuteOut = _transformLayoutPermutation(len(tensorOut.shape), spatialDims, channels_first)
        graph.nodes.append(_prependTranspose(tensorOut, node, permuteOut))

        _maxPoolGradXDone = False  # RW merge: devel's shared-transpose path below owns MaxPoolGrad's X
        if node.op == "MaxPoolGrad" and len(node.inputs) > 1 and node.inputs[1].shape is not None \
                and len(node.inputs[1].shape) == len(tensorIn.shape):
            _maxPoolGradXDone = True
            # MaxPoolGrad(dY, X): the kernel reads X in the same layout as dY. X is the
            # forward pool's input, which that pool already transposed; read that copy
            # rather than transposing the (large) activation a second time.
            x = node.inputs[1]
            permX = _transformLayoutPermutation(len(x.shape), spatialDims, default_channels_first)
            shared = next((n for n in graph.nodes if n.op == "Transpose" and n.inputs and n.inputs[0] is x
                           and list(n.attrs.get("perm", [])) == list(permX) and n.outputs), None)
            if shared is not None:
                node.inputs[1] = shared.outputs[0]
            else:
                graph.nodes.append(_appendTranspose(x, node, permX))

        if node.op in ["Conv", "RequantizedConv", "RequantizedDualConv"]:  # QW exp5: +RequantizedDualConv (inputs[1:] = weight, mul, add, wmul, bmul) -- QW
            # QW (exp16a): a PRE-ENCODED NE16 weight is already in the accelerator's bit-serial
            #     layout -- it is NOT a logical [cout, cin, *spatial] tensor and must never be
            #     layout-permuted. Skip input[1] for such nodes; mul/add/shift still get handled.
            # _skipWeight = "ne16_weight_preencoded" in node.attrs  # -- QW (exp16a original)
            # QW (dual_pass_fusion exp7): the 7-input RequantizedDualConv takes its weight in RAW CHW order
            #     (attr weight_chw) and perturbs it in that order inside the kernel -- leave it untransposed.
            #     Its x- (input 6, rank 4) needs no special case: the loop below transposes it with the 4-D
            #     NCHW->NHWC permutation, which is the activation's. -- QW
            _skipWeight = "ne16_weight_preencoded" in node.attrs or bool(int(node.attrs.get("weight_chw", 0)))  # -- QW
            # In the case of Conv: [weights, opt. bias], RequantizedConv: [weights, mul, add, opt. shift]
            for _tIdx, tensor in enumerate(node.inputs[1:]):
                if _skipWeight and _tIdx == 0:  # -- QW
                    continue  # -- QW
                if isinstance(tensor, gs.Constant):
                    # Inference graph: weight is a fixed constant — permute its data in-place.
                    _transformLayoutConst(tensor, spatialDims, default_channels_first)
                elif isinstance(tensor, gs.Variable) and tensor.shape is not None and len(tensor.shape) >= 2:
                    # ZO (MeZO): weight is a Perturb-node output. Use a runtime Transpose (exactly like the
                    # BP/training path), NOT _foldLayoutIntoPerturb. Folding the NCHW->NHWC permutation into
                    # the perturb's constant base weight makes the perturb kernel generate its Rademacher z
                    # over the TRANSPOSED flat buffer, so for in_ch>1 conv weights z lands on different
                    # elements than the reference (which perturbs in logical NCHW order) -> device perturbed
                    # weight != reference -> wrong loss (verified: conv1-4 diverge, conv0 in_ch=1 does not).
                    # The perturb MUST run in logical order to match the reference; the Transpose then
                    # converts the already-perturbed weight to NHWC for the conv. (The fold's "~100x slowdown"
                    # rationale was a misdiagnosis — that was the FP-on-FPU-less-FC illegal-instr trap, now
                    # fixed in deeploymezotest.c.) -- QW
                    perm = _transformLayoutPermutation(len(tensor.shape), spatialDims, default_channels_first)
                    graph.nodes.append(_appendTranspose(tensor, node, perm))

        # QW: transpose the 2nd activation input to NHWC like the primary input.
        #     MaxPoolGrad: inputs[1] = forward X (recompute source); MaxPoolGradMask (Part-4):
        #     inputs[1] = the argmax mask (pooled shape). Both are 4D activations that must
        #     match the NHWC layout, else the channel-tiler tiles the wrong axis. -- QW
        if node.op == "MaxPoolGradMask" or (node.op == "MaxPoolGrad" and not _maxPoolGradXDone):  # RW merge: no double transpose
            # MaxPoolGrad has a SECOND activation input (inputs[1] = the forward input X,
            # used to recompute the argmax). The HWC kernel reads it in NHWC, so it must be
            # transposed like the primary input. Without this, X stays NCHW while dY/dX are
            # NHWC, so the kernel reads wrong window values and routes the gradient to the
            # wrong positions — a silent correctness bug for non-square / asymmetric pools.
            tensorX = node.inputs[1]
            permuteX = _transformLayoutPermutation(len(tensorX.shape), spatialDims, default_channels_first)
            graph.nodes.append(_appendTranspose(tensorX, node, permuteX))  # -- QW

        node.attrs["channels_first"] = default_channels_first

    return graph


@contextagnostic
class NCHWtoNHWCMaxPoolPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "MaxPool")
        name = "_NCHW_TO_NHWC_MAXPOOL_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCQuantPass(ReplaceSequentialPatternPass):
    # QW: wrap elementwise Quant in the NHWC layout so MaxPool→Quant→RequantizedConv are all NHWC and the
    #     redundant round-trip transposes cancel (fixes C>1 inter-block quant on device). -- QW
    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "Quant")
        name = "_NCHW_TO_NHWC_QUANT_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCDequantPass(ReplaceSequentialPatternPass):  # QW -- QW
    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "Dequant")
        name = "_NCHW_TO_NHWC_DEQUANT_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


def _bnFrozenNHWC_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):
    # RW 2026-10-13: only when the generator was asked for frozen-stat BN (DEEPLOY_BN_FROZEN_NHWC, set by the MeZO
    # runner for -D BN_FROZEN_STATS=ON) and nothing reads the statistics outputs (forward-only graphs; BP graphs
    # feed saved_mean / saved_inv_std to BatchNormalizationGrad and keep NCHW). Then BN is an elementwise
    # per-channel affine map and channels-last lets the transposes around it cancel with the conv's / pool's.
    node = next(iter((match.nodes_map.values())))
    if os.environ.get("DEEPLOY_BN_FROZEN_NHWC", "0") != "1":
        return graph
    if len(node.inputs[0].shape) != 4 or any(len(o.outputs) > 0 for o in node.outputs[1:]):
        return graph
    return _NCHWtoNHWC_fun(graph, match, name, default_channels_first)


def _dequantNHWC_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):
    # RW 2026-10-13: per-tensor Dequant between a channels-last conv and a channels-last BN; 4-D only.
    node = next(iter((match.nodes_map.values())))
    if os.environ.get("DEEPLOY_BN_FROZEN_NHWC", "0") != "1" or len(node.inputs[0].shape) != 4:
        return graph
    if not all(c.op == "BatchNormInternal" for c in node.outputs[0].outputs):
        return graph
    return _NCHWtoNHWC_fun(graph, match, name, default_channels_first)


@contextagnostic
class NCHWtoNHWCBatchNormInternalPass(Pass):
    # A plain node walk: the single-node pattern matcher does not match BatchNormInternal's five outputs.

    def __init__(self, default_channels_first: bool = True):
        super().__init__()
        self.default_channels_first = default_channels_first
        self.name = "_NCHW_TO_NHWC_BATCHNORMINTERNAL_PASS"

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        for node in [n for n in graph.nodes if n.op == "BatchNormInternal"]:
            graph = _bnFrozenNHWC_fun(graph, Match(None, {"bn": node}), self.name, self.default_channels_first)
        graph.cleanup().toposort()
        return graph


@contextagnostic
class NCHWtoNHWCDequantBeforeBNPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "Dequant")
        name = "_NCHW_TO_NHWC_DEQUANT_BEFORE_BN_PASS"
        super().__init__(graph, partial(_dequantNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCAveragePoolPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "AveragePool")
        name = "_NCHW_TO_NHWC_AVERAGEPOOL_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCAveragePoolGradPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "AveragePoolGrad")
        name = "_NCHW_TO_NHWC_AVERAGEPOOLGRAD_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCMaxPoolGradPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "MaxPoolGrad")
        name = "_NCHW_TO_NHWC_MAXPOOLGRAD_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


# QW: Part-4 layout passes for the argmax-mask ops (mirror MaxPool/MaxPoolGrad); without these
# their activations stay NCHW and the HWC channel-tiler tiles the wrong axis. -- QW
@contextagnostic
class NCHWtoNHWCMaxPoolArgmaxPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "MaxPoolArgmax")
        name = "_NCHW_TO_NHWC_MAXPOOLARGMAX_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCMaxPoolGradMaskPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "MaxPoolGradMask")
        name = "_NCHW_TO_NHWC_MAXPOOLGRADMASK_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "Conv|RequantizedConv|RequantizedDualConv")  # QW exp5 -- QW
        name = "_NCHW_TO_NHWC_CONV_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name,
                         NonBranchingMatcher(regex_op = True))


@contextagnostic
class NCHWtoNHWCPadPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "Pad")
        name = "_NCHW_TO_NHWC_PAD_PASS"
        super().__init__(graph, partial(_NCHWtoNHWC_fun, default_channels_first = default_channels_first), name)


def _NCWHtoNHWC_dw_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool) -> gs.Graph:
    node = next(iter((match.nodes_map.values())))

    if not _isDepthwise(node):
        return graph

    channels_first = node.attrs.get("channels_first", True)
    if (channels_first != default_channels_first):
        tensorIn = node.inputs[0]
        tensorOut = node.outputs[0]

        spatialDims = len(node.inputs[1].shape) - 2

        permuteIn = _transformLayoutPermutation(len(tensorIn.shape), spatialDims, default_channels_first)
        permuteOut = _transformLayoutPermutation(len(tensorOut.shape), spatialDims, channels_first)

        graph.nodes.append(_appendTranspose(tensorIn, node, permuteIn))
        graph.nodes.append(_prependTranspose(tensorOut, node, permuteOut))

        _transformLayoutDwWeightConst(node.inputs[1], default_channels_first)  # weights

        if len(node.inputs) > 2:
            # In the case of Conv: [opt. bias], RequantizedConv: [mul, add, opt. shift]
            for tensor in node.inputs[2:]:
                _transformLayoutConst(tensor, spatialDims, default_channels_first)  # bias

        node.attrs["channels_first"] = default_channels_first

    return graph


@contextagnostic
class NCHWtoNHWCDwConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "Conv|RequantizedConv")
        name = "_NCHW_TO_NHWC_DW_CONV_PASS"
        super().__init__(graph, partial(_NCWHtoNHWC_dw_fun, default_channels_first = default_channels_first), name,
                         NonBranchingMatcher(regex_op = True))


def _PULP_NCHWtoNHWC_dw_fun(graph: gs.Graph, match: Match, name: str, default_channels_first: bool = True):
    node = next(iter((match.nodes_map.values())))

    if not _isDepthwise(node):
        return graph

    channels_first = node.attrs.get("channels_first", True)
    if (channels_first != default_channels_first):
        tensorOut = node.outputs[0]

        spatialDims = len(node.inputs[1].shape) - 2

        # LMACAN: PULP DW doesn't transpose the input

        permuteOut = _transformLayoutPermutation(len(tensorOut.shape), spatialDims, channels_first)
        graph.nodes.append(_prependTranspose(tensorOut, node, permuteOut))

        # RequantizedConv: [weights, mul, add, opt. shift]
        for tensor in node.inputs[1:]:
            _transformLayoutConst(tensor, spatialDims, default_channels_first)

        node.attrs["channels_first"] = default_channels_first

    return graph


@contextagnostic
class PULPNCHWtoNHWCDwConvPass(ReplaceSequentialPatternPass):

    def __init__(self, default_channels_first: bool = True):
        graph = _singleNodePattern(op = "RequantizedConv")
        name = "_PULP_NCHW_TO_NHWC_DW_CONV_PASS"
        super().__init__(graph, partial(_PULP_NCHWtoNHWC_dw_fun, default_channels_first = default_channels_first), name)


@contextagnostic
class NCHWtoNHWCPass(SequentialPass):

    def __init__(self, default_channels_first: bool = True):
        passes = [
            NCHWtoNHWCPadPass(default_channels_first),
            NCHWtoNHWCMaxPoolPass(default_channels_first),
            NCHWtoNHWCAveragePoolPass(default_channels_first),
            NCHWtoNHWCAveragePoolGradPass(default_channels_first),
            NCHWtoNHWCMaxPoolGradPass(default_channels_first),
            NCHWtoNHWCDwConvPass(default_channels_first),
            NCHWtoNHWCConvPass(default_channels_first),
        ]
        super().__init__(*passes)


@contextagnostic
class PULPNCHWtoNHWCPass(SequentialPass):

    def __init__(self, default_channels_first: bool = True):
        passes = [
            NCHWtoNHWCPadPass(default_channels_first),
            NCHWtoNHWCMaxPoolPass(default_channels_first),
            NCHWtoNHWCAveragePoolPass(default_channels_first),
            NCHWtoNHWCAveragePoolGradPass(default_channels_first),
            NCHWtoNHWCMaxPoolGradPass(default_channels_first),
            NCHWtoNHWCMaxPoolArgmaxPass(default_channels_first),   # QW: Part-4 -- QW
            NCHWtoNHWCMaxPoolGradMaskPass(default_channels_first),  # QW: Part-4 -- QW
            # QW: NCHWtoNHWCQuantPass DISABLED — cancels the MaxPool→Quant→Conv transposes structurally but
            #     changes the numerics net-worse (L- 10.36→20.89); the transposes (verified correct perms)
            #     were NOT the root cause. Left the pass class defined for future use. -- QW
            PULPNCHWtoNHWCDwConvPass(default_channels_first),
            NCHWtoNHWCConvPass(default_channels_first),
            NCHWtoNHWCDequantBeforeBNPass(default_channels_first),  # RW 2026-10-13: before the BN pass (checks its consumer)
            NCHWtoNHWCBatchNormInternalPass(default_channels_first),  # RW 2026-10-13: frozen-stat BN only
        ]
        super().__init__(*passes)


# QW: Part-4. Merge sibling Transpose nodes (same input tensor + same perm) into a single
# Transpose feeding all consumers. The layout passes create a SEPARATE input transpose for each
# op that reads the same block activation (MaxPool, MaxPoolArgmax, MaxPoolGradMask), and each costs
# a full-size buffer. Deduping them removes the redundant 314 KB-class transpose buffers. Must run
# AFTER the last TransposeSplitPass so it is not re-split. -- QW
@contextagnostic
class MergeSiblingTransposesPass(Pass):

    def __init__(self):
        super().__init__()
        self.name = "_MERGE_SIBLING_TRANSPOSES_PASS"

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        seen: dict = {}
        for node in list(graph.nodes):
            if node.op != "Transpose" or not node.inputs or not node.outputs:
                continue
            key = (node.inputs[0].name, tuple(node.attrs.get("perm", [])))
            if key in seen:
                keepOut = seen[key].outputs[0]
                dupOut = node.outputs[0]
                for consumer in list(dupOut.outputs):
                    consumer.inputs = [keepOut if inp is dupOut else inp for inp in consumer.inputs]
                node.inputs = []
                node.outputs = []
            else:
                seen[key] = node
        graph.cleanup().toposort()
        return graph


def _requantized_gemm_to_pw_fun(graph: gs.Graph, match: Match, name: str):
    node = next(iter((match.nodes_map.values())))

    matrixA: gs.Variable = node.inputs[0]
    matrixB: gs.Constant = node.inputs[1]
    matrixY: gs.Variable = node.outputs[0]

    # Check matrixB is a constant, otherwise don't transform
    if not isinstance(matrixB, gs.Constant):
        return graph

    assert len(matrixA.shape) in [
        2, 3
    ], f"Unsupported number of dimensions for input matrix A of GEMM operation: {len(matrixA.shape)}; shape: {matrixA.shape}"
    assert len(matrixY.shape) in [
        2, 3
    ], f"Unsupported number of dimensions for output matrix of GEMM operation: {len(matrixY.shape)}; shape: {matrixY.shape}"

    # Pointwise with HWC layout (channels_first == False)

    # Defaults
    node.attrs['transA'] = node.attrs.get('transA', 0)
    node.attrs['transB'] = node.attrs.get('transB', 0)
    node.attrs['alpha'] = node.attrs.get('alpha', 1.0)
    node.attrs['beta'] = node.attrs.get('beta', 1.0)

    # If transA is set then the matrix is of shape [B x K x M] and it needs to be transposed, otherwise its shape is  [B x M x K]
    if node.attrs['transA'] == 1:
        perm = _swapLastTwoDimsPermutation(len(matrixA.shape))
        graph.nodes.append(_appendTranspose(matrixA, node, perm))
        matrixA = node.inputs[0]

    # If transB is set then the matrix is of shape [N x K] and it doesn't need to be transposed, otherwise its shape is [K x N] and it has to be transposed
    if node.attrs['transB'] == 0:
        perm = _swapLastTwoDimsPermutation(len(matrixB.shape))
        matrixB.values = matrixB.values.transpose(perm)

    # Align dimensions for convolution
    expandAxis = []
    # Align the batch dimension
    if len(matrixA.shape) == 2:
        expandAxis.append(0)
    # Expand the height dimension
    expandAxis.append(1)
    # pwIn, shape [B x 1 x M x K]
    matrixAExpandDimsNode, pwIn = _appendExpandDims(matrixA, name, axis = expandAxis)
    graph.nodes.append(matrixAExpandDimsNode)

    # pwWeight, shape [N x 1 x 1 x K]
    matrixBExpandDimsNode, pwWeight = _appendExpandDims(matrixB, name, axis = (1, 2))
    graph.nodes.append(matrixBExpandDimsNode)

    if len(matrixY.shape) == 2:
        # matrixY, shape [M x N]
        squeezeDims = (0, 1)
    else:
        # matrixY, shape [B x M x N]
        squeezeDims = (1,)
    # pwOut, shape [B x 1 x M x N]
    matrixYSqueezeDimsNode, pwOut = _prependSqueezeDims(matrixY, name, squeezeDims)
    graph.nodes.append(matrixYSqueezeDimsNode)

    pwAttrs = {
        'channels_first': False,
        'dilations': [1, 1],
        'group': 1,
        'kernel_shape': [1, 1],
        'pads': [0, 0, 0, 0],
        'strides': [1, 1],
        'div': node.attrs['div'],
        'n_levels_out': node.attrs['n_levels_out'],
        'shift': node.attrs['shift'],
        'signed': node.attrs['signed'],
    }

    add = node.inputs[2]
    mul = node.inputs[3]

    _inputs = [pwIn, pwWeight, mul, add]

    pw = gs.Node(op = 'RequantizedConv',
                 name = name + "_RequantizedPwConv",
                 inputs = _inputs,
                 outputs = [pwOut],
                 attrs = pwAttrs)
    graph.nodes.append(pw)

    node.inputs.clear()
    node.outputs.clear()
    graph.nodes.remove(node)

    return graph


@contextagnostic
class RequantizedGemmToPwPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = _singleNodePattern("RequantizedGemm")
        super().__init__(graph, _requantized_gemm_to_pw_fun, "_REQUANTIZED_GEMM_TO_PW_PASS")


def _remove_global_output_reshape_fun(graph: gs.Graph, match: Match, name: str):
    node = next(iter((match.nodes_map.values())))

    isGlobalOutput = len(node.outputs[0].outputs) == 0
    if isGlobalOutput:
        graph.deleteNode(node)

    return graph


@contextagnostic
class RemoveGlobalOutputReshapePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = _singleNodePattern("Reshape")
        super().__init__(graph, _remove_global_output_reshape_fun, "_REMOVE_GLOBAL_OUTPUT_RESHAPE_PASS")


def _remove_empty_conv_bias_fun(graph: gs.Graph, match: Match, name: str):
    node = next(iter((match.nodes_map.values())))

    # Check if the node has an all-zero bias and remove it
    if len(node.inputs) == 3:
        bias = node.inputs[2]
        if isinstance(bias, gs.Constant) and np.all(bias.values == 0):
            del node.inputs[2]

    return graph


@contextagnostic
class RemoveEmptyConvBiasPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = _singleNodePattern("Conv")
        name = "_REMOVE_EMPTY_CONV_BIAS_PASS"
        super().__init__(graph, _remove_empty_conv_bias_fun, name)


def _remove_only_singleton_reduce_mean(graph: gs.Graph, match: Match, name: str):
    node = next(iter((match.nodes_map.values())))

    # Keep node if only one in the graph
    if len(graph.nodes) == 1:
        return graph

    # Delete node if only reduction over singleton dimensions
    if 'axis' in node.attrs:
        axis = node.attrs['axis']
    elif len(node.inputs) > 1 and node.inputs[1] is not None and hasattr(
            node.inputs[1], 'values') and node.inputs[1].values is not None:
        axis = node.inputs[1].values
    else:
        return graph  # axis unknown, skip

    # Check if shape information is available
    if node.inputs[0].shape is not None and all(node.inputs[0].shape[ax] == 1 for ax in axis):
        graph.deleteNode(node)

    return graph


@contextagnostic
class RemoveOnlySingletonReduceMeanPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = _singleNodePattern("ReduceMean")
        name = "_REMOVE_ONLY_SINGLETON_REDUCE_MEAN_PASS"
        super().__init__(graph, _remove_only_singleton_reduce_mean, name)
