# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import OrderedDict

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import BranchingMatcher, Match
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import Pass, ReplaceSequentialPatternPass, contextagnostic


@contextagnostic
class PULPConvKeepCHWPass(Pass):
    """Tag every forward Conv node with attr ``keep_channels_first = True``.

    Used by the GAP9 channels-first conv path (deployer ``conv_channels_first=True``).
    The tag has two effects, both gated entirely by the tag so non-CHW networks
    are untouched:

      1. ``_NCHWtoNHWC_fun`` early-returns on tagged nodes, so NO NCHW->NHWC
         transpose is inserted around the conv — its activations stay channels-first
         (native ONNX NCHW), which removes the L1-floor-setting stem transpose and
         the per-conv input/output transpose pair.
      2. The CHW conv mappers (which sit ahead of the HWC mappers in the 'Conv'
         mapping) only accept tagged nodes, so tagged convs bind to the *_CHW
         kernels + NCHW tile constraints while every other conv falls through to
         the unchanged HWC path.

    Must run BEFORE PULPNCHWtoNHWCPass. No-op on graphs with no Conv nodes.
    """

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        for node in graph.nodes:
            if node.op == "Conv":
                node.attrs["keep_channels_first"] = True
        return graph


@contextagnostic
class TransposeGemmSquashPass(Pass):
    """Eliminate the materialised transposed weight ``W^T`` of a linear layer.

    A training graph emits a linear layer's forward as ``MatMul(X, Transpose(W))`` and
    its input-gradient as ``Gemm(dY, Transpose(W), transB=1)``. Both read the SAME
    materialised ``W^T = Transpose_[1,0](W)`` -- a full extra weight copy per linear
    layer that is kept alive across the whole forward->backward step. But the GEMM
    kernel already transposes its B operand in-place via ``transB``, so ``W^T`` never
    needs to exist:

      * forward  ``MatMul(X, T[1,0](W))``          == ``Gemm(X,  W, transB=1)``
      * backward ``Gemm(dY, T[1,0](W), transB=1)``  == ``Gemm(dY, W, transB=0)``

    This pass rewires both consumers to read ``W`` directly (converting the forward
    MatMul to a Gemm). Once no consumer references the Transpose output, ``cleanup()``
    drops the Transpose and ``W^T`` disappears from the arena entirely. Deeploy's GEMM
    parser handles the forward's batched 3-D activation (``[1, seq, dim]``) natively via
    its batch dims, so no reshape is needed.

    Measured (GAP9, CCT training): removing the backward's ``W^T`` dependency alone gives
    ``MEMORYARENA_L3`` 3528 -> 3173 KB; folding the forward too removes the remaining
    forward-side ``W^T``. Exact, ``@contextagnostic``; only fires on a plain 2-D
    ``perm=[1,0]`` Transpose of a weight leaf (graph input / Constant) feeding a
    Gemm/MatMul's B input.
    """

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        producers = {o.name: n for n in graph.nodes for o in n.outputs if o.name}
        for node in graph.nodes:
            if node.op not in ("Gemm", "MatMul") or len(node.inputs) < 2:
                continue
            tp = producers.get(node.inputs[1].name)
            if tp is None or tp.op != "Transpose":
                continue
            perm = tp.attrs.get("perm", None)
            if perm is None or list(perm) != [1, 0]:
                continue
            # only fold a transposed WEIGHT (a leaf: graph input or Constant), not a
            # transposed activation -- keeps the rewrite to the W^T-redundancy case.
            weight = tp.inputs[0]
            if weight.name in producers:
                continue
            if node.op == "Gemm":
                if int(node.attrs.get("transB", 0)) != 1:
                    continue
                node.inputs[1] = weight  # Gemm(A, T(W), transB=1) == Gemm(A, W, transB=0)
                node.attrs["transB"] = 0
            else:  # MatMul(A, T(W)) == Gemm(A, W, transB=1)
                node.op = "Gemm"
                node.inputs[1] = weight
                node.attrs["transB"] = 1
                node.attrs.setdefault("transA", 0)
                node.attrs.setdefault("alpha", 1)
                node.attrs.setdefault("beta", 1)
        graph.cleanup()
        return graph


def _squash_transpose_add_fun(graph: gs.Graph, match: Match, name: str):

    nodes_map = match.nodes_map

    # SCHEREMO: Check that perms are equal
    if not nodes_map['transpose1'].attrs['perm'] == nodes_map['transpose2'].attrs['perm']:
        return graph

    # SCHEREMO: Make sure we are requantizing layerwise
    if not (isinstance(nodes_map['add'].attrs['rqs1_add'], int)
            and isinstance(nodes_map['add'].attrs['rqs1_mul'], int)):
        return graph

    addNode = nodes_map['add']

    transposeAttrs = copy.deepcopy(nodes_map['transpose1'].attrs)
    newInputs = [nodes_map['transpose1'].inputs[0], nodes_map['transpose2'].inputs[0]]
    newOutputs = [addNode.outputs[0]]

    graph.deleteNode(nodes_map['transpose1'])
    graph.deleteNode(nodes_map['transpose2'])
    newAddOut = gs.Variable(name = addNode.outputs[0].name + "_tp")
    newAddOut.shape = newInputs[0].shape
    newAddOut.dtype = newOutputs[0].dtype

    addNode.outputs = [newAddOut]
    graph.layer(inputs = [newAddOut],
                outputs = [newOutputs[0]],
                op = "Transpose",
                name = addNode.name + "_transpose",
                attrs = transposeAttrs)

    return graph


@contextagnostic
class RQAddTransposeSquashPass(ReplaceSequentialPatternPass):

    def __init__(self):
        _input1 = gs.Variable(name = 'input_1')
        _input2 = gs.Variable(name = 'input_2')
        _addIn1 = gs.Variable(name = 'addIn1')
        _addIn2 = gs.Variable(name = 'addIn2')
        _addOut = gs.Variable(name = 'addOut')
        _rqs = gs.Variable(name = 'rqs')

        anyIn1 = gs.Node(inputs = [_input1], outputs = [_addIn1], op = r'Transpose', name = 'transpose1')
        anyIn2 = gs.Node(inputs = [_input2], outputs = [_addIn2], op = r'Transpose', name = 'transpose2')

        addOut = gs.Node(inputs = [_addIn1, _addIn2], outputs = [_addOut], op = 'RequantizedAdd', name = 'add')

        graph = gs.Graph(nodes = [anyIn1, anyIn2, addOut], inputs = [_input1, _input2], outputs = [_rqs])

        super().__init__(graph,
                         replacement_fn = _squash_transpose_add_fun,
                         name = "_SQUASH_TRANSPOSE_RQADD_PASS",
                         matcher = BranchingMatcher(regex_op = True))


def _merge_add_rq_fun(graph: gs.Graph, match: Match, name: str):

    nodes_map = match.nodes_map
    addNode = nodes_map['add']

    rqDict = OrderedDict([("rqs1", None), ("rqs2", None), ("rqsOut", None)])

    for key, node in nodes_map.items():

        if node.outputs[0].name == addNode.inputs[0].name:
            rqDict['rqs1'] = node
        elif node.outputs[0].name == addNode.inputs[1].name:
            rqDict['rqs2'] = node
        elif node.inputs[0].name == addNode.outputs[0].name:
            rqDict['rqsOut'] = node

    newAttrs = copy.copy(addNode.attrs)
    newInputs = []

    if rqDict['rqsOut'] is not None:
        newOutputs = rqDict['rqsOut'].outputs
    else:
        newOutputs = addNode.outputs

    defaultAttrs = {
        "mul": 1,
        "add": 0,
        "div": gs.Constant('div', np.array(1)),
        'shift': gs.Constant('div', np.array(0))
    }
    guessAttrs = {"n_levels_out": 256, "signed": np.array([True])}
    for idx, (rqKey, node) in enumerate(rqDict.items()):
        if node.op == "RequantShift":
            for key, attr in node.attrs.items():
                newAttrs[f"{rqKey}_{key}"] = attr

            if np.prod(node.inputs[1].values.shape) != 1:
                return graph

            if np.prod(node.inputs[2].values.shape) != 1:
                return graph

            if rqKey != 'rqsOut':
                newInputs.append(node.inputs[0])

            newAttrs[f"{rqKey}_mul"] = int(node.inputs[1].values.item())
            newAttrs[f"{rqKey}_add"] = int(node.inputs[2].values.item() + newAttrs[f"{rqKey}_div"].values.item() // 2)
            newAttrs[f"{rqKey}_shift"] = int(np.log2(newAttrs[f"{rqKey}_div"].values.item()))

        else:
            for key, attr in defaultAttrs.items():
                newAttrs[f"{rqKey}_{key}"] = attr

            for key, attr in guessAttrs.items():
                if not key in node.attrs:
                    newAttrs[f"{rqKey}_{key}"] = attr
                else:
                    newAttrs[f"{rqKey}_{key}"] = node.attrs[key]
            if rqKey != 'rqsOut':
                newInputs.append(addNode.inputs[idx])

    rqAdd = gs.Node(op = "RequantizedAdd", name = name, attrs = newAttrs)
    graph.replaceInsertNode(newInputs, newOutputs, rqAdd)

    return graph


@contextagnostic
class PULPAddRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        _input1 = gs.Variable(name = 'input_1')
        _input2 = gs.Variable(name = 'input_2')
        _addIn1 = gs.Variable(name = 'addIn1')
        _addIn2 = gs.Variable(name = 'addIn2')
        _addOut = gs.Variable(name = 'addOut')
        _rqs = gs.Variable(name = 'rqs')

        anyIn1 = gs.Node(inputs = [_input1], outputs = [_addIn1], op = r'.*', name = 'any1')
        anyIn2 = gs.Node(inputs = [_input2], outputs = [_addIn2], op = r'.*', name = 'any2')

        addOut = gs.Node(inputs = [_addIn1, _addIn2], outputs = [_addOut], op = 'Add', name = 'add')
        output = gs.Node(inputs = [_addOut], outputs = [_rqs], op = r'RequantShift', name = 'rqsOut')

        graph = gs.Graph(nodes = [anyIn1, anyIn2, addOut, output], inputs = [_input1, _input2], outputs = [_rqs])

        super().__init__(graph,
                         replacement_fn = _merge_add_rq_fun,
                         name = "_MERGE_ADDRQ_PASS",
                         matcher = BranchingMatcher(regex_op = True))


def _merge_conv_rq_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    conv = matched_nodes[0]
    rqs = matched_nodes[1]

    totalShift = int(np.log2(rqs.attrs['div'].values))

    # Artifically add half the shift division value to implement rounding
    rounding = 2**(totalShift - 1) if totalShift > 0 else 0

    rqs.inputs[-1].values = copy.deepcopy(rqs.inputs[-1].values) + rounding

    _inputs = list(conv.inputs) + list(rqs.inputs[1:])

    _outputs = rqs.outputs

    rqsConv = gs.Node(op = 'RequantizedConv', name = name, attrs = {**conv.attrs, **rqs.attrs, "shift": totalShift})
    graph.replaceInsertNode(_inputs, _outputs, rqsConv)

    return graph


@contextagnostic
class PULPConvRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['conv_out'], op = 'Conv', name = 'conv1')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_CONVRQ_PASS"
        super().__init__(graph, _merge_conv_rq_fun, name)


def _merge_gemm_rq_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    gemm = matched_nodes[0]
    rqs = matched_nodes[1]

    totalShift = int(np.log2(rqs.attrs['div'].values))

    rqs.inputs[-1].values = copy.deepcopy(rqs.inputs[-1].values) + 2**(totalShift - 1)

    # GEMM has add
    if len(list(gemm.inputs)) == 3:

        gemm.inputs[2].values = np.round(gemm.inputs[2].values * (rqs.inputs[1].values)) + rqs.inputs[2].values

        #gemm.inputs[2].values = gemm.inputs[2].values + np.round(rqs.inputs[2].values / (rqs.inputs[1].values + 1e-3))
        # Keep input, weight from GEMM
        # Take mul from RQS
        _inputs = list(gemm.inputs) + list(rqs.inputs[1:2])
    else:
        _inputs = list(gemm.inputs) + list(rqs.inputs[2:]) + list(rqs.inputs[1:2])
    _outputs = rqs.outputs
    attrs = {**gemm.attrs, **rqs.attrs}
    attrs['shift'] = gs.Constant(name = 'shift', values = np.array(totalShift))
    #attrs['mul']=gs.Constant(name='mul',values = np.array(rqs.inputs[1].values))
    rqsGemm = gs.Node(op = 'RequantizedGemm', name = name, attrs = attrs)
    graph.replaceInsertNode(_inputs, _outputs, rqsGemm)

    return graph


@contextagnostic
class PULPGEMMRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['gemm_out'], op = 'Gemm', name = 'gemm')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_GEMMRQ_PASS"
        super().__init__(graph, _merge_gemm_rq_fun, name)


@contextagnostic
class PULPMatMulRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['gemm_out'], op = 'MatMul', name = 'gemm')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_GEMM_MATMUL_RQ_PASS"
        super().__init__(graph, _merge_gemm_rq_fun, name)


@contextagnostic
class FoldActivationTransposeIntoGemmPass(Pass):
    """Fold a 2-D activation transpose into the Gemm that consumes it.

    ``Gemm(A, T_[1,0](X), transB=0)`` and ``Gemm(A, X, transB=1)`` compute the same
    thing, so a Transpose feeding a Gemm's B input only to be multiplied can be replaced
    by a flag the kernel already honours. Nothing new is asked of the backend: transB=1
    is the path a linear layer's backward already takes.

    Mirror image of TransposeGemmSquashPass, which rewrites transB 1 -> 0 to drop a
    WEIGHT transpose the Gemm was undoing anyway. Here the transpose is of an ACTIVATION
    and the flag goes the other way. What decides whether a transpose can be folded is
    not where its data comes from but whether the permutation is expressible as the
    flag, and a 2-D perm=[1,0] is, constant or not.

    Motivation, CCT on GAP9. At the training peak six [128,128] fp32 transposes are
    live, 384KB, every one perm=[1,0] feeding a Gemm as B with transB=0 -- 43% of that
    peak and, together with a 256KB NCHW->NHWC transpose into MaxPool, the reason
    rematerialisation cannot reach it: those transposes were on the recompute ban list,
    which made every budget below 880KB infeasible in under 7 seconds. Unbanning them
    and letting the ILP recompute instead was measured and is the worse trade: 2896KB
    against a 2960KB baseline, 64KB for 23 recomputes, because a recomputed tensor still
    has to be materialised somewhere. Folding removes the buffer instead of shortening
    its life.

    The Transpose node is left in place. graph.cleanup() drops it once the rewire has
    taken its last consumer and keeps it when something else still reads X^T.
    """

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        # Diagnostic escape hatch: lets the same tree be built with and without
        # the fold so a failure can be attributed to it rather than argued about.
        import os
        if os.environ.get("DISABLE_TPFOLD") == "1":
            return graph
        producers = {o.name: n for n in graph.nodes for o in n.outputs if o is not None and o.name}
        folded = 0
        for node in graph.nodes:
            if node.op != "Gemm" or len(node.inputs) < 2 or node.inputs[1] is None:
                continue
            if int(node.attrs.get("transB", 0)) != 0:
                continue
            transpose = producers.get(node.inputs[1].name)
            if transpose is None or transpose.op != "Transpose":
                continue
            if not transpose.inputs or transpose.inputs[0] is None:
                continue
            perm = transpose.attrs.get("perm", None)
            if perm is None or list(perm) != [1, 0]:
                continue
            node.inputs[1] = transpose.inputs[0]
            node.attrs["transB"] = 1
            folded += 1
        graph.cleanup()
        return graph


@contextagnostic
class FoldDequantIntoMatMulPass(Pass):
    """Fold a weight's Dequant into the MatMul or Gemm nodes that consume it.

    Weight-only quantisation stores a frozen weight as int8 and needs it as float
    for an fp32 matmul. Left as its own node, the Dequant materialises the whole
    dequantised matrix, which then has to live from the forward pass to the
    backward one or be recomputed -- on CCT-QLoRA, 1088 KB of fp32 tensors whose
    Dequant nodes account for 239.95M cycles, only 48.47M of which is the kernel.
    QLoRA instead dequantises inside the matmul and discards the value;
    PULP_MatMul_fp32_i8_fp32_unroll1x7 does that, given the per-tensor scale and
    zero point this pass attaches.

    Written as a plain graph walk rather than a pattern replacement. The Dequant
    worth folding is read by TWO nodes, the forward matmul and its gradient, and
    that fan-out is what the sequential matcher skips; a Dequant with one consumer
    is the cheap tail. Each consumer then dequantises independently, which repeats
    the conversion on purpose -- it is register-level work inside the kernel, and
    cheaper than keeping a materialised fp32 copy alive across the step.

    Only a win alongside constant deduplication: removing the Dequant gives the
    int8 constant two direct consumers, and _duplicateConstants stores one copy per
    consumer unless byte-identical constants are shared.
    """

    # Conv is here because at this stage -- before parsing inserts the NCHW->NHWC
    # transposes -- the tokenizer's quantised weight reaches its Conv directly.
    # Reading the post-parsing graph suggests otherwise and sent three attempts at
    # this after a Transpose that does not exist yet when the pass runs.
    # Which input index carries the WEIGHT, per consumer op. It is not always 1: a
    # convolution's gradient is ConvGrad(dY, X, W), so the weight is third. Hard-coding
    # index 1 made the all() below false for every Conv whose weight is also read by its
    # gradient -- which is every frozen weight in an adapter-tuned CNN -- so all 18
    # Dequants in MobileNetV1-QLoRA were skipped and the fp32 copies stayed materialised.
    # CCT never showed it because a Gemm's gradient is another Gemm, weight at index 1.
    # ConvGradX, not ConvGrad: this pass runs AFTER lowering has split the fused
    # gradient into its input-gradient and weight-gradient halves, and only the input
    # gradient reads the weight. Reading the exported graph says ConvGrad(dY, X, W) with
    # the weight third, and that is not what the pass sees -- ConvGradX takes it at 1,
    # like the forward. Checking the graph the pass actually runs on is the only way to
    # get this right; the same mistake sent three earlier attempts at the Conv case.
    WEIGHT_INPUT_IDX = {'MatMul': 1, 'Gemm': 1, 'Conv': 1, 'ConvGradX': 1}
    FOLDABLE_OPS = tuple(WEIGHT_INPUT_IDX)

    def run_pass(self, graph: gs.Graph):
        folded = 0
        for node in list(graph.nodes):
            if node.op != 'Dequant' or not node.outputs or node.outputs[0] is None:
                continue
            dequantOut = node.outputs[0]
            consumers = list(dequantOut.outputs)
            if not consumers:
                continue
            # Every consumer must take it as the weight operand, which is the one
            # the kernels dequantise. A Dequant feeding anything else is left alone.
            def readsAsWeight(consumer):
                idx = self.WEIGHT_INPUT_IDX.get(consumer.op)
                return (idx is not None and len(consumer.inputs) > idx and consumer.inputs[idx] is dequantOut)

            if not all(readsAsWeight(c) for c in consumers):
                continue
            quantised = node.inputs[0] if node.inputs else None
            if quantised is None:
                continue

            # The stored array is int8 but the tensor can carry a wider dtype from
            # the exporter; with the Dequant in place that never mattered, because
            # its output was explicitly float32. Once folded, this dtype is what the
            # binding checker sees, and a wrong one silently selects the fp32 kernel
            # and reads the int8 weight as float -- which runs, reports no error, and
            # computes nonsense four times faster.
            # gs.Constant derives dtype from values and does not let it be set, so
            # rewriting the array is both necessary and sufficient. A Variable that
            # is not a Constant cannot be corrected here and is left for the fp32
            # path rather than folded into a kernel that would misread it.
            if not isinstance(quantised, gs.Constant):
                continue
            quantised.values = np.asarray(quantised.values).astype(np.int8)

            # Each consumer takes the int8 tensor directly and carries the scale and
            # zero point as attributes, so it dequantises inside the kernel. Several
            # consumers each repeat that conversion on purpose: it is register-level
            # work, and cheaper than keeping a materialised fp32 copy alive between a
            # forward use and its backward one.
            # Every consumer reads the same int8 constant. A ConstantBuffer is global
            # and read-only, so sharing one across the forward MatMul and its backward
            # Gemm is sound, and it is what keeps the memory win: giving each consumer
            # its own copy charges the full constant again per clone, which on the int8
            # QLoRA CCT took persistent from 662 KB to 938 KB.
            for consumer in consumers:
                consumer.inputs[self.WEIGHT_INPUT_IDX[consumer.op]] = quantised
                consumer.attrs['dequant_scale'] = float(node.attrs.get('scale', 1.0))
                consumer.attrs['dequant_zero_point'] = int(node.attrs.get('zero_point', 0))

            node.inputs.clear()
            node.outputs.clear()
            folded += 1

        if folded:
            graph.cleanup()
        return graph


def _fold_dequant_fun(graph: gs.Graph, match: Match, name: str):
    """Rewire a Dequant's consumer to read the int8 tensor and dequantise in-kernel."""
    matched = [m for k, m in match.nodes_map.items()]
    dequant, consumer = matched[0], matched[1]

    dequantOut = dequant.outputs[0]
    # Only the weight operand is folded: the kernels dequantise their second input.
    if len(consumer.inputs) < 2 or consumer.inputs[1] is not dequantOut:
        return graph
    quantised = dequant.inputs[0] if dequant.inputs else None
    if quantised is None:
        return graph

    # Every consumer of this Dequant takes the int8 tensor and dequantises it
    # itself. That repeats the conversion on purpose: it is register-level work
    # inside the kernel, and cheaper than keeping a materialised fp32 copy alive
    # between a forward use and the backward one that reads it again.
    consumers = list(dequantOut.outputs)
    if not all(len(c.inputs) >= 2 and c.inputs[1] is dequantOut for c in consumers):
        return graph

    for c in consumers:
        c.inputs[1] = quantised
        c.attrs['dequant_scale'] = float(dequant.attrs.get('scale', 1.0))
        c.attrs['dequant_zero_point'] = int(dequant.attrs.get('zero_point', 0))

    # deleteNode() reconnects a node's output to its input, which assumes the node
    # is a pass-through. This one is not any more: its consumers now read the int8
    # tensor directly, so its output is an orphan. Detach it and let cleanup()
    # collect the node instead.
    dequant.inputs.clear()
    dequant.outputs.clear()
    graph.cleanup()
    return graph
