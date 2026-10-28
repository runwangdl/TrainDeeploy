# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp9, memory): `FrozenBNPoolReluQuant` -- the fp32 tail of a quantized conv block in ONE node.

After lowering, a frozen-stat QZO block (Dequant already folded into the BN by FoldDequantIntoFrozenBNPass) reads

    BN(int8 HWC -> fp32 HWC) -> [MaxPool HWC] -> Transpose(0,3,1,2) -> Relu -> Quant(int8) -> Transpose(0,2,3,1)

i.e. int8 HWC in, int8 HWC out, with up to four fp32 tensors in between (block 0 of SpeechNet: the 628 KB BN output is
73 % of the L2 peak). Every step is per element or a max over a window, so one kernel computes the chain per output
element in registers and the fp32 tensors never exist:

    FrozenBNPoolReluQuant(x [1,H,W,C] i8, gamma, beta, running_mean, running_var [C] f32) -> y [1,Ho,Wo,C] i8

Bit-identical to the unfused chain: each step uses the expression of the kernel it replaces
(PULP_BatchNormInternalHWC_i8_fp32, PULP_MaxPool2d_fp32_fp32_HWC, PULP_Relu_fp32_fp32, PULPQuantTemplate); the two
Transposes only move data. Kernel: TargetLibraries/PULPOpen/src/BatchNorm.c (same file and flags as the BN kernel).

Only for frozen-stat BN (DEEPLOY_BN_FROZEN_NHWC=1, set by the MeZO runner for -D BN_FROZEN_STATS=ON). Opt out with
DEEPLOY_FUSE_BN_TAIL=0 (A/B measurement). -- QW
"""

import os
from typing import Dict, List, Optional, Tuple

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float32_t, int8_t, uint16_t
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import Pass, contextagnostic
from Deeploy.DeeployTypes import NetworkContext, NodeBinding, NodeParser, NodeTemplate, ONNXLayer, \
    OperatorRepresentation
from Deeploy.Targets.Generic.TypeCheckers import SignPropTypeChecker
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme

OP = "FrozenBNPoolReluQuant"

# ---------------------------------------------------------------------------------------------------------------------
# Topology pass (runs at the end of lowering, after the layout transposes exist)
# ---------------------------------------------------------------------------------------------------------------------


def _only_consumer(t: gs.Tensor, graph: gs.Graph) -> Optional[gs.Node]:
    # a node without outputs computes nothing for anyone; skip it (ZODualPassFusion used to leave such phantom
    # consumers behind, fixed there in exp9 -- kept as a guard)
    users = [n for n in t.outputs if len(n.outputs) > 0]
    if t in graph.outputs or len(users) != 1:
        return None
    return users[0]


def _perm(n: gs.Node) -> List[int]:
    return [int(p) for p in n.attrs.get("perm", [])]


def _match_tail(bn: gs.Node, graph: gs.Graph) -> Optional[Tuple[Optional[gs.Node], List[gs.Node], gs.Node]]:
    """BN -> [MaxPool] -> Transpose(0,3,1,2) -> Relu -> Quant -> Transpose(0,2,3,1); returns (pool, chain, quant)."""
    if int(bn.attrs.get("bn_nhwc", 0)) != 1 or "dq_scale" not in bn.attrs or len(bn.outputs) != 5:
        return None
    if any(len(o.outputs) > 0 or o in graph.outputs for o in bn.outputs[1:]):
        return None
    chain, pool = [], None
    nxt = _only_consumer(bn.outputs[0], graph)
    if nxt is not None and nxt.op == "MaxPool":
        a = nxt.attrs
        ks, st = list(a.get("kernel_shape", [])), list(a.get("strides", a.get("kernel_shape", [])))
        if len(ks) != 2 or ks != st or any(int(p) != 0 for p in a.get("pads", [0, 0, 0, 0])) \
                or any(int(d) != 1 for d in a.get("dilations", [1, 1])) or int(a.get("ceil_mode", 0)) != 0:
            return None
        pool = nxt
        chain.append(nxt)
        nxt = _only_consumer(nxt.outputs[0], graph)
    expect = [("Transpose", [0, 3, 1, 2]), ("Relu", None), ("Quant", None), ("Transpose", [0, 2, 3, 1])]
    quant = None
    for i, (op, perm) in enumerate(expect):
        if nxt is None or nxt.op != op or (perm is not None and _perm(nxt) != perm):
            return None
        if op == "Quant":
            quant = nxt
        chain.append(nxt)
        if i < len(expect) - 1:
            nxt = _only_consumer(nxt.outputs[0], graph)
    # zero points 0: the unfused code may contract x*s + zp into one fma; with zp = 0 both forms round identically
    if float(quant.attrs.get("zero_point", 0.0)) != 0.0 or float(bn.attrs.get("dq_zero_point", 0.0)) != 0.0:
        return None
    if int(quant.attrs.get("bit_width", 8)) != 8:
        return None
    return pool, chain, quant


@contextagnostic
class FuseFrozenBNTailPass(Pass):
    """QW exp9: BN -> [MaxPool] -> Transpose -> Relu -> Quant -> Transpose  ==>  FrozenBNPoolReluQuant."""

    def __init__(self):
        super().__init__()
        self.name = "_FUSE_FROZEN_BN_TAIL_PASS"

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        if os.environ.get("DEEPLOY_BN_FROZEN_NHWC", "0") != "1" or os.environ.get("DEEPLOY_FUSE_BN_TAIL", "1") == "0":
            return graph
        graph.cleanup().toposort()  # drop nodes whose outputs nobody reads (left by earlier lowering passes)
        for bn in [n for n in graph.nodes if n.op == "BatchNormInternal"]:
            m = _match_tail(bn, graph)
            if m is None:
                continue
            pool, chain, quant = m
            y = chain[-1].outputs[0]
            attrs = {
                "epsilon": float(bn.attrs.get("epsilon", 1e-5)),
                "dq_scale": float(bn.attrs["dq_scale"]),
                "dq_zero_point": float(bn.attrs.get("dq_zero_point", 0.0)),
                "pool_kernel": [int(k) for k in pool.attrs["kernel_shape"]] if pool is not None else [1, 1],
                "q_scale": float(quant.attrs["scale"]),
                "q_zero_point": float(quant.attrs.get("zero_point", 0.0)),
                "q_bit_width": int(quant.attrs.get("bit_width", 8)),
                "q_signed": int(quant.attrs.get("signed", 1)),
            }
            if "engine" in bn.attrs:
                attrs["engine"] = bn.attrs["engine"]
            fused = gs.Node(op = OP, name = bn.name + "_tail", inputs = list(bn.inputs), outputs = [y], attrs = attrs)
            for n in [bn] + chain:
                n.inputs.clear()
                n.outputs.clear()
            graph.nodes.append(fused)
        graph.cleanup().toposort()
        return graph


# ---------------------------------------------------------------------------------------------------------------------
# Parser / type checker / layer / template
# ---------------------------------------------------------------------------------------------------------------------


class FrozenBNPoolReluQuantParser(NodeParser):

    def parseNode(self, node: gs.Node) -> bool:
        if node.op != OP or len(node.inputs) != 5 or len(node.outputs) != 1:
            return False
        r = self.operatorRepresentation
        r['epsilon'] = float(node.attrs['epsilon'])
        r['dq_scale'] = float(node.attrs['dq_scale'])
        r['dq_zero_point'] = float(node.attrs['dq_zero_point'])
        r['kh'], r['kw'] = (int(k) for k in node.attrs['pool_kernel'])
        # exactly as QuantParser: the template multiplies by the reciprocal of the ONNX scale
        r['q_mul'] = 1.0 / float(node.attrs['q_scale'])
        r['q_zero_point'] = float(node.attrs['q_zero_point'])
        r['bit_width'] = int(node.attrs['q_bit_width'])
        r['signed'] = bool(node.attrs['q_signed'])
        bw = r['bit_width']
        r['q_min'], r['q_max'] = (-(2**(bw - 1)), 2**(bw - 1) - 1) if r['signed'] else (0, 2**bw - 1)
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        r = self.operatorRepresentation
        for key, t in zip(['data_in', 'scale', 'bias', 'running_mean', 'running_var'], node.inputs):
            r[key] = ctxt.lookup(t.name).name
        r['data_out'] = ctxt.lookup(node.outputs[0].name).name
        _, H, W, C = (int(v) for v in ctxt.lookup(node.inputs[0].name).shape)
        _, Ho, Wo, Co = (int(v) for v in ctxt.lookup(node.outputs[0].name).shape)
        if Co != C or Ho != H // r['kh'] or Wo != (W - r['kw']) // r['kw'] + 1:
            return ctxt, False
        r['H_in'], r['W_in'], r['C'], r['H_out'], r['W_out'] = H, W, C, Ho, Wo
        r['size'] = Ho * Wo * C
        return ctxt, True


class FrozenBNPoolReluQuantChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs, operatorRepresentation):
        return [2**operatorRepresentation['bit_width']]

    def _inferSignedness(self, inputs, operatorRepresentation):
        return [bool(operatorRepresentation['signed'])]


class FrozenBNPoolReluQuantLayer(ONNXLayer):

    def __init__(self, maps):
        super().__init__(maps)

    def computeOps(self):
        r = self.mapper.parser.operatorRepresentation
        return r['H_out'] * r['kh'] * r['W_out'] * r['kw'] * r['C']


referenceTemplate = NodeTemplate("""
// QW exp9: frozen BN + MaxPool + Relu + Quant, int8 HWC -> int8 HWC (Name: ${nodeName}, Op: ${nodeOp})
PULP_FrozenBNPoolReluQuantHWC_i8_i8(${data_in}, ${scale}, ${bias}, ${running_mean}, ${running_var}, ${data_out},
    ${H_out}, ${W_in}, ${W_out}, ${C}, ${kh}, ${kw}, ${epsilon}f, (float32_t)${dq_scale}, (float32_t)${dq_zero_point},
    (float32_t)${q_mul}, (float32_t)${q_zero_point}, ${q_min}, ${q_max});
""")

# -- the fp32 vectors stay fp32; same transformer as the other PULPOpen cluster kernels (DualHalf runs this way on GAP9)
from Deeploy.Targets.PULPOpen.Bindings import ForkTransformer  # noqa: E402

FrozenBNPoolReluQuantBindings = [
    NodeBinding(
        FrozenBNPoolReluQuantChecker([PointerClass(int8_t)] + [PointerClass(float32_t)] * 4, [PointerClass(int8_t)]),
        referenceTemplate, ForkTransformer)
]

# ---------------------------------------------------------------------------------------------------------------------
# Tile constraint: whole rows (all W, all C) along H, like MaxPoolHCTileConstraint; the [C] vectors stay whole
# ---------------------------------------------------------------------------------------------------------------------

_VECS = ['scale', 'bias', 'running_mean', 'running_var']


class FrozenBNPoolReluQuantTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inName, outName = parseDict['data_in'], parseDict['data_out']
        for name in [inName, outName] + [parseDict[v] for v in _VECS]:
            tilerModel.addTensorDimToModel(ctxt, name)
        dim = lambda n, i: tilerModel.getTensorDimVar(tensorName = n, dimIdx = i)
        inShape, outShape = ctxt.lookup(inName).shape, ctxt.lookup(outName).shape
        for k in (0, 2, 3):
            tilerModel.addConstraint(dim(inName, k) == inShape[k])
            tilerModel.addConstraint(dim(outName, k) == outShape[k])
        tilerModel.addConstraint(dim(inName, 1) == dim(outName, 1) * parseDict['kh'])
        for v in _VECS:
            tilerModel.addConstraint(dim(parseDict[v], 0) == inShape[3])
        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        r = operatorRepresentation
        # global parameters may be outside the solution (_checkResolve), as in BatchNormInternalTileConstraint
        vecs = [v for v in _VECS if r[v] in tilingSolution.tensorMemoryConstraints]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel, r,
                                                                  ['data_in', 'data_out'] + vecs)
        kh, W, C = r['kh'], r['W_in'], r['C']
        replacements, replacementTypes = {"H_out": []}, {"H_out": PointerClass(uint16_t)}
        inputLoadSchedule, outputLoadSchedule = [], []
        for cube in [c.rectangle for c in absoluteOutputCubes]:
            (b0, h0, _, _), (bs, hs, _, _) = cube.offset, cube.dims
            entry = {"data_in": HyperRectangle((b0, h0 * kh, 0, 0), (bs, hs * kh, W, C))}
            for v in vecs:
                entry[v] = HyperRectangle((0,), (C,))
            replacements["H_out"].append(hs)
            inputLoadSchedule.append(entry)
            outputLoadSchedule.append({"data_out": cube})
        return VariableReplacementScheme(replacements, replacementTypes), \
            TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)


def tilingReadyBindings():
    from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings
    return TilingReadyNodeBindings(nodeBindings = FrozenBNPoolReluQuantBindings,
                                   tileConstraint = FrozenBNPoolReluQuantTileConstraint())
