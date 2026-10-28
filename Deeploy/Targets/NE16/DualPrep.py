# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp6, task 2): the two cluster-side nodes that feed the FUSED dual-pass conv on NE16.

    NE16DualPerturbEncode : W_RAW, wmul                    -> weight_enc  uint8 [2*cout, cinMajor, 16]
    NE16DualBias          : W_RAW, wmul, add, bmul, mul    -> add2        int32 [2*cout]

Both reproduce exp5's phase 1 (both eps-signs from one Rademacher draw, the standalone perturb nodes'
chunking / seeding / rounding / clamp -- `kernel/DualPerturb.h`) and then do the NE16-specific part:
the first bit-serial-encodes the stacked [W+; W-] as ONE 1x1 weight with 2*cout output channels and
K = cin*kh*kw input channels (the im2col route's folded weight); the second produces the stacked
requant-add with exp16c's signed-input correction per row. The NE16 conv itself is the ordinary
pointwise path with Cout = 2*cout, so A+ lands in output channels [0, cout) and A- in [cout, 2*cout)
-- exp5's channel-stacked output, same reference tensor.

Why two nodes: Deeploy's tiler assumes single-output nodes (exp5 Findings §4). The second node
re-derives W+/W- (deterministic, same seeds) -- 32 weights, a few hundred cycles.

Attributes (both nodes): seed_w, idx_w, div_w, and for the bias node also seed_b, idx_b, div_b and
ne16_input_offset -- the same values the RequantizedDualConv node carries (exp5). The seed expression
emitted here is the one DualConvTemplate.py emits, so the device draws exp5's noise stream.
Kernels: TargetLibraries/GAP9/src/NE16DualPrep.c. -- QW
"""

import re
from typing import Dict, List, Tuple, Union

import numpy as np
import onnx_graphsurgeon as gs
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import NetworkContext, NodeBinding, NodeParser, NodeTemplate, ONNXLayer, \
    OperatorRepresentation
from Deeploy.Targets.Generic.TypeCheckers import SignPropTypeChecker
from Deeploy.Targets.PULPOpen.Bindings import ForkTransformer

NE16_CIN_SUBTILE = 16


def _hoistPrepCache(ctxt: NetworkContext, node: gs.Node, out_bytes: int) -> str:
    """-- QW exp21c (opt H): a per-node L2 cache for the prep output of the current update step (header of 7 words +
    the output), emitted as a static global like Layernorm's stash (Generic/Parsers.py). The kernel keys it on the
    seeds, the eps scale and zo_weight_version (NE16DualPrep.c); -D NE16DUAL_NOOPT_H=ON compiles its use out (the
    buffer is then unused). -- QW exp21c"""
    name = ctxt._mangle(re.sub(r'[^A-Za-z0-9_]', '_', node.name)) + "_x21c_hcache"
    if name not in ctxt.globalObjects:
        ctxt.hoistGlobalDefinition(name, f"static PI_L2 uint32_t {name}[{7 + (out_bytes + 3) // 4}];")
    return name


def _log2(v: int) -> int:
    l = int(np.log2(v))
    assert (1 << l) == v, f"div must be a power of two, got {v}"
    return l


# ---------------------------------------------------------------------------------------------
# NE16DualPerturbEncode
# ---------------------------------------------------------------------------------------------


class NE16DualPerturbEncodeParser(NodeParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not all([
                node.op == "NE16DualPerturbEncode",
                len(node.inputs) == 2,
                len(node.outputs) == 1,
                all(a in node.attrs for a in ("seed_w", "idx_w", "div_w")),
        ]):
            return False
        r = self.operatorRepresentation
        r['seed_w'], r['idx_w'] = int(node.attrs['seed_w']), int(node.attrs['idx_w'])
        r['log2D_w'] = _log2(int(node.attrs['div_w']))
        # QW exp7: 0 = stacked [W+; W-] (block 0), 1 = W+ only, 2 = W- only (blocks 1-4: one NE16 job per pass)
        r['sign_sel'] = int(node.attrs.get('ne16_sign_sel', 0))
        r['hwc'] = int(node.attrs.get('ne16_hwc', 0))  # -- QW exp7: rows in (ky, kx, ci) order (pipelined NE16 conv)
        return r['sign_sel'] in (0, 1, 2)

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        w = ctxt.lookup(node.inputs[0].name)
        wmul = ctxt.lookup(node.inputs[1].name)
        out = ctxt.lookup(node.outputs[0].name)
        wShape = tuple(int(s) for s in w.shape)
        if len(wShape) != 4:
            return ctxt, False
        cout, K = wShape[0], int(np.prod(wShape[1:]))
        cinMajor = (K + NE16_CIN_SUBTILE - 1) // NE16_CIN_SUBTILE
        rows = 2 * cout if self.operatorRepresentation['sign_sel'] == 0 else cout  # -- QW exp7
        if tuple(int(s) for s in out.shape) != (rows, cinMajor, 16) or int(np.prod(wmul.shape)) != cout:
            return ctxt, False
        r = self.operatorRepresentation
        r['weight'], r['wmul'], r['data_out'] = w.name, wmul.name, out.name
        r['ch_out'], r['K'] = cout, K
        r['w_ci'], r['w_kh'], r['w_kw'] = wShape[1], wShape[2], wShape[3]  # -- QW exp7 (CHW weight)
        r['size'] = cout * K
        r['nodeName'] = node.name  # the tiler sizes the scratch (computeTransientBuffersSize) before binding fills this in
        r['hcache'] = _hoistPrepCache(ctxt, node, rows * cinMajor * 16)  # -- QW exp21c (opt H)
        return ctxt, True


class _DualPrepTemplate(NodeTemplate):
    """Hoists the int8 scratch ([W+; W-] and, for the bias node, add+/add-) like DualConvTemplate.
    `computeTransientBuffersSize` must be a STATIC method: the tiler calls it on the class with
    (ctxt, operatorRepresentation). Subclasses set WITH_ADD."""

    WITH_ADD = False

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        operatorRepresentation['tile_seed_offset'] = 0  # the weight is never tiled
        return ctxt, operatorRepresentation, []

    @classmethod
    def _scratchSize(cls, operatorRepresentation: OperatorRepresentation) -> int:
        wsize = operatorRepresentation['ch_out'] * operatorRepresentation['K']
        size = ((2 * wsize + 3) // 4) * 4
        if cls.WITH_ADD:
            size += 8 * operatorRepresentation['ch_out']
        return size

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        name, size = type(self).computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(name, size)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size
        return ctxt, operatorRepresentation, [name]


class _DualPerturbEncodeTemplate(_DualPrepTemplate):
    WITH_ADD = False

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        return [(operatorRepresentation['nodeName'] + "_buffer",
                 _DualPerturbEncodeTemplate._scratchSize(operatorRepresentation))]


class _DualBiasTemplate(_DualPrepTemplate):
    WITH_ADD = True

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        return [(operatorRepresentation['nodeName'] + "_buffer", _DualBiasTemplate._scratchSize(operatorRepresentation))]


dualPerturbEncodeTemplate = _DualPerturbEncodeTemplate("""
// NE16DualPerturbEncode (Name: ${nodeName}, Op: ${nodeOp}) -- W+/W- from one draw, encoded as one 2*cout 1x1 weight -- QW exp6
{
uint32_t ${nodeName}_seed_w = ((${seed_w} + perturb_seed_base) + NUM_CORES * ${idx_w});
uint32_t ${nodeName}_tile_xor = (${tile_seed_offset} * 0x9E3779B1u);
float32_t ${nodeName}_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
PULP_NE16DualPerturbEncode((const int8_t *) ${weight}, (const int32_t *) ${wmul}, (uint8_t *) ${data_out},
    (int8_t *) ${ctxtBuffer}, ${nodeName}_seed_w, ${nodeName}_tile_xor, ${log2D_w}, ${nodeName}_eps_scale,
    NUM_CORES, ${ch_out}, ${K}, ${sign_sel}, ${hwc}, ${w_ci}, ${w_kh}, ${w_kw}, (uint32_t *) ${hcache});
}
""")


class NE16DualPerturbEncodeChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs, operatorRepresentation):
        return [256]

    def _inferSignedness(self, inputs, operatorRepresentation):
        return [False]


NE16DualPerturbEncodeBindings = [
    NodeBinding(NE16DualPerturbEncodeChecker([PointerClass(int8_t), PointerClass(int32_t)], [PointerClass(uint8_t)]),
                dualPerturbEncodeTemplate, ForkTransformer)
]


class NE16DualPerturbEncodeLayer(ONNXLayer):

    def __init__(self, maps):
        super().__init__(maps)

    def computeOps(self):
        return 10 * self.mapper.parser.operatorRepresentation['size']


# ---------------------------------------------------------------------------------------------
# NE16DualBias
# ---------------------------------------------------------------------------------------------


class NE16DualBiasParser(NodeParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not all([
                node.op == "NE16DualBias",
                len(node.inputs) == 5,
                len(node.outputs) == 1,
                all(a in node.attrs for a in ("seed_w", "idx_w", "div_w", "seed_b", "idx_b", "div_b")),
        ]):
            return False
        r = self.operatorRepresentation
        r['seed_w'], r['idx_w'] = int(node.attrs['seed_w']), int(node.attrs['idx_w'])
        r['seed_b'], r['idx_b'] = int(node.attrs['seed_b']), int(node.attrs['idx_b'])
        r['log2D_w'], r['log2D_b'] = _log2(int(node.attrs['div_w'])), _log2(int(node.attrs['div_b']))
        r['input_offset'] = int(node.attrs.get('ne16_input_offset', 128))
        r['sign_sel'] = int(node.attrs.get('ne16_sign_sel', 0))  # -- QW exp7: 0 stacked, 1 add+', 2 add-'
        return r['sign_sel'] in (0, 1, 2)

    def parseNodeCtxt(self, ctxt: NetworkContext, node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        names = ['weight', 'wmul', 'add', 'bmul', 'mul']
        bufs = [ctxt.lookup(i.name) for i in node.inputs]
        out = ctxt.lookup(node.outputs[0].name)
        wShape = tuple(int(s) for s in bufs[0].shape)
        if len(wShape) != 4:
            return ctxt, False
        cout, K = wShape[0], int(np.prod(wShape[1:]))
        rows = 2 * cout if self.operatorRepresentation['sign_sel'] == 0 else cout  # -- QW exp7
        if tuple(int(s) for s in out.shape) != (rows,) or \
           any(int(np.prod(b.shape)) != cout for b in bufs[1:]):
            return ctxt, False
        r = self.operatorRepresentation
        for k, b in zip(names, bufs):
            r[k] = b.name
        r['data_out'] = out.name
        r['ch_out'], r['K'] = cout, K
        r['size'] = rows
        r['nodeName'] = node.name
        r['hcache'] = _hoistPrepCache(ctxt, node, rows * 4)  # -- QW exp21c (opt H)
        return ctxt, True


dualBiasTemplate = _DualBiasTemplate("""
// NE16DualBias (Name: ${nodeName}, Op: ${nodeOp}) -- add+/add- with the signed-input correction, stacked -- QW exp6
{
uint32_t ${nodeName}_seed_w = ((${seed_w} + perturb_seed_base) + NUM_CORES * ${idx_w});
uint32_t ${nodeName}_seed_b = ((${seed_b} + perturb_seed_base) + NUM_CORES * ${idx_b});
uint32_t ${nodeName}_tile_xor = (${tile_seed_offset} * 0x9E3779B1u);
float32_t ${nodeName}_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
PULP_NE16DualBias((const int8_t *) ${weight}, (const int32_t *) ${wmul}, (const int32_t *) ${add},
    (const int32_t *) ${bmul}, (const int32_t *) ${mul}, (int32_t *) ${data_out}, (int8_t *) ${ctxtBuffer},
    ${nodeName}_seed_w, ${nodeName}_seed_b, ${nodeName}_tile_xor, ${log2D_w}, ${log2D_b}, ${nodeName}_eps_scale,
    NUM_CORES, ${ch_out}, ${K}, ${input_offset}, ${sign_sel}, (uint32_t *) ${hcache});
}
""")


class NE16DualBiasChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs, operatorRepresentation):
        return [2**32]

    def _inferSignedness(self, inputs, operatorRepresentation):
        return [True]


NE16DualBiasBindings = [
    NodeBinding(
        NE16DualBiasChecker([PointerClass(int8_t), PointerClass(int32_t), PointerClass(int32_t),
                             PointerClass(int32_t), PointerClass(int32_t)], [PointerClass(int32_t)]),
        dualBiasTemplate, ForkTransformer)
]


class NE16DualBiasLayer(ONNXLayer):

    def __init__(self, maps):
        super().__init__(maps)

    def computeOps(self):
        return self.mapper.parser.operatorRepresentation['size'] * self.mapper.parser.operatorRepresentation['K']
