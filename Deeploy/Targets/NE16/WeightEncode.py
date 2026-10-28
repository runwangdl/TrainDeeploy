# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (exp16c / blocker 1b): the `NE16WeightEncode` op -- bit-serial weight encoding ON DEVICE.

exp16a and exp16b both let the HOST bit-plane-transpose the conv weight, via
`NE16AdjustWeightMemoryLayoutPass` -> `_weightEncode`
(`Deeploy/Targets/NE16/TopologyOptimizationPasses/Passes.py:24`). That pass rewrites
`weightTensor.values`, so it only applies when the weight is a `gs.Constant`.

In the QZO training graph the conv weight is the output of `RQSPerturbRademacher`, recomputed on
device twice per ZO step. This node closes that gap:

    RQSPerturbRademacher --int8 (cout,cin,H,W)--> NE16WeightEncode --uint8 (taps*cout,cinMajor,16)--> Conv

It runs on the CLUSTER cores (it is plain data movement), not on NE16 -- so it is registered in the
GAP9 cluster Mapping, and `NE16Engine.canExecute` (which only ever claims `Conv`) leaves it alone.

`weight_offset` is FIXED at -128, never `values.min()`: a data-dependent offset cannot work for a
weight that changes every step, and -128 makes the offset step a single XOR with 0x80 on device.
The node therefore carries no offset attribute; the consuming Conv keeps `weight_offset = -128`.

Attributes: `ne16_taps` (H*W of the original kernel), `ne16_bits` (8, the only supported width).
-- QW
"""

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import NetworkContext, NodeBinding, NodeParser, NodeTemplate, ONNXLayer, \
    OperatorRepresentation
from Deeploy.Targets.Generic.TypeCheckers import SignPropTypeChecker
from Deeploy.Targets.PULPOpen.Bindings import ForkTransformer

NE16_CIN_SUBTILE = 16

# ---------------------------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------------------------


class NE16WeightEncodeParser(NodeParser):

    def parseNode(self, node: gs.Node) -> bool:
        return all([
            node.op == "NE16WeightEncode",
            len(node.inputs) == 1,
            len(node.outputs) == 1,
            'ne16_taps' in node.attrs,
        ])

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)

        # The producer emits the weight in NCHW: (cout, cin, H, W) with H*W == taps.
        shape = tuple(int(s) for s in data_in.shape)
        if len(shape) != 4:
            return ctxt, False
        cout, cin, kh, kw = shape
        taps = int(node.attrs['ne16_taps'])
        # QW (dual_pass_fusion exp6): `ne16_fold=1` folds the kernel taps INTO the input channels --
        #     the (cout, cin, H, W) weight is encoded as a 1x1 weight with cin*H*W input channels.
        #     Memory order is unchanged (cout-major, then ci, ky, kx), so the C kernel needs no change:
        #     with taps = 1 it reads src[co*cin' + ci'] for ci' in 0..cin*H*W. This is the weight half
        #     of the GAP9 SDK's im2col route (Ker_MM_Conv1DSmallv2_NE16); the column buffer built by
        #     NE16Im2ColU8 uses the same (ci, ky, kx) order. Requires ne16_taps == 1. -- QW
        if int(node.attrs.get('ne16_fold', 0)) == 1:
            if taps != 1:
                return ctxt, False
            cin, kh, kw = cin * kh * kw, 1, 1
        if kh * kw != taps:
            return ctxt, False

        cinMajor = (cin + NE16_CIN_SUBTILE - 1) // NE16_CIN_SUBTILE
        bits = int(node.attrs.get('ne16_bits', 8))
        if bits != 8:
            return ctxt, False

        expected = (taps * cout, cinMajor, bits * NE16_CIN_SUBTILE // 8)
        if tuple(int(s) for s in data_out.shape) != expected:
            return ctxt, False

        self.operatorRepresentation['data_in'] = data_in.name
        self.operatorRepresentation['data_out'] = data_out.name
        self.operatorRepresentation['ne16_cout'] = cout
        self.operatorRepresentation['ne16_cin'] = cin
        self.operatorRepresentation['ne16_taps'] = taps
        self.operatorRepresentation['ne16_rows'] = taps * cout  # the parallelised dimension
        self.operatorRepresentation['size'] = int(np.prod(shape))
        return ctxt, True


# ---------------------------------------------------------------------------------------------
# Type checker -- int8 weight in, uint8 encoded bytes out
# ---------------------------------------------------------------------------------------------


class NE16WeightEncodeChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs, operatorRepresentation):
        return [256]

    def _inferSignedness(self, inputs, operatorRepresentation):
        return [False]  # the encoded buffer is opaque packed bits, read as uint8 by NE16


# ---------------------------------------------------------------------------------------------
# Template -- chunk the (taps*cout) rows over the cluster cores, exactly like
# RQSPerturbRademacherTemplate chunks the tensor. Each row is independent.
# ---------------------------------------------------------------------------------------------

referenceTemplate = NodeTemplate("""
// NE16WeightEncode (Name: ${nodeName}, Op: ${nodeOp}) -- bit-serial encode on the cluster -- QW
{
    uint32_t ${nodeName}_nrows = ${ne16_rows};
    uint32_t ${nodeName}_chunk = (${nodeName}_nrows + NUM_CORES - 1) / NUM_CORES;
    uint32_t ${nodeName}_start = MIN(${nodeName}_chunk * (uint32_t) pi_core_id(), ${nodeName}_nrows);
    uint32_t ${nodeName}_stop  = MIN(${nodeName}_start + ${nodeName}_chunk, ${nodeName}_nrows);

    NE16WeightEncode_i8_u8((const int8_t *) ${data_in},
                           (uint8_t *) ${data_out},
                           ${ne16_cout},
                           ${ne16_cin},
                           ${ne16_taps},
                           ${nodeName}_start,
                           ${nodeName}_stop - ${nodeName}_start);
}
""")

# ---------------------------------------------------------------------------------------------
# Binding + layer
# ---------------------------------------------------------------------------------------------

NE16WeightEncodeBindings = [
    NodeBinding(NE16WeightEncodeChecker([PointerClass(int8_t)], [PointerClass(uint8_t)]), referenceTemplate,
                ForkTransformer)
]


class NE16WeightEncodeLayer(ONNXLayer):

    def __init__(self, maps):
        super().__init__(maps)

    def computeOps(self):
        # one pass over the source weight, 8 bit-extractions per element
        return 8 * self.mapper.parser.operatorRepresentation['size']


# ---------------------------------------------------------------------------------------------
# QW (exp16c phase 4 / BLOCKER 3): NE16SignedInputBias
#
# NE16 reads activations as UNSIGNED, and CONFIG0 bit 26 (PR #183's `input_signed`) is undecoded
# by the hardware. SpeechNet block 0's activation is genuinely signed ([-66, 127]), so it is fed
# as x_u = x + 128 and the induced per-output-channel term removed afterwards:
#
#     sum w*x = sum w*x_u - 128 * sum w
#
# The RequantShift computes (acc*mul + add) >> log2(div), i.e. `add` lands AFTER the multiply, so
# correcting the accumulator by -128*sum_w is the same as correcting `add` by -128*sum_w*mul.
#
# This has to run on device: in QZO the weight is an RQSPerturbRademacher output that changes
# every ZO step, so `sum_w` is not a host-time constant.
# ---------------------------------------------------------------------------------------------


class NE16SignedInputBiasParser(NodeParser):

    def parseNode(self, node: gs.Node) -> bool:
        return all([
            node.op == "NE16SignedInputBias",
            len(node.inputs) == 3,
            len(node.outputs) == 1,
        ])

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        weight = ctxt.lookup(node.inputs[0].name)
        mul = ctxt.lookup(node.inputs[1].name)
        add = ctxt.lookup(node.inputs[2].name)
        out = ctxt.lookup(node.outputs[0].name)

        wShape = tuple(int(s) for s in weight.shape)
        if len(wShape) != 4:
            return ctxt, False
        cout = wShape[0]
        if tuple(int(s) for s in out.shape) != (cout,) or \
           int(np.prod(mul.shape)) != cout or int(np.prod(add.shape)) != cout:
            return ctxt, False

        self.operatorRepresentation['weight'] = weight.name
        self.operatorRepresentation['mul'] = mul.name
        self.operatorRepresentation['add'] = add.name
        self.operatorRepresentation['data_out'] = out.name
        self.operatorRepresentation['ne16_cout'] = cout
        self.operatorRepresentation['ne16_cin_taps'] = int(np.prod(wShape[1:]))
        self.operatorRepresentation['ne16_input_offset'] = int(node.attrs.get('ne16_input_offset', 128))
        self.operatorRepresentation['size'] = cout
        return ctxt, True


class NE16SignedInputBiasChecker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs, operatorRepresentation):
        return [2**32]

    def _inferSignedness(self, inputs, operatorRepresentation):
        return [True]


signedInputBiasTemplate = NodeTemplate("""
// NE16SignedInputBias (Name: ${nodeName}, Op: ${nodeOp}) -- BLOCKER 3 correction -- QW
{
    uint32_t ${nodeName}_ncout = ${ne16_cout};
    uint32_t ${nodeName}_chunk = (${nodeName}_ncout + NUM_CORES - 1) / NUM_CORES;
    uint32_t ${nodeName}_start = MIN(${nodeName}_chunk * (uint32_t) pi_core_id(), ${nodeName}_ncout);
    uint32_t ${nodeName}_stop  = MIN(${nodeName}_start + ${nodeName}_chunk, ${nodeName}_ncout);

    NE16SignedInputBias_i32((const int8_t *) ${weight},
                            (const int32_t *) ${mul},
                            (const int32_t *) ${add},
                            (int32_t *) ${data_out},
                            ${ne16_cout},
                            ${ne16_cin_taps},
                            ${ne16_input_offset},
                            ${nodeName}_start,
                            ${nodeName}_stop - ${nodeName}_start);
}
""")

NE16SignedInputBiasBindings = [
    NodeBinding(
        NE16SignedInputBiasChecker([PointerClass(int8_t), PointerClass(int32_t),
                                    PointerClass(int32_t)], [PointerClass(int32_t)]),
        signedInputBiasTemplate, ForkTransformer)
]


class NE16SignedInputBiasLayer(ONNXLayer):

    def __init__(self, maps):
        super().__init__(maps)

    def computeOps(self):
        return self.mapper.parser.operatorRepresentation['ne16_cout'] * \
            self.mapper.parser.operatorRepresentation['ne16_cin_taps']
