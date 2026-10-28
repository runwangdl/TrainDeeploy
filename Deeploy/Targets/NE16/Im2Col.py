# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp6): the `NE16Im2ColU8` op -- the cluster half of the GAP9 SDK's im2col route.

The GAP9 SDK runs a 1xK convolution on NE16 as (a) the cluster cores building a column buffer --
for every output pixel the `ch_in*K` input values under the kernel window, as one contiguous row --
and (b) NE16 running an ordinary 1x1 convolution over `ch_in*K` input channels
(`Ker_MM_Conv1DSmallv2_NE16`, CNN_BasicKernels_NE16.c:4551). This op is (a). (b) is the existing
NE16 pointwise path, unchanged, fed with the folded weight through `NE16WeightEncode(ne16_taps=1)`.

Doing the copy on the cores lets two things that NE16 cannot do happen for free:
  * padding -- padded taps are written as `pad_value` (NE16's 1x1 mode cannot pad, and GVSoC's
    guard for it is disabled, so a padded 1x1 job fails silently);
  * the +128 offset a SIGNED activation needs (NE16 reads inputs as unsigned; GAP9's NE16 has no
    signed-input bit) -- `x ^ 0x80` while copying; the induced term is removed by
    `NE16SignedInputBias`, exactly as in exp16c.

Input : NCHW `[1, ch_in, H, W]` int8 (the activation as the graph stores it).
Output: NHWC `[1, Ho, Wo, ch_in*kh*kw]` uint8, column order (ci, ky, kx) == the flattening order of
        an NCHW weight `[co][ci][ky][kx]`, so the folded 1x1 weight is the perturbed weight reshaped.
Attributes: `kernel_shape`, `pads` [top, left, bottom, right], `strides`, `ne16_input_offset` (128).

Runs on the CLUSTER (registered in GAP9Mapping like NE16WeightEncode); NE16Engine never claims it.
Kernel: TargetLibraries/PULPOpen/src/Im2ColU8.c (compiled into both the PULPOpen and GAP9 libs). -- QW
"""

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t, uint8_t
from Deeploy.DeeployTypes import NetworkContext, NodeBinding, NodeParser, NodeTemplate, ONNXLayer
from Deeploy.Targets.Generic.TypeCheckers import SignPropTypeChecker
from Deeploy.Targets.PULPOpen.Bindings import ForkTransformer


class NE16Im2ColU8Parser(NodeParser):

    def parseNode(self, node: gs.Node) -> bool:
        if not all([
                node.op == "NE16Im2ColU8",
                len(node.inputs) == 1,
                len(node.outputs) == 1,
                'kernel_shape' in node.attrs,
                'pads' in node.attrs,
        ]):
            return False
        kh, kw = (int(v) for v in node.attrs['kernel_shape'])
        pads = [int(v) for v in node.attrs['pads']]
        strides = [int(v) for v in node.attrs.get('strides', [1, 1])]
        if len(pads) != 4 or len(strides) != 2:
            return False
        self.operatorRepresentation['ky_n'] = kh
        self.operatorRepresentation['kx_n'] = kw
        self.operatorRepresentation['pad_top'], self.operatorRepresentation['pad_left'] = pads[0], pads[1]
        self.operatorRepresentation['pad_bottom'], self.operatorRepresentation['pad_right'] = pads[2], pads[3]
        self.operatorRepresentation['stride_y'], self.operatorRepresentation['stride_x'] = strides
        self.operatorRepresentation['input_offset'] = int(node.attrs.get('ne16_input_offset', 128))
        return True

    def parseNodeCtxt(self,
                      ctxt: NetworkContext,
                      node: gs.Node,
                      channels_first: bool = True) -> Tuple[NetworkContext, bool]:
        data_in = ctxt.lookup(node.inputs[0].name)
        data_out = ctxt.lookup(node.outputs[0].name)
        inShape = tuple(int(s) for s in data_in.shape)
        outShape = tuple(int(s) for s in data_out.shape)
        if len(inShape) != 4 or len(outShape) != 4 or inShape[0] != 1 or outShape[0] != 1:
            return ctxt, False
        _, ch_in, H, W = inShape
        _, Ho, Wo, K = outShape
        r = self.operatorRepresentation
        if K != ch_in * r['ky_n'] * r['kx_n']:
            return ctxt, False
        if Ho != (H + r['pad_top'] + r['pad_bottom'] - r['ky_n']) // r['stride_y'] + 1 or \
           Wo != (W + r['pad_left'] + r['pad_right'] - r['kx_n']) // r['stride_x'] + 1:
            return ctxt, False
        r['data_in'] = data_in.name
        r['data_out'] = data_out.name
        r['ch_in'], r['in_rows'], r['in_cols'] = ch_in, H, W
        r['out_rows'], r['out_cols'], r['col_len'] = Ho, Wo, K
        r['size'] = int(np.prod(outShape))
        # 128 == 0x80: the offset is a single XOR on int8; the pad value is the offset itself (= 0 signed)
        r['xor_mask'] = r['input_offset'] & 0xFF
        r['pad_value'] = r['input_offset'] & 0xFF
        return ctxt, True


class NE16Im2ColU8Checker(SignPropTypeChecker):

    def _inferNumLevels(self, inputs, operatorRepresentation):
        return [256]

    def _inferSignedness(self, inputs, operatorRepresentation):
        return [False]  # x + 128, read by NE16 as uint8


referenceTemplate = NodeTemplate("""
// NE16Im2ColU8 (Name: ${nodeName}, Op: ${nodeOp}) -- column buffer for the NE16 im2col route -- QW exp6
PULP_Im2ColU8_NCHW_to_HWC((const int8_t *) ${data_in}, (uint8_t *) ${data_out}, NUM_CORES,
    ${in_rows}, ${in_cols}, ${ch_in}, ${out_rows}, ${out_cols}, ${ky_n}, ${kx_n},
    ${pad_top}, ${pad_bottom}, ${pad_left}, ${pad_right}, ${stride_y}, ${stride_x},
    (uint8_t) ${xor_mask}, (uint8_t) ${pad_value});
""")

NE16Im2ColU8Bindings = [
    NodeBinding(NE16Im2ColU8Checker([PointerClass(int8_t)], [PointerClass(uint8_t)]), referenceTemplate,
                ForkTransformer)
]


class NE16Im2ColU8Layer(ONNXLayer):

    def __init__(self, maps):
        super().__init__(maps)

    def computeOps(self):
        return self.mapper.parser.operatorRepresentation['size']
