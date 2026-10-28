# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp5a): template for the single-pass control kernel `PULP_FlatConv2d_i8_i8_i8_HWC`
(TargetLibraries/PULPOpen/src/FlatConv.c). Binds to the ordinary `RequantizedConv` node in place of
pulp_nn_conv_i8_i8_i8 -- the switch is the `Conv2DMapper` line in PULPOpen/Platform.py (see exp5a Plan §4).

Argument keys are the RQ conv template's (ConvTemplate.PULPConv2D_8_Template), so RQConv2DTileConstraint's
per-tile replacements apply unchanged. Deeploy's dim_im_in_y is the kernel's dim_in_x, as for pulp-nn.
Scratch: one Kpad-byte patch per core (pulp-nn's template hoists a 2*8*K im2col buffer instead).
"""
from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

_MAX_CORES = 8  # PULP2DConvTemplate hardcodes the same 8


class _FlatConvTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        K = operatorRepresentation['ch_im_in'] * operatorRepresentation['dim_kernel_x'] * operatorRepresentation[
            'dim_kernel_y']
        Kpad = ((K + 3) // 4) * 4
        name = operatorRepresentation['nodeName'] + "_buffer"
        return [(name, _MAX_CORES * Kpad)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        name, size = _FlatConvTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(name, size)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size
        return ctxt, operatorRepresentation, [name]


referenceTemplate = _FlatConvTemplate("""
// PULP FLAT CONV (single-pass control kernel for exp5a; Name: ${nodeName}, Op: ${nodeOp})  -- QW
PULP_FlatConv2d_i8_i8_i8_HWC(${data_in}, (int8_t *)${ctxtBuffer}, ${data_out}, ${weight}, ${mul}, ${add}, NUM_CORES,
    ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in}, ${dim_im_out_y}, ${dim_im_out_x}, ${ch_im_out},
    ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
    ${stride_y}, ${stride_x});
""")
