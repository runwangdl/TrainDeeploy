# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp7a, step 2a): template for the single-pass control kernel `PULP_Conv4x2_i8_i8_i8_HWC`
(TargetLibraries/PULPOpen/src/Conv4x2.c). Binds to the ordinary `RequantizedConv` node in place of
pulp_nn_conv_i8_i8_i8; selected with DEEPLOY_RQCONV_KERNEL=4x2 (PULPOpen/Platform.py).

Argument keys and order are FlatConvTemplate's (= the RQ conv template's), so RQConv2DTileConstraint's per-tile
replacements apply unchanged. Deeploy's dim_im_in_y is the kernel's dim_in_x, as for pulp-nn.
Scratch: two K-byte patches per core (for the two pixels of a 4x2 step that touch the padding).
"""
from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

_MAX_CORES = 8  # PULP2DConvTemplate hardcodes the same 8


class _Conv4x2Template(NodeTemplate):

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
        return [(name, _MAX_CORES * 2 * Kpad)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        name, size = _Conv4x2Template.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(name, size)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size
        return ctxt, operatorRepresentation, [name]


referenceTemplate = _Conv4x2Template("""
// PULP CONV 4x2 (single-pass control kernel for exp7a step 2a; Name: ${nodeName}, Op: ${nodeOp})  -- QW
PULP_Conv4x2_i8_i8_i8_HWC(${data_in}, (int8_t *)${ctxtBuffer}, ${data_out}, ${weight}, ${mul}, ${add}, NUM_CORES,
    ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in}, ${dim_im_out_y}, ${dim_im_out_x}, ${ch_im_out},
    ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
    ${stride_y}, ${stride_x});
""")
