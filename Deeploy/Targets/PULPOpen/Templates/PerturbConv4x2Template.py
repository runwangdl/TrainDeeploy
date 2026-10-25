# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp11): template for `RequantizedDualConv` with `single_pass` = 1 -- ONE call to
`PULP_PerturbConv4x2_i8_i8_i8_HWC` (TargetLibraries/PULPOpen/src/PerturbConv4x2.c): the raw CHW weight and the requant
add perturbed in the kernel for this pass's sign (runtime `perturbation_sign`, as RQSPerturbRademacher), then the
Conv4x2 step-5 conv. Seeds as DualConv2InTemplate (= the two RQSPerturbRademacher nodes it replaces).
"""
from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

_MAX_CORES = 8


class _PerturbConv4x2Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        operatorRepresentation['tile_seed_offset'] = 0  # whole weight in one tile (tile constraint)
        return ctxt, operatorRepresentation, []

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # MUST match PerturbConv4x2.c: W (HWC) | add (4-aligned) | two K-byte patches per core
        K = operatorRepresentation['in_ch'] * operatorRepresentation['dim_kernel_x'] * operatorRepresentation[
            'dim_kernel_y']
        cout = operatorRepresentation['ch_out_pass']
        size = ((cout * K + 3) // 4) * 4 + 4 * cout + _MAX_CORES * 2 * K
        return [(operatorRepresentation['nodeName'] + "_buffer", size)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        name, size = _PerturbConv4x2Template.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(name, size)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size
        return ctxt, operatorRepresentation, [name]


referenceTemplate = _PerturbConv4x2Template("""
// PULP SINGLE-PASS QZO CONV, PERTURBATION IN THE KERNEL (Name: ${nodeName}, Op: ${nodeOp})  -- QW exp11
uint32_t ${nodeName}_seed_w = ((${seed_w} + perturb_seed_base) + NUM_CORES * ${idx_w});
uint32_t ${nodeName}_seed_b = ((${seed_b} + perturb_seed_base) + NUM_CORES * ${idx_b});
uint32_t ${nodeName}_tile_xor = (${tile_seed_offset} * 0x9E3779B1u);
float32_t ${nodeName}_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
PULP_PerturbConv4x2_i8_i8_i8_HWC(${data_in}, ${weight}, ${mul}, ${add}, ${wmul}, ${bmul}, ${data_out},
    (int8_t *)${ctxtBuffer}, ${sign_expr}, ${nodeName}_seed_w, ${nodeName}_seed_b, ${nodeName}_tile_xor,
    ${log2D_w}, ${log2D_b}, ${nodeName}_eps_scale, NUM_CORES,
    ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${in_ch}, ${dim_im_out_y}, ${dim_im_out_x}, ${ch_im_out},
    ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
    ${stride_y}, ${stride_x});
""")
