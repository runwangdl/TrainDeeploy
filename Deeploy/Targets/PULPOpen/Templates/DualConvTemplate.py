# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp5): template for the fused dual-pass QZO conv (`RequantizedDualConv`).

Emits ONE call to `PULP_DualConv2d_i8_i8_i8_HWC` (TargetLibraries/PULPOpen/src/DualConv.c), which
perturbs the UNPERTURBED weight / requant-add internally (both signs from one Rademacher stream) and
produces A+ and A- in a single traversal. The seed expression is the one the two standalone
RQSPerturbRademacher nodes would have used -- `(seed + perturb_seed_base) + NUM_CORES*idx + core_id`,
XOR the per-tile offset -- so the perturbed weights are bit-identical to the two-node graph.

Argument order mirrors ConvTemplate.PULPConv2D_8_Template exactly (Deeploy's dim_im_in_y is the
kernel's dim_in_x, etc.), so the RQ conv tile constraint's replacement keys apply unchanged. The
output tensor carries 2*ch_out channels; the kernel takes the per-pass count, hence `>> 1`.
"""
from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

# Scratch is sized for this many cores' private patches. PULP2DConvTemplate hardcodes the same 8.
_MAX_CORES = 8


class _DualConvTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Per-tile seed offset; 0 when untiled. The tile constraint forces the whole weight into one
        # tile, so it stays 0 -- kept for parity with RQSPerturbRademacherTemplate.
        operatorRepresentation['tile_seed_offset'] = 0
        return ctxt, operatorRepresentation, []

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # MUST match the layout in DualConv.c: W+ | W- | add+ (4-aligned) | add- | one patch per core.
        K = operatorRepresentation['ch_im_in'] * operatorRepresentation['dim_kernel_x'] * operatorRepresentation[
            'dim_kernel_y']
        Kpad = ((K + 3) // 4) * 4
        cout = operatorRepresentation['ch_out_pass']
        wsize = cout * K
        off_addp = ((2 * wsize + 3) // 4) * 4
        size = off_addp + 8 * cout + _MAX_CORES * Kpad
        name = operatorRepresentation['nodeName'] + "_buffer"
        return [(name, size)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        name, size = _DualConvTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(name, size)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size
        return ctxt, operatorRepresentation, [name]


referenceTemplate = _DualConvTemplate("""
// PULP DUAL-PASS QZO CONV (Name: ${nodeName}, Op: ${nodeOp})  -- QW exp5
// Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
uint32_t ${nodeName}_seed_w = ((${seed_w} + perturb_seed_base) + NUM_CORES * ${idx_w});
uint32_t ${nodeName}_seed_b = ((${seed_b} + perturb_seed_base) + NUM_CORES * ${idx_b});
uint32_t ${nodeName}_tile_xor = (${tile_seed_offset} * 0x9E3779B1u);
float32_t ${nodeName}_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
PULP_DualConv2d_i8_i8_i8_HWC(${data_in}, ${weight}, ${mul}, ${add}, ${wmul}, ${bmul}, ${data_out}, (int8_t *)${ctxtBuffer},
    ${nodeName}_seed_w, ${nodeName}_seed_b, ${nodeName}_tile_xor, ${log2D_w}, ${log2D_b}, ${nodeName}_eps_scale, NUM_CORES,
    ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in}, ${dim_im_out_y}, ${dim_im_out_x}, ((${ch_im_out}) >> 1),
    ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
    ${stride_y}, ${stride_x});
""")
