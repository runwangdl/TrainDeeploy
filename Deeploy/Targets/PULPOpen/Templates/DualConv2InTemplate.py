# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (dual_pass_fusion exp7, step 3): template for the 7-input `RequantizedDualConv` (SpeechNet blocks 1-4).

Emits ONE call to `PULP_DualConv2d_2in_i8_i8_i8_HWC` (TargetLibraries/PULPOpen/src/DualConv2In.c): x+ and x-
as two inputs, the UNPERTURBED weight in raw CHW order (perturbed inside, in CHW order, written HWC), A+ and A-
into the stacked output [A+ | A-] (pOutP = out, pOutM = out + ch_out, pixel stride 2*ch_out).

Seeds are DualConvTemplate's (= the two standalone RQSPerturbRademacher nodes'), and the argument order after
the pointers mirrors it, so the tile constraint's replacement keys apply unchanged. The output tensor carries
2*ch_out channels; the kernel takes the per-pass count, hence `>> 1`.
"""
from typing import Dict, List, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

_MAX_CORES = 8  # PULP2DConvTemplate hardcodes the same 8


class _DualConv2InTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # The tile constraint keeps the whole weight in one tile, so the per-tile seed offset stays 0.
        operatorRepresentation['tile_seed_offset'] = 0
        return ctxt, operatorRepresentation, []

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # MUST match DualConv2In.c: W+ | W- | add+ (4-aligned) | add- | four K-byte patches per core.
        K = operatorRepresentation['in_ch'] * operatorRepresentation['dim_kernel_x'] * operatorRepresentation[
            'dim_kernel_y']  # per-pass input channels (exp8: ch_im_in is 2*Ci for a stacked input)
        cout = operatorRepresentation['ch_out_pass']
        wsize = cout * K
        off_addp = ((2 * wsize + 3) // 4) * 4
        size = off_addp + 8 * cout + _MAX_CORES * 4 * K
        # exp8: stacked input with 1xK rows -> the kernel de-interleaves the input tile into x+ | x- here (whole
        # tensor = upper bound of any tile); weight is raw CHW [co, ci, kh, kw]
        if operatorRepresentation.get('stacked_input', 0) and ctxt.lookup(operatorRepresentation['weight']).shape[3] != 1:
            size += int(np.prod(ctxt.lookup(operatorRepresentation['data_in']).shape))
        name = operatorRepresentation['nodeName'] + "_buffer"
        return [(name, size)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        name, size = _DualConv2InTemplate.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(name, size)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size
        return ctxt, operatorRepresentation, [name]


referenceTemplate = _DualConv2InTemplate("""
// PULP DUAL-PASS QZO CONV, TWO INPUTS (Name: ${nodeName}, Op: ${nodeOp})  -- QW exp7
// Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
uint32_t ${nodeName}_seed_w = ((${seed_w} + perturb_seed_base) + NUM_CORES * ${idx_w});
uint32_t ${nodeName}_seed_b = ((${seed_b} + perturb_seed_base) + NUM_CORES * ${idx_b});
uint32_t ${nodeName}_tile_xor = (${tile_seed_offset} * 0x9E3779B1u);
float32_t ${nodeName}_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
PULP_DualConv2d_2in_i8_i8_i8_HWC(${data_in}, ${data_in_m}, ${weight}, ${mul}, ${add}, ${wmul}, ${bmul},
    ${data_out}, ${data_out} + ((${ch_im_out}) >> 1), (${ch_im_out}), ${in_stride}, (int8_t *)${ctxtBuffer},
    ${nodeName}_seed_w, ${nodeName}_seed_b, ${nodeName}_tile_xor, ${log2D_w}, ${log2D_b}, ${nodeName}_eps_scale, NUM_CORES,
    ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${in_ch}, ${dim_im_out_y}, ${dim_im_out_x}, ((${ch_im_out}) >> 1),
    ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
    ${stride_y}, ${stride_x});
""")

# -- QW (exp8): the stacked-input form (dual-pass graph): x- = x + Ci, pixel stride 2*Ci. -- QW
stackedTemplate = _DualConv2InTemplate("""
// PULP DUAL-PASS QZO CONV, STACKED [x+ | x-] INPUT (Name: ${nodeName}, Op: ${nodeOp})  -- QW exp8
// Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
uint32_t ${nodeName}_seed_w = ((${seed_w} + perturb_seed_base) + NUM_CORES * ${idx_w});
uint32_t ${nodeName}_seed_b = ((${seed_b} + perturb_seed_base) + NUM_CORES * ${idx_b});
uint32_t ${nodeName}_tile_xor = (${tile_seed_offset} * 0x9E3779B1u);
float32_t ${nodeName}_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
PULP_DualConv2d_2in_i8_i8_i8_HWC(${data_in}, ${data_in} + ${in_ch}, ${weight}, ${mul}, ${add}, ${wmul}, ${bmul},
    ${data_out}, ${data_out} + ((${ch_im_out}) >> 1), (${ch_im_out}), ${in_stride}, (int8_t *)${ctxtBuffer},
    ${nodeName}_seed_w, ${nodeName}_seed_b, ${nodeName}_tile_xor, ${log2D_w}, ${log2D_b}, ${nodeName}_eps_scale, NUM_CORES,
    ${log2D}, ${dim_im_in_y}, ${dim_im_in_x}, ${in_ch}, ${dim_im_out_y}, ${dim_im_out_x}, ((${ch_im_out}) >> 1),
    ${dim_kernel_y}, ${dim_kernel_x}, ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right},
    ${stride_y}, ${stride_x});
""")
