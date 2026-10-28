# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (exp16b): execute a 1xK dense convolution as ceil(K/3) NE16 **3x3 DENSE** dispatches.

The alternative to exp16a's all-pointwise decomposition. A 1x3 slice of the kernel is embedded in
a 3x3 kernel with rows 0 and 2 zeroed:

        [  0     0      0   ]
        [ w3c  w3c+1  w3c+2 ]   <- chunk c holds taps 3c, 3c+1, 3c+2
        [  0     0      0   ]

so chunk c is an ordinary NE16 3x3 job with `infeat_addr += 3*c*ch_im_in` (the same address-offset
trick as the pointwise path, stepping by 3 instead of 1). Chunks 1..N-1 carry CONFIG0[14] streamin,
so the partial sums accumulate inside NE16 exactly as in exp16a; output stays int32 and the
RequantShift remains a separate cluster node.

Unlike 1x1 mode, **padding is legal in 3x3 mode** (gvsoc fsm.cpp:53 restricts filter masking and
padding to fs==3), so H padding 1/1 is used natively to keep Hout = Hin. The zero kernel rows make
whatever the padded values are irrelevant.

Cost model (ne16_matrixvec.cpp:316): 3x3 mode spends the 9 row-slots per column on the 9 spatial
TAPS and therefore walks the 8 bitplanes sequentially (`mv_qw_lim = qw`), where 1x1 mode spends
them on the bitplanes and finishes in ONE pass. With only 3 of 9 taps real, utilisation is 3/9 =
54 MAC/cycle against pointwise's 8/9 = 144. exp16b exists to MEASURE that rather than assume it.

Uses only fields pulp-nnx's ne16_task_t already has -- no register, no ISA change. -- QW
"""

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.NE16.Templates.ConvTemplate import NE162DDenseConvTemplate

NE16_FLAG_STREAMIN = 1 << 14  # pulp-nnx ne16/hal/ne16_task_defs.h:79
NE16_CHUNK = 3  # a 3x3 kernel covers 3 taps of a 1xK filter


class NE162D3x3ChunkConvTemplate(NE162DDenseConvTemplate):
    """ceil(K/3) dense 3x3 dispatches with per-chunk pointer offsets and streamin accumulation."""

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Counters, strides and conf0 for ONE 3x3 job come from the dense path.
        ctxt, operatorRepresentation, _ = super().alignToContext(ctxt, operatorRepresentation)

        weight = ctxt.lookup(operatorRepresentation['weight'])
        chunks = int(operatorRepresentation['ne16_chunks'])
        assert len(weight.shape) == 4 and int(weight.shape[0]) % chunks == 0, \
            f"3x3-chunk weight must be (chunks*cout, cinMajor, bits, HW*cinMinorBytes); got {weight.shape}"

        operatorRepresentation['weight_chunk_bytes'] = int(np.prod(weight.shape)) // chunks
        # NHWC: stepping NE16_CHUNK pixels along W is NE16_CHUNK * ch_im_in input elements
        operatorRepresentation['input_chunk_bytes'] = NE16_CHUNK * int(operatorRepresentation['ch_im_in']) * \
            int(operatorRepresentation['input_typeWidth_bytes'])
        operatorRepresentation['conf0_streamin'] = int(operatorRepresentation['conf0']) | NE16_FLAG_STREAMIN

        # QW: filter masking was TRIED and REVERTED. `load_filter_masking` (ne16_load.cpp:424)
        #     builds row_enable, which `__weightoffs` also receives -- so masking should have
        #     skipped the zero taps in the Wmin*Sum(x) pass too. Measured: errors went 431 -> 557
        #     (WORSE). The mask field's top/bottom may index columns rather than rows given
        #     `W_mask[i + j*fs]`, or masking interacts with padding. Left at 0; see Findings.
        operatorRepresentation['ne16_filter_mask'] = 0  # -- QW

        assert operatorRepresentation['mul'] == 'NULL', \
            "3x3-chunk decomposition needs int32 output (streamin requires quantization_bits==32)"
        return ctxt, operatorRepresentation, []


# QW (exp16c_SDK_port phase 1): pipelined over NE16's 2-deep job queue, same rationale and same
# safety argument as Conv1xKTemplate.py -- see the long comment there. -- QW
NE163x3ChunkTaskTemplateStr = """
// NE16 1xK dense conv as ${ne16_chunks} DENSE 3x3 dispatches (streamin accumulation, exp16b)
// Pipelined over NE16's 2-deep job queue -- see Conv1xKTemplate.py. -- QW
{
ne16_task_t task;
% for _chunk in range(ne16_chunks):
{
    task = (ne16_task_t) {
        .data = (ne16_task_data_t) {
            .weights_addr = (uint32_t)${weight} + ${_chunk} * ${weight_chunk_bytes},
            .infeat_addr = (uint32_t)${data_in} + ${_chunk} * ${input_chunk_bytes} - ${input_addr_offset},
            .outfeat_addr = (uint32_t)${data_out},
            .scale_addr = (uint32_t)${mul},
            .scale_shift_addr = (uint32_t)${shift},
            .scale_bias_addr = (uint32_t)${add},
            .cfg = (ne16_cfg_t) {
                .input_stride = (ne16_stride_t) {
                    .d0 = ${dim_im_in_y_stride},
                    .d1 = ${dim_im_in_x_stride},
                    .d2 = 0
                },
                .output_stride = (ne16_stride_t) {
                    .d0 = NE16_OUTPUT_BANDWIDTH_BYTES,
                    .d1 = ${dim_im_out_y_stride},
                    .d2 = ${dim_im_out_x_stride}
                },
                .weights_stride = (ne16_stride_t) {
                    .d0 = ${weightStrideD0},
                    .d1 = ${weightStrideD1},
                    .d2 = ${weightStrideD2}
                },
                .subtile = (ne16_subtile_t) {
                    .number = {
                        .KoKi = nnx_concat_half(${nKo}, ${nKi}),
                        .HoWo = nnx_concat_half(${nHo}, ${nWo})
                    },
                    .remainder = {
                        .KoKi = nnx_concat_half(${bKo}, ${bKi}),
                        .HoWo = nnx_concat_half(${bHo}, ${bWo}),
                        .HiWi = nnx_concat_half(${bHi}, ${bWi})
                    }
                },
                // H padding 1/1 keeps Hout = Hin; legal because this is 3x3 mode, and harmless
                // because the kernel's top and bottom rows are zero.
                .padding = (${padding_y_top} << 28) + (${padding_x_right} << 24) + (${padding_y_bottom} << 20) + (${padding_x_left} << 16),
                .weight_offset_factor = ${weight_offset},
                .filter_mask = ${ne16_filter_mask},
                .conf0 = ${conf0 if loop.index == 0 else conf0_streamin},
            }
        }
    };
    task.weight_d0_stride = NE16_WEIGHT_D0_STRIDE_MODE8;
    task.qw = ${weight_bits};
    task.subtile_output_channel = ${ne16_subtile_output_channel};
    task.kernel_shape = ${ne16_kernel_shape};
    task.depthwise = ${ne16_depthwise};

    // Block only while both job contexts are busy, then program this chunk into the free one.
    ne16_nnx_dispatch_wait(ne16_pulp_get_dev());
    ne16_nnx_dispatch(ne16_pulp_get_dev(), &task);
}
% endfor
// Wait for the whole chunk chain exactly once, not once per chunk.
ne16_nnx_resolve_wait(ne16_pulp_get_dev(), &task);
}
"""

NE163x3ChunkConv2D_Template = NE162D3x3ChunkConvTemplate(NE163x3ChunkTaskTemplateStr)
