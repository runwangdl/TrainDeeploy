# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""QW (exp16a / STEP 2b): execute a 1xK (or Kx1) dense convolution as K POINTWISE NE16 dispatches.

NE16 has only three filter modes (CONFIG0[6:5]: 3x3, 3x3-depthwise, 1x1), so SpeechNet's
1x4 / 1x16 / 1x8 / 7x1 kernels have no native mode. But a 1xK convolution is exactly K pointwise
convolutions over K input windows shifted along W:

    conv_{1xK}(W, X)[co,h,w] = sum_{j=0..K-1} conv_{1x1}(W[:,:,0,j], X)[co,h,w+j]

and in NHWC the shift along W is simply `+ j * ch_im_in` bytes on the input pointer. So each tap
is an ordinary NE16 1x1 job with two pointers moved:

    infeat_addr  += j * ch_im_in * input_typeWidth_bytes
    weights_addr += j * <one tap's encoded block>

The K partial sums accumulate INSIDE NE16 via CONFIG0[14] streamin, which preloads the
accumulator from the output buffer instead of zeroing it (gvsoc ne16_streamin.cpp:52). streamin
requires 32-bit output (`assert(!(streamin && quantization_bits != 32))`, gvsoc fsm.cpp:50), so
this template is for the **int32-output `Conv`** only -- the matching `RequantShift` stays on the
cluster. That is also why this template never enables normquant.

Only the two pointers and one config bit vary across the K dispatches: every counter, stride and
the rest of conf0 are computed once by NE162DPWConvTemplate.alignToContext. The weight is
expected PRE-ENCODED per tap, stacked (K, cout, cinMajor, bits*cinMinorBytes), so tap j's block is
byte-identical to a standalone 1x1 conv's weight at offset j * prod(shape[1:]).

Uses only fields pulp-nnx's ne16_task_t already has -- no register, no ISA change, so this runs on
real GAP9 silicon exactly as on gvsoc. -- QW
"""

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.NE16.Templates.ConvTemplate import NE162DPWConvTemplate

NE16_FLAG_STREAMIN = 1 << 14  # pulp-nnx ne16/hal/ne16_task_defs.h:79


class NE162D1xKConvTemplate(NE162DPWConvTemplate):
    """K pointwise dispatches with per-tap pointer offsets and streamin accumulation."""

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Everything a single 1x1 job needs -- counters, strides, conf0 -- comes from the PW path.
        ctxt, operatorRepresentation, _ = super().alignToContext(ctxt, operatorRepresentation)

        weight = ctxt.lookup(operatorRepresentation['weight'])
        taps = int(operatorRepresentation['ne16_taps'])
        # RANK 3 on purpose: (taps*cout, cinMajor, encBytes). A rank-4 weight would be permuted
        # by PULPNCHWtoNHWCPass, which treats any 4-D conv input as an activation and would
        # destroy the bit-serial encoding.
        assert len(weight.shape) == 3 and int(weight.shape[0]) % taps == 0, \
            f"1xK weight must be (taps*cout, cinMajor, encBytes); got {weight.shape} for taps={taps}"

        # one tap's encoded block, in bytes (weights are encoded as bytes -> no typeWidth scaling)
        operatorRepresentation['weight_tap_bytes'] = int(np.prod(weight.shape)) // taps
        # NHWC: stepping one pixel along W is exactly ch_im_in input elements
        operatorRepresentation['input_tap_bytes'] = int(operatorRepresentation['ch_im_in']) * \
            int(operatorRepresentation['input_typeWidth_bytes'])
        operatorRepresentation['conf0_streamin'] = int(operatorRepresentation['conf0']) | NE16_FLAG_STREAMIN
        operatorRepresentation['ne16_streamin_flag'] = NE16_FLAG_STREAMIN

        # QW (exp16c / STEP 3): SpeechNet blocks 3 and 4 are 7x1, i.e. the taps run along H, not W.
        #     Stepping one tap is then stepping one ROW, not one pixel. The row stride is
        #     `dim_im_in_x_stride` (`ioStridesFromDimensions` returns (height_stride, width_stride)
        #     and the constraint assigns them to x_stride, y_stride in that order -- the names are
        #     the transposed ones). It is a per-TILE value, so the template has to reference the
        #     substituted variable rather than a constant computed here. -- QW
        kh, kw = (int(v) for v in operatorRepresentation['kernel_shape'])
        assert 1 in (kh, kw), f"1xK / Kx1 decomposition only; got kernel_shape {(kh, kw)}"
        operatorRepresentation['ne16_tap_axis'] = 2 if kh == 1 else 1  # 2 = W, 1 = H

        assert operatorRepresentation['mul'] == 'NULL', \
            "1xK decomposition needs int32 output (streamin requires quantization_bits==32); " \
            "the RequantShift must stay a separate cluster node"
        return ctxt, operatorRepresentation, []


# One ne16_task_t is built per tap. Only weights_addr, infeat_addr and conf0 differ; the struct is
# otherwise identical to the single-dispatch pointwise task in ConvTemplate.py.
#
# QW (exp16c_SDK_port phase 1): the dispatch loop is PIPELINED, matching the GAP9 SDK's own 1-D
# NE16 kernel (`KerConv1D_StrideS_NE16`, CNN_BasicKernels_NE16.c:1643, which does
# "already commit and trigger NE16 computation, while programming the next one").
#
# NE16 has a 2-deep job queue (NE16_TASK_QUEUE_SIZE, pulp-nnx ne16/hal/ne16.h:27).
# `ne16_nnx_dispatch_wait` blocks only while the queue is FULL, so the core can program tap j+1's
# registers into the free context while tap j is still computing. The previous revision called
# `ne16_nnx_resolve_wait` after EVERY dispatch, and under GVSoC that reduces to
# `ne16_task_queue_empty(dev)` (pulp_nnx_ne16.c:57) -- a full drain -- so every tap ran serialised
# and the second job context was never used.
#
# This is SAFE with streamin, which creates a RAW dependency from tap j's output to tap j+1's
# accumulator preload: GVSoC's `fsm_end_handler` (gap/ne16/src/fsm.cpp:88) decrements job_pending,
# flips cxt_use_ptr and only THEN enqueues the next job, while `fsm_start_handler` sets
# job_running = 1. Queued jobs therefore execute ONE AT A TIME, IN ORDER -- the queue is a
# program-ahead buffer, not concurrency. Verified before this change was made.
#
# The task struct is declared ONCE outside the loop and reassigned per tap: `ne16_nnx_dispatch`
# copies the descriptor into the accelerator's register context immediately
# (`hwpe_task_queue_write_task`), so reuse is safe, and keeping the last task alive lets the final
# `ne16_nnx_resolve_wait` carry a valid task id on real silicon (where, unlike GVSoC, the check
# uses task->id). -- QW
NE161xKTaskTemplateStr = """
// NE16 1x${ne16_taps} dense conv as ${ne16_taps} pointwise dispatches (streamin accumulation)
// Pipelined over NE16's 2-deep job queue -- see Conv1xKTemplate.py. -- QW
{
ne16_task_t task;
% for _tap in range(ne16_taps):
{
    task = (ne16_task_t) {
        .data = (ne16_task_data_t) {
            .weights_addr = (uint32_t)${weight} + ${_tap} * ${weight_tap_bytes},
% if ne16_tap_axis == 1:
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr = (uint32_t)${data_in} + ${_tap} * ${dim_im_in_x_stride} - ${input_addr_offset},
% else:
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)${data_in} + ${_tap} * ${input_tap_bytes} - ${input_addr_offset},
% endif
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
                .padding = 0,
                .weight_offset_factor = ${weight_offset},
                .filter_mask = 0,
                // tap 0 zeroes the accumulator; taps 1..K-1 stream the running int32 partial
                // sums back in from outfeat_addr (CONFIG0[14]).
                .conf0 = ${conf0 if loop.index == 0 else conf0_streamin},
            }
        }
    };
    task.weight_d0_stride = NE16_WEIGHT_D0_STRIDE_MODE8;
    task.qw = ${weight_bits};
    task.subtile_output_channel = ${ne16_subtile_output_channel};
    task.kernel_shape = ${ne16_kernel_shape};
    task.depthwise = ${ne16_depthwise};

    // Block only while both job contexts are busy, then program this tap into the free one.
    ne16_nnx_dispatch_wait(ne16_pulp_get_dev());
    ne16_nnx_dispatch(ne16_pulp_get_dev(), &task);
}
% endfor
// Wait for the whole tap chain exactly once, not once per tap.
ne16_nnx_resolve_wait(ne16_pulp_get_dev(), &task);
}
"""

NE161xKConv2D_Template = NE162D1xKConvTemplate(NE161xKTaskTemplateStr)
