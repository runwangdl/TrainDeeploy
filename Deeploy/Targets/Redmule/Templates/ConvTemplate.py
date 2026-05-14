# ----------------------------------------------------------------------
#
# File: ConvTemplate.py
#
# Last edited: 09.05.2025
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Run Wang, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class RedmuleFloatConvIm2ColTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # Streaming im2col buffer: IM2COL_CHUNK_ROWS rows of K = C*P*Q FP32
        # values.  Must stay in sync with the IM2COL_CHUNK_ROWS macro in
        # Conv2d_Im2Col_fp32_Redmule.c.  A full-image im2col would blow L1
        # for non-trivial Conv layers (e.g. ResNet8 with H_out*W_out=1024
        # and K=144 -> 576 KiB), which made the tiler infeasible; capping
        # the buffer at 16 rows keeps every Conv layer tilable, at the cost
        # of a few extra RedMulE MMIO triggers per layer.
        IM2COL_CHUNK_ROWS = 16
        k_per_row = (operatorRepresentation['ch_im_in'] * operatorRepresentation['dim_kernel_x'] *
                     operatorRepresentation['dim_kernel_y'])
        im2col_dim = 4 * IM2COL_CHUNK_ROWS * k_per_row
        im2col_name = operatorRepresentation['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = RedmuleFloatConvIm2ColTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


reference2DIm2ColTemplate = RedmuleFloatConvIm2ColTemplate("""
// 2D FP Conv HWC Parallel with Im2Col (Name: ${nodeName}, Op: ${nodeOp})
${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {

    Conv2d_Im2Col_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC_8_Redmule(
        ref_${data_out}_${data_in},
        ${dim_im_in_y},
        ${dim_im_in_x},
        ${ch_im_in},
        ${weight},
        ${dim_kernel_y},
        ${dim_kernel_x},
        ${stride_y},
        ${stride_x},
        ${bias},
        ${has_bias},
        ref_${data_out}_${data_out},
        ${ch_im_out},
        ${padding_y_top},
        ${padding_y_bottom},
        ${padding_x_left},
        ${padding_x_right},
        ${ctxtBuffer}
    );

    ref_${data_out}_${data_in} += ${ch_im_in} * ${dim_im_in_x} * ${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
}
""")
