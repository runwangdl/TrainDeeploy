# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Templates that route Pointwise (1x1) ConvGradW / ConvGradX to RedMulE.

Both kernels reuse the existing PULPOpen tile constraints
(PWConvGradWTileConstraint / PWConvGradXTileConstraint) so the tile-shape
search is identical to the pulp-trainlib variants; only the kernel body
calls into PWConvGrad_fp32_Redmule.c instead, which materialises the
necessary transpose into a transient buffer and then fires a single
RedMulE GEMM.
"""

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class RedmulePWConvGradWTemplate(NodeTemplate):
    """RedMulE pointwise ConvGradW: dW = dY @ X^T (1x1 kernel).

    Reserves a C_in * H_in * W_in transient buffer in L1 to hold the
    transposed input that the RedMulE GEMM consumes.  At the kernel side
    PWConvGradW2d_fp32_fp32_fp32_CHW_Redmule (in
    TargetLibraries/PULPOpen/src/PWConvGrad_fp32_Redmule.c) builds the
    transpose in parallel across the 8 cluster cores and then triggers
    one RedMulE call.
    """

    def __init__(self, templateStr: str):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        bt_dim = (operatorRepresentation["data_in_type"].typeWidth // 8) * \
                 operatorRepresentation['ch_im_in'] * \
                 operatorRepresentation['dim_im_in_x'] * \
                 operatorRepresentation['dim_im_in_y']
        bt_name = operatorRepresentation['nodeName'] + "_transpose_buffer"
        return [(bt_name, bt_dim)]

    def hoistTransientBuffers(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        bt_name, bt_dim = RedmulePWConvGradWTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(bt_name, bt_dim)
        operatorRepresentation['transposeBuffer'] = bt_name
        operatorRepresentation['transposeBufferSize'] = bt_dim
        return ctxt, operatorRepresentation, [bt_name]


class RedmulePWConvGradXTemplate(NodeTemplate):
    """RedMulE pointwise ConvGradX: dX = W^T @ dY (1x1 kernel).

    Reserves a C_in * C_out transient buffer in L1 to hold the
    transposed weight (identical size to the pulp-trainlib version's
    transpose buffer, so the existing PWConvGradXTileConstraint keeps
    working unchanged).
    """

    def __init__(self, templateStr: str):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        bt_dim = (operatorRepresentation["weight_type"].typeWidth // 8) * \
                 operatorRepresentation['ch_im_in'] * \
                 operatorRepresentation['ch_im_out']
        bt_name = operatorRepresentation['nodeName'] + "_transpose_buffer"
        return [(bt_name, bt_dim)]

    def hoistTransientBuffers(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        bt_name, bt_dim = RedmulePWConvGradXTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(bt_name, bt_dim)
        operatorRepresentation['transposeBuffer'] = bt_name
        operatorRepresentation['transposeBufferSize'] = bt_dim
        return ctxt, operatorRepresentation, [bt_name]


referencePWConvGradW2DTemplate = RedmulePWConvGradWTemplate("""
// 2D FP Pointwise ConvGradW (1x1) CHW via RedMulE (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName} ref_${grad_weight}_${grad_out} = ${grad_out};
${data_in_type.typeName} ref_${grad_weight}_${data_in} = ${data_in};
${grad_weight_type.typeName} ref_${grad_weight}_out = ${grad_weight};

for (uint32_t n = 0; n < ${batch}; ++n) {
    PWConvGradW2d_fp${grad_out_type.referencedType.typeWidth}_fp${data_in_type.referencedType.typeWidth}_fp${grad_weight_type.referencedType.typeWidth}_CHW_Redmule(
        ref_${grad_weight}_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${grad_weight}_${data_in},
        ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
        ref_${grad_weight}_out,
        ${transposeBuffer}
    );

    ref_${grad_weight}_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${grad_weight}_${data_in} += ${ch_im_in} * ${dim_im_in_y} * ${dim_im_in_x};
}
""")

referencePWConvGradX2DTemplate = RedmulePWConvGradXTemplate("""
// 2D FP Pointwise ConvGradX (1x1) CHW via RedMulE (Name: ${nodeName}, Op: ${nodeOp})
${grad_out_type.typeName}  ref_${grad_in}_${grad_out} = ${grad_out};
${weight_type.typeName}    ref_${grad_in}_${weight}  = ${weight};
${grad_in_type.typeName}   ref_${grad_in}_out        = ${grad_in};

for (uint32_t n = 0; n < ${batch}; ++n) {
    PWConvGradX2d_fp${grad_out_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${grad_in_type.referencedType.typeWidth}_CHW_Redmule(
        ref_${grad_in}_${grad_out},
        ${dim_im_out_x}, ${dim_im_out_y}, ${ch_im_out},
        ref_${grad_in}_${weight},
        ${ch_im_in},
        ref_${grad_in}_out,
        ${dim_im_in_x}, ${dim_im_in_y},
        ${transposeBuffer}, ${transposeBufferSize}
    );

    ref_${grad_in}_${grad_out} += ${ch_im_out} * ${dim_im_out_y} * ${dim_im_out_x};
    ref_${grad_in}_out        += ${ch_im_in}  * ${dim_im_in_y}  * ${dim_im_in_x};
}
""")
