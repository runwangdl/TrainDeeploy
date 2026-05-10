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

    # Must stay in sync with PWGW_CHUNK_P in PWConvGrad_fp32_Redmule.c.
    PWGW_CHUNK_P = 16

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        # Fixed-size chunk scratch: PWGW_CHUNK_P rows of [C_in] for the
        # X-sampled-and-transposed slice + PWGW_CHUNK_P rows of [C_out] for
        # the dY view (used by the multi-chunk path when P > CHUNK_P).
        # Independent of the layer's feature-map area -- crucial on
        # MobileNetV1 early blocks where H_out * W_out can hit 48*48 and a
        # full transpose buffer would blow L1.
        wbytes = operatorRepresentation["data_in_type"].typeWidth // 8
        chunk = RedmulePWConvGradWTemplate.PWGW_CHUNK_P
        bt_dim = wbytes * chunk * (operatorRepresentation['ch_im_in'] +
                                   operatorRepresentation['ch_im_out'])
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
    """RedMulE pointwise ConvGradX: dX = scatter(W^T @ dY) (1x1 kernel).

    For stride 1 the transpose buffer only holds C_in * C_out floats (the
    transposed weight matrix); the RedMulE GEMM writes the [C_in, H*W]
    result straight into pGradIn.

    For stride > 1 the GEMM output is the *dense* [C_in, H_out * W_out]
    matrix and must be scattered into the [C_in, H_in, W_in] dX tensor at
    the strided positions (the rest of dX stays zero).  In that case the
    transpose buffer is also reused to hold the dense GEMM result, so the
    template reserves C_in * C_out + C_in * H_out * W_out floats.  At
    stride 1 the dense buffer is unused but the over-allocation is small
    enough to keep the worst-case size simple.
    """

    def __init__(self, templateStr: str):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        wt_elts = operatorRepresentation['ch_im_in'] * operatorRepresentation['ch_im_out']
        dense_elts = operatorRepresentation['ch_im_in'] * operatorRepresentation[
            'dim_im_out_x'] * operatorRepresentation['dim_im_out_y']
        bt_dim = (operatorRepresentation["weight_type"].typeWidth // 8) * (wt_elts + dense_elts)
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
