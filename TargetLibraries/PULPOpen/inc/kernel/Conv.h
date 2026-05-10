/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_CONV_KERNEL_HEADER_
#define __DEEPLOY_MATH_CONV_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_Conv2d_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right);

void PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer);

// RedMulE-accelerated FP32 Conv2d.  Expects weight already permuted from the
// ONNX [F, P, Q, C] layout to [P, Q, C, F] (a flat [P*Q*C, F] matrix);
// RedMuleAdjustWeightMemoryLayoutPass handles that.  pIm2ColBuf must hold
// H_out * W_out * (C*P*Q) FP32 elements; its size is reserved by
// RedmuleFloatConvIm2ColTemplate.computeTransientBuffersSize.
void Conv2d_Im2Col_fp32_fp32_fp32_HWC_8_Redmule(
    const float32_t *__restrict__ pIn, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pWeight, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, const float32_t *__restrict__ pBias, const bool has_bias,
    float32_t *__restrict__ pOut, uint32_t F, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pIm2ColBuf);

void PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer);

// ============================================================================
// Minimal pulp-trainlib interface
// ============================================================================

void PULP_ConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

void PULP_ConvGradX2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_Im2Col_tiled(
    const float *__restrict__ pGradOut, uint32_t dim_im_out_x,
    uint32_t dim_im_out_y, uint32_t ch_im_out,
    const float *__restrict__ pWeight, uint32_t ch_im_in, uint32_t dim_kernel_x,
    uint32_t dim_kernel_y, uint32_t stride_h, uint32_t stride_w,
    float *__restrict__ pGradIn, uint32_t dim_im_in_x, uint32_t dim_im_in_y,
    uint32_t padding_y_top, uint32_t padding_y_bottom, uint32_t padding_x_left,
    uint32_t padding_x_right, uint16_t offset_grad_in_h,
    uint16_t offset_grad_in_w, uint16_t offset_grad_out_h,
    uint16_t offset_grad_out_w, float *__restrict__ ctxtBuffer,
    uint32_t ctxtBufferSize, float *__restrict__ btBuffer,
    uint32_t btBufferSize);

void PULP_ConvGradW2d_fp32_fp32_fp32_CHW_Im2Col(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    float *__restrict__ ctxtBuffer, uint32_t ctxtBufferSize);

void PULP_DWConvTrans2d_fp32_fp32_fp32_HWC(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_total, const float *__restrict__ pWeight, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, float *__restrict__ pGradIn, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

void PULP_DWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

// Pointwise (1x1) Convolution Gradient Functions
void PULP_PWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, float *__restrict__ pGradWeight);

// RedMulE-accelerated pointwise (1x1) Conv backward weight gradient.
// Same arg order as PULP_PWConvGradW2d_fp32_fp32_fp32_CHW plus a
// pTransposeBuffer of C_in * H_in * W_in FP32 elements (reserved by
// RedmulePWConvGradW2DTemplate.computeTransientBuffersSize) used to
// materialise X^T before firing one RedMulE GEMM.
void PWConvGradW2d_fp32_fp32_fp32_CHW_Redmule(
    const float32_t *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float32_t *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, float32_t *__restrict__ pGradWeight,
    float32_t *__restrict__ pTransposeBuffer);

// RedMulE-accelerated pointwise (1x1) Conv backward input gradient.
// Mirrors PULP_PWConvGradX2d_fp32_fp32_fp32_CHW signature; the C_in*C_out
// transpose buffer is reused for W^T before firing one RedMulE GEMM.
void PWConvGradX2d_fp32_fp32_fp32_CHW_Redmule(
    const float32_t *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float32_t *__restrict__ pWeight, uint32_t C_in,
    float32_t *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in,
    float32_t *__restrict__ pTransposeBuffer,
    uint32_t transposeBufferSize);

void PULP_PWConvGradX2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in,
    float *__restrict__ pTransposeBuffer, uint32_t transposeBufferSize);

void PULP_DWConvGradX2d_fp32_fp32_fp32_CHW_tiled(
    const float *__restrict__ pGradOut,
    uint32_t dim_im_out_x, // H_out (tile)
    uint32_t dim_im_out_y, // W_out (tile)
    uint32_t ch_im_out,    // C_out (full)
    const float *__restrict__ pWeight,
    uint32_t ch_im_in,     // C_in (full)
    uint32_t dim_kernel_x, // P (kernel H)
    uint32_t dim_kernel_y, // Q (kernel W)
    uint32_t stride_h,     // SH
    uint32_t stride_w,     // SW
    float *__restrict__ pGradIn,
    uint32_t dim_im_in_x,      // H_in (tile)
    uint32_t dim_im_in_y,      // W_in (tile)
    uint32_t padding_x_left,   // pad_top
    uint32_t padding_x_right,  // pad_bottom (unused here)
    uint32_t padding_y_top,    // pad_left
    uint32_t padding_y_bottom, // pad_right (unused here)
    uint16_t offset_grad_in_h, uint16_t offset_grad_in_w,
    uint16_t offset_grad_out_h, uint16_t offset_grad_out_w);
#endif // __DEEPLOY_MATH_CONV_KERNEL_HEADER_