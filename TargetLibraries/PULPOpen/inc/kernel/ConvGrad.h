/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_CONVGRAD_KERNEL_HEADER_
#define __DEEPLOY_MATH_CONVGRAD_KERNEL_HEADER_

#include "DeeployPULPMath.h"

// ============================================================================
// Regular Conv — GradW (weight gradient via pulp-trainlib)
// ============================================================================

void PULP_ConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

void PULP_ConvGradW2d_fp32_fp32_fp32_CHW_Im2Col(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    float *__restrict__ ctxtBuffer, uint32_t ctxtBufferSize);

// ============================================================================
// Regular Conv — GradX (input gradient)
// ============================================================================

// Non-tiled: delegates to pulp-trainlib naive kernel
void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_trainlib(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

// Non-tiled im2col: delegates to pulp-trainlib im2col kernel
void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_Im2Col(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    float *__restrict__ ctxtBuffer, uint32_t ctxtBufferSize,
    float *__restrict__ btBuffer, uint32_t btBufferSize);

// Tiled naive: scatter pattern with tile offsets (ForkTransformer)
void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_tiled(
    const float *__restrict__ pGradOut,
    uint32_t dim_im_out_x, uint32_t dim_im_out_y, uint32_t ch_im_out,
    const float *__restrict__ pWeight, uint32_t ch_im_in,
    uint32_t dim_kernel_x, uint32_t dim_kernel_y,
    uint32_t stride_h, uint32_t stride_w,
    float *__restrict__ pGradIn,
    uint32_t dim_im_in_x, uint32_t dim_im_in_y,
    uint32_t padding_x_left, uint32_t padding_x_right,
    uint32_t padding_y_top, uint32_t padding_y_bottom,
    uint16_t offset_grad_in_h, uint16_t offset_grad_in_w,
    uint16_t offset_grad_out_h, uint16_t offset_grad_out_w);

// Tiled non-tiled-offset variant (ForkTransformer)
void PULP_ConvGradX2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut,
    uint32_t dim_im_out_x, uint32_t dim_im_out_y, uint32_t ch_im_out,
    const float *__restrict__ pWeight, uint32_t ch_im_in,
    uint32_t dim_kernel_x, uint32_t dim_kernel_y,
    uint32_t stride_h, uint32_t stride_w,
    float *__restrict__ pGradIn,
    uint32_t dim_im_in_x, uint32_t dim_im_in_y,
    uint32_t padding_x_left, uint32_t padding_x_right,
    uint32_t padding_y_top, uint32_t padding_y_bottom);

// Tiled im2col+GEMM with co_block (ForkTransformer)
void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_Im2Col_tiled(
    const float *__restrict__ pGradOut,
    uint32_t dim_im_out_x, uint32_t dim_im_out_y, uint32_t ch_im_out,
    const float *__restrict__ pWeight, uint32_t ch_im_in,
    uint32_t dim_kernel_x, uint32_t dim_kernel_y,
    uint32_t stride_h, uint32_t stride_w,
    float *__restrict__ pGradIn,
    uint32_t dim_im_in_x, uint32_t dim_im_in_y,
    uint32_t padding_y_top, uint32_t padding_y_bottom,
    uint32_t padding_x_left, uint32_t padding_x_right,
    uint16_t offset_grad_in_h, uint16_t offset_grad_in_w,
    uint16_t offset_grad_out_h, uint16_t offset_grad_out_w,
    float *__restrict__ ctxtBuffer, uint32_t ctxtBufferSize,
    float *__restrict__ btBuffer, uint32_t btBufferSize);

// ============================================================================
// Depthwise Conv — GradW (via pulp-trainlib, ClusterTransformer)
// ============================================================================

void PULP_DWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

// ============================================================================
// Depthwise Conv — GradX (via pulp-trainlib gather kernel, ClusterTransformer)
// ============================================================================

void PULP_DWConvGradX2d_fp32_fp32_fp32_CHW_trainlib_tiled(
    const float *__restrict__ pGradOut,
    uint32_t dim_im_out_x, uint32_t dim_im_out_y, uint32_t ch_im_out,
    const float *__restrict__ pWeight, uint32_t ch_im_in,
    uint32_t dim_kernel_x, uint32_t dim_kernel_y,
    uint32_t stride_h, uint32_t stride_w,
    float *__restrict__ pGradIn,
    uint32_t dim_im_in_x, uint32_t dim_im_in_y,
    uint32_t padding_x_left, uint32_t padding_x_right,
    uint32_t padding_y_top, uint32_t padding_y_bottom,
    uint16_t offset_grad_in_h, uint16_t offset_grad_in_w,
    uint16_t offset_grad_out_h, uint16_t offset_grad_out_w);

// ============================================================================
// Pointwise Conv — GradW (via pulp-trainlib, ClusterTransformer)
// ============================================================================

void PULP_PWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, float *__restrict__ pGradWeight);

// ============================================================================
// Pointwise Conv — GradX (direct AXPY kernel, ClusterTransformer)
// ============================================================================

void PULP_PWConvGradX2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in);

#endif // __DEEPLOY_MATH_CONVGRAD_KERNEL_HEADER_
