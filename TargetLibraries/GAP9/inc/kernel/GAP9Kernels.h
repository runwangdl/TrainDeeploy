/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// RW: Prototypes for the PULP training gradient kernels that GAP9 links from
// PULPOpen/src (ConvGrad.c / DWConvGrad.c / PWConvGrad.c / BatchNorm.c, pulled
// into the deeploygap9 library via the CMake source GLOB).
//
// These are declared here, rather than by including PULPOpen's ConvGrad.h /
// BatchNorm.h, because those headers transitively #include DeeployPULPMath.h,
// which unconditionally redefines the SINGLE_CORE macros (-Werror) and pulls in
// forward-kernel prototypes whose signatures conflict with GAP9's own kernels.
// Forward-declaring just the gradient kernels avoids every such collision.
//
// Without these declarations the generated TrainingNetwork.c called the
// high-arity gradient kernels (>8 args, stack-passed) with implicit (int)
// declarations, mismatching the ABI and corrupting memory after training -- a
// gvsoc "not indexable" abort that happened during the post-training epilogue.
//
// Signatures are kept byte-for-byte in sync with
// TargetLibraries/PULPOpen/inc/kernel/{ConvGrad.h,BatchNorm.h}.

#ifndef __DEEPLOY_GAP9_KERNEL_HEADER_
#define __DEEPLOY_GAP9_KERNEL_HEADER_

#include <stdint.h>

#include "types.h" // float32_t

// --- Regular (dense) Conv -- GradW ---
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

// --- Regular (dense) Conv -- GradX ---
void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_scatter_tiled(
    const float *__restrict__ pGradOut, uint32_t dim_im_out_x,
    uint32_t dim_im_out_y, uint32_t ch_im_out,
    const float *__restrict__ pWeight, uint32_t ch_im_in, uint32_t dim_kernel_x,
    uint32_t dim_kernel_y, uint32_t stride_h, uint32_t stride_w,
    float *__restrict__ pGradIn, uint32_t dim_im_in_x, uint32_t dim_im_in_y,
    uint32_t padding_x_left, uint32_t padding_x_right, uint32_t padding_y_top,
    uint32_t padding_y_bottom, uint16_t offset_grad_in_h,
    uint16_t offset_grad_in_w, uint16_t offset_grad_out_h,
    uint16_t offset_grad_out_w);

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

// --- Depthwise Conv -- GradW / GradX ---
void PULP_DWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

void PULP_DWConvGradX2d_fp32_fp32_fp32_CHW_trainlib_tiled(
    const float *__restrict__ pGradOut, uint32_t dim_im_out_x,
    uint32_t dim_im_out_y, uint32_t ch_im_out,
    const float *__restrict__ pWeight, uint32_t ch_im_in, uint32_t dim_kernel_x,
    uint32_t dim_kernel_y, uint32_t stride_h, uint32_t stride_w,
    float *__restrict__ pGradIn, uint32_t dim_im_in_x, uint32_t dim_im_in_y,
    uint32_t padding_x_left, uint32_t padding_x_right, uint32_t padding_y_top,
    uint32_t padding_y_bottom, uint16_t offset_grad_in_h,
    uint16_t offset_grad_in_w, uint16_t offset_grad_out_h,
    uint16_t offset_grad_out_w);

// --- Pointwise Conv -- GradW / GradX ---
void PULP_PWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, float *__restrict__ pGradWeight);

void PULP_PWConvGradX2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in);

// --- BatchNorm -- Grad ---
void PULP_BatchNormGrad_fp32(const float32_t *dY, const float32_t *X,
                             const float32_t *gamma,
                             const float32_t *saved_mean,
                             const float32_t *saved_inv_std, float32_t *dX,
                             float32_t *dgamma, float32_t *dbeta, uint32_t N,
                             uint32_t C, uint32_t H_in, uint32_t W_in,
                             float32_t epsilon);

// --- Forward + non-conv gradient kernels also linked from PULPOpen/src but
//     not surfaced by any GAP9-reachable header (same implicit-decl risk). ---
void PULP_BatchNormInternal_fp32(
    const float32_t *X, const float32_t *gamma, const float32_t *beta,
    const float32_t *running_mean, const float32_t *running_var, float32_t *Y,
    float32_t *saved_mean, float32_t *saved_inv_std, uint32_t N, uint32_t C,
    uint32_t H_in, uint32_t W_in, float32_t epsilon, float32_t momentum);

void PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer);

// Channels-first (NCHW) forward conv variants (conv_channels_first path).
void PULP_Conv2d_Im2Col_fp32_fp32_fp32_CHW(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer);

void PULP_DW_Conv2d_Im2Col_fp32_fp32_fp32_CHW(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer);

void PULP_Gemm_fp32_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                                   const float32_t *__restrict__ pSrcB,
                                   const float32_t *__restrict__ pDstC,
                                   float32_t *__restrict__ pDstY, uint32_t M,
                                   uint32_t N, uint32_t O, uint32_t transA,
                                   uint32_t transB, uint32_t biasStride);

void PULP_GlobalAveragePoolGrad_fp32(const float32_t *dY, float32_t *dX,
                                     uint32_t N, uint32_t C, uint32_t H,
                                     uint32_t W);

void PULP_GlobalAveragePool_fp32(const float32_t *input, float32_t *output,
                                 uint32_t N, uint32_t C, uint32_t H,
                                 uint32_t W);

void PULP_ReluGrad_fp32_fp32(float32_t *grad_out, float32_t *data_in,
                             float32_t *grad_in, uint32_t size);

void PULP_Relu_fp32_fp32(float32_t *input, float32_t *output, uint32_t size);

#endif // __DEEPLOY_GAP9_KERNEL_HEADER_
