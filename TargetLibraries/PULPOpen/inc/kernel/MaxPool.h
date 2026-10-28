/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_MAXPOOL_KERNEL_HEADER_
#define __DEEPLOY_MATH_MAXPOOL_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_MaxPool2d_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA,
                                  uint32_t W, uint32_t H, uint32_t C,
                                  uint32_t Q, uint32_t P, uint32_t SQ,
                                  uint32_t SP, float32_t *__restrict__ pDstC,
                                  uint32_t pad_top, uint32_t pad_bottom,
                                  uint32_t pad_left, uint32_t pad_right);

void PULP_MaxPoolGrad2d_fp32_fp32_HWC(
    const float32_t *__restrict__ pGradOut,
    const float32_t *__restrict__ pInput, uint32_t H_out, uint32_t W_out,
    uint32_t C, uint32_t H_in, uint32_t W_in, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, float32_t *__restrict__ pGradIn, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

/* QW: Part-4 argmax-mask kernels (see MaxPool.c). MaxPoolArgmax emits the within-window
 * argmax offset (uint8) so MaxPoolGradMask can scatter without the forward activation. -- QW */
void PULP_MaxPoolArgmax2d_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA,
                                        uint32_t W, uint32_t H, uint32_t C,
                                        uint32_t Q, uint32_t P, uint32_t SQ,
                                        uint32_t SP, float32_t *__restrict__ pMask,
                                        uint32_t pad_top, uint32_t pad_bottom,
                                        uint32_t pad_left, uint32_t pad_right);

void PULP_MaxPoolGradMask2d_fp32_fp32_HWC(
    const float32_t *__restrict__ pGradOut,
    const float32_t *__restrict__ pMask, uint32_t H_out, uint32_t W_out,
    uint32_t C, uint32_t H_in, uint32_t W_in, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, float32_t *__restrict__ pGradIn, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right);

#endif // __DEEPLOY_MATH_MAXPOOL_KERNEL_HEADER_