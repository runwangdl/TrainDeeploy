/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_GEMM_KERNEL_HEADER_
#define __DEEPLOY_MATH_GEMM_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_Gemm_fp32_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                                   const float32_t *__restrict__ pSrcB,
                                   const float32_t *__restrict__ pDstC,
                                   float32_t *__restrict__ pDstY, uint32_t M,
                                   uint32_t N, uint32_t O, uint32_t transA,
                                   uint32_t transB);

// Gemm with an int8 weight, dequantised in-kernel; see Gemm.c.
void PULP_Gemm_fp32_i8_fp32_fp32(const float32_t *__restrict__ pSrcA,
                                 const int8_t *__restrict__ pSrcB,
                                 const float32_t *__restrict__ pDstC,
                                 float32_t *__restrict__ pDstY, uint32_t M,
                                 uint32_t N, uint32_t O, uint32_t transA,
                                 uint32_t transB, uint32_t biasStride,
                                 float32_t scale, int32_t zeroPoint);

#endif // __DEEPLOY_MATH_GEMM_KERNEL_HEADER_