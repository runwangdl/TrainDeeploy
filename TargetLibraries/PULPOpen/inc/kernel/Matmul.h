/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_
#define __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_MatMul_fp32_fp32_fp32_unroll1x7(const float32_t *__restrict__ pSrcA,
                                          const float32_t *__restrict__ pSrcB,
                                          float32_t *__restrict__ pDstY,
                                          uint32_t M, uint32_t N, uint32_t O);

// Matmul against an int8 weight matrix with per-tensor affine dequantisation
// folded into the loop; see Matmul.c for why the scale factors out.
void PULP_MatMul_fp32_i8_fp32_unroll1x7(const float32_t *__restrict__ pSrcA,
                                        const int8_t *__restrict__ pSrcB,
                                        float32_t *__restrict__ pDstY,
                                        uint32_t M, uint32_t N, uint32_t O,
                                        float32_t scale, int32_t zeroPoint);

#endif // __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_