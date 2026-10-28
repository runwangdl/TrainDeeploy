/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp7a, step 2a): single-pass requantised int8 conv with 4 ch x 2 px register blocking,
 * in-place input reads and a flat pixel split. Control kernel for the blocks 1-4 fused kernel. See Conv4x2.c.
 */
#ifndef __DEEPLOY_MATH_CONV4X2_KERNEL_HEADER_
#define __DEEPLOY_MATH_CONV4X2_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_Conv4x2_i8_i8_i8_HWC(const int8_t *pIn, int8_t *pScratch, int8_t *pOut, const int8_t *pWeight,
                               const int32_t *pMul, const int32_t *pAdd, uint32_t num_cores, uint16_t out_shift,
                               uint16_t dim_in_x, uint16_t dim_in_y, uint16_t ch_in, uint16_t dim_out_x,
                               uint16_t dim_out_y, uint16_t ch_out, uint16_t dim_kernel_x, uint16_t dim_kernel_y,
                               uint16_t pad_y_top, uint16_t pad_y_bottom, uint16_t pad_x_left,
                               uint16_t pad_x_right, uint16_t stride_x, uint16_t stride_y);

#endif
