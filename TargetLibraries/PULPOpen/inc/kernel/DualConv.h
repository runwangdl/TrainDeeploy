/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp5): fused dual-pass (theta +/- eps*z) quantized conv. See DualConv.c.
 */
#ifndef __DEEPLOY_MATH_DUALCONV_KERNEL_HEADER_
#define __DEEPLOY_MATH_DUALCONV_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_DualConv2d_i8_i8_i8_HWC(
    const int8_t *pIn, const int8_t *pWeight, const int32_t *pMul, const int32_t *pAdd,
    const int32_t *pWMul, const int32_t *pBMul, int8_t *pOut, int8_t *pScratch,
    uint32_t seed_w_base, uint32_t seed_b_base, uint32_t tile_seed_xor, uint32_t log2D_w,
    uint32_t log2D_b, float32_t eps_scale, uint32_t num_cores, uint16_t out_shift,
    uint16_t dim_in_x, uint16_t dim_in_y, uint16_t ch_in, uint16_t dim_out_x, uint16_t dim_out_y,
    uint16_t ch_out, uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t pad_y_top,
    uint16_t pad_y_bottom, uint16_t pad_x_left, uint16_t pad_x_right, uint16_t stride_x,
    uint16_t stride_y);

#endif
