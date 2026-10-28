/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp11): single-pass QZO conv with in-kernel perturbation (one sign), Conv4x2 step-5 loop.
 * The unfused reference of DualConv2In.c without the dual-pass fusion. See PerturbConv4x2.c.
 */
#ifndef __DEEPLOY_MATH_PERTURBCONV4X2_KERNEL_HEADER_
#define __DEEPLOY_MATH_PERTURBCONV4X2_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_PerturbConv4x2_i8_i8_i8_HWC(const int8_t *pIn, const int8_t *pWeight, const int32_t *pMul,
                                       const int32_t *pAdd, const int32_t *pWMul, const int32_t *pBMul, int8_t *pOut,
                                       int8_t *pScratch, uint32_t sign, uint32_t seed_w_base, uint32_t seed_b_base,
                                       uint32_t tile_seed_xor, uint32_t log2D_w, uint32_t log2D_b, float32_t eps_scale,
                                       uint32_t num_cores, uint16_t out_shift, uint16_t dim_in_x, uint16_t dim_in_y,
                                       uint16_t ch_in, uint16_t dim_out_x, uint16_t dim_out_y, uint16_t ch_out,
                                       uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t pad_y_top,
                                       uint16_t pad_y_bottom, uint16_t pad_x_left, uint16_t pad_x_right,
                                       uint16_t stride_x, uint16_t stride_y);

#endif
