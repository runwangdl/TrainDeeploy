/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp5a): single-pass requantised int8 conv with DualConv.c's structure
 * (flat pixel partition, inline gather, no calls in the pixel loop). A/B control for exp5. See FlatConv.c.
 */
#ifndef __DEEPLOY_MATH_FLATCONV_KERNEL_HEADER_
#define __DEEPLOY_MATH_FLATCONV_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_FlatConv2d_i8_i8_i8_HWC(const int8_t *pIn, int8_t *pScratch, int8_t *pOut,
                                  const int8_t *pWeight, const int32_t *pMul, const int32_t *pAdd,
                                  uint32_t num_cores, uint16_t out_shift, uint16_t dim_in_x,
                                  uint16_t dim_in_y, uint16_t ch_in, uint16_t dim_out_x,
                                  uint16_t dim_out_y, uint16_t ch_out, uint16_t dim_kernel_x,
                                  uint16_t dim_kernel_y, uint16_t pad_y_top, uint16_t pad_y_bottom,
                                  uint16_t pad_x_left, uint16_t pad_x_right, uint16_t stride_x,
                                  uint16_t stride_y);

#endif
