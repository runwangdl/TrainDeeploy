/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp6): column-buffer builder for the NE16 im2col route. See Im2ColU8.c.
 */
#ifndef __DEEPLOY_MATH_IM2COLU8_KERNEL_HEADER_
#define __DEEPLOY_MATH_IM2COLU8_KERNEL_HEADER_

#include <stdint.h>

void PULP_Im2ColU8_NCHW_to_HWC(const int8_t *pIn, uint8_t *pOut, uint32_t num_cores, uint16_t in_rows,
                               uint16_t in_cols, uint16_t ch_in, uint16_t out_rows, uint16_t out_cols,
                               uint16_t ky_n, uint16_t kx_n, uint16_t pad_top, uint16_t pad_bottom,
                               uint16_t pad_left, uint16_t pad_right, uint16_t stride_y,
                               uint16_t stride_x, uint8_t xor_mask, uint8_t pad_value);

#endif
