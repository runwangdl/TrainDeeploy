/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Internal header for ConvGrad kernel family — shared trainlib interface
 * structs used by ConvGrad.c, DWConvGrad.c, and PWConvGrad.c.
 */

#ifndef __CONVGRAD_INTERNAL_H_
#define __CONVGRAD_INTERNAL_H_

#include "DeeployPULPMath.h"
#include "pmsis.h"

// ============================================================================
// pulp-trainlib interface structs (local forward declarations to avoid
// pulling in pulp_train_defines.h which conflicts with Deeploy headers)
// ============================================================================

struct blob {
  float *data;
  float *diff;
  int dim;
  int W;
  int H;
  int C;
};

// --- Regular Conv2D ---
void pulp_conv2d_fp32_bw_param_grads_cl(void *Conv2D_args);
void pulp_conv2d_fp32_bw_input_grads_cl(void *Conv2D_args);

struct Conv2D_args {
  struct blob *input;
  struct blob *coeff;
  struct blob *bias;
  struct blob *output;
  int Lpad;  int Rpad;  int Upad;  int Dpad;
  int stride_h;  int stride_w;
  float *i2c_buffer;  float *bt_buffer;
  int skip_wg_grad;  int skip_in_grad;  int HWC;
  int opt_matmul_type_fw;  int opt_matmul_type_wg;  int opt_matmul_type_ig;
  int USE_BIASES;  int USE_IM2COL;  int USE_DMA_IM2COL;
};

// --- Depthwise Conv ---
void pulp_conv_dw_fp32_bw_param_grads_cl(void *DepthWise_Conv_args);
void pulp_conv_dw_fp32_bw_input_grads_tiled_cl(void *DepthWise_Conv_args);

struct DepthWise_Conv_args {
  struct blob *input;  struct blob *coeff;  struct blob *output;
  int stride_h;  int stride_w;
  int Lpad;  int Rpad;  int Upad;  int Dpad;
  int skip_wg_grad;  int skip_in_grad;  int HWC;
  int offset_in_h;  int offset_in_w;  int offset_out_h;  int offset_out_w;
};

// --- Pointwise Conv (GradW via trainlib) ---
void pulp_conv_pw_fp32_bw_param_grads_cl(void *PointWise_Conv_args);

struct PointWise_Conv_args {
  struct blob *input;  struct blob *coeff;  struct blob *output;
  float *transpose_buffer;
  int skip_wg_grad;  int skip_in_grad;  int HWC;
  int opt_matmul_type_fw;  int opt_matmul_type_wg;  int opt_matmul_type_ig;
};

// ============================================================================
// Utility
// ============================================================================

static inline int32_t max_i32(int32_t a, int32_t b) { return (a > b) ? a : b; }
static inline int32_t min_i32(int32_t a, int32_t b) { return (a < b) ? a : b; }

#endif // __CONVGRAD_INTERNAL_H_
