/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_Conv2d_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right) {

  // Compute core
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  // Compute the chunk size for each core
  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  // Pointer to the weights for the current core
  const float32_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;

  // Compute the output dimensions
  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  // Compute the output
  if (has_bias) {
    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {
        for (uint32_t f = 0; f < ch_out_count; ++f) {
          float32_t sum = 0.0f;

          for (uint32_t p = 0; p < P; ++p) {
            for (uint32_t q = 0; q < Q; ++q) {
              for (uint32_t c = 0; c < C; ++c) {
                int32_t h_in = h * SP + p - pad_top;
                int32_t w_in = w * SQ + q - pad_left;

                if (h_in < 0 || h_in >= (int32_t)H || w_in < 0 ||
                    w_in >= (int32_t)W) {
                  continue;
                }

                uint32_t input_idx = (h_in * W + w_in) * C + c;
                uint32_t weight_idx = f * (P * Q * C) + p * (Q * C) + q * C + c;

                sum += pSrcA[input_idx] * weight_ptr[weight_idx];
              }
            }
          }

          uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
          pDstC[output_idx] = sum + pSrcBias[f + ch_out_start];
        }
      }
    }
  } else {
    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {
        for (uint32_t f = 0; f < ch_out_count; ++f) {
          float32_t sum = 0.0f;

          for (uint32_t p = 0; p < P; ++p) {
            for (uint32_t q = 0; q < Q; ++q) {
              for (uint32_t c = 0; c < C; ++c) {
                int32_t h_in = h * SP + p - pad_top;
                int32_t w_in = w * SQ + q - pad_left;

                if (h_in < 0 || h_in >= (int32_t)H || w_in < 0 ||
                    w_in >= (int32_t)W) {
                  continue;
                }

                uint32_t input_idx = (h_in * W + w_in) * C + c;
                uint32_t weight_idx = f * (P * Q * C) + p * (Q * C) + q * C + c;

                sum += pSrcA[input_idx] * weight_ptr[weight_idx];
              }
            }
          }

          uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
          pDstC[output_idx] = sum;
        }
      }
    }
  }
}

// RW: CHW (channels-first) forward conv via pulp-trainlib (HWC=0). Mirrors the
// CHW grad wrappers in ConvGrad.c. Lets GAP9 keep activations channels-first so
// the NCHW->NHWC input transpose is not needed (lowers L1 floor + cuts forks).
// trainlib pulp_conv2d_fp32_fw_cl forks internally (master-core dispatch),
// matching the ClusterTransformer binding. The im2col transient buffer must be
// sized for the FULL CHW matrix (C*P*Q*H_out*W_out), see the CHW template.
void PULP_Conv2d_Im2Col_fp32_fp32_fp32_CHW(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer) {

  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  struct blob input_blob = {0};
  struct blob coeff_blob = {0};
  struct blob bias_blob = {0};
  struct blob output_blob = {0};

  input_blob.data = (float *)pSrcA;
  input_blob.diff = NULL;
  input_blob.W = (int)W;
  input_blob.H = (int)H;
  input_blob.C = (int)C;
  input_blob.dim = (int)(C * H * W);

  coeff_blob.data = (float *)pSrcB;
  coeff_blob.diff = NULL;
  coeff_blob.W = (int)Q;
  coeff_blob.H = (int)P;
  coeff_blob.C = (int)F_total;
  coeff_blob.dim = (int)(F_total * C * P * Q);

  bias_blob.data = (float *)pSrcBias;
  bias_blob.diff = NULL;
  bias_blob.W = 1;
  bias_blob.H = 1;
  bias_blob.C = (int)F_total;
  bias_blob.dim = (int)F_total;

  output_blob.data = pDstC;
  output_blob.diff = NULL;
  output_blob.W = (int)W_out;
  output_blob.H = (int)H_out;
  output_blob.C = (int)F_total;
  output_blob.dim = (int)(F_total * H_out * W_out);

  struct Conv2D_args conv_args;
  memset(&conv_args, 0, sizeof(conv_args));
  conv_args.input = &input_blob;
  conv_args.coeff = &coeff_blob;
  conv_args.bias = &bias_blob;
  conv_args.output = &output_blob;
  conv_args.Lpad = (int)pad_left;
  conv_args.Rpad = (int)pad_right;
  conv_args.Upad = (int)pad_top;
  conv_args.Dpad = (int)pad_bottom;
  conv_args.stride_h = (int)SP;
  conv_args.stride_w = (int)SQ;
  conv_args.i2c_buffer = pContextBuffer;
  conv_args.bt_buffer = NULL;
  conv_args.skip_wg_grad = 1;
  conv_args.skip_in_grad = 1;
  conv_args.HWC = 0;
  conv_args.USE_BIASES = has_bias ? 1 : 0;
  conv_args.USE_IM2COL = 1;
  conv_args.USE_DMA_IM2COL = 0;
  conv_args.opt_matmul_type_fw = 0;

  pulp_conv2d_fp32_fw_cl(&conv_args);
}

void PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer) {

  // Compute core
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  // Compute the chunk size for each core
  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  // Pointer to the weights for the current core
  const float32_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;

  uint32_t im2col_size_per_core = C * P * Q;
  float32_t *im2col_buffer = pContextBuffer + core_id * im2col_size_per_core;

  // Compute the output dimensions
  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
  uint32_t kernel_size = P * Q * C;

  // Compute the output
  if (has_bias) {
    for (uint32_t h_out = 0; h_out < H_out; h_out++) {
      for (uint32_t w_out = 0; w_out < W_out; w_out++) {
        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;

        for (uint32_t p = 0; p < P; p++) {
          int32_t h_in = h_in_start + p;

          for (uint32_t q = 0; q < Q; q++) {
            int32_t w_in = w_in_start + q;

            for (uint32_t c = 0; c < C; c++) {
              if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
                  w_in < (int32_t)W) {
                uint32_t in_idx = (h_in * W + w_in) * C + c;
                im2col_buffer[p * Q * C + q * C + c] = pSrcA[in_idx];
              } else {
                im2col_buffer[p * Q * C + q * C + c] = 0.0f;
              }
            }
          }
        }

        for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
          float32_t sum = 0.0f;
          const float32_t *local_weight_ptr =
              weight_ptr + (f - ch_out_start) * kernel_size;

          for (uint32_t k = 0; k < kernel_size; k++) {
            sum += im2col_buffer[k] * local_weight_ptr[k];
          }

          uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;

          pDstC[out_idx] = sum + pSrcBias[f];
        }
      }
    }
  } else {
    for (uint32_t h_out = 0; h_out < H_out; h_out++) {
      for (uint32_t w_out = 0; w_out < W_out; w_out++) {
        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;

        for (uint32_t p = 0; p < P; p++) {
          int32_t h_in = h_in_start + p;

          for (uint32_t q = 0; q < Q; q++) {
            int32_t w_in = w_in_start + q;

            for (uint32_t c = 0; c < C; c++) {
              if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
                  w_in < (int32_t)W) {
                uint32_t in_idx = (h_in * W + w_in) * C + c;
                im2col_buffer[p * Q * C + q * C + c] = pSrcA[in_idx];
              } else {
                im2col_buffer[p * Q * C + q * C + c] = 0.0f;
              }
            }
          }
        }

        for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
          float32_t sum = 0.0f;
          const float32_t *local_weight_ptr =
              weight_ptr + (f - ch_out_start) * kernel_size;

          for (uint32_t k = 0; k < kernel_size; k++) {
            sum += im2col_buffer[k] * local_weight_ptr[k];
          }

          uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;

          pDstC[out_idx] = sum;
        }
      }
    }
  }
}

// HWC im2col convolution against an int8 weight, dequantising inside the loop.
//
// Same reasoning as PULP_MatMul_fp32_i8_fp32_unroll1x7: a weight-only-quantised
// model would otherwise materialise the dequantised weight as its own tensor.
// For the tokenizer's conv that is 288 KB of fp32 for a 72 KB int8 constant,
// and profiling puts Conv's pre-kernel time at 143.90M cycles against 0.87M for
// the same model unquantised -- the kernel itself is unchanged at 46.20M
// vs 45.85M, so the entire difference is moving that weight.
//
// The per-tensor affine dequantisation factors out of the accumulation:
//
//   sum_k a_k * (w_k - zp) * scale  ==  scale * (sum_k a_k * w_k - zp * sum_k
//   a_k)
//
// so it costs one multiply per output element. sum_k a_k is over the im2col
// window and is shared by every output channel, hence hoisted above the channel
// loop; with zeroPoint 0 it is skipped entirely.
void PULP_Conv2d_Im2Col_fp32_i8_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const int8_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, const float32_t *__restrict__ pSrcBias,
    const bool has_bias, float32_t *__restrict__ pDstC, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer, float32_t scale,
    int32_t zeroPoint) {

  // Compute core
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  // Compute the chunk size for each core
  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  // Pointer to the weights for the current core
  const int8_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;

  uint32_t im2col_size_per_core = C * P * Q;
  float32_t *im2col_buffer = pContextBuffer + core_id * im2col_size_per_core;

  // Compute the output dimensions
  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
  uint32_t kernel_size = P * Q * C;

  // Compute the output
  if (has_bias) {
    for (uint32_t h_out = 0; h_out < H_out; h_out++) {
      for (uint32_t w_out = 0; w_out < W_out; w_out++) {
        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;

        for (uint32_t p = 0; p < P; p++) {
          int32_t h_in = h_in_start + p;

          for (uint32_t q = 0; q < Q; q++) {
            int32_t w_in = w_in_start + q;

            for (uint32_t c = 0; c < C; c++) {
              if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
                  w_in < (int32_t)W) {
                uint32_t in_idx = (h_in * W + w_in) * C + c;
                im2col_buffer[p * Q * C + q * C + c] = pSrcA[in_idx];
              } else {
                im2col_buffer[p * Q * C + q * C + c] = 0.0f;
              }
            }
          }
        }

        float32_t im2col_sum = 0.0f;
        if (zeroPoint != 0) {
          for (uint32_t k = 0; k < kernel_size; k++) {
            im2col_sum += im2col_buffer[k];
          }
        }
        for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
          float32_t sum = 0.0f;
          const int8_t *local_weight_ptr =
              weight_ptr + (f - ch_out_start) * kernel_size;

          for (uint32_t k = 0; k < kernel_size; k++) {
            sum += im2col_buffer[k] * (float32_t)local_weight_ptr[k];
          }

          uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;

          pDstC[out_idx] =
              scale * (sum - (float32_t)zeroPoint * im2col_sum) + pSrcBias[f];
        }
      }
    }
  } else {
    for (uint32_t h_out = 0; h_out < H_out; h_out++) {
      for (uint32_t w_out = 0; w_out < W_out; w_out++) {
        int32_t h_in_start = h_out * SP - pad_top;
        int32_t w_in_start = w_out * SQ - pad_left;

        for (uint32_t p = 0; p < P; p++) {
          int32_t h_in = h_in_start + p;

          for (uint32_t q = 0; q < Q; q++) {
            int32_t w_in = w_in_start + q;

            for (uint32_t c = 0; c < C; c++) {
              if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
                  w_in < (int32_t)W) {
                uint32_t in_idx = (h_in * W + w_in) * C + c;
                im2col_buffer[p * Q * C + q * C + c] = pSrcA[in_idx];
              } else {
                im2col_buffer[p * Q * C + q * C + c] = 0.0f;
              }
            }
          }
        }

        float32_t im2col_sum = 0.0f;
        if (zeroPoint != 0) {
          for (uint32_t k = 0; k < kernel_size; k++) {
            im2col_sum += im2col_buffer[k];
          }
        }
        for (uint32_t f = ch_out_start; f < ch_out_stop; f++) {
          float32_t sum = 0.0f;
          const int8_t *local_weight_ptr =
              weight_ptr + (f - ch_out_start) * kernel_size;

          for (uint32_t k = 0; k < kernel_size; k++) {
            sum += im2col_buffer[k] * (float32_t)local_weight_ptr[k];
          }

          uint32_t out_idx = (h_out * W_out + w_out) * F_total + f;

          pDstC[out_idx] = scale * (sum - (float32_t)zeroPoint * im2col_sum);
        }
      }
    }
  }
}
