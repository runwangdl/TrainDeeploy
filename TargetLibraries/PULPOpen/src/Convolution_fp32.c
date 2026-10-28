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

// RW 2026-10-13: direct convolution for a single input channel (SpeechNet block 0: 14x700 input, 1x4 kernel).
// The im2col path costs as much copying as the P*Q MACs it feeds, and inlined into the 17-argument im2col kernel
// it ran out of registers: every pixel reloaded ~10 values from the stack, which on GAP9 (slave stacks in L2)
// made the node 3.1x slower than with L1 stacks. Here the loop nest keeps its state in registers: channel
// outer (weights loaded once), output pixels inner, the row split into left edge / interior / right edge
// ranges so the interior MAC loop has fixed bounds and no branch (a version with the bound checks inside the
// MAC loop was miscompiled by the Siracusa clang at -O3 -ffast-math). Per output the taps are summed in
// k = p*Q + q order with padded taps skipped (the im2col path adds them as 0*w = +-0), so the result is
// bit-identical to the im2col path.
typedef struct {
  const float32_t *in;
  const float32_t *w;
  const float32_t *bias;
  float32_t *out;
  int32_t H, W, F, P, Q, SP, SQ, pad_top, pad_left, f0, f1, h0, h1, W_out;
} conv1ch_args_t;

// n interior outputs of one row and one channel, P == 1: everything lives in registers (<= 8 arguments).
// Accumulation order per output is q = 0..Q-1, as in the general path; bias added last (has_b only).
static void __attribute__((noinline)) conv1ch_row_interior(const float32_t *in, const float32_t *wf,
                                                           float32_t *o, int32_t n, int32_t Q, int32_t in_step,
                                                           int32_t out_step, const float32_t *bias) {
  for (int32_t i = 0; i < n; i++) {
    float32_t sum = 0.0f;
    for (int32_t q = 0; q < Q; q++) {
      sum += in[q] * wf[q];
    }
    *o = bias ? sum + *bias : sum;
    in += in_step;
    o += out_step;
  }
}

static void __attribute__((noinline)) conv1ch_rows(const conv1ch_args_t *a) {
  const int32_t H = a->H, W = a->W, F = a->F, P = a->P, Q = a->Q, SP = a->SP, SQ = a->SQ;
  const int32_t W_out = a->W_out, K = a->P * a->Q;
  // interior output columns: the whole 1..Q window lies inside the input row
  int32_t w_lo = (a->pad_left + SQ - 1) / SQ;
  int32_t w_hi = (W + a->pad_left - Q) / SQ + 1;
  if (w_hi > W_out) w_hi = W_out;
  if (w_lo > w_hi) w_lo = w_hi;
  for (int32_t h = a->h0; h < a->h1; h++) {
    const int32_t h_in0 = h * SP - a->pad_top;
    const int32_t p_lo = h_in0 < 0 ? -h_in0 : 0;
    const int32_t p_hi = (H - h_in0) < P ? (H - h_in0) : P;
    for (int32_t f = a->f0; f < a->f1; f++) {
      const float32_t *wf = a->w + (f - a->f0) * K;
      const float32_t b = a->bias ? a->bias[f] : 0.0f;
      const int32_t has_b = a->bias != NULL;
      float32_t *o = a->out + (h * W_out) * F + f;
      if (P == 1 && p_lo == 0 && p_hi == 1 && w_hi > w_lo) {
        // dominant case: interior columns through the register-only helper, edges below
        conv1ch_row_interior(a->in + h_in0 * W + w_lo * SQ - a->pad_left, wf, o + w_lo * F, w_hi - w_lo, Q, SQ, F,
                             has_b ? &a->bias[f] : NULL);
      }
      for (int32_t part = 0; part < 3; part++) {
        if (part == 1 && P == 1 && p_lo == 0 && p_hi == 1) {
          continue;  // done by conv1ch_row_interior
        }
        const int32_t ws = part == 0 ? 0 : (part == 1 ? w_lo : w_hi);
        const int32_t we = part == 0 ? w_lo : (part == 1 ? w_hi : W_out);
        for (int32_t wo = ws; wo < we; wo++) {
          const int32_t w_in0 = wo * SQ - a->pad_left;
          int32_t q_lo = 0, q_hi = Q;
          if (part != 1) {
            q_lo = w_in0 < 0 ? -w_in0 : 0;
            q_hi = (W - w_in0) < Q ? (W - w_in0) : Q;
          }
          float32_t sum = 0.0f;
          for (int32_t p = p_lo; p < p_hi; p++) {
            const float32_t *row = a->in + (h_in0 + p) * W + w_in0;
            const float32_t *wr = wf + p * Q;
            for (int32_t q = q_lo; q < q_hi; q++) {
              sum += row[q] * wr[q];
            }
          }
          o[wo * F] = has_b ? sum + b : sum;
        }
      }
    }
  }
}

static inline void Conv2d_1ch_fp32_HWC_rows(const float32_t *in, const float32_t *w, const float32_t *bias,
                                            float32_t *out, uint32_t H, uint32_t W, uint32_t F, uint32_t P,
                                            uint32_t Q, uint32_t SP, uint32_t SQ, uint32_t pad_top,
                                            uint32_t pad_left, uint32_t f0, uint32_t f1, uint32_t h0, uint32_t h1,
                                            uint32_t W_out) {
  conv1ch_args_t a = {in, w, bias, out, (int32_t)H, (int32_t)W, (int32_t)F, (int32_t)P, (int32_t)Q,
                      (int32_t)SP, (int32_t)SQ, (int32_t)pad_top, (int32_t)pad_left, (int32_t)f0, (int32_t)f1,
                      (int32_t)h0, (int32_t)h1, (int32_t)W_out};
  conv1ch_rows(&a);
}

/* RW 2026-10-13: previous channel-split version kept for reference (comment out, do not delete)
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

  // Compute the output dimensions
  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
  uint32_t kernel_size = P * Q * C;

  // Compute the chunk size for each core
  uint16_t ch_out_chunk =
      (F_total >> log2Core) + ((F_total & (NUM_CORES - 1)) != 0);
  uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
  uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
  uint32_t h_out_start = 0;
  uint32_t h_out_stop = H_out;

  // RW 2026-10-12: with fewer output channels than cores (e.g. SpeechNet block 0, F = 8 split by the
  // tiler into 1-2 channels per tile) the channel split leaves most cores idle and every active core
  // rebuilds the same im2col rows. Split output rows instead; each core computes all F_total channels.
  // Per-output summation order is unchanged, so results are bit-identical to the channel split.
  if (F_total < NUM_CORES) {
    uint32_t h_chunk = (H_out + NUM_CORES - 1) / NUM_CORES;
    ch_out_start = 0;
    ch_out_stop = F_total;
    h_out_start = MIN(h_chunk * core_id, H_out);
    h_out_stop = MIN(h_out_start + h_chunk, H_out);
    if (h_out_start == h_out_stop) {
      return;
    }
  }
  uint16_t ch_out_count = ch_out_stop - ch_out_start;

  if (ch_out_count == 0) {
    return;
  }

  // Pointer to the weights for the current core
  const float32_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;

  uint32_t im2col_size_per_core = C * P * Q;
  float32_t *im2col_buffer = pContextBuffer + core_id * im2col_size_per_core;

  // RW 2026-10-13: single input channel (SpeechNet block 0) -> register-light direct kernel, see above.
  if (C == 1) {
    Conv2d_1ch_fp32_HWC_rows(pSrcA, weight_ptr, has_bias ? pSrcBias : NULL, pDstC, H, W, F_total, P, Q, SP, SQ,
                             pad_top, pad_left, ch_out_start, ch_out_stop, h_out_start, h_out_stop, W_out);
    return;
  }

  // Compute the output
  if (has_bias) {
    for (uint32_t h_out = h_out_start; h_out < h_out_stop; h_out++) {
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
    for (uint32_t h_out = h_out_start; h_out < h_out_stop; h_out++) {
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
*/

// RW 2026-10-13: the im2col conv now splits OUTPUT PIXELS (flattened h*W_out + w) across the cores and each core
// computes all F_total channels. The old output-channel split made every core rebuild the same im2col row for
// every pixel (8x redundant gather), and left cores idle whenever a tile carried fewer than 8 channels (the
// tiler cuts SpeechNet's F=8/16 convs to 1..16 channels per tile). Per output the dot product is unchanged
// (k = 0..kernel_size-1 over the same im2col row), so results are bit-identical to the channel split.
// The work is in a separate noinline function so the single-channel path (conv1ch_rows) does not stack its
// frame on top of this one (slave stacks live in L1 and are sized to the deepest call chain).
typedef struct {
  const float32_t *in;
  const float32_t *w;
  const float32_t *bias;
  float32_t *out;
  float32_t *im2col;
  int32_t H, W, C, F, P, Q, SP, SQ, pad_top, pad_left, W_out, p0, p1;
} conv_im2col_args_t;

static void __attribute__((noinline)) conv_im2col_pixels(const conv_im2col_args_t *a) {
  const int32_t H = a->H, W = a->W, C = a->C, F = a->F, P = a->P, Q = a->Q, W_out = a->W_out;
  const int32_t K = P * Q * C;
  float32_t *col = a->im2col;
  int32_t h_out = a->p0 / W_out, w_out = a->p0 % W_out;
  for (int32_t pix = a->p0; pix < a->p1; pix++) {
    const int32_t h_in0 = h_out * a->SP - a->pad_top;
    const int32_t w_in0 = w_out * a->SQ - a->pad_left;
    float32_t *dst = col;
    for (int32_t p = 0; p < P; p++) {
      const int32_t h_in = h_in0 + p;
      for (int32_t q = 0; q < Q; q++) {
        const int32_t w_in = w_in0 + q;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
          const float32_t *src = a->in + (h_in * W + w_in) * C;
          for (int32_t c = 0; c < C; c++) dst[c] = src[c];
        } else {
          for (int32_t c = 0; c < C; c++) dst[c] = 0.0f;
        }
        dst += C;
      }
    }
    float32_t *o = a->out + pix * F;
    const float32_t *wf = a->w;
    for (int32_t f = 0; f < F; f++) {
      float32_t sum = 0.0f;
      for (int32_t k = 0; k < K; k++) sum += col[k] * wf[k];
      o[f] = a->bias ? sum + a->bias[f] : sum;
      wf += K;
    }
    if (++w_out == W_out) {
      w_out = 0;
      h_out++;
    }
  }
}

void PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total, uint32_t P,
    uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer) {
  const uint32_t core_id = pi_core_id();
  const uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  const uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
  const float32_t *bias = has_bias ? pSrcBias : NULL;

  if (C == 1) {
    // single input channel: direct convolution, rows split across cores
    uint32_t h_chunk = (H_out + NUM_CORES - 1) / NUM_CORES;
    uint32_t h0 = MIN(h_chunk * core_id, H_out);
    uint32_t h1 = MIN(h0 + h_chunk, H_out);
    if (h0 < h1) {
      Conv2d_1ch_fp32_HWC_rows(pSrcA, pSrcB, bias, pDstC, H, W, F_total, P, Q, SP, SQ, pad_top, pad_left, 0,
                               F_total, h0, h1, W_out);
    }
    return;
  }

  const uint32_t n_pix = H_out * W_out;
  const uint32_t chunk = (n_pix + NUM_CORES - 1) / NUM_CORES;
  const uint32_t p0 = MIN(chunk * core_id, n_pix);
  const uint32_t p1 = MIN(p0 + chunk, n_pix);
  if (p0 >= p1) {
    return;
  }
  conv_im2col_args_t a = {pSrcA, pSrcB, bias, pDstC, pContextBuffer + core_id * C * P * Q,
                          (int32_t)H, (int32_t)W, (int32_t)C, (int32_t)F_total, (int32_t)P, (int32_t)Q,
                          (int32_t)SP, (int32_t)SQ, (int32_t)pad_top, (int32_t)pad_left, (int32_t)W_out,
                          (int32_t)p0, (int32_t)p1};
  conv_im2col_pixels(&a);
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
