/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pmsis.h"

#include "DeeployPULPMath.h"

#include <math.h>

/* QW: frozen-stat BN switch. 0 (default) = training-mode batch statistics (unchanged
 * behaviour for all existing tests). 1 = normalize the training forward with the FROZEN
 * pretrained running stats (train == inference), and use the matching affine backward
 * (dX = gamma*inv_std*dY). Set by the training harness (extern) only when built with
 * -D BN_FROZEN_STATS. Enables paper-faithful full-model FT at batch-1 without the
 * batch-stat corruption. -- QW */
uint32_t g_bn_frozen_stats = 0u;
uint32_t g_bn_debug = 0u; /* QW: runtime-gated frozen-BN kernel debug print -- QW */

/*
 * Training-mode Batch Normalization forward pass (BatchNormInternal).
 *
 * Computes per-channel batch statistics over (N, H, W) and normalizes.
 * Saves batch mean and 1/sqrt(var+eps) into stash buffers for the backward
 * pass. Running statistics are NOT updated (their outputs have no consumers in
 * the graph).
 *
 * Layout: NCHW — element [n, c, h, w] lives at offset (n*C + c)*N_hw + h*W_in +
 * w.
 *
 * Parallelism: channels are split evenly across cores.
 */
void PULP_BatchNormInternal_fp32(
    const float32_t *X, const float32_t *gamma, const float32_t *beta,
    const float32_t *running_mean, const float32_t *running_var, float32_t *Y,
    float32_t *saved_mean, float32_t *saved_inv_std, uint32_t N, uint32_t C,
    uint32_t H_in, uint32_t W_in, float32_t epsilon, float32_t momentum) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t N_hw = H_in * W_in;

  /* RW 2026-10-13: frozen stats make BN a per-element affine map, so split the flattened (n, c, hw) range across
   * the cores instead of the channels -- the tiler hands block 0 one channel per tile, which left 7 of 8 cores idle
   * (1.12 M of a 5.9 M QZO step). Same per-element expression as below, so the output is bit-identical. */
  if (g_bn_frozen_stats) {
    const uint32_t total = N * C * N_hw;
    const uint32_t chunk_e = (total + NUM_CORES - 1) / NUM_CORES;
    const uint32_t e0 = MIN(chunk_e * (uint32_t)core_id, total);
    const uint32_t e1 = MIN(e0 + chunk_e, total);
    uint32_t e = e0;
    while (e < e1) {  // one segment per channel run, constants hoisted out of the inner loop
      const uint32_t c = (e / N_hw) % C;
      const uint32_t seg_end = MIN(e1, (e / N_hw + 1) * N_hw);
      const float32_t mean = running_mean[c], inv_std = 1.0f / sqrtf(running_var[c] + epsilon);
      const float32_t g = gamma[c], b = beta[c];
      const float32_t *x = X + e;
      float32_t *y = Y + e;
      for (uint32_t k = seg_end - e; k > 0; k--) {
        *y++ = (*x++ - mean) * inv_std * g + b;
      }
      e = seg_end;
    }
    for (uint32_t c = core_id; c < C; c += NUM_CORES) {
      saved_mean[c] = running_mean[c];
      saved_inv_std[c] = 1.0f / sqrtf(running_var[c] + epsilon);
    }
    return;
  }
  uint32_t N_total = N * N_hw;
  float32_t inv_N = 1.0f / (float32_t)N_total;

  /* Split channels across cores */
  int32_t chunk =
      ((int32_t)C >> log2Core) + (((int32_t)C & (NUM_CORES - 1)) != 0);
  int32_t c_start = MIN(chunk * core_id, (int32_t)C);
  int32_t c_end = MIN(c_start + chunk, (int32_t)C);

  for (int32_t c = c_start; c < c_end; c++) {
    float32_t mean, inv_std;
    if (g_bn_frozen_stats) {
      /* QW: frozen-stat BN — normalize with the pretrained running stats (constants),
       * exactly what inference uses, so training and inference are consistent. -- QW */
      mean = running_mean[c];
      inv_std = 1.0f / sqrtf(running_var[c] + epsilon);
#ifdef BN_DEBUG_PRINT  // RW 2026-10-13: the float printf chain needs ~600 B of slave stack
      if (g_bn_debug && core_id == 0) { /* QW: dump what the frozen kernel reads per tile -- QW */
        printf("[BN_DBG] C=%u c=%d gamma=%f beta=%f rmean=%f rvar=%f inv_std=%f x0=%f\n",
               (unsigned)C, (int)c, gamma[c], beta[c], running_mean[c], running_var[c],
               inv_std, X[(0 * C + c) * N_hw]);
      }
#endif
    } else {
      /* ── Compute batch mean ───────────────────────────────────────────── */
      mean = 0.0f;
      for (uint32_t n = 0; n < N; n++) {
        const float32_t *x_nc = X + (n * C + c) * N_hw;
        for (uint32_t hw = 0; hw < N_hw; hw++) {
          mean += x_nc[hw];
        }
      }
      mean *= inv_N;

      /* ── Compute batch variance (unbiased=False) ──────────────────────── */
      float32_t var = 0.0f;
      for (uint32_t n = 0; n < N; n++) {
        const float32_t *x_nc = X + (n * C + c) * N_hw;
        for (uint32_t hw = 0; hw < N_hw; hw++) {
          float32_t diff = x_nc[hw] - mean;
          var += diff * diff;
        }
      }
      var *= inv_N;
      inv_std = 1.0f / sqrtf(var + epsilon);
    }
    saved_mean[c] = mean;
    saved_inv_std[c] = inv_std;

    float32_t g = gamma[c];
    float32_t b = beta[c];

    /* ── Normalize and apply affine transform ───────────────────────────── */
    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      float32_t *y_nc = Y + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        y_nc[hw] = (x_nc[hw] - mean) * inv_std * g + b;
      }
    }
#ifdef BN_DEBUG_PRINT
    if (g_bn_frozen_stats && g_bn_debug && core_id == 0) { /* QW: dump BN output for first channel -- QW */
      float32_t *y0 = Y + (0 * C + c) * N_hw;
      printf("[BN_OUT] C=%u c=%d N_hw=%u y[0]=%f y[1]=%f y[last]=%f\n",
             (unsigned)C, (int)c, (unsigned)N_hw, y0[0], y0[1], y0[N_hw - 1]);
    }
#endif
  }
}

/*
 * Batch Normalization backward pass.
 *
 * Uses saved_mean / saved_inv_std stash from the forward pass to compute:
 *   dbeta[c]    = sum_{n,h,w} dY[n,c,h,w]
 *   dgamma[c]   = sum_{n,h,w} dY[n,c,h,w] * x_hat[n,c,h,w]
 *   dX[n,c,h,w] = inv_std / N_total * (N_total * dY[n,c,h,w] * gamma[c]
 *                                       - dbeta[c]
 *                                       - x_hat[n,c,h,w] * dgamma[c])
 *
 * Parallelism: channels are split evenly across cores.
 * Each core independently computes its channel slice of dgamma, dbeta, dX.
 */
/*
 * Welford reduction: compute per-channel mean and 1/sqrt(var+eps).
 * Parallelism: channels split across cores.
 */
void PULP_WelfordReduce_fp32(const float32_t *X, float32_t *saved_mean,
                             float32_t *saved_inv_std, uint32_t N, uint32_t C,
                             uint32_t H_in, uint32_t W_in, float32_t epsilon) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t N_hw = H_in * W_in;
  uint32_t N_total = N * N_hw;
  float32_t inv_N = 1.0f / (float32_t)N_total;

  int32_t chunk =
      ((int32_t)C >> log2Core) + (((int32_t)C & (NUM_CORES - 1)) != 0);
  int32_t c_start = MIN(chunk * core_id, (int32_t)C);
  int32_t c_end = MIN(c_start + chunk, (int32_t)C);

  for (int32_t c = c_start; c < c_end; c++) {
    float32_t mean = 0.0f;
    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        mean += x_nc[hw];
      }
    }
    mean *= inv_N;
    saved_mean[c] = mean;

    float32_t var = 0.0f;
    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        float32_t diff = x_nc[hw] - mean;
        var += diff * diff;
      }
    }
    var *= inv_N;
    saved_inv_std[c] = 1.0f / sqrtf(var + epsilon);
  }
}

/*
 * Channel normalize: Y = (X - mean) * inv_std * gamma + beta.
 * Freely spatially tileable. Parallelism: channels split across cores.
 */
void PULP_ChannelNormalize_fp32(const float32_t *X, const float32_t *saved_mean,
                                const float32_t *saved_inv_std,
                                const float32_t *gamma, const float32_t *beta,
                                float32_t *Y, uint32_t N, uint32_t C,
                                uint32_t H_in, uint32_t W_in) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t N_hw = H_in * W_in;

  int32_t chunk =
      ((int32_t)C >> log2Core) + (((int32_t)C & (NUM_CORES - 1)) != 0);
  int32_t c_start = MIN(chunk * core_id, (int32_t)C);
  int32_t c_end = MIN(c_start + chunk, (int32_t)C);

  for (int32_t c = c_start; c < c_end; c++) {
    float32_t mean = saved_mean[c];
    float32_t inv_std = saved_inv_std[c];
    float32_t g = gamma[c];
    float32_t b = beta[c];

    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      float32_t *y_nc = Y + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        y_nc[hw] = (x_nc[hw] - mean) * inv_std * g + b;
      }
    }
  }
}

/*
 * BN gradient reduction: compute dgamma and dbeta.
 * Parallelism: channels split across cores.
 */
void PULP_BNGradReduce_fp32(const float32_t *dY, const float32_t *X,
                            const float32_t *saved_mean,
                            const float32_t *saved_inv_std, float32_t *dgamma,
                            float32_t *dbeta, uint32_t N, uint32_t C,
                            uint32_t H_in, uint32_t W_in) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t N_hw = H_in * W_in;

  int32_t chunk =
      ((int32_t)C >> log2Core) + (((int32_t)C & (NUM_CORES - 1)) != 0);
  int32_t c_start = MIN(chunk * core_id, (int32_t)C);
  int32_t c_end = MIN(c_start + chunk, (int32_t)C);

  for (int32_t c = c_start; c < c_end; c++) {
    float32_t mean = saved_mean[c];
    float32_t inv_std = saved_inv_std[c];

    float32_t sum_dbeta = 0.0f;
    float32_t sum_dgamma = 0.0f;

    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      const float32_t *dy_nc = dY + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        float32_t x_hat = (x_nc[hw] - mean) * inv_std;
        sum_dbeta += dy_nc[hw];
        sum_dgamma += dy_nc[hw] * x_hat;
      }
    }
    dgamma[c] = sum_dgamma;
    dbeta[c] = sum_dbeta;
  }
}

/*
 * BN gradient normalize: compute dX using pre-computed dgamma, dbeta.
 * Freely spatially tileable. Parallelism: channels split across cores.
 *
 * N_total_inv = 1.0f / (N * H_full * W_full), pre-computed with FULL spatial
 * dims.
 */
void PULP_BNGradNormalize_fp32(const float32_t *dY, const float32_t *X,
                               const float32_t *saved_mean,
                               const float32_t *saved_inv_std,
                               const float32_t *gamma, const float32_t *dgamma,
                               const float32_t *dbeta, float32_t *dX,
                               uint32_t N, uint32_t C, uint32_t H_in,
                               uint32_t W_in, float32_t N_total_inv) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t N_hw = H_in * W_in;
  /* N_total for the BN formula uses FULL spatial extents (passed as
   * N_total_inv). */
  float32_t N_total_f = 1.0f / N_total_inv;

  int32_t chunk =
      ((int32_t)C >> log2Core) + (((int32_t)C & (NUM_CORES - 1)) != 0);
  int32_t c_start = MIN(chunk * core_id, (int32_t)C);
  int32_t c_end = MIN(c_start + chunk, (int32_t)C);

  for (int32_t c = c_start; c < c_end; c++) {
    float32_t mean = saved_mean[c];
    float32_t inv_std = saved_inv_std[c];
    float32_t g = gamma[c];
    float32_t dg = dgamma[c];
    float32_t db = dbeta[c];
    /* scale = gamma * inv_std / N_total; gamma applies to all three terms of
       the canonical BN backward formula
         dX = (g * inv_std / N) * (N*dY - dbeta - x_hat * dgamma) */
    float32_t scale = g * inv_std * N_total_inv;

    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      const float32_t *dy_nc = dY + (n * C + c) * N_hw;
      float32_t *dx_nc = dX + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        float32_t x_hat = (x_nc[hw] - mean) * inv_std;
        dx_nc[hw] = scale * (N_total_f * dy_nc[hw] - db - x_hat * dg);
      }
    }
  }
}

void PULP_BatchNormGrad_fp32(const float32_t *dY, const float32_t *X,
                             const float32_t *gamma,
                             const float32_t *saved_mean,
                             const float32_t *saved_inv_std, float32_t *dX,
                             float32_t *dgamma, float32_t *dbeta, uint32_t N,
                             uint32_t C, uint32_t H_in, uint32_t W_in,
                             float32_t epsilon) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t N_hw = H_in * W_in;
  uint32_t N_total = N * N_hw;
  float32_t inv_N = 1.0f / (float32_t)N_total;

  /* Split channels across cores */
  int32_t chunk =
      ((int32_t)C >> log2Core) + (((int32_t)C & (NUM_CORES - 1)) != 0);
  int32_t c_start = MIN(chunk * core_id, (int32_t)C);
  int32_t c_end = MIN(c_start + chunk, (int32_t)C);

  for (int32_t c = c_start; c < c_end; c++) {
    float32_t mean = saved_mean[c];
    float32_t inv_std = saved_inv_std[c];
    float32_t g = gamma[c];

    /* ── First pass: accumulate dbeta and dgamma ─────────────────────────── */
    float32_t sum_dbeta = 0.0f;
    float32_t sum_dgamma = 0.0f;

    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      const float32_t *dy_nc = dY + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        float32_t x_hat = (x_nc[hw] - mean) * inv_std;
        sum_dbeta += dy_nc[hw];
        sum_dgamma += dy_nc[hw] * x_hat;
      }
    }
    dgamma[c] = sum_dgamma;
    dbeta[c] = sum_dbeta;

    /* ── Second pass: compute dX ─────────────────────────────────────────── */
    if (g_bn_frozen_stats) {
      /* QW: frozen stats are constants w.r.t. X, so the BN reduces to a plain per-channel
       * affine y = gamma*inv_std*(x-mean)+beta -> dX = gamma*inv_std*dY. None of the
       * batch-stat Jacobian terms (-dbeta, -x_hat*dgamma) apply. -- QW */
      float32_t s = g * inv_std;
      for (uint32_t n = 0; n < N; n++) {
        const float32_t *dy_nc = dY + (n * C + c) * N_hw;
        float32_t *dx_nc = dX + (n * C + c) * N_hw;
        for (uint32_t hw = 0; hw < N_hw; hw++) {
          dx_nc[hw] = s * dy_nc[hw];
        }
      }
    } else {
      /* scale = gamma * inv_std / N_total; gamma applies to all three terms:
           dX = (g * inv_std / N) * (N*dY - dbeta - x_hat * dgamma) */
      float32_t scale = g * inv_std * inv_N;
      for (uint32_t n = 0; n < N; n++) {
        const float32_t *x_nc = X + (n * C + c) * N_hw;
        const float32_t *dy_nc = dY + (n * C + c) * N_hw;
        float32_t *dx_nc = dX + (n * C + c) * N_hw;
        for (uint32_t hw = 0; hw < N_hw; hw++) {
          float32_t x_hat = (x_nc[hw] - mean) * inv_std;
          dx_nc[hw] = scale * ((float32_t)N_total * dy_nc[hw] - sum_dbeta -
                               x_hat * sum_dgamma);
        }
      }
    }
  }
}

/*
 * RW 2026-10-13: BatchNormInternal on a channels-last tile X[P][C] (P = N*H*W pixels), emitted for frozen-stat BN
 * (NCHWtoNHWCBatchNormInternalPass, MeZO graphs with -D BN_FROZEN_STATS=ON). Output element by element as the
 * NCHW kernel computes it -- (x - mean) * inv_std * gamma + beta -- so the result is bit-identical; the pixels
 * are split across the cores. With g_bn_frozen_stats == 0 the statistics would come from this tile only, which
 * is only right for an untiled tensor: the pass never emits this kernel in that configuration.
 */

void PULP_BatchNormInternalHWC_fp32(const float32_t *X, const float32_t *gamma, const float32_t *beta,
                                    const float32_t *running_mean, const float32_t *running_var, float32_t *Y,
                                    float32_t *saved_mean, float32_t *saved_inv_std, uint32_t P, uint32_t C,
                                    float32_t epsilon) {
  const int32_t core_id = pi_core_id();
  const uint32_t chunk = (P + NUM_CORES - 1) / NUM_CORES;
  const uint32_t p0 = MIN(chunk * (uint32_t)core_id, P), p1 = MIN(p0 + chunk, P);

  // channel-outer, pixel-inner with stride C (L1, no per-pixel constant arrays on the 768 B slave stack)
  for (uint32_t c = 0; c < C; c++) {
    float32_t mean, inv_std;
    if (g_bn_frozen_stats) {
      mean = running_mean[c];
      inv_std = 1.0f / sqrtf(running_var[c] + epsilon);
    } else {
      float32_t s = 0.0f, v = 0.0f;
      for (uint32_t p = 0; p < P; p++) s += X[p * C + c];
      s *= 1.0f / (float32_t)P;
      for (uint32_t p = 0; p < P; p++) {
        const float32_t d = X[p * C + c] - s;
        v += d * d;
      }
      v *= 1.0f / (float32_t)P;
      mean = s;
      inv_std = 1.0f / sqrtf(v + epsilon);
    }
    const float32_t g = gamma[c], b = beta[c];
    const float32_t *x = X + p0 * C + c;
    float32_t *y = Y + p0 * C + c;
    for (uint32_t p = p0; p < p1; p++) {
      *y = (*x - mean) * inv_std * g + b;
      x += C;
      y += C;
    }
#ifndef BN_HWC_NO_SAVED
    if ((uint32_t)core_id == c % NUM_CORES) {
      saved_mean[c] = mean;
      saved_inv_std[c] = inv_std;
    }
#endif
  }
}

/*
 * RW 2026-10-15: channels-last frozen-stat BatchNormInternal on an int8 tile (a Dequant folded in by
 * FoldDequantIntoFrozenBNPass). Each element is dequantized with the Dequant kernel's own expression,
 *   dq = ((float)q - zero_point) * scale,
 * and the empty asm makes dq a rounded fp32 value in a register before BN uses it -- without it the compiler may fuse
 * the multiply with BN's subtraction (fmsub), which rounds once instead of twice and changes the bits. Then BN as in
 * PULP_BatchNormInternalHWC_fp32. Bit-identical to Dequant followed by that kernel; the fp32 tensor between them is
 * never materialised.
 */
void PULP_BatchNormInternalHWC_i8_fp32(const int8_t *X, const float32_t *gamma, const float32_t *beta,
                                       const float32_t *running_mean, const float32_t *running_var, float32_t *Y,
                                       float32_t *saved_mean, float32_t *saved_inv_std, uint32_t P, uint32_t C,
                                       float32_t epsilon, float32_t dq_scale, float32_t dq_zero_point) {
  const int32_t core_id = pi_core_id();
  const uint32_t chunk = (P + NUM_CORES - 1) / NUM_CORES;
  const uint32_t p0 = MIN(chunk * (uint32_t)core_id, P), p1 = MIN(p0 + chunk, P);
  for (uint32_t c = 0; c < C; c++) {
    /* frozen statistics only: the pass folds the Dequant only for -D BN_FROZEN_STATS=ON builds */
    const float32_t mean = running_mean[c];
    const float32_t inv_std = 1.0f / sqrtf(running_var[c] + epsilon);
    const float32_t g = gamma[c], b = beta[c];
    const int8_t *x = X + p0 * C + c;
    float32_t *y = Y + p0 * C + c;
    for (uint32_t p = p0; p < p1; p++) {
      float32_t dq = ((float32_t)(int32_t)*x - dq_zero_point) * dq_scale;
#if defined(__riscv_flen) && !defined(__riscv_zfinx)
      __asm__ volatile("" : "+f"(dq));
#else
      __asm__ volatile("" : "+r"(dq));
#endif
      *y = (dq - mean) * inv_std * g + b;
      x += C;
      y += C;
    }
    if ((uint32_t)core_id == c % NUM_CORES) {
      saved_mean[c] = mean;
      saved_inv_std[c] = inv_std;
    }
  }
}

/*
 * QW (dual_pass_fusion exp9, memory): FrozenBNPoolReluQuant -- the fp32 tail of a frozen-stat quantized block in one
 * pass, int8 HWC in, int8 HWC out:
 *   x [H_out*kh, W_in, C] i8  ->  BN (folded Dequant)  ->  MaxPool kh x kw, stride = kernel, no pad  ->  Relu
 *     ->  Quant  ->  y [H_out, W_out, C] i8
 * Replaces PULP_BatchNormInternalHWC_i8_fp32 + PULP_MaxPool2d_fp32_fp32_HWC + 2 Transposes + PULP_Relu_fp32_fp32 +
 * the Quant template, whose fp32 tensors (block 0: 628 KB BN output) are never stored. Bit-identical: every step is
 * written with the expression of the kernel it replaces --
 *   BN     dq = ((float)q - dq_zp) * dq_scale (barrier), y = (dq - mean) * inv_std * g + b   (as above, same file)
 *   pool   m = -inf; m = fmax.s(v, m) over the window                        (PULP_MaxPool2d_fp32_fp32_HWC)
 *   Relu   r = MAX(m, 0.0f)                                                  (PULP_Relu_fp32_fp32)
 *   Quant  s = r * q_mul + q_zp; q = (int32)(s + 0.5f * (s >= 0 ? 1 : -1)), clamped   (PULPQuantTemplate)
 * The y barrier keeps BN's result a rounded fp32 value, as it was when stored. Cores split the output pixels; the
 * channel loop is outside, as in the BN kernel, so the per-channel constants stay in registers.
 */
void PULP_FrozenBNPoolReluQuantHWC_i8_i8(const int8_t *X, const float32_t *gamma, const float32_t *beta,
                                         const float32_t *running_mean, const float32_t *running_var, int8_t *Y,
                                         uint32_t H_out, uint32_t W_in, uint32_t W_out, uint32_t C, uint32_t kh,
                                         uint32_t kw, float32_t epsilon, float32_t dq_scale, float32_t dq_zero_point,
                                         float32_t q_mul, float32_t q_zero_point, int32_t q_min, int32_t q_max) {
  const int32_t core_id = pi_core_id();
  const uint32_t n_pix = H_out * W_out;
  const uint32_t chunk = (n_pix + NUM_CORES - 1) / NUM_CORES;
  const uint32_t p0 = MIN(chunk * (uint32_t)core_id, n_pix), p1 = MIN(p0 + chunk, n_pix);
  for (uint32_t c = 0; c < C; c++) {
    const float32_t mean = running_mean[c];
    const float32_t inv_std = 1.0f / sqrtf(running_var[c] + epsilon);
    const float32_t g = gamma[c], b = beta[c];
    for (uint32_t pix = p0; pix < p1; pix++) {
      const uint32_t ho = pix / W_out, wo = pix % W_out;
      float32_t m = -inf;
      for (uint32_t dy = 0; dy < kh; dy++) {
        const int8_t *x = X + ((ho * kh + dy) * W_in + wo * kw) * C + c;
        for (uint32_t dx = 0; dx < kw; dx++) {
          float32_t dq = ((float32_t)(int32_t)*x - dq_zero_point) * dq_scale;
#if defined(__riscv_flen) && !defined(__riscv_zfinx)
          __asm__ volatile("" : "+f"(dq));
#else
          __asm__ volatile("" : "+r"(dq));
#endif
          float32_t v = (dq - mean) * inv_std * g + b;
#if defined(__riscv_flen) && !defined(__riscv_zfinx)
          __asm__ volatile("" : "+f"(v));
          __asm__("fmax.s %0, %1, %2" : "=f"(m) : "f"(v), "f"(m));
#else
          __asm__ volatile("" : "+r"(v));
          m = v > m ? v : m;
#endif
          x += C;
        }
      }
      const float32_t r = MAX(m, 0.0f);
      const float32_t s = r * q_mul + q_zero_point;
      int32_t q = (int32_t)(s + 0.5f * (s >= 0 ? 1 : -1));
      if (q < q_min)
        q = q_min;
      if (q > q_max)
        q = q_max;
      Y[pix * C + c] = (int8_t)q;
    }
  }
}
