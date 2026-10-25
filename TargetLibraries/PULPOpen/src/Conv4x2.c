/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp7a, step 2a): SINGLE-PASS requantised int8 conv, the control kernel for the
 * blocks 1-4 fused kernel. It receives exactly what pulp_nn_conv_i8_i8_i8 receives (the already perturbed,
 * already HWC-transposed weight; one input) and must be bit-identical to it, so its cycles against pulp-nn
 * measure the kernel structure alone, before any fusion.
 *
 * What it keeps from FlatConv.c (exp5a) and what it changes, after the block-1 measurement
 * (exp7a/block1_unfused: FlatConv 652 k vs pulp-nn 401 k conv cycles at K = 128):
 *   kept     - a contiguous range of the FLATTENED output-pixel index per core (100 % balance on block 1,
 *              vs pulp-nn's 4/4/4/2 row split at 87.5 %);
 *            - no calls in the pixel loop; the requant inline, exactly as pulp_nn_bn_quant_i8;
 *   changed  - register blocking: 4 output channels x 2 output pixels per inner iteration (8 accumulators,
 *              2 activation + 4 weight words per 8 sdotsp4), the shape of pulp_nn_matmul_i8_i8_i8.
 *              FlatConv did 1 x 1 (2 loads per sdotsp4);
 *            - no im2col copy for interior pixels: in HWC one kernel row of a receptive field is
 *              Kw*Ci CONTIGUOUS input bytes, so the dot loop reads the input in place, row by row
 *              (one row for 1xK kernels, Kh rows for Kx1). Only pixels whose field touches the padding
 *              are gathered into a zero-filled per-core patch (block 1: 16 of 88 output columns).
 *
 * Preconditions (all SpeechNet convs satisfy them): K = Ci*Kh*Kw and the row length Kw*Ci are multiples
 * of 4. In-place reads additionally need Ci % 4 == 0 and a word-aligned input; otherwise every pixel is
 * gathered (block 0, Ci = 1) -- still correct, just slower.
 *
 * Axis names follow pulp_nn_conv_i8_i8_i8 so the Deeploy template passes the same arguments:
 * `x` is the axis Deeploy calls dim_im_in_y (the long / time axis), `y` is dim_im_in_x.
 * Input  : HWC  [dim_in_y][dim_in_x][ch_in]
 * Weight : [ch_out][dim_kernel_y][dim_kernel_x][ch_in]   (Deeploy's NHWC-transposed layout)
 * Output : HWC  [dim_out_y][dim_out_x][ch_out]
 * Scratch: num_cores * 2 * K bytes (two patches per core; Conv4x2Template.computeTransientBuffersSize).
 *
 * QW (2026-10-22, per-block breakdown): this kernel now carries every NON-fusion optimisation that DualConv2In.c
 * received in exp7 steps 5c-5d, so that 2 x Conv4x2 vs DualConv2In isolates the fusion alone:
 *   - step 5d: word-wise, out-of-line border gather (c42_gather = d2_gather's word path for one input);
 *   - step 5c: phase-2 split by PIXELS or by OUTPUT CHANNELS (same rule as DualConv2In.c; override
 *     -D CONV4X2_SPLIT=1 (pixels) | 2 (channels));
 *   - same pixel loop shape as DualConv2In.c (both field addresses first, then the 4x2 loop).
 * Not ported because they belong to the fusion: phase 1 (in-kernel perturbation), the de-interleave of the
 * channel-stacked [x+ | x-] input (exp8), the second pass. The step-2a kernel is kept below and selected with
 * -D CONV4X2_STEP2A=ON (the version behind exp7a's 653,328 cycles).
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

typedef signed char c42_v4s __attribute__((vector_size(4)));

/* requant exactly as pulp_nn_bn_quant_i8 (32-bit variant) and FlatConv.c: int32 wrap, arithmetic shift, clip */
#define C42_RQ(acc, c)                                                                                       \
  ((int8_t)__builtin_pulp_clip_r((int32_t)((uint32_t)(acc) * (uint32_t)pMul[c] + (uint32_t)pAdd[c]) >> out_shift, \
                                 127))

#ifdef CONV4X2_STEP2A /* -- QW 2026-10-22: original step-2a kernel, kept for A/B (-D CONV4X2_STEP2A=ON) */
/* Address of the receptive field's first kernel row and the distance between its rows. In place when the
 * field lies inside the input; otherwise the field is gathered into `patch` (rows contiguous, zeros where a
 * tap falls into the padding). */
static inline const int8_t *c42_receptive_field(const int8_t *pIn, int8_t *patch, int32_t iy0, int32_t ix0, uint32_t in_place,
                                      uint16_t dim_in_x, uint16_t dim_in_y, uint16_t ch_in, uint16_t dim_kernel_x,
                                      uint16_t dim_kernel_y, uint32_t *row_stride) {
  if (in_place && iy0 >= 0 && iy0 + (int32_t)dim_kernel_y <= (int32_t)dim_in_y && ix0 >= 0 &&
      ix0 + (int32_t)dim_kernel_x <= (int32_t)dim_in_x) {
    *row_stride = (uint32_t)dim_in_x * ch_in;
    return pIn + ((uint32_t)iy0 * dim_in_x + (uint32_t)ix0) * ch_in;
  }
  int8_t *dst = patch;
  for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
    const int32_t iy = iy0 + (int32_t)ky;
    for (uint32_t kx = 0; kx < dim_kernel_x; kx++) {
      const int32_t ix = ix0 + (int32_t)kx;
      if (iy < 0 || iy >= (int32_t)dim_in_y || ix < 0 || ix >= (int32_t)dim_in_x) {
        for (uint32_t ci = 0; ci < ch_in; ci++)
          *dst++ = 0;
      } else {
        const int8_t *src = pIn + ((uint32_t)iy * dim_in_x + (uint32_t)ix) * ch_in;
        for (uint32_t ci = 0; ci < ch_in; ci++)
          *dst++ = src[ci];
      }
    }
  }
  *row_stride = (uint32_t)dim_kernel_x * ch_in;
  return patch;
}

void PULP_Conv4x2_i8_i8_i8_HWC(const int8_t *pIn, int8_t *pScratch, int8_t *pOut, const int8_t *pWeight,
                               const int32_t *pMul, const int32_t *pAdd, uint32_t num_cores, uint16_t out_shift,
                               uint16_t dim_in_x, uint16_t dim_in_y, uint16_t ch_in, uint16_t dim_out_x,
                               uint16_t dim_out_y, uint16_t ch_out, uint16_t dim_kernel_x, uint16_t dim_kernel_y,
                               uint16_t pad_y_top, uint16_t pad_y_bottom, uint16_t pad_x_left,
                               uint16_t pad_x_right, uint16_t stride_x, uint16_t stride_y) {
  const uint32_t core_id = pi_core_id();
  const uint32_t K = (uint32_t)ch_in * dim_kernel_x * dim_kernel_y;   // block 1: 8*16*1 = 128
  const uint32_t row_len = (uint32_t)dim_kernel_x * ch_in;           // bytes of one kernel row, block 1: 128
  const uint32_t row_words = row_len >> 2;
  const uint32_t in_place = ((ch_in & 3u) == 0u) && ((((uint32_t)pIn) & 3u) == 0u);
  int8_t *patchA = pScratch + core_id * 2u * K;
  int8_t *patchB = patchA + K;

  /* ---------------- flattened output-pixel range of this core, walked two pixels at a time --------- */
  const uint32_t n_out_px = (uint32_t)dim_out_y * dim_out_x;           // block 1: 14*88 = 1232
  const uint32_t px_chunk = (n_out_px + num_cores - 1u) / num_cores;   // block 1: 154
  uint32_t px_begin = px_chunk * core_id;
  if (px_begin > n_out_px)
    px_begin = n_out_px;
  uint32_t px_end = px_begin + px_chunk;
  if (px_end > n_out_px)
    px_end = n_out_px;
  uint32_t oy = px_begin / dim_out_x;
  uint32_t ox = px_begin - oy * dim_out_x;

  for (uint32_t px = px_begin; px < px_end; px += 2u) {
    const uint32_t has_px_b = (px + 1u < px_end); // the last pixel of an odd range runs alone (B duplicates A)
    uint32_t rowStrideA, rowStrideB;
    const int8_t *fieldA =
        c42_receptive_field(pIn, patchA, (int32_t)(oy * stride_y) - (int32_t)pad_y_top,
                            (int32_t)(ox * stride_x) - (int32_t)pad_x_left, in_place, dim_in_x, dim_in_y, ch_in,
                            dim_kernel_x, dim_kernel_y, &rowStrideA);
    if (++ox == dim_out_x) {
      ox = 0;
      oy++;
    }
    const int8_t *fieldB = fieldA;
    rowStrideB = rowStrideA;
    if (has_px_b) {
      fieldB = c42_receptive_field(pIn, patchB, (int32_t)(oy * stride_y) - (int32_t)pad_y_top,
                                   (int32_t)(ox * stride_x) - (int32_t)pad_x_left, in_place, dim_in_x, dim_in_y,
                                   ch_in, dim_kernel_x, dim_kernel_y, &rowStrideB);
      if (++ox == dim_out_x) {
        ox = 0;
        oy++;
      }
    }
    int8_t *outA = pOut + px * ch_out;
    int8_t *outB = outA + ch_out;

    /* ---- 4 output channels x 2 pixels per iteration (pulp_nn_matmul_i8_i8_i8's 4x2 shape) ----
     * accumulators a<j> / b<j>: pixel A / B, output channel c + j; v<j> / w<j>: weight word / row of channel c + j */
    uint32_t c = 0;
    for (; c + 4u <= ch_out; c += 4u) {
      int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
      const int8_t *w = pWeight + c * K;
      for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
        const c42_v4s *xa = (const c42_v4s *)(fieldA + ky * rowStrideA);
        const c42_v4s *xb = (const c42_v4s *)(fieldB + ky * rowStrideB);
        const c42_v4s *w0 = (const c42_v4s *)(w + ky * row_len);
        const c42_v4s *w1 = (const c42_v4s *)(w + K + ky * row_len);
        const c42_v4s *w2 = (const c42_v4s *)(w + 2u * K + ky * row_len);
        const c42_v4s *w3 = (const c42_v4s *)(w + 3u * K + ky * row_len);
        for (uint32_t k = 0; k < row_words; k++) {
          const c42_v4s va = *xa++;
          const c42_v4s vb = *xb++;
          const c42_v4s v0 = *w0++;
          const c42_v4s v1 = *w1++;
          const c42_v4s v2 = *w2++;
          const c42_v4s v3 = *w3++;
          a0 = __builtin_pulp_sdotsp4(va, v0, a0);
          b0 = __builtin_pulp_sdotsp4(vb, v0, b0);
          a1 = __builtin_pulp_sdotsp4(va, v1, a1);
          b1 = __builtin_pulp_sdotsp4(vb, v1, b1);
          a2 = __builtin_pulp_sdotsp4(va, v2, a2);
          b2 = __builtin_pulp_sdotsp4(vb, v2, b2);
          a3 = __builtin_pulp_sdotsp4(va, v3, a3);
          b3 = __builtin_pulp_sdotsp4(vb, v3, b3);
        }
      }
      outA[c] = C42_RQ(a0, c);
      outA[c + 1u] = C42_RQ(a1, c + 1u);
      outA[c + 2u] = C42_RQ(a2, c + 2u);
      outA[c + 3u] = C42_RQ(a3, c + 3u);
      if (has_px_b) {
        outB[c] = C42_RQ(b0, c);
        outB[c + 1u] = C42_RQ(b1, c + 1u);
        outB[c + 2u] = C42_RQ(b2, c + 2u);
        outB[c + 3u] = C42_RQ(b3, c + 3u);
      }
    }
    /* ---- channel tail (ch_out % 4), 1 x 2 ---- */
    for (; c < ch_out; c++) {
      int32_t a = 0, b = 0;
      const int8_t *w = pWeight + c * K;
      for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
        const c42_v4s *xa = (const c42_v4s *)(fieldA + ky * rowStrideA);
        const c42_v4s *xb = (const c42_v4s *)(fieldB + ky * rowStrideB);
        const c42_v4s *w0 = (const c42_v4s *)(w + ky * row_len);
        for (uint32_t k = 0; k < row_words; k++) {
          const c42_v4s v0 = *w0++;
          a = __builtin_pulp_sdotsp4(*xa++, v0, a);
          b = __builtin_pulp_sdotsp4(*xb++, v0, b);
        }
      }
      outA[c] = C42_RQ(a, c);
      if (has_px_b)
        outB[c] = C42_RQ(b, c);
    }
  }
  pi_cl_team_barrier();
}
#else /* -- QW 2026-10-22: step-5 optimisations of DualConv2In.c, without the fusion */

/* Gather one receptive field into `patch` (rows contiguous, zeros in the padding). Out of line: only pixels whose
 * field touches the padding come here. Word path = DualConv2In.c's d2_gather (exp7 step 5d) for one separate input
 * (pixel stride = ch_in): per kernel row zero-fill the padded words and copy the valid run word by word. */
static void __attribute__((noinline)) c42_gather(const int8_t *pIn, int8_t *patch, int32_t iy0, int32_t ix0,
                                                 uint16_t dim_in_x, uint16_t dim_in_y, uint16_t ch_in,
                                                 uint16_t dim_kernel_x, uint16_t dim_kernel_y) {
  if (((ch_in & 3u) == 0u) && ((((uint32_t)pIn) & 3u) == 0u)) {
    const uint32_t tap_words = (uint32_t)ch_in >> 2; /* words per tap */
    int32_t kx_lo = -ix0, kx_hi = (int32_t)dim_in_x - ix0;
    if (kx_lo < 0)
      kx_lo = 0;
    if (kx_hi > (int32_t)dim_kernel_x)
      kx_hi = (int32_t)dim_kernel_x;
    if (kx_hi < kx_lo)
      kx_hi = kx_lo;
    int32_t *dst_word = (int32_t *)patch;
    for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
      const int32_t iy = iy0 + (int32_t)ky;
      if (iy < 0 || iy >= (int32_t)dim_in_y) {
        for (uint32_t k = 0; k < (uint32_t)dim_kernel_x * tap_words; k++)
          *dst_word++ = 0;
        continue;
      }
      for (uint32_t k = 0; k < (uint32_t)kx_lo * tap_words; k++)
        *dst_word++ = 0;
      const int32_t *src = (const int32_t *)(pIn + ((uint32_t)iy * dim_in_x + (uint32_t)(ix0 + kx_lo)) * ch_in);
      for (uint32_t k = 0; k < (uint32_t)(kx_hi - kx_lo) * tap_words; k++)
        *dst_word++ = *src++;
      for (uint32_t k = 0; k < (uint32_t)((int32_t)dim_kernel_x - kx_hi) * tap_words; k++)
        *dst_word++ = 0;
    }
    return;
  }
  int8_t *dst = patch;
  for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
    const int32_t iy = iy0 + (int32_t)ky;
    for (uint32_t kx = 0; kx < dim_kernel_x; kx++) {
      const int32_t ix = ix0 + (int32_t)kx;
      if (iy < 0 || iy >= (int32_t)dim_in_y || ix < 0 || ix >= (int32_t)dim_in_x) {
        for (uint32_t ci = 0; ci < ch_in; ci++)
          *dst++ = 0;
      } else {
        const int8_t *src = pIn + ((uint32_t)iy * dim_in_x + (uint32_t)ix) * ch_in;
        for (uint32_t ci = 0; ci < ch_in; ci++)
          *dst++ = src[ci];
      }
    }
  }
}

void PULP_Conv4x2_i8_i8_i8_HWC(const int8_t *pIn, int8_t *pScratch, int8_t *pOut, const int8_t *pWeight,
                               const int32_t *pMul, const int32_t *pAdd, uint32_t num_cores, uint16_t out_shift,
                               uint16_t dim_in_x, uint16_t dim_in_y, uint16_t ch_in, uint16_t dim_out_x,
                               uint16_t dim_out_y, uint16_t ch_out, uint16_t dim_kernel_x, uint16_t dim_kernel_y,
                               uint16_t pad_y_top, uint16_t pad_y_bottom, uint16_t pad_x_left,
                               uint16_t pad_x_right, uint16_t stride_x, uint16_t stride_y) {
  const uint32_t core_id = pi_core_id();
  const uint32_t K = (uint32_t)ch_in * dim_kernel_x * dim_kernel_y;   // block 1: 8*16*1 = 128
  const uint32_t row_len = (uint32_t)dim_kernel_x * ch_in;           // bytes of one kernel row, block 1: 128
  const uint32_t row_words = row_len >> 2;
  const uint32_t in_place = ((ch_in & 3u) == 0u) && ((((uint32_t)pIn) & 3u) == 0u);
  const uint32_t in_row_stride = (uint32_t)dim_in_x * ch_in;
  int8_t *patch = pScratch + core_id * 2u * K; /* pixel A | pixel B */

  /* ---------------- split: PIXELS or OUTPUT CHANNELS, the rule of DualConv2In.c (exp7 step 5c) ---------- */
  const uint32_t n_out_px = (uint32_t)dim_out_y * dim_out_x;           // block 1: 14*88 = 1232
  const uint32_t n_ch_blocks = ch_out >> 2;
  const uint32_t px_chunk = (n_out_px + num_cores - 1u) / num_cores;   // block 1: 154
  const uint32_t px_busy_cores = px_chunk ? (n_out_px + px_chunk - 1u) / px_chunk : 0u;
  const uint32_t ch_blocks_per_core = (n_ch_blocks + num_cores - 1u) / num_cores;
  const uint32_t ch_busy_cores = ch_blocks_per_core ? (n_ch_blocks + ch_blocks_per_core - 1u) / ch_blocks_per_core : 0u;
#if defined(CONV4X2_SPLIT) && CONV4X2_SPLIT == 1 /* measurement override: 1 = pixels, 2 = channels */
  const uint32_t split_by_channel = 0u;
  (void)px_busy_cores;
  (void)ch_busy_cores;
#elif defined(CONV4X2_SPLIT) && CONV4X2_SPLIT == 2
  const uint32_t split_by_channel = ((ch_out & 3u) == 0u);
  (void)px_busy_cores;
  (void)ch_busy_cores;
#else
  const uint32_t split_by_channel =
      ((ch_out & 3u) == 0u) && (px_busy_cores < num_cores) && (ch_busy_cores > px_busy_cores);
#endif
  uint32_t px_begin, px_end, co_begin, co_end;
  if (split_by_channel) {
    px_begin = 0;
    px_end = n_out_px;
    co_begin = 4u * ch_blocks_per_core * core_id;
    if (co_begin > ch_out)
      co_begin = ch_out;
    co_end = co_begin + 4u * ch_blocks_per_core;
    if (co_end > ch_out)
      co_end = ch_out;
    if (co_begin == co_end)
      px_end = 0; /* no channels for this core */
  } else {
    co_begin = 0;
    co_end = ch_out;
    px_begin = px_chunk * core_id;
    if (px_begin > n_out_px)
      px_begin = n_out_px;
    px_end = px_begin + px_chunk;
    if (px_end > n_out_px)
      px_end = n_out_px;
  }
  uint32_t oy = px_begin / dim_out_x;
  uint32_t ox = px_begin - oy * dim_out_x;

  for (uint32_t px = px_begin; px < px_end; px += 2u) {
    const uint32_t has_px_b = (px + 1u < px_end); // the last pixel of an odd range runs alone (B duplicates A)
    const int8_t *field[2]; /* [0] = pixel A, [1] = pixel B */
    uint32_t fieldRowStride[2];
    for (uint32_t pix = 0; pix < 2u; pix++) {
      if (pix == 1u && !has_px_b) {
        field[1] = field[0];
        fieldRowStride[1] = fieldRowStride[0];
        break;
      }
      const int32_t iy0 = (int32_t)(oy * stride_y) - (int32_t)pad_y_top;
      const int32_t ix0 = (int32_t)(ox * stride_x) - (int32_t)pad_x_left;
      if (in_place && iy0 >= 0 && iy0 + (int32_t)dim_kernel_y <= (int32_t)dim_in_y && ix0 >= 0 &&
          ix0 + (int32_t)dim_kernel_x <= (int32_t)dim_in_x) {
        field[pix] = pIn + ((uint32_t)iy0 * dim_in_x + (uint32_t)ix0) * ch_in;
        fieldRowStride[pix] = in_row_stride;
      } else {
        int8_t *patchPix = patch + pix * K;
        c42_gather(pIn, patchPix, iy0, ix0, dim_in_x, dim_in_y, ch_in, dim_kernel_x, dim_kernel_y);
        field[pix] = patchPix;
        fieldRowStride[pix] = row_len;
      }
      if (++ox == dim_out_x) {
        ox = 0;
        oy++;
      }
    }
    const int8_t *fieldA = field[0], *fieldB = field[1];
    const uint32_t rowStrideA = fieldRowStride[0], rowStrideB = fieldRowStride[1];
    int8_t *outA = pOut + px * ch_out;
    int8_t *outB = outA + ch_out;

    /* ---- 4 output channels x 2 pixels per iteration (pulp_nn_matmul_i8_i8_i8's 4x2 shape) ----
     * accumulators a<j> / b<j>: pixel A / B, output channel c + j; v<j> / w<j>: weight word / row of channel c + j */
    uint32_t c = co_begin;
    for (; c + 4u <= co_end; c += 4u) {
      int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
      const int8_t *w = pWeight + c * K;
      for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
        const c42_v4s *xa = (const c42_v4s *)(fieldA + ky * rowStrideA);
        const c42_v4s *xb = (const c42_v4s *)(fieldB + ky * rowStrideB);
        const c42_v4s *w0 = (const c42_v4s *)(w + ky * row_len);
        const c42_v4s *w1 = (const c42_v4s *)(w + K + ky * row_len);
        const c42_v4s *w2 = (const c42_v4s *)(w + 2u * K + ky * row_len);
        const c42_v4s *w3 = (const c42_v4s *)(w + 3u * K + ky * row_len);
        for (uint32_t k = 0; k < row_words; k++) {
          const c42_v4s va = *xa++;
          const c42_v4s vb = *xb++;
          const c42_v4s v0 = *w0++;
          const c42_v4s v1 = *w1++;
          const c42_v4s v2 = *w2++;
          const c42_v4s v3 = *w3++;
          a0 = __builtin_pulp_sdotsp4(va, v0, a0);
          b0 = __builtin_pulp_sdotsp4(vb, v0, b0);
          a1 = __builtin_pulp_sdotsp4(va, v1, a1);
          b1 = __builtin_pulp_sdotsp4(vb, v1, b1);
          a2 = __builtin_pulp_sdotsp4(va, v2, a2);
          b2 = __builtin_pulp_sdotsp4(vb, v2, b2);
          a3 = __builtin_pulp_sdotsp4(va, v3, a3);
          b3 = __builtin_pulp_sdotsp4(vb, v3, b3);
        }
      }
      outA[c] = C42_RQ(a0, c);
      outA[c + 1u] = C42_RQ(a1, c + 1u);
      outA[c + 2u] = C42_RQ(a2, c + 2u);
      outA[c + 3u] = C42_RQ(a3, c + 3u);
      if (has_px_b) {
        outB[c] = C42_RQ(b0, c);
        outB[c + 1u] = C42_RQ(b1, c + 1u);
        outB[c + 2u] = C42_RQ(b2, c + 2u);
        outB[c + 3u] = C42_RQ(b3, c + 3u);
      }
    }
    /* ---- channel tail (ch_out % 4, pixel split only), 1 x 2 -- empty for every SpeechNet block ---- */
    for (; c < co_end; c++) {
      int32_t a = 0, b = 0;
      const int8_t *w = pWeight + c * K;
      for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
        const c42_v4s *xa = (const c42_v4s *)(fieldA + ky * rowStrideA);
        const c42_v4s *xb = (const c42_v4s *)(fieldB + ky * rowStrideB);
        const c42_v4s *w0 = (const c42_v4s *)(w + ky * row_len);
        for (uint32_t k = 0; k < row_words; k++) {
          const c42_v4s v0 = *w0++;
          a = __builtin_pulp_sdotsp4(*xa++, v0, a);
          b = __builtin_pulp_sdotsp4(*xb++, v0, b);
        }
      }
      outA[c] = C42_RQ(a, c);
      if (has_px_b)
        outB[c] = C42_RQ(b, c);
    }
  }
  pi_cl_team_barrier();
}
#endif /* CONV4X2_STEP2A -- QW */
