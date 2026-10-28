/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp11, 2026-10-23): SINGLE-PASS quantized QZO conv with the perturbation IN THE KERNEL -- the
 * unfused reference that carries every optimisation of the fused DualConv2In.c except the dual-pass fusion itself:
 *
 *   phase 1  every core perturbs ITS chunk of the UNPERTURBED raw-CHW weight for ONE sign (perturbation_sign of this
 *            pass), in CHW order (noise bit i on CHW element i, as the RQSPerturbRademacher node), writing W at its
 *            HWC position (dc_perturb1_i8_chw: per-channel up/dn, bit select, branch-free runs = DualConv2In's step 5b);
 *            and its chunk of the requant add (dc_perturb1_i32). No weight-transpose node, no perturbation nodes.
 *   phase 2  Conv4x2.c's step-5 loop (= DualConv2In's phase 2 for one pass): flat pixel range or output-channel range
 *            (exp7 step 5c rule), 4 ch x 2 px, interior fields in place, border fields gathered word-wise (step 5d).
 *
 * What it does NOT have (the fusion): both signs from one draw, one pixel loop for both passes. Run once per pass.
 * Bit-identical to RQSPerturbRademacher (weight, bias) + Conv + RequantShift of that pass (fixture QZO_Conv_B<i>).
 *
 * Layouts: x HWC [dim_in_y][dim_in_x][ch_in]; weight CHW [ch_out][ch_in][kh][kw] (unperturbed); out HWC.
 * Scratch (PerturbConv4x2Template.computeTransientBuffersSize -- keep in sync):
 *   [0, wsize) W (HWC)   off_add (4-aligned) add [ch_out] i32   off_patch: two K-byte patches per core.
 * -D DUALCONV2IN_PROBE=ON: per-core work cycles of phase 1 / phase 2 as perf records (PC_PROBE_*). -- QW
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"
#include <math.h>

#include "kernel/DualPerturb.h"
#ifdef DUALCONV2IN_PROBE
#include "perf_utils.h"
static uint32_t pc_probe_work[2][8];
static const char *const pc_probe_label[2][8] = {
    {"PC_PROBE_p1_core0", "PC_PROBE_p1_core1", "PC_PROBE_p1_core2", "PC_PROBE_p1_core3", "PC_PROBE_p1_core4",
     "PC_PROBE_p1_core5", "PC_PROBE_p1_core6", "PC_PROBE_p1_core7"},
    {"PC_PROBE_p2_core0", "PC_PROBE_p2_core1", "PC_PROBE_p2_core2", "PC_PROBE_p2_core3", "PC_PROBE_p2_core4",
     "PC_PROBE_p2_core5", "PC_PROBE_p2_core6", "PC_PROBE_p2_core7"}};
#endif

typedef signed char pc_v4s __attribute__((vector_size(4)));

/* requant exactly as pulp_nn_bn_quant_i8 (32-bit variant), Conv4x2.c and DualConv2In.c */
#define PC_RQ(acc, c)                                                                                         \
  ((int8_t)__builtin_pulp_clip_r((int32_t)((uint32_t)(acc) * (uint32_t)pMul[c] + (uint32_t)addq[c]) >> out_shift, \
                                 127))

/* Gather one receptive field into `patch` (rows contiguous, zeros in the padding). Out of line: only pixels whose
 * field touches the padding come here. Word path = DualConv2In.c's d2_gather (exp7 step 5d) for one separate input
 * (pixel stride = ch_in): per kernel row zero-fill the padded words and copy the valid run word by word. */
static void __attribute__((noinline)) pc_gather(const int8_t *pIn, int8_t *patch, int32_t iy0, int32_t ix0,
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

void PULP_PerturbConv4x2_i8_i8_i8_HWC(const int8_t *pIn, const int8_t *pWeight, const int32_t *pMul,
                                       const int32_t *pAdd, const int32_t *pWMul, const int32_t *pBMul, int8_t *pOut,
                                       int8_t *pScratch, uint32_t sign, uint32_t seed_w_base, uint32_t seed_b_base,
                                       uint32_t tile_seed_xor, uint32_t log2D_w, uint32_t log2D_b, float32_t eps_scale,
                                       uint32_t num_cores, uint16_t out_shift, uint16_t dim_in_x, uint16_t dim_in_y,
                                       uint16_t ch_in, uint16_t dim_out_x, uint16_t dim_out_y, uint16_t ch_out,
                                       uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t pad_y_top,
                                       uint16_t pad_y_bottom, uint16_t pad_x_left, uint16_t pad_x_right,
                                       uint16_t stride_x, uint16_t stride_y) {
  const uint32_t core_id = pi_core_id();
  const uint32_t K = (uint32_t)ch_in * dim_kernel_x * dim_kernel_y;
  const uint32_t row_len = (uint32_t)dim_kernel_x * ch_in;           // bytes of one kernel row (HWC)
  const uint32_t row_words = row_len >> 2;
  const uint32_t wsize = (uint32_t)ch_out * K;
  const uint32_t off_add = (wsize + 3u) & ~3u;
  const uint32_t off_patch = off_add + 4u * ch_out;
  int8_t *wq = pScratch;
  int32_t *addq = (int32_t *)(pScratch + off_add);
  int8_t *patch = pScratch + off_patch + core_id * 2u * K; /* pixel A | pixel B */
  const uint32_t in_place = ((ch_in & 3u) == 0u) && ((((uint32_t)pIn) & 3u) == 0u);
  const uint32_t in_row_stride = (uint32_t)dim_in_x * ch_in;

#ifdef DUALCONV2IN_PROBE
  uint32_t probe_t0, probe_t1;
  pi_perf_conf(1 << PI_PERF_CYCLES);
  pi_perf_reset();
  pi_perf_start();
  probe_t0 = pi_perf_read(PI_PERF_CYCLES);
#endif
  /* ---------------- phase 1: perturb my CHW chunk for this pass's sign, write HWC; my chunk of add ----------- */
  uint32_t chunk_begin, chunk_len;
  dc_chunk(wsize, num_cores, core_id, &chunk_begin, &chunk_len);
  if (chunk_len)
    dc_perturb1_i8_chw(pWeight, wq, pWMul, ch_in, dim_kernel_y, dim_kernel_x, (seed_w_base + core_id) ^ tile_seed_xor,
                       chunk_begin, chunk_len, log2D_w, eps_scale, sign);
  dc_chunk((uint32_t)ch_out, num_cores, core_id, &chunk_begin, &chunk_len);
  if (chunk_len)
    dc_perturb1_i32(pAdd, addq, pBMul, 1u, (seed_b_base + core_id) ^ tile_seed_xor, chunk_begin, chunk_len, log2D_b,
                    eps_scale, sign);
#ifdef DUALCONV2IN_PROBE
  if (core_id < 8u)
    pc_probe_work[0][core_id] = pi_perf_read(PI_PERF_CYCLES) - probe_t0;
#endif
  pi_cl_team_barrier();
#ifdef DUALCONV2IN_PROBE
  probe_t1 = pi_perf_read(PI_PERF_CYCLES);
#endif

  /* ---------------- phase 2: Conv4x2.c step 5 on (x, W, add) ------------------------------------------- */
  /* ---------------- split: PIXELS or OUTPUT CHANNELS, the rule of DualConv2In.c (exp7 step 5c) ---------- */
  const uint32_t n_out_px = (uint32_t)dim_out_y * dim_out_x;           // block 1: 14*88 = 1232
  const uint32_t n_ch_blocks = ch_out >> 2;
  const uint32_t px_chunk = (n_out_px + num_cores - 1u) / num_cores;   // block 1: 154
  const uint32_t px_busy_cores = px_chunk ? (n_out_px + px_chunk - 1u) / px_chunk : 0u;
  const uint32_t ch_blocks_per_core = (n_ch_blocks + num_cores - 1u) / num_cores;
  const uint32_t ch_busy_cores = ch_blocks_per_core ? (n_ch_blocks + ch_blocks_per_core - 1u) / ch_blocks_per_core : 0u;
#if defined(PERTURBCONV_SPLIT) && PERTURBCONV_SPLIT == 1 /* measurement override: 1 = pixels, 2 = channels */
  const uint32_t split_by_channel = 0u;
  (void)px_busy_cores;
  (void)ch_busy_cores;
#elif defined(PERTURBCONV_SPLIT) && PERTURBCONV_SPLIT == 2
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
        pc_gather(pIn, patchPix, iy0, ix0, dim_in_x, dim_in_y, ch_in, dim_kernel_x, dim_kernel_y);
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
      const int8_t *w = wq + c * K;
      for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
        const pc_v4s *xa = (const pc_v4s *)(fieldA + ky * rowStrideA);
        const pc_v4s *xb = (const pc_v4s *)(fieldB + ky * rowStrideB);
        const pc_v4s *w0 = (const pc_v4s *)(w + ky * row_len);
        const pc_v4s *w1 = (const pc_v4s *)(w + K + ky * row_len);
        const pc_v4s *w2 = (const pc_v4s *)(w + 2u * K + ky * row_len);
        const pc_v4s *w3 = (const pc_v4s *)(w + 3u * K + ky * row_len);
        for (uint32_t k = 0; k < row_words; k++) {
          const pc_v4s va = *xa++;
          const pc_v4s vb = *xb++;
          const pc_v4s v0 = *w0++;
          const pc_v4s v1 = *w1++;
          const pc_v4s v2 = *w2++;
          const pc_v4s v3 = *w3++;
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
      outA[c] = PC_RQ(a0, c);
      outA[c + 1u] = PC_RQ(a1, c + 1u);
      outA[c + 2u] = PC_RQ(a2, c + 2u);
      outA[c + 3u] = PC_RQ(a3, c + 3u);
      if (has_px_b) {
        outB[c] = PC_RQ(b0, c);
        outB[c + 1u] = PC_RQ(b1, c + 1u);
        outB[c + 2u] = PC_RQ(b2, c + 2u);
        outB[c + 3u] = PC_RQ(b3, c + 3u);
      }
    }
    /* ---- channel tail (ch_out % 4, pixel split only), 1 x 2 -- empty for every SpeechNet block ---- */
    for (; c < co_end; c++) {
      int32_t a = 0, b = 0;
      const int8_t *w = wq + c * K;
      for (uint32_t ky = 0; ky < dim_kernel_y; ky++) {
        const pc_v4s *xa = (const pc_v4s *)(fieldA + ky * rowStrideA);
        const pc_v4s *xb = (const pc_v4s *)(fieldB + ky * rowStrideB);
        const pc_v4s *w0 = (const pc_v4s *)(w + ky * row_len);
        for (uint32_t k = 0; k < row_words; k++) {
          const pc_v4s v0 = *w0++;
          a = __builtin_pulp_sdotsp4(*xa++, v0, a);
          b = __builtin_pulp_sdotsp4(*xb++, v0, b);
        }
      }
      outA[c] = PC_RQ(a, c);
      if (has_px_b)
        outB[c] = PC_RQ(b, c);
    }
  }
#ifdef DUALCONV2IN_PROBE
  if (core_id < 8u)
    pc_probe_work[1][core_id] = pi_perf_read(PI_PERF_CYCLES) - probe_t1;
#endif
  pi_cl_team_barrier();
#ifdef DUALCONV2IN_PROBE
  if (core_id == 0)
    for (uint32_t ph = 0; ph < 2u; ph++)
      for (uint32_t c = 0; c < num_cores && c < 8u; c++)
        if (perf_record_count < PERF_MAX_RECORDS) {
          perf_records[perf_record_count].label = pc_probe_label[ph][c];
          perf_records[perf_record_count].stats = (perf_stats_t){0};
          perf_records[perf_record_count].stats.cycles = pc_probe_work[ph][c];
          perf_record_count++;
        }
#endif
}
