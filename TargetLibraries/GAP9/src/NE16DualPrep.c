/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp6, task 2): the cluster-side preparation for the FUSED dual-pass conv on NE16.
 *
 * On NE16 the two eps-passes fuse by stacking the output channels: one 1x1 job with Cout' = 2*Cout,
 * rows [0, Cout) = W+ and rows [Cout, 2*Cout) = W-, over the SAME column buffer (block 0's input is
 * pass-independent). NE16 retires TP_OUT = 32 output channels per pass whether 8 or 16 are used, so
 * the second pass rides along in lanes that were idle. Two small cluster kernels feed that job:
 *
 *   PULP_NE16DualPerturbEncode : W (int8, NCHW [cout][K]) + wmul  ->  weight_enc (uint8, [2*cout][cinMajor][16])
 *       phase 1  every core perturbs ITS chunk of W for both signs (DualPerturb.h == exp5's phase 1)
 *                into a scratch [2*cout][K] int8 matrix (W+ rows, then W- rows); barrier;
 *       phase 2  every core bit-serial-encodes ITS share of the 2*cout rows (NE16WeightEncode_i8_u8,
 *                taps = 1, cin = K: the folded 1x1 weight of the im2col route).
 *
 *   PULP_NE16DualBias : W, wmul, add, bmul, mul  ->  add2 (int32, [2*cout])
 *       re-derives the same W+/W- (deterministic: same seeds, same chunking) and add+/add-, then
 *       applies the signed-input correction of exp16c per row:
 *           add2[c]        = add+[c] - off * mul[c] * sum_k W+[c][k]
 *           add2[cout + c] = add-[c] - off * mul[c] * sum_k W-[c][k]
 *       (NE16 reads the +off column buffer; the induced term is per output row.)
 *
 * Two nodes instead of one because Deeploy's tiler assumes single-output nodes (exp5 Findings §4);
 * the duplicated perturbation of 32 weights costs a few hundred cycles. Both run once per step,
 * untiled (tensors are tens of bytes). -- QW
 */

#include "DeeployGAP9Math.h"
#include "kernel/DualPerturb.h"
#include "pmsis.h"

#ifdef NE16DUAL_PROBE /* -- QW exp11: per-core work cycles of the two phases of each prep kernel, as perf records */
#include "perf_utils.h"
static uint32_t n16p_work[2][8];
static const char *const n16p_label[2][2][8] = {
    {{"N16PROBE_wenc_perturb_core0", "N16PROBE_wenc_perturb_core1", "N16PROBE_wenc_perturb_core2",
      "N16PROBE_wenc_perturb_core3", "N16PROBE_wenc_perturb_core4", "N16PROBE_wenc_perturb_core5",
      "N16PROBE_wenc_perturb_core6", "N16PROBE_wenc_perturb_core7"},
     {"N16PROBE_wenc_encode_core0", "N16PROBE_wenc_encode_core1", "N16PROBE_wenc_encode_core2",
      "N16PROBE_wenc_encode_core3", "N16PROBE_wenc_encode_core4", "N16PROBE_wenc_encode_core5",
      "N16PROBE_wenc_encode_core6", "N16PROBE_wenc_encode_core7"}},
    {{"N16PROBE_bias_perturb_core0", "N16PROBE_bias_perturb_core1", "N16PROBE_bias_perturb_core2",
      "N16PROBE_bias_perturb_core3", "N16PROBE_bias_perturb_core4", "N16PROBE_bias_perturb_core5",
      "N16PROBE_bias_perturb_core6", "N16PROBE_bias_perturb_core7"},
     {"N16PROBE_bias_rowsum_core0", "N16PROBE_bias_rowsum_core1", "N16PROBE_bias_rowsum_core2",
      "N16PROBE_bias_rowsum_core3", "N16PROBE_bias_rowsum_core4", "N16PROBE_bias_rowsum_core5",
      "N16PROBE_bias_rowsum_core6", "N16PROBE_bias_rowsum_core7"}}};
static inline void n16p_start(uint32_t *t) {
  pi_perf_conf(1 << PI_PERF_CYCLES);
  pi_perf_reset();
  pi_perf_start();
  *t = pi_perf_read(PI_PERF_CYCLES);
}
static inline void n16p_mark(uint32_t ph, uint32_t core_id, uint32_t *t) { /* work since *t, before the barrier */
  const uint32_t now = pi_perf_read(PI_PERF_CYCLES);
  if (core_id < 8u)
    n16p_work[ph][core_id] = now - *t;
}
static inline void n16p_records(uint32_t kernel, uint32_t num_cores) { /* core 0, after the last barrier */
  for (uint32_t ph = 0; ph < 2u; ph++)
    for (uint32_t c = 0; c < num_cores && c < 8u; c++)
      if (perf_record_count < PERF_MAX_RECORDS) {
        perf_records[perf_record_count].label = n16p_label[kernel][ph][c];
        perf_records[perf_record_count].stats = (perf_stats_t){0};
        perf_records[perf_record_count].stats.cycles = n16p_work[ph][c];
        perf_record_count++;
      }
}
#endif

/* QW (dual_pass_fusion exp7, step 7): `sign_sel` selects what is encoded -- 0: the stacked [W+; W-] (2*ch_out rows,
 * block 0, exp6, unchanged); 1: W+ only; 2: W- only (ch_out rows). Blocks 1-4 have x+ != x-, so the two passes are two
 * NE16 jobs over two column buffers, each with its own weight. The raw weight is CHW [co][ci][ky][kx] and the im2col
 * column is gathered in (ci, ky, kx) order, so perturbing in CHW MEMORY order is the reference order for any Ci (no
 * transpose involved, unlike the cluster kernel). For sign_sel != 0 the faster exp7 helper is used with
 * (ch_in = 1, kh = 1, kw = K), i.e. the identity mapping in runs of K -- bit-identical to dc_perturb_i8. -- QW */
void PULP_NE16DualPerturbEncode(const int8_t *pW, const int32_t *pWMul, uint8_t *pWEnc, int8_t *pScratch,
                                uint32_t seed_w_base, uint32_t tile_seed_xor, uint32_t log2D_w,
                                float eps_scale, uint32_t num_cores, uint32_t ch_out, uint32_t K, uint32_t sign_sel,
                                uint32_t hwc, uint32_t ch_in, uint32_t kh, uint32_t kw) {
  const uint32_t core_id = pi_core_id();
  const uint32_t wsize = ch_out * K;
  int8_t *wp = pScratch;         /* rows [0, ch_out)        */
  int8_t *wm = pScratch + wsize; /* rows [ch_out, 2*ch_out) */
#ifdef NE16DUAL_PROBE /* -- QW exp11 */
  uint32_t n16p_t;
  n16p_start(&n16p_t);
#endif

  uint32_t chunk_begin, chunk_len;
  dc_chunk(wsize, num_cores, core_id, &chunk_begin, &chunk_len);
  if (chunk_len) {
    /* QW (exp7 step 7, pipelined NE16 conv): hwc = 1 writes W+/W- rows in (ky, kx, ci) order -- the column order of
     * the word-wise HWC column buffer (NE16DualConv2in.c) -- still consuming the noise in CHW order. -- QW */
    if (hwc)
      dc_perturb_i8_chw(pW, wp, wm, pWMul, ch_in, kh, kw, (seed_w_base + core_id) ^ tile_seed_xor, chunk_begin,
                        chunk_len, log2D_w, eps_scale);
    else if (sign_sel == 0u)
      dc_perturb_i8(pW, wp, wm, pWMul, K, (seed_w_base + core_id) ^ tile_seed_xor, chunk_begin, chunk_len, log2D_w,
                    eps_scale);
    else
      dc_perturb_i8_chw(pW, wp, wm, pWMul, 1u, 1u, K, (seed_w_base + core_id) ^ tile_seed_xor, chunk_begin, chunk_len,
                        log2D_w, eps_scale);
  }
#ifdef NE16DUAL_PROBE /* -- QW exp11: phase 1 = perturbation */
  n16p_mark(0, core_id, &n16p_t);
#endif
  pi_cl_team_barrier();
#ifdef NE16DUAL_PROBE
  n16p_t = pi_perf_read(PI_PERF_CYCLES);
#endif

  /* encode the stacked [2*ch_out][K] matrix (or one half of it) as a 1x1 weight with K input channels */
  const uint32_t rows = (sign_sel == 0u) ? 2u * ch_out : ch_out;
  const int8_t *src = (sign_sel == 2u) ? wm : pScratch;
  const uint32_t row_chunk = (rows + num_cores - 1u) / num_cores;
  uint32_t row_begin = row_chunk * core_id;
  if (row_begin > rows)
    row_begin = rows;
  uint32_t row_end = row_begin + row_chunk;
  if (row_end > rows)
    row_end = rows;
  if (row_end > row_begin)
    NE16WeightEncode_i8_u8(src, pWEnc, rows, K, 1u, row_begin, row_end - row_begin);
#ifdef NE16DUAL_PROBE /* -- QW exp11: phase 2 = bit-plane encode */
  n16p_mark(1, core_id, &n16p_t);
#endif
  pi_cl_team_barrier();
#ifdef NE16DUAL_PROBE
  if (core_id == 0)
    n16p_records(0, num_cores);
#endif
}

void PULP_NE16DualBias(const int8_t *pW, const int32_t *pWMul, const int32_t *pAdd, const int32_t *pBMul,
                       const int32_t *pMul, int32_t *pAddOut, int8_t *pScratch, uint32_t seed_w_base,
                       uint32_t seed_b_base, uint32_t tile_seed_xor, uint32_t log2D_w, uint32_t log2D_b,
                       float eps_scale, uint32_t num_cores, uint32_t ch_out, uint32_t K, int32_t input_offset,
                       uint32_t sign_sel) {
  const uint32_t core_id = pi_core_id();
  const uint32_t wsize = ch_out * K;
  int8_t *wp = pScratch;
  int8_t *wm = pScratch + wsize;
  const uint32_t off_add = (2u * wsize + 3u) & ~3u;
  int32_t *addp = (int32_t *)(pScratch + off_add);
  int32_t *addm = addp + ch_out;
#ifdef NE16DUAL_PROBE /* -- QW exp11 */
  uint32_t n16p_t;
  n16p_start(&n16p_t);
#endif

  uint32_t chunk_begin, chunk_len;
  dc_chunk(wsize, num_cores, core_id, &chunk_begin, &chunk_len);
  if (chunk_len) {
    /* exp7: the faster helper (identity map in runs of K, same values) unless sign_sel == 0 with K < 16 -- block 0
     * (K = 4) keeps exp6's code path; there the runs would be 4 elements long. */
    if (sign_sel == 0u && K < 16u)
      dc_perturb_i8(pW, wp, wm, pWMul, K, (seed_w_base + core_id) ^ tile_seed_xor, chunk_begin, chunk_len, log2D_w,
                    eps_scale);
    else
      dc_perturb_i8_chw(pW, wp, wm, pWMul, 1u, 1u, K, (seed_w_base + core_id) ^ tile_seed_xor, chunk_begin, chunk_len,
                        log2D_w, eps_scale);
  }
  dc_chunk(ch_out, num_cores, core_id, &chunk_begin, &chunk_len);
  if (chunk_len)
    dc_perturb_i32(pAdd, addp, addm, pBMul, 1u, (seed_b_base + core_id) ^ tile_seed_xor, chunk_begin, chunk_len,
                   log2D_b, eps_scale);
#ifdef NE16DUAL_PROBE /* -- QW exp11: phase 1 = re-perturbation of W+/W- and add+/add- */
  n16p_mark(0, core_id, &n16p_t);
#endif
  pi_cl_team_barrier();
#ifdef NE16DUAL_PROBE
  n16p_t = pi_perf_read(PI_PERF_CYCLES);
#endif

  /* exp7: sign_sel 0 = stacked [add+'; add-'] (2*ch_out rows); 1 / 2 = add+' / add-' only (ch_out rows) */
  const uint32_t rows = (sign_sel == 0u) ? 2u * ch_out : ch_out;
  const uint32_t row_base = (sign_sel == 2u) ? ch_out : 0u; /* row of the stacked matrix that output row 0 is */
  const uint32_t row_chunk = (rows + num_cores - 1u) / num_cores;
  uint32_t row_begin = row_chunk * core_id;
  if (row_begin > rows)
    row_begin = rows;
  uint32_t row_end = row_begin + row_chunk;
  if (row_end > rows)
    row_end = rows;
  for (uint32_t out_row = row_begin; out_row < row_end; out_row++) {
    const uint32_t stacked_row = out_row + row_base;
    const uint32_t co = (stacked_row < ch_out) ? stacked_row : stacked_row - ch_out;
    const int8_t *w_row = pScratch + stacked_row * K; /* row of the stacked [W+; W-] matrix */
    int32_t w_row_sum = 0;
    for (uint32_t k = 0; k < K; k++)
      w_row_sum += (int32_t)w_row[k];
    const int32_t add_pass = (stacked_row < ch_out) ? addp[co] : addm[co];
    pAddOut[out_row] = add_pass - input_offset * w_row_sum * pMul[co];
  }
#ifdef NE16DUAL_PROBE /* -- QW exp11: phase 2 = row sums + signed-input correction */
  n16p_mark(1, core_id, &n16p_t);
#endif
  pi_cl_team_barrier();
#ifdef NE16DUAL_PROBE
  if (core_id == 0)
    n16p_records(1, num_cores);
#endif
}
