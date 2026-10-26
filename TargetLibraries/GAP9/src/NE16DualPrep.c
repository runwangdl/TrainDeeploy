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

/* -- QW exp21c: NE16DUAL_PROBE_WALL=ON (default off) = exp11's NE16DUAL_PROBE plus the WALL time of every phase of the
 *    two prep kernels, measured on worker core 0 from kernel entry / barrier to barrier (so it includes the per-core
 *    imbalance and the barrier), and the W re-perturbation of NE16DualBias separated from its add perturbation (core-0
 *    work). Only worker counters are reset here; the master core's Runtime counter is untouched. NE16DualConv2in.c
 *    reads the master counter without resetting it under this switch, so the fixture Runtime stays valid. -- QW exp21c */
#if defined(NE16DUAL_PROBE_WALL) && !defined(NE16DUAL_PROBE)
#define NE16DUAL_PROBE
#endif
#ifdef NE16DUAL_PROBE_WALL
#include "perf_utils.h"
static inline void x21c_rec(const char *label, uint32_t v) { /* -- QW exp21c: one record, core 0 only; 2 stores (the
                                                              * zero-initialised table needs no struct clear) */
  if (perf_record_count < PERF_MAX_RECORDS) {
    perf_records[perf_record_count].label = label;
    perf_records[perf_record_count].stats.cycles = v;
    perf_record_count++;
  }
}
#endif

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

/* -- QW exp21c (opt H): hoist the prep to once per UPDATE STEP.
 *    Both prep nodes are pure functions of (W, add, wmul, bmul, mul, seeds, eps scale, input_offset). z is drawn from
 *    perturb_seed_base, which the harness sets once per update step (deeploymezotest.c: seed_base outside the
 *    accumulation loop), and W / add change only in the update graph. So the outputs are identical in the n_accum
 *    windows of an update step. Each node owns an L2 cache `uint32_t hc[X21C_HC_HDR + out_bytes/4]` (a static global
 *    emitted by DualPrep.py): header = {valid, seed_w, seed_b, tile_xor, eps bits, zo_weight_version, out_bytes}.
 *    Hit (all fields equal): copy the cached output and return. Miss: compute as before, then copy the output into the
 *    cache and commit the header (core 0, after a barrier). Everything the output depends on that can change at run
 *    time is in the key (weights via zo_weight_version, which the harness bumps after every zo_update / resume).
 *    -D NE16DUAL_NOOPT_H=ON compiles the cache out (recompute every window). -- QW exp21c */
#define X21C_HC_HDR 7u
static inline uint32_t x21c_f2u(float f) {
  uint32_t u;
  __builtin_memcpy(&u, &f, 4);
  return u;
}
static inline uint32_t x21c_hc_hit(const uint32_t *hc, uint32_t sw, uint32_t sb, uint32_t tx, uint32_t eb,
                                   uint32_t bytes) {
  return hc && hc[0] == 1u && hc[1] == sw && hc[2] == sb && hc[3] == tx && hc[4] == eb &&
         hc[5] == zo_weight_version && hc[6] == bytes;
}
static inline void x21c_hc_commit(uint32_t *hc, uint32_t sw, uint32_t sb, uint32_t tx, uint32_t eb, uint32_t bytes) {
  hc[1] = sw, hc[2] = sb, hc[3] = tx, hc[4] = eb, hc[5] = zo_weight_version, hc[6] = bytes;
  hc[0] = 1u; /* last */
}
static inline void x21c_copy(void *dst, const void *src, uint32_t bytes, uint32_t core_id, uint32_t num_cores) {
  uint32_t b, n;
  if (((((uintptr_t)dst) | ((uintptr_t)src) | bytes) & 3u) == 0u) {
    dc_chunk(bytes >> 2, num_cores, core_id, &b, &n);
    uint32_t *d = (uint32_t *)dst + b;
    const uint32_t *s = (const uint32_t *)src + b;
    for (uint32_t i = 0; i < n; i++)
      d[i] = s[i];
  } else {
    dc_chunk(bytes, num_cores, core_id, &b, &n);
    for (uint32_t i = 0; i < n; i++)
      ((uint8_t *)dst)[b + i] = ((const uint8_t *)src)[b + i];
  }
}

/* QW (dual_pass_fusion exp7, step 7): `sign_sel` selects what is encoded -- 0: the stacked [W+; W-] (2*ch_out rows,
 * block 0, exp6, unchanged); 1: W+ only; 2: W- only (ch_out rows). Blocks 1-4 have x+ != x-, so the two passes are two
 * NE16 jobs over two column buffers, each with its own weight. The raw weight is CHW [co][ci][ky][kx] and the im2col
 * column is gathered in (ci, ky, kx) order, so perturbing in CHW MEMORY order is the reference order for any Ci (no
 * transpose involved, unlike the cluster kernel). For sign_sel != 0 the faster exp7 helper is used with
 * (ch_in = 1, kh = 1, kw = K), i.e. the identity mapping in runs of K -- bit-identical to dc_perturb_i8. -- QW */
void PULP_NE16DualPerturbEncode(const int8_t *pW, const int32_t *pWMul, uint8_t *pWEnc, int8_t *pScratch,
                                uint32_t seed_w_base, uint32_t tile_seed_xor, uint32_t log2D_w,
                                float eps_scale, uint32_t num_cores, uint32_t ch_out, uint32_t K, uint32_t sign_sel,
                                uint32_t hwc, uint32_t ch_in, uint32_t kh, uint32_t kw, uint32_t *pHCache) {
  const uint32_t core_id = pi_core_id();
  const uint32_t wsize = ch_out * K;
  int8_t *wp = pScratch;         /* rows [0, ch_out)        */
  int8_t *wm = pScratch + wsize; /* rows [ch_out, 2*ch_out) */
#ifndef NE16DUAL_NOOPT_H /* -- QW exp21c (opt H): cached output of this update step? */
  const uint32_t x21c_bytes = ((sign_sel == 0u) ? 2u * ch_out : ch_out) * ((K + 15u) / 16u) * 16u;
  const uint32_t x21c_eb = x21c_f2u(eps_scale);
  if (x21c_hc_hit(pHCache, seed_w_base, 0u, tile_seed_xor, x21c_eb, x21c_bytes)) {
    x21c_copy(pWEnc, pHCache + X21C_HC_HDR, x21c_bytes, core_id, num_cores);
    return; /* the fork's join is the barrier */
  }
#else
  (void)pHCache;
#endif
#ifdef NE16DUAL_PROBE /* -- QW exp11 */
  uint32_t n16p_t;
  n16p_start(&n16p_t);
#endif
#ifdef NE16DUAL_PROBE_WALL /* -- QW exp21c: wall marks on core 0 */
  const uint32_t x21c_t0 = n16p_t;
  uint32_t x21c_t1 = 0, x21c_t2 = 0;
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
#ifdef NE16DUAL_PROBE_WALL /* -- QW exp21c */
  x21c_t1 = n16p_t;
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
#ifdef NE16DUAL_PROBE_WALL /* -- QW exp21c: phase walls (core 0, entry -> barrier 1 -> barrier 2) */
  x21c_t2 = pi_perf_read(PI_PERF_CYCLES);
  if (core_id == 0) {
    x21c_rec("X21C_wenc_perturb_wall", x21c_t1 - x21c_t0);
    x21c_rec("X21C_wenc_encode_wall", x21c_t2 - x21c_t1);
  }
#endif
#ifndef NE16DUAL_NOOPT_H /* -- QW exp21c (opt H): fill the cache for the remaining windows of this update step */
  if (pHCache) {
    x21c_copy(pHCache + X21C_HC_HDR, pWEnc, x21c_bytes, core_id, num_cores);
    pi_cl_team_barrier();
    if (core_id == 0)
      x21c_hc_commit(pHCache, seed_w_base, 0u, tile_seed_xor, x21c_eb, x21c_bytes);
  }
#endif
// #ifdef NE16DUAL_PROBE   -- QW exp21c: original guard, replaced by the line below
#if defined(NE16DUAL_PROBE) && !defined(NE16DUAL_PROBE_WALL) /* -- QW exp21c: the wall probe skips exp11's 16 per-core
                                                               records (their L2 writes cost ~17 k cycles in the node) */
  if (core_id == 0)
    n16p_records(0, num_cores);
#endif
}

void PULP_NE16DualBias(const int8_t *pW, const int32_t *pWMul, const int32_t *pAdd, const int32_t *pBMul,
                       const int32_t *pMul, int32_t *pAddOut, int8_t *pScratch, uint32_t seed_w_base,
                       uint32_t seed_b_base, uint32_t tile_seed_xor, uint32_t log2D_w, uint32_t log2D_b,
                       float eps_scale, uint32_t num_cores, uint32_t ch_out, uint32_t K, int32_t input_offset,
                       uint32_t sign_sel, uint32_t *pHCache) {
  const uint32_t core_id = pi_core_id();
  const uint32_t wsize = ch_out * K;
  int8_t *wp = pScratch;
  int8_t *wm = pScratch + wsize;
#ifndef NE16DUAL_NOOPT_H /* -- QW exp21c (opt H): cached add2 of this update step? (input_offset is a per-node constant) */
  const uint32_t x21c_bytes = ((sign_sel == 0u) ? 2u * ch_out : ch_out) * 4u;
  const uint32_t x21c_eb = x21c_f2u(eps_scale);
  if (x21c_hc_hit(pHCache, seed_w_base, seed_b_base, tile_seed_xor, x21c_eb, x21c_bytes)) {
    x21c_copy(pAddOut, pHCache + X21C_HC_HDR, x21c_bytes, core_id, num_cores);
    return;
  }
#else
  (void)pHCache;
#endif
  const uint32_t off_add = (2u * wsize + 3u) & ~3u;
  int32_t *addp = (int32_t *)(pScratch + off_add);
  int32_t *addm = addp + ch_out;
#ifdef NE16DUAL_PROBE /* -- QW exp11 */
  uint32_t n16p_t;
  n16p_start(&n16p_t);
#endif
#ifdef NE16DUAL_PROBE_WALL /* -- QW exp21c */
  const uint32_t x21c_t0 = n16p_t;
  uint32_t x21c_tw = 0, x21c_t1 = 0, x21c_t2 = 0;
#endif

  /* -- QW exp21c (opt B): with input_offset == 0 (stacked input behind a Relu, ZODualPassFusion._nonnegative: network
   *    blocks 1-4) the row-sum correction below is `add - 0 * sum * mul` == add, so the W re-perturbation that only
   *    feeds the row sums is dead work (~93 % of the wenc perturbation, exp21c E_breakdown). Skip it and the row sums.
   *    Bit-identical: the dropped term is exactly 0 in int32. -D NE16DUAL_NOOPT_B=ON restores the original. -- QW exp21c */
#ifndef NE16DUAL_NOOPT_B
  const uint32_t skip_w = (input_offset == 0);
#else
  const uint32_t skip_w = 0u;
#endif
  uint32_t chunk_begin, chunk_len;
  dc_chunk(wsize, num_cores, core_id, &chunk_begin, &chunk_len);
  // if (chunk_len) {   -- QW exp21c: original condition
  if (chunk_len && !skip_w) { /* -- QW exp21c (opt B) */
    /* exp7: the faster helper (identity map in runs of K, same values) unless sign_sel == 0 with K < 16 -- block 0
     * (K = 4) keeps exp6's code path; there the runs would be 4 elements long. */
    if (sign_sel == 0u && K < 16u)
      dc_perturb_i8(pW, wp, wm, pWMul, K, (seed_w_base + core_id) ^ tile_seed_xor, chunk_begin, chunk_len, log2D_w,
                    eps_scale);
    else
      dc_perturb_i8_chw(pW, wp, wm, pWMul, 1u, 1u, K, (seed_w_base + core_id) ^ tile_seed_xor, chunk_begin, chunk_len,
                        log2D_w, eps_scale);
  }
#ifdef NE16DUAL_PROBE_WALL /* -- QW exp21c: core-0 work of the W re-perturbation alone */
  x21c_tw = pi_perf_read(PI_PERF_CYCLES);
#endif
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
#ifdef NE16DUAL_PROBE_WALL /* -- QW exp21c */
  x21c_t1 = n16p_t;
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
  if (skip_w) { /* -- QW exp21c (opt B): input_offset == 0 -> add2 = add+/add- unchanged */
    for (uint32_t out_row = row_begin; out_row < row_end; out_row++) {
      const uint32_t stacked_row = out_row + row_base;
      pAddOut[out_row] = (stacked_row < ch_out) ? addp[stacked_row] : addm[stacked_row - ch_out];
    }
  } else
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
#ifdef NE16DUAL_PROBE_WALL /* -- QW exp21c: W re-perturb (core-0 work), phase-1 wall, row-sum phase wall */
  x21c_t2 = pi_perf_read(PI_PERF_CYCLES);
  if (core_id == 0) {
    x21c_rec("X21C_bias_wrepert_work", x21c_tw - x21c_t0);
    x21c_rec("X21C_bias_perturb_wall", x21c_t1 - x21c_t0);
    x21c_rec("X21C_bias_rowsum_wall", x21c_t2 - x21c_t1);
  }
#endif
#ifndef NE16DUAL_NOOPT_H /* -- QW exp21c (opt H) */
  if (pHCache) {
    x21c_copy(pHCache + X21C_HC_HDR, pAddOut, x21c_bytes, core_id, num_cores);
    pi_cl_team_barrier();
    if (core_id == 0)
      x21c_hc_commit(pHCache, seed_w_base, seed_b_base, tile_seed_xor, x21c_eb, x21c_bytes);
  }
#endif
// #ifdef NE16DUAL_PROBE   -- QW exp21c: original guard, replaced by the line below
#if defined(NE16DUAL_PROBE) && !defined(NE16DUAL_PROBE_WALL) /* -- QW exp21c: the wall probe skips exp11's 16 per-core
                                                               records (their L2 writes cost ~17 k cycles in the node) */
  if (core_id == 0)
    n16p_records(1, num_cores);
#endif
}
