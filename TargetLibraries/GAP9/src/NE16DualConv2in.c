/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp7, step 7): BOTH ZO passes of a 1xK / Kx1 quantized conv (SpeechNet blocks 1-4) on NE16
 * in ONE node, with the GAP9 SDK's pipeline (Ker_MM_Conv1DSmallv2_NE16):
 *
 *   - the column buffer is built from the HWC input, WORD-WISE: in HWC the valid taps of one kernel row are ONE
 *     contiguous run of (kx_hi - kx_lo)*ch_in bytes, so per row: fill the padded words with the pad value 0x80808080,
 *     copy the run as words XOR 0x80808080 (= x + 128, NE16 reads unsigned). Column order (ky, kx, ci), so the weight
 *     is encoded in the same order (NE16DualPerturbEncode with ne16_hwc = 1). exp6's Im2ColU8.c built the column from
 *     NCHW byte by byte (block 1: 291 k cycles per pass on GAP9);
 *   - one NE16 job = 3 x w consecutive output pixels of one pass (w = job_w, from an L1 budget; a flat pixel range
 *     presented to NE16 as a 3 x w block, so its 3x3 output subtiles are full), a 1x1 conv over K = ch_in*kh*kw input
 *     channels; the remainder of a pass is at most one 3 x (r/3) and one 1 x (r%3) job. Pass + first;
 *   - DOUBLE BUFFERED (pipelined = 1): two column buffers; while NE16 computes job j from one, the 8 workers build the
 *     column of job j+1 into the other. Job j+1 is queued behind job j (NE16's 2-deep job queue), so NE16 starts it
 *     without waiting for the cores. Buffer reuse needs no job ids (GVSoC's running_id is broken): right after job j
 *     is dispatched, ne16_nnx_dispatch_wait() returns only when a queue slot is free, i.e. when job j-1 -- the last
 *     user of the buffer job j+1 will fill -- has finished. pipelined = 0 runs the same jobs strictly in sequence
 *     (build, dispatch, wait) to measure what the overlap is worth.
 *
 * Output: HWC [dim_out_y][dim_out_x][2*ch_out], A+ in channels [0, ch_out), A- in [ch_out, 2*ch_out) -- NE16 writes each
 * pass with an output pixel stride of 2*ch_out. Requant on NE16 (32-bit scale per channel, bias, shift), configured
 * exactly as Deeploy's pointwise template (conf0 is computed by the Deeploy template and passed in).
 * Inputs: x+ / x- HWC int8; pWEnc2 = [W+; W-] encoded, 2*ch_out rows; pAdd2 = [add+'; add-'] (signed-input corrected).
 * Runs on the cluster MASTER core (NE16 engine); pColBuf = 2 * 3 * job_w * K bytes of L1.
 * Built only on GAP9_w_NE16 (pulp-nnx present). -- QW
 */

#if __has_include("pulp_nnx_ne16.h")

#include "DeeployGAP9Math.h"
#include "ne16.h"
#include "ne16_pulp_bsp.h"
#include "ne16_task.h"
#include "pmsis.h"
#include "pulp_nnx_ne16.h"
#include "pulp_nnx_util.h"

#define N16D_PAD_WORD 0x80808080u
#ifdef NE16DUAL_PROBE /* -- QW exp11: master-core (= wall) cycles of the node: column builds, NE16 waits, the rest */
#include "perf_utils.h"
static inline uint32_t n16d_now(void) { return pi_perf_read(PI_PERF_CYCLES); }
static inline void n16d_rec(const char *label, uint32_t v) {
  if (perf_record_count < PERF_MAX_RECORDS) {
    perf_records[perf_record_count].label = label;
    perf_records[perf_record_count].stats = (perf_stats_t){0};
    perf_records[perf_record_count].stats.cycles = v;
    perf_record_count++;
  }
}
#define N16D_T0() uint32_t _n16d_t = n16d_now()
#define N16D_ACC(acc) do { const uint32_t _n = n16d_now(); (acc) += _n - _n16d_t; _n16d_t = _n; } while (0)
#define N16D_SKIP() (_n16d_t = n16d_now())
#endif

typedef struct {
  const int8_t *in;
  uint8_t *col;
  uint32_t px_begin, px_count; /* flat output-pixel range [px_begin, px_begin+px_count) of the job, may span rows */
  uint32_t in_stride; /* bytes between input pixels: ch_in, or 2*ch_in for the stacked [x+ | x-] input (exp8) */
  uint32_t xor_word;  /* RW 2026-10-15: 0x80808080 (signed input + 128) or 0 (input known >= 0, no offset) */
  uint16_t dim_in_y, dim_in_x, ch_in, dim_out_x, kh, kw, pad_top, pad_left, stride_y, stride_x;
} n16d_col_args_t;

/* worker body: the columns of a flat range of output pixels, split over the cores */
static void n16d_build_col(void *arg) {
  const n16d_col_args_t *a = (const n16d_col_args_t *)arg;
  const uint32_t core_id = pi_core_id(), num_cores = pi_cl_cluster_nb_cores();
  const uint32_t tap_words = (uint32_t)a->ch_in >> 2; /* words per tap */
  const uint32_t K = (uint32_t)a->kh * a->kw * a->ch_in;
  const uint32_t px_chunk = (a->px_count + num_cores - 1u) / num_cores;
  uint32_t px_lo = px_chunk * core_id, px_hi = px_lo + px_chunk; /* relative to the job's first pixel */
  if (px_lo > a->px_count)
    px_lo = a->px_count;
  if (px_hi > a->px_count)
    px_hi = a->px_count;
  uint32_t oy = (a->px_begin + px_lo) / a->dim_out_x;
  uint32_t ox = (a->px_begin + px_lo) - oy * a->dim_out_x;
  for (uint32_t px = px_lo; px < px_hi; px++) {
    uint32_t *dst = (uint32_t *)(a->col + px * K);
    const int32_t iy0 = (int32_t)(oy * a->stride_y) - (int32_t)a->pad_top;
    const int32_t ix0 = (int32_t)(ox * a->stride_x) - (int32_t)a->pad_left;
    int32_t kx_lo = -ix0, kx_hi = (int32_t)a->dim_in_x - ix0;
    if (kx_lo < 0)
      kx_lo = 0;
    if (kx_hi > (int32_t)a->kw)
      kx_hi = (int32_t)a->kw;
    if (kx_hi < kx_lo)
      kx_hi = kx_lo;
    for (uint32_t ky = 0; ky < a->kh; ky++) {
      const int32_t iy = iy0 + (int32_t)ky;
      if (iy < 0 || iy >= (int32_t)a->dim_in_y) {
        for (uint32_t k = 0; k < (uint32_t)a->kw * tap_words; k++)
          *dst++ = a->xor_word;
        continue;
      }
      for (uint32_t k = 0; k < (uint32_t)kx_lo * tap_words; k++)
        *dst++ = a->xor_word;
      if (a->in_stride == a->ch_in) { /* one contiguous run */
        const uint32_t *src =
            (const uint32_t *)(a->in + ((uint32_t)iy * a->dim_in_x + (uint32_t)(ix0 + kx_lo)) * a->ch_in);
        for (uint32_t k = 0; k < (uint32_t)(kx_hi - kx_lo) * tap_words; k++)
          *dst++ = *src++ ^ a->xor_word;
      } else { /* exp8: stacked input -- tap_words words per tap, taps in_stride apart */
        for (int32_t kx = kx_lo; kx < kx_hi; kx++) {
          const uint32_t *src =
              (const uint32_t *)(a->in + ((uint32_t)iy * a->dim_in_x + (uint32_t)(ix0 + kx)) * a->in_stride);
          for (uint32_t k = 0; k < tap_words; k++)
            *dst++ = src[k] ^ a->xor_word;
        }
      }
      for (uint32_t k = 0; k < (uint32_t)((int32_t)a->kw - kx_hi) * tap_words; k++)
        *dst++ = a->xor_word;
    }
    if (++ox == a->dim_out_x) {
      ox = 0;
      oy++;
    }
  }
}

/* one NE16 job: a 1x1 conv over the columns of job_rows x job_cols consecutive output pixels (K input channels),
 * presented to NE16 as a job_rows x job_cols block -- the field values are Deeploy's NE162DPWConvTemplate for such a
 * tile, with the output pixel stride 2*ch_out (A+ and A- interleaved per pixel). The output pixels of a tile are
 * contiguous in HWC, so a flat pixel range is a valid block with row stride job_cols * pixel stride. */
static void n16d_task(ne16_task_t *task, const uint8_t *col, const uint8_t *w_enc, int8_t *out, const int32_t *mul,
                      const int32_t *add, uint32_t job_rows, uint32_t job_cols, uint32_t K, uint32_t ch_out,
                      uint32_t out_px_stride, uint32_t conf0, int32_t weight_offset) {
  const uint32_t nKi = (K + 15u) / 16u, nKo = (ch_out + 31u) / 32u, nHo = (job_rows + 2u) / 3u,
                 nWo = (job_cols + 2u) / 3u;
  const uint32_t bKi = (K % 16u) ? (K % 16u) : 16u, bKo = (ch_out % 32u) ? (ch_out % 32u) : 32u;
  const uint32_t bHo = (job_rows % 3u) ? (job_rows % 3u) : 3u, bWo = (job_cols % 3u) ? (job_cols % 3u) : 3u;
  task->data.weights_addr = (uint32_t)w_enc;
  task->data.infeat_addr = (uint32_t)col;
  task->data.outfeat_addr = (uint32_t)out;
  task->data.scale_addr = (uint32_t)mul;
  task->data.scale_shift_addr = (uint32_t)NULL;
  task->data.scale_bias_addr = (uint32_t)add;
  task->data.cfg.input_stride = (ne16_stride_t){.d0 = K, .d1 = job_cols * K, .d2 = 0};
  task->data.cfg.output_stride =
      (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = out_px_stride, .d2 = job_cols * out_px_stride};
  task->data.cfg.weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16 * nKi, .d2 = 0};
  task->data.cfg.subtile.number.KoKi = nnx_concat_half(nKo, nKi);
  task->data.cfg.subtile.number.HoWo = nnx_concat_half(nHo, nWo);
  task->data.cfg.subtile.remainder.KoKi = nnx_concat_half(bKo, bKi);
  task->data.cfg.subtile.remainder.HoWo = nnx_concat_half(bHo, bWo);
  task->data.cfg.subtile.remainder.HiWi = nnx_concat_half(bHo, bWo);
  task->data.cfg.padding = 0;
  task->data.cfg.weight_offset_factor = weight_offset;
  task->data.cfg.filter_mask = 0;
  task->data.cfg.conf0 = conf0;
  task->weight_d0_stride = NE16_WEIGHT_D0_STRIDE_MODE8;
  task->qw = 8;
  task->subtile_output_channel = 32;
  task->kernel_shape = 1;
  task->depthwise = 0;
}

/* RW 2026-10-15: GAP9 silicon NE16 (not GVSoC) gets column 2 of the first 3x3 output subtile of a job wrong (+1 LSB on
 * ~0.4 % of block 1's outputs) when the job is wider than 3 and its width is not a multiple of 3. So every job is
 * either a multiple of 3 wide or narrower than 3: n_full jobs of 3 x job_cols_full (job_cols_full a multiple of 3,
 * DualConv2in.py), then the remainder r as 3 x 3k (k = r / 9), 1 x 3m (m = (r - 9k) / 3, i.e. 3 or 6), 1 x (r % 3).
 * (Was: 3 x (r/3) and 1 x (r%3).) Board, block-1 fixture: 152 -> 5 -> 0 wrong outputs of 39,424. */
static void n16d_rem_split(uint32_t r, uint32_t *a, uint32_t *b, uint32_t *c) {
  *a = (r / 9u) * 3u;          /* columns of the 3-row job */
  *b = ((r - 3u * *a) / 3u) * 3u; /* columns of the first 1-row job (multiple of 3) */
  *c = r - 3u * *a - *b;       /* columns of the last 1-row job (< 3) */
}

static uint32_t n16d_jobs_per_pass(uint32_t n_out_px, uint32_t job_cols_full) {
  const uint32_t n_full = n_out_px / (3u * job_cols_full), r = n_out_px - n_full * 3u * job_cols_full;
  uint32_t a, b, c;
  n16d_rem_split(r, &a, &b, &c);
  return n_full + (a ? 1u : 0u) + (b ? 1u : 0u) + (c ? 1u : 0u);
}

static void n16d_job_geometry(uint32_t j, uint32_t n_out_px, uint32_t job_cols_full, uint32_t *px_begin,
                              uint32_t *job_rows, uint32_t *job_cols) {
  const uint32_t n_full = n_out_px / (3u * job_cols_full), r = n_out_px - n_full * 3u * job_cols_full;
  uint32_t a, b, c, base = n_full * 3u * job_cols_full;
  n16d_rem_split(r, &a, &b, &c);
  if (j < n_full) {
    *px_begin = j * 3u * job_cols_full;
    *job_rows = 3u;
    *job_cols = job_cols_full;
    return;
  }
  j -= n_full;
  if (a) {
    if (j == 0u) {
      *px_begin = base;
      *job_rows = 3u;
      *job_cols = a;
      return;
    }
    j--;
    base += 3u * a;
  }
  if (b) {
    if (j == 0u) {
      *px_begin = base;
      *job_rows = 1u;
      *job_cols = b;
      return;
    }
    j--;
    base += b;
  }
  *px_begin = base;
  *job_rows = 1u;
  *job_cols = c;
}

void PULP_NE16DualConv2in(const int8_t *pInP, const int8_t *pInM, const uint8_t *pWEnc2, const int32_t *pMul,
                          const int32_t *pAdd2, int8_t *pOut, uint8_t *pColBuf, uint32_t num_cores, uint32_t conf0,
                          uint16_t dim_in_y, uint16_t dim_in_x, uint16_t ch_in, uint16_t dim_out_y, uint16_t dim_out_x,
                          uint16_t ch_out, uint16_t dim_kernel_y, uint16_t dim_kernel_x, uint16_t pad_top,
                          uint16_t pad_bottom, uint16_t pad_left, uint16_t pad_right, uint16_t stride_y,
                          uint16_t stride_x, int32_t weight_offset, uint32_t pipelined, uint32_t job_w,
                          uint32_t in_stride, int32_t input_offset) {
  (void)pad_bottom;
  (void)pad_right;
  const ne16_dev_t *dev = ne16_pulp_get_dev();
  const uint32_t K = (uint32_t)ch_in * dim_kernel_y * dim_kernel_x;
  const uint32_t wenc_pass_bytes = (uint32_t)ch_out * ((K + 15u) / 16u) * 16u; /* encoded bytes of one pass */
  const uint32_t out_px_stride = 2u * ch_out;
  /* job = 3 x job_w output pixels (NE16 computes 3x3 output subtiles); job_w is set by the template from an L1 budget
   * (two buffers of 3 * job_w * K bytes) */
  const uint32_t job_cols_full = job_w ? job_w : 1u;
  uint8_t *col[2] = {pColBuf, pColBuf + 3u * job_cols_full * K};
  const uint32_t n_out_px = (uint32_t)dim_out_y * dim_out_x;
  const uint32_t jobs_per_pass = n16d_jobs_per_pass(n_out_px, job_cols_full);
  const uint32_t n_jobs = 2u * jobs_per_pass;
  uint32_t px_begin, job_rows, job_cols;
  n16d_job_geometry(0, n_out_px, job_cols_full, &px_begin, &job_rows, &job_cols);
  n16d_col_args_t col_args = {.in = pInP, .col = col[0], .px_begin = px_begin, .px_count = job_rows * job_cols,
                              .in_stride = in_stride, .xor_word = input_offset ? N16D_PAD_WORD : 0u, .dim_in_y = dim_in_y, .dim_in_x = dim_in_x,
                              .ch_in = ch_in, .dim_out_x = dim_out_x, .kh = dim_kernel_y, .kw = dim_kernel_x,
                              .pad_top = pad_top, .pad_left = pad_left, .stride_y = stride_y, .stride_x = stride_x};
  ne16_task_t task;
#ifdef NE16DUAL_PROBE /* -- QW exp11 */
  pi_perf_conf(1 << PI_PERF_CYCLES);
  pi_perf_reset();
  pi_perf_start();
  const uint32_t n16d_t_begin = n16d_now();
  uint32_t n16d_col = 0, n16d_wait = 0;
  N16D_T0();
#endif

  /* columns of job 0 */
  pi_cl_team_fork(num_cores, n16d_build_col, &col_args);
#ifdef NE16DUAL_PROBE
  N16D_ACC(n16d_col);
#endif

  for (uint32_t j = 0; j < n_jobs; j++) {
    const uint32_t pass = j / jobs_per_pass;
    n16d_job_geometry(j - pass * jobs_per_pass, n_out_px, job_cols_full, &px_begin, &job_rows, &job_cols);
    n16d_task(&task, col[j & 1u], pWEnc2 + pass * wenc_pass_bytes, pOut + px_begin * out_px_stride + pass * ch_out,
              pMul, pAdd2 + pass * ch_out, job_rows, job_cols, K, ch_out, out_px_stride, conf0, weight_offset);
#ifdef NE16DUAL_PROBE
    N16D_SKIP();
#endif
    ne16_nnx_dispatch_wait(dev);
#ifdef NE16DUAL_PROBE
    N16D_ACC(n16d_wait);
#endif
    ne16_nnx_dispatch(dev, &task);
#ifdef NE16DUAL_PROBE
    N16D_SKIP();
#endif
    if (!pipelined)
      ne16_nnx_resolve_wait(dev, &task); /* sequential: job j done before its successor's columns are built */
#ifdef NE16DUAL_PROBE
    N16D_ACC(n16d_wait);
#endif
    if (j + 1u < n_jobs) {
      if (pipelined)
        ne16_nnx_dispatch_wait(dev); /* a free slot => job j-1 done => col[(j+1)&1] is free; job j keeps running */
#ifdef NE16DUAL_PROBE
      N16D_ACC(n16d_wait);
#endif
      const uint32_t next_pass = (j + 1u) / jobs_per_pass;
      n16d_job_geometry(j + 1u - next_pass * jobs_per_pass, n_out_px, job_cols_full, &px_begin, &job_rows, &job_cols);
      col_args.in = next_pass ? pInM : pInP;
      col_args.px_begin = px_begin;
      col_args.px_count = job_rows * job_cols;
      col_args.col = col[(j + 1u) & 1u];
#ifdef NE16DUAL_PROBE
      N16D_SKIP();
#endif
      pi_cl_team_fork(num_cores, n16d_build_col, &col_args);
#ifdef NE16DUAL_PROBE
      N16D_ACC(n16d_col);
#endif
    }
  }
#ifdef NE16DUAL_PROBE
  N16D_SKIP();
#endif
  ne16_nnx_resolve_wait(dev, &task); /* GVSoC: waits until the queue is empty */
#ifdef NE16DUAL_PROBE /* -- QW exp11: wall = col + wait + rest (task set-up, dispatch, loop) */
  N16D_ACC(n16d_wait);
  const uint32_t n16d_total = n16d_now() - n16d_t_begin;
  n16d_rec("N16PROBE_conv_jobs", n_jobs);
  n16d_rec("N16PROBE_conv_colbuild", n16d_col);
  n16d_rec("N16PROBE_conv_ne16wait", n16d_wait);
  n16d_rec("N16PROBE_conv_rest", n16d_total - n16d_col - n16d_wait);
  n16d_rec("N16PROBE_conv_total", n16d_total);
#endif
}

#endif /* pulp-nnx present */
