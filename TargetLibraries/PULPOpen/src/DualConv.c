/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp5): FUSED DUAL-PASS quantized convolution for zeroth-order training.
 *
 * A ZO step runs the network twice, on theta+eps*z and theta-eps*z, and today that means TWO
 * separate graph executions in which the block's conv is:
 *
 *      RQSPerturbRademacher(W)  -> W_pert     (one sign per execution, `perturbation_sign` global)
 *      RQSPerturbRademacher(b)  -> b_pert
 *      Transpose(W_pert)                       (NCHW -> NHWC)
 *      pulp_nn_conv_i8_i8_i8(X, W_pert, ..)    (im2col + matmul call every 2 pixels + requant)
 *
 * This kernel does BOTH passes in ONE traversal of the output pixel space:
 *
 *   phase 1  every core perturbs ITS chunk of the UNPERTURBED weight and requant-add into an L1
 *            scratch, producing W+, W-, add+, add- from ONE Rademacher stream -- the same per-core
 *            chunking, seeding (LCG warm-up + Xorshift32, LSB-first), per-output-channel multiplier,
 *            rounding, shift and int8 clamp as ApplyPerturbQuantRademacher_{CHW,i32}, so the
 *            perturbed weights are BIT-IDENTICAL to what the two standalone nodes would produce.
 *            The "-" pass is obtained the way the device does it (negate r, i.e. `sgn = -1`), NOT
 *            by negating the rounded "+" noise -- those differ when the rounding is asymmetric.
 *   phase 2  each core walks a contiguous range of the FLATTENED pixel index (99.9 % load balance
 *            instead of pulp-nn's 4/4/4/2 row split, 87.4 %), gathers each receptive field ONCE
 *            (it is identical for both passes), and accumulates all `ch_out` channels for W+ and
 *            W- from that one patch, then requantises each pass with its own add. Output is HWC
 *            with 2*ch_out channels per pixel: [0, ch_out) = A+, [ch_out, 2*ch_out) = A-.
 *
 * Why it should be faster than two pulp_nn passes (exp3/exp4 measurements):
 *   - the code footprint is walked ONCE instead of twice (exp3: 22.8 % of the conv's cycles are
 *     I-cache refill, 54 line refills per pixel against a 512 B L0);
 *   - no im2col helper call and no matmul call per pixel pair -- this file makes NO calls in its
 *     pixel loop (the fp32 Deeploy conv, which also makes none, measured 0.04 % imiss);
 *   - the activation patch is loaded once for 2*ch_out MACs, not once per pass;
 *   - the two perturb nodes, the weight transpose, and one full L1 copy of a perturbed weight per
 *     pass disappear from the graph.
 * What it does NOT do: reduce MACs. Both passes' dot products are computed in full. Fusing the
 * arithmetic (X*W base + X*Delta correction) is a different idea and is blocked by the int8 clamp
 * (see the worklog, Step 5.4); this kernel is deliberately exact instead.
 *
 * Axis names follow pulp_nn_conv_i8_i8_i8 so the Deeploy template can pass the same arguments:
 * `x` is the axis Deeploy calls dim_im_in_y (the long / time axis), `y` is dim_im_in_x.
 * Input  : HWC  [dim_in_y][dim_in_x][ch_in]
 * Weight : [ch_out][dim_kernel_y][dim_kernel_x][ch_in]   (Deeploy's NHWC-transposed layout; a row
 *          of K = ch_in*dim_kernel_x*dim_kernel_y bytes is in the SAME order as the gathered patch)
 * Output : HWC  [dim_out_y][dim_out_x][2*ch_out]
 *
 * Scratch (L1, sized by DualConvTemplate.computeTransientBuffersSize -- keep the two in sync):
 *   [0, wsize)            W+       wsize = ch_out*K
 *   [wsize, 2*wsize)      W-
 *   off_addp (4-aligned)  add+     ch_out * int32
 *   off_addm              add-     ch_out * int32
 *   off_patch             one Kpad-byte patch per core, Kpad = K rounded up to 4
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"
#include <math.h>

typedef signed char dc_v4s __attribute__((vector_size(4)));

/* QW (exp6): dc_xorshift32 / dc_chunk / dc_perturb_i8 / dc_perturb_i32 moved VERBATIM to
 * inc/kernel/DualPerturb.h so the NE16 fused path (GAP9) shares them. -- QW */
#include "kernel/DualPerturb.h"

void PULP_DualConv2d_i8_i8_i8_HWC(
    const int8_t *pIn, const int8_t *pWeight, const int32_t *pMul, const int32_t *pAdd,
    const int32_t *pWMul, const int32_t *pBMul, int8_t *pOut, int8_t *pScratch,
    uint32_t seed_w_base, uint32_t seed_b_base, uint32_t tile_seed_xor, uint32_t log2D_w,
    uint32_t log2D_b, float32_t eps_scale, uint32_t num_cores, uint16_t out_shift,
    uint16_t dim_in_x, uint16_t dim_in_y, uint16_t ch_in, uint16_t dim_out_x, uint16_t dim_out_y,
    uint16_t ch_out, uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t pad_y_top,
    uint16_t pad_y_bottom, uint16_t pad_x_left, uint16_t pad_x_right, uint16_t stride_x,
    uint16_t stride_y) {
  const uint32_t core_id = pi_core_id();
  const uint32_t K = (uint32_t)ch_in * dim_kernel_x * dim_kernel_y; //block 0, (1,1,4), K=4
  const uint32_t Kpad = (K + 3u) & ~3u; // rounds up to multiple of 4
  const uint32_t wsize = (uint32_t)ch_out * K; // block 0, (8,1,1,4), wsize = 32

  /* scratch layout -- must match DualConvTemplate.computeTransientBuffersSize */
  const uint32_t off_addp = (2u * wsize + 3u) & ~3u;
  const uint32_t off_addm = off_addp + 4u * ch_out;
  const uint32_t off_patch = off_addm + 4u * ch_out;
  int8_t *wp = pScratch;
  int8_t *wm = pScratch + wsize;
  int32_t *addp = (int32_t *)(pScratch + off_addp);
  int32_t *addm = (int32_t *)(pScratch + off_addm);
  int8_t *patch = pScratch + off_patch + core_id * Kpad;

  /* ---------------- phase 1: perturb my chunk, both signs, exactly as the two nodes would ---- */
  uint32_t chunk_begin, chunk_len;
  dc_chunk(wsize, num_cores, core_id, &chunk_begin, &chunk_len);
  if (chunk_len)
    dc_perturb_i8(pWeight, wp, wm, pWMul, K, (seed_w_base + core_id) ^ tile_seed_xor, chunk_begin, chunk_len,
                  log2D_w, eps_scale);
  dc_chunk((uint32_t)ch_out, num_cores, core_id, &chunk_begin, &chunk_len);
  if (chunk_len)
    dc_perturb_i32(pAdd, addp, addm, pBMul, 1u, (seed_b_base + core_id) ^ tile_seed_xor, chunk_begin, chunk_len,
                   log2D_b, eps_scale);
  for (uint32_t k = K; k < Kpad; k++)
    patch[k] = 0; /* zero tail: the vector loop may read up to Kpad */
  pi_cl_team_barrier();
#ifdef DUALCONV_DEBUG_PRINT
  if (core_id == 0) {
    printf("[DC] scratch=%p num_cores=%u K=%u ch_out=%u in(%u,%u,%u) out(%u,%u) pad(%u,%u,%u,%u) shift=%u\n", (void*)pScratch,
           (unsigned)num_cores, (unsigned)K, (unsigned)ch_out, dim_in_x, dim_in_y, ch_in, dim_out_x, dim_out_y,
           pad_y_top, pad_y_bottom, pad_x_left, pad_x_right, out_shift);
    for (uint32_t c = 0; c < ch_out; c++)
      printf("[DC] c%u W=[%d,%d,%d,%d] W+=[%d,%d,%d,%d] W-=[%d,%d,%d,%d] add=%ld add+=%ld add-=%ld mul=%ld wmul=%ld bmul=%ld\n", (unsigned)c,
             pWeight[4*c],pWeight[4*c+1],pWeight[4*c+2],pWeight[4*c+3], wp[4*c],wp[4*c+1],wp[4*c+2],wp[4*c+3],
             wm[4*c],wm[4*c+1],wm[4*c+2],wm[4*c+3], (long)pAdd[c], (long)addp[c], (long)addm[c], (long)pMul[c], (long)pWMul[c], (long)pBMul[c]);
  }
  pi_cl_team_barrier();
#endif

  /* ---------------- phase 2: one pass over the pixels, both weight sets per patch ------------ */
  const uint32_t n_out_px = (uint32_t)dim_out_y * dim_out_x;
  const uint32_t px_chunk = (n_out_px + num_cores - 1u) / num_cores; /* ceiling: every core gets the same count */
  uint32_t px_begin = px_chunk * core_id;
  if (px_begin > n_out_px)
    px_begin = n_out_px;
  uint32_t px_end = px_begin + px_chunk;
  if (px_end > n_out_px)
    px_end = n_out_px;
  uint32_t oy = px_begin / dim_out_x;
  uint32_t ox = px_begin - oy * dim_out_x;
  const uint32_t out_px_stride = 2u * ch_out;

  for (uint32_t px = px_begin; px < px_end; px++) {
    /* gather the receptive field once (shared by both passes) */
    int8_t *dst = patch;
    const int32_t iy0 = (int32_t)oy * (int32_t)stride_y - (int32_t)pad_y_top;
    const int32_t ix0 = (int32_t)ox * (int32_t)stride_x - (int32_t)pad_x_left;
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

    int8_t *out_px = pOut + px * out_px_stride;
    for (uint32_t co = 0; co < ch_out; co++) {
      const int8_t *wp_row = wp + co * K;
      const int8_t *wm_row = wm + co * K;
      int32_t acc_p = 0, acc_m = 0;
      for (uint32_t k = 0; k < Kpad; k += 4u) {
        const dc_v4s x_word = *(const dc_v4s *)(patch + k);
        acc_p = __builtin_pulp_sdotsp4(x_word, *(const dc_v4s *)(wp_row + k), acc_p);
        acc_m = __builtin_pulp_sdotsp4(x_word, *(const dc_v4s *)(wm_row + k), acc_m);
      }
      /* requant exactly as pulp_nn_bn_quant_i8 (32-bit variant): int32 wrap, arithmetic shift, clip */
      const int32_t y_p = (int32_t)((uint32_t)acc_p * (uint32_t)pMul[co] + (uint32_t)addp[co]) >> out_shift;
      const int32_t y_m = (int32_t)((uint32_t)acc_m * (uint32_t)pMul[co] + (uint32_t)addm[co]) >> out_shift;
      out_px[co] = (int8_t)__builtin_pulp_clip_r(y_p, 127);
      out_px[ch_out + co] = (int8_t)__builtin_pulp_clip_r(y_m, 127);
    }
    if (++ox == dim_out_x) {
      ox = 0;
      oy++;
    }
  }
  pi_cl_team_barrier();
}
