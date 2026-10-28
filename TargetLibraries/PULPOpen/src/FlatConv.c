/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp5a): SINGLE-PASS requantised int8 conv with the structure of DualConv.c.
 *
 * Purpose: an A/B control for exp5. DualConv.c changed two things at once versus pulp_nn_conv_i8_i8_i8
 * (the kernel structure, and the fusion of both eps-passes). This kernel is DualConv.c's phase 2 with
 * ONE weight set and NO phase 1 -- it receives the already-perturbed, already-transposed weight from
 * the graph exactly as pulp-nn does -- so running it twice (one graph execution per sign, the deployed
 * flow) measures the structure change alone, and the gap to DualConv measures the fusion alone.
 *
 * FAIRNESS RULE: this is a mechanical deletion from DualConv.c. Same generic loops, same byte-wise
 * bounds-tested gather, same unhoisted parameters, no unrolling. Do NOT tune it beyond DualConv.c.
 *
 * Structure (per core):
 *   - a contiguous range of the FLATTENED output-pixel index (ceil(H*W/cores) px each: 99.8 %
 *     balance for block 0, vs pulp-nn's 4/4/4/2 row split at 87.4 %);
 *   - per pixel: gather the receptive field once into a private Kpad-byte patch (zeros where a tap
 *     falls in the padding), then for every output channel one sdotsp4 loop, the requant exactly as
 *     pulp_nn_bn_quant_i8 (32-bit variant), clip, store. No calls in the pixel loop.
 *
 * Axis names follow pulp_nn_conv_i8_i8_i8 so the Deeploy template passes the same arguments:
 * `x` is the axis Deeploy calls dim_im_in_y (the long / time axis), `y` is dim_im_in_x.
 * Input  : HWC  [dim_in_y][dim_in_x][ch_in]
 * Weight : [ch_out][dim_kernel_y][dim_kernel_x][ch_in]   (Deeploy's NHWC-transposed layout)
 * Output : HWC  [dim_out_y][dim_out_x][ch_out]
 * Scratch: num_cores * Kpad bytes (sized by FlatConvTemplate.computeTransientBuffersSize).
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

typedef signed char fc_v4s __attribute__((vector_size(4)));

void PULP_FlatConv2d_i8_i8_i8_HWC(const int8_t *pIn, int8_t *pScratch, int8_t *pOut,
                                  const int8_t *pWeight, const int32_t *pMul, const int32_t *pAdd,
                                  uint32_t num_cores, uint16_t out_shift, uint16_t dim_in_x,
                                  uint16_t dim_in_y, uint16_t ch_in, uint16_t dim_out_x,
                                  uint16_t dim_out_y, uint16_t ch_out, uint16_t dim_kernel_x,
                                  uint16_t dim_kernel_y, uint16_t pad_y_top, uint16_t pad_y_bottom,
                                  uint16_t pad_x_left, uint16_t pad_x_right, uint16_t stride_x,
                                  uint16_t stride_y) {
  const uint32_t core_id = pi_core_id();
  const uint32_t K = (uint32_t)ch_in * dim_kernel_x * dim_kernel_y; // block 0, (1,1,4), K=4
  const uint32_t Kpad = (K + 3u) & ~3u; //rounds K up to multiple of 4, because sdotsp4 reads 4 bytes at a time, patch four bytes at a time 
  int8_t *patch = pScratch + core_id * Kpad;

  for (uint32_t k = K; k < Kpad; k++)
    patch[k] = 0; 

  /* ---------------- one pass over the pixels, flattened index ------------------------------- */
  const uint32_t n_out_px = (uint32_t)dim_out_y * dim_out_x;
  const uint32_t px_chunk = (n_out_px + num_cores - 1u) / num_cores;
  uint32_t px_begin = px_chunk * core_id;
  if (px_begin > n_out_px)
    px_begin = n_out_px;
  uint32_t px_end = px_begin + px_chunk;
  if (px_end > n_out_px)
    px_end = n_out_px;
  uint32_t oy = px_begin / dim_out_x;
  uint32_t ox = px_begin - oy * dim_out_x; /* first output pixel of this core */

  for (uint32_t px = px_begin; px < px_end; px++) {
    /* gather the receptive field once */
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

    int8_t *out_px = pOut + px * ch_out;
    for (uint32_t co = 0; co < ch_out; co++) {
      const int8_t *w_row = pWeight + co * K;
      int32_t acc = 0;
      for (uint32_t k = 0; k < Kpad; k += 4u) {
        const fc_v4s x_word = *(const fc_v4s *)(patch + k);
        acc = __builtin_pulp_sdotsp4(x_word, *(const fc_v4s *)(w_row + k), acc);
      }
      /* requant exactly as pulp_nn_bn_quant_i8 (32-bit variant): int32 wrap, arithmetic shift, clip */
      const int32_t y = (int32_t)((uint32_t)acc * (uint32_t)pMul[co] + (uint32_t)pAdd[co]) >> out_shift;
      out_px[co] = (int8_t)__builtin_pulp_clip_r(y, 127);
    }
    if (++ox == dim_out_x) {
      ox = 0;
      oy++;
    }
  }
  pi_cl_team_barrier();
}
