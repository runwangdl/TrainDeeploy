/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp6): column-buffer ("im2col") builder for the NE16 im2col route.
 *
 * This is the cluster half of the GAP9 SDK's preferred way to run a 1xK convolution on NE16
 * (`Ker_MM_Conv1DSmallv2_NE16`, autotiler_v3/CNN_Libraries_HWPE/CNN_BasicKernels_NE16.c:4551):
 * the cores write, for every output pixel, the K*ch_in input values under the kernel window as one
 * contiguous row, and NE16 then runs an ordinary 1x1 convolution over `ch_in*K` input channels.
 * The SDK's `ZeroBuffer(..., PadValChar)` + `CopySeq(...)` per pixel is what the inner loop below
 * does, with two SpeechNet-specific additions folded in:
 *   - the +128 offset NE16 needs for a SIGNED activation (it reads inputs as unsigned; GAP9's NE16
 *     has no signed-input bit) is applied while copying: x_u = x ^ 0x80 == x + 128 for int8;
 *   - padded taps are written as `pad_value` (128 == signed 0 after the offset), so the NE16 job is
 *     UNPADDED -- NE16's 1x1 mode cannot pad, and GVSoC's guard for it is disabled (silent errors).
 *
 * Column order per pixel is (ci, ky, kx) -- the flattening order of an NCHW weight [co][ci][ky][kx]
 * -- so that the folded 1x1 weight row `W[co][ci*K + ky*kx_n + kx]` is just the original weight
 * reshaped, no permutation. (DualConv.c gathers in (ky, kx, ci) order because it consumes Deeploy's
 * NHWC-transposed weight; here the weight goes straight from the perturb node into the NE16 encoder
 * in NCHW order.)
 *
 * Work split: flat output-pixel index over the cores (exp5's split, 99.8 % balance on block 0).
 *
 * Input : NCHW  [ch_in][in_rows][in_cols]   (the graph's activation as Deeploy stores it; for
 *         ch_in == 1 this is also NHWC byte-for-byte)
 * Output: NHWC  [out_rows][out_cols][ch_in*ky_n*kx_n]  uint8 -- the NE16 node consumes it as-is.
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_Im2ColU8_NCHW_to_HWC(const int8_t *pIn, uint8_t *pOut, uint32_t num_cores, uint16_t in_rows,
                               uint16_t in_cols, uint16_t ch_in, uint16_t out_rows, uint16_t out_cols,
                               uint16_t ky_n, uint16_t kx_n, uint16_t pad_top, uint16_t pad_bottom,
                               uint16_t pad_left, uint16_t pad_right, uint16_t stride_y,
                               uint16_t stride_x, uint8_t xor_mask, uint8_t pad_value) {
  (void)pad_bottom;
  (void)pad_right;
  const uint32_t core_id = pi_core_id();
  const uint32_t K = (uint32_t)ch_in * ky_n * kx_n;
  const uint32_t plane = (uint32_t)in_rows * in_cols; /* NCHW: one input channel plane */

  const uint32_t n_out_px = (uint32_t)out_rows * out_cols;
  const uint32_t px_chunk = (n_out_px + num_cores - 1u) / num_cores;
  uint32_t px_begin = px_chunk * core_id;
  if (px_begin > n_out_px)
    px_begin = n_out_px;
  uint32_t px_end = px_begin + px_chunk;
  if (px_end > n_out_px)
    px_end = n_out_px;
  uint32_t oy = px_begin / out_cols;
  uint32_t ox = px_begin - oy * out_cols;

  for (uint32_t px = px_begin; px < px_end; px++) {
    uint8_t *dst = pOut + px * K;
    const int32_t iy0 = (int32_t)oy * (int32_t)stride_y - (int32_t)pad_top;
    const int32_t ix0 = (int32_t)ox * (int32_t)stride_x - (int32_t)pad_left;
    for (uint32_t ci = 0; ci < ch_in; ci++) {
      const int8_t *src_ch = pIn + ci * plane;
      for (uint32_t ky = 0; ky < ky_n; ky++) {
        const int32_t iy = iy0 + (int32_t)ky;
        const int row_in_bounds = (iy >= 0) && (iy < (int32_t)in_rows);
        for (uint32_t kx = 0; kx < kx_n; kx++) {
          const int32_t ix = ix0 + (int32_t)kx;
          if (row_in_bounds && ix >= 0 && ix < (int32_t)in_cols)
            *dst++ = (uint8_t)((uint8_t)src_ch[(uint32_t)iy * in_cols + (uint32_t)ix] ^ xor_mask);
          else
            *dst++ = pad_value;
        }
      }
    }
    if (++ox == out_cols) {
      ox = 0;
      oy++;
    }
  }
  pi_cl_team_barrier();
}
