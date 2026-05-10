// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#include "DeeployPULPMath.h"
#include "pmsis.h"

// RedMulE matmul kernels live in Matmul_fp32_Redmule.c and have no header
// of their own; forward-declare the two we need rather than adding a
// cross-file include.
extern void MatMul_fp32_fp32_fp32_Redmule(const float32_t *__restrict__ pSrcA,
                                          const float32_t *__restrict__ pSrcB,
                                          float32_t *__restrict__ pDstY,
                                          uint32_t M, uint32_t N, uint32_t O);
extern void Gemm_fp32_fp32_fp32_fp32_Redmule(
    const float32_t *__restrict__ pSrcA, const float32_t *__restrict__ pSrcB,
    const float32_t *__restrict__ pBias, float32_t *__restrict__ pDstY,
    uint32_t M, uint32_t N, uint32_t O);

// Chunk size for the streaming im2col + RedMulE pipeline.  Chosen to be 16
// because RedMulE's FP32 mode wants M divisible by 16 for full 4x12-array
// utilisation, and 16 rows × K columns fits comfortably in L1 for any K we
// reasonably expect from a Conv layer (e.g. C·P·Q = 576 for a 3x3 Conv with
// 64 input channels -> 16*576*4 = 36 KiB).  The transient buffer hoisted by
// RedmuleFloatConvIm2ColTemplate.computeTransientBuffersSize is sized to
// exactly this many rows.
#define IM2COL_CHUNK_ROWS 16

// Layout assumptions:
//   pIn      : input  in HWC, shape [H, W, C]
//   pWeight  : weight after RedMuleAdjustWeightMemoryLayoutPass, which
//              transposes the ONNX [F, P, Q, C] weight into [P, Q, C, F].
//              In a flat im2col-style view that is a [P*Q*C, F] matrix,
//              i.e. exactly the right operand of (im2col @ W).
//   pOut     : output in HWC, shape [H_out, W_out, F]
//   pBias    : optional bias of shape [F], broadcast across all output
//              positions when has_bias is true.
//   pIm2ColBuf: transient L1 scratch of size IM2COL_CHUNK_ROWS * (C*P*Q)
//              floats, hoisted by ConvTemplate.computeTransientBuffersSize.
//
// Compute (streaming):
//   For each chunk of IM2COL_CHUNK_ROWS output positions:
//     1. All cluster cores cooperatively build the chunk's im2col rows
//        into pIm2ColBuf (zero-pad when h_in/w_in fall outside the input).
//     2. Cluster barrier.
//     3. Master core triggers one RedMulE GEMM:
//            [chunk_rows, K] @ [K, F]  ->  [chunk_rows, F]
//        written directly into the corresponding stripe of pOut.  When
//        has_bias is set, the [F] bias is broadcast into that stripe
//        first and then Gemm is called with y_addr = z_addr = stripe
//        (same y=z aliasing pattern Matmul_fp32_Redmule already uses).
//     4. Cluster barrier.
//
// Streaming was chosen over whole-image im2col because larger Conv layers
// (e.g. ResNet8 middle layers with H_out*W_out ≥ 1024) would otherwise
// blow the L1 budget: a 1024-row im2col with K=144 is 576 KiB, far above
// the 128 KiB L1 tile budget.  16 rows per chunk costs a few extra RedMulE
// triggers (~200 cycles each) but lets the tiler keep working at any
// reasonable Conv size.
void Conv2d_Im2Col_fp32_fp32_fp32_HWC_8_Redmule(
    const float32_t *__restrict__ pIn, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pWeight, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, const float32_t *__restrict__ pBias, const bool has_bias,
    float32_t *__restrict__ pOut, uint32_t F, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pIm2ColBuf) {

  const int8_t core_id = pi_core_id();

  const uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  const uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
  const uint32_t N_out = H_out * W_out;
  const uint32_t K = C * P * Q;

  for (uint32_t row_start = 0; row_start < N_out;
       row_start += IM2COL_CHUNK_ROWS) {
    const uint32_t this_chunk =
        ((N_out - row_start) < IM2COL_CHUNK_ROWS) ? (N_out - row_start)
                                                  : IM2COL_CHUNK_ROWS;

    // ---- 1. Parallel im2col over this chunk's rows ----------------------
    // Each core fills a contiguous slice of the chunk; with CHUNK_ROWS=16
    // and NUM_CORES=8, every core handles exactly 2 rows when the chunk is
    // full.  A short tail chunk (e.g. last 5 rows) leaves the higher-numbered
    // cores idle.
    const uint32_t local_chunk =
        (this_chunk + NUM_CORES - 1) / NUM_CORES;
    const uint32_t local_start =
        ((uint32_t)core_id * local_chunk < this_chunk)
            ? ((uint32_t)core_id * local_chunk)
            : this_chunk;
    const uint32_t local_end = ((local_start + local_chunk) < this_chunk)
                                   ? (local_start + local_chunk)
                                   : this_chunk;

    for (uint32_t r = local_start; r < local_end; ++r) {
      const uint32_t pos = row_start + r;
      const uint32_t h_out = pos / W_out;
      const uint32_t w_out = pos % W_out;
      float32_t *row = pIm2ColBuf + r * K;
      uint32_t k = 0;
      for (uint32_t p = 0; p < P; ++p) {
        const int32_t h_in = (int32_t)(h_out * SP + p) - (int32_t)pad_top;
        const bool h_in_range = (h_in >= 0) && (h_in < (int32_t)H);
        for (uint32_t q = 0; q < Q; ++q) {
          const int32_t w_in = (int32_t)(w_out * SQ + q) - (int32_t)pad_left;
          if (h_in_range && (w_in >= 0) && (w_in < (int32_t)W)) {
            const uint32_t in_base = ((uint32_t)h_in * W + (uint32_t)w_in) * C;
            for (uint32_t c = 0; c < C; ++c) {
              row[k++] = pIn[in_base + c];
            }
          } else {
            for (uint32_t c = 0; c < C; ++c) {
              row[k++] = 0.0f;
            }
          }
        }
      }
    }

    pi_cl_team_barrier(0);

    // ---- 2. RedMulE GEMM for this chunk's output stripe -----------------
    if (core_id == 0) {
      float32_t *out_stripe = pOut + row_start * F;
      if (has_bias) {
        for (uint32_t i = 0; i < this_chunk; ++i) {
          for (uint32_t f = 0; f < F; ++f) {
            out_stripe[i * F + f] = pBias[f];
          }
        }
        Gemm_fp32_fp32_fp32_fp32_Redmule(pIm2ColBuf, pWeight, out_stripe,
                                         out_stripe, this_chunk, K, F);
      } else {
        MatMul_fp32_fp32_fp32_Redmule(pIm2ColBuf, pWeight, out_stripe,
                                      this_chunk, K, F);
      }
    }

    pi_cl_team_barrier(0);
  }
}
