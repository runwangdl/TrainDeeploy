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

// Layout assumptions:
//   pIn      : input  in HWC, shape [H, W, C]
//   pWeight  : weight after RedMuleAdjustWeightMemoryLayoutPass, which
//              transposes the ONNX [F, P, Q, C] weight into [P, Q, C, F].
//              In a flat im2col-style view that is a [P*Q*C, F] matrix,
//              i.e. exactly the right operand of (im2col @ W).
//   pOut     : output in HWC, shape [H_out, W_out, F]
//   pBias    : optional bias of shape [F], broadcast across all output
//              positions when has_bias is true.
//   pIm2ColBuf: transient L1 scratch of size H_out * W_out * (C*P*Q) floats,
//              hoisted by RedmuleFloatConvIm2ColTemplate.
//
// Compute:
//   1. All cluster cores cooperatively build the full im2col matrix
//      [N_out, K] in pIm2ColBuf, where each row is the (p,q,c)-flattened
//      receptive field for one output position.  Out-of-bounds positions
//      from padding are filled with 0.
//   2. Cluster barrier.
//   3. Master core triggers a single RedMulE GEMM:
//          [N_out, K] @ [K, F]  ->  [N_out, F]
//      with bias broadcast in-place when has_bias is set.
//   4. Cluster barrier so the rest of the network sees a valid pOut.
//
// The whole-image im2col + one-shot RedMulE call is preferred over a
// per-pixel call because RedMulE's setup cost (register writes + the
// blocking *wait_reg read) is several hundred cycles -- amortizing it
// across one large matmul is a big win.  L1 budget for the im2col buf
// is reserved by computeTransientBuffersSize in ConvTemplate.py.
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

  // ---- 1. Parallel im2col -------------------------------------------------
  // Each core handles a contiguous slice of output positions.  The slice
  // is sized so that no core writes past the buffer end even when N_out
  // is not divisible by NUM_CORES.
  const uint32_t chunk = (N_out + NUM_CORES - 1) / NUM_CORES;
  const uint32_t pos_start = MIN((uint32_t)core_id * chunk, N_out);
  const uint32_t pos_end = MIN(pos_start + chunk, N_out);

  for (uint32_t pos = pos_start; pos < pos_end; ++pos) {
    const uint32_t h_out = pos / W_out;
    const uint32_t w_out = pos % W_out;
    float32_t *row = pIm2ColBuf + pos * K;
    uint32_t k = 0;
    for (uint32_t p = 0; p < P; ++p) {
      const int32_t h_in = (int32_t)(h_out * SP + p) - (int32_t)pad_top;
      const bool h_in_range = (h_in >= 0) && (h_in < (int32_t)H);
      for (uint32_t q = 0; q < Q; ++q) {
        const int32_t w_in = (int32_t)(w_out * SQ + q) - (int32_t)pad_left;
        if (h_in_range && (w_in >= 0) && (w_in < (int32_t)W)) {
          const uint32_t in_base =
              ((uint32_t)h_in * W + (uint32_t)w_in) * C;
          for (uint32_t c = 0; c < C; ++c) {
            row[k++] = pIn[in_base + c];
          }
        } else {
          // Zero-padding.
          for (uint32_t c = 0; c < C; ++c) {
            row[k++] = 0.0f;
          }
        }
      }
    }
  }

  // Synchronise all cores before handing the matrix to RedMulE.
  pi_cl_team_barrier(0);

  // ---- 2. RedMulE GEMM ----------------------------------------------------
  if (core_id == 0) {
    if (has_bias) {
      // RedMulE Gemm computes Z = X*W + Y where Y is read element-wise as
      // an [M, O] matrix.  Our bias is just [F]; broadcast it into pOut
      // first so y_addr = pOut points at a per-output-row replica, and
      // then have Z alias pOut too (the existing MatMul_*_Redmule kernel
      // already shows that y_addr == z_addr is supported).
      for (uint32_t i = 0; i < N_out; ++i) {
        for (uint32_t f = 0; f < F; ++f) {
          pOut[i * F + f] = pBias[f];
        }
      }
      Gemm_fp32_fp32_fp32_fp32_Redmule(pIm2ColBuf, pWeight, pOut, pOut, N_out,
                                       K, F);
    } else {
      // MatMul_*_Redmule zeroes pOut internally before triggering, so we
      // do not need to clear it here.
      MatMul_fp32_fp32_fp32_Redmule(pIm2ColBuf, pWeight, pOut, N_out, K, F);
    }
  }

  pi_cl_team_barrier(0);
}
