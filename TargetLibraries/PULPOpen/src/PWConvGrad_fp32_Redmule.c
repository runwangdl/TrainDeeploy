// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#include "DeeployPULPMath.h"
#include "pmsis.h"

extern void MatMul_fp32_fp32_fp32_Redmule(const float32_t *__restrict__ pSrcA,
                                          const float32_t *__restrict__ pSrcB,
                                          float32_t *__restrict__ pDstY,
                                          uint32_t M, uint32_t N, uint32_t O);
extern void Gemm_fp32_fp32_fp32_fp32_Redmule(
    const float32_t *__restrict__ pSrcA, const float32_t *__restrict__ pSrcB,
    const float32_t *__restrict__ pBias, float32_t *__restrict__ pDstY,
    uint32_t M, uint32_t N, uint32_t O);

// Chunk over P = H_out * W_out positions to keep the L1 transient buffer
// fixed-small regardless of the network's feature-map area.  Each chunk
// runs one RedMulE call; chunk-to-chunk accumulation rides on Gemm's
// y_addr = bias = previous dW pattern (same trick the MatMul driver uses
// for its Y=Z=pDstY zero-init).
#define PWGW_CHUNK_P 16

// Pointwise (1x1) Conv backward weight gradient, RedMulE-accelerated.
//
// Forward (1x1, stride (SP, SQ)):
//   Y[F, h_out, w_out] = sum_c X[c, h_out * SP, w_out * SQ] * W[F, c, 0, 0]
// Backward dW:
//   dW[F, C] = sum_{n, h_out, w_out} dY[F, h_out, w_out]
//              * X[C, h_out * SP, w_out * SQ]
//
// Mathematically dW = dY_reshape[F, P] @ X_sampled^T[P, C] with
// P = H_out * W_out.  A full P-row transpose buffer doesn't scale -- early
// MobileNet blocks would need a 32 * 48 * 48 = 72 KiB buffer and the
// pattern-memory solver runs out of L1 budget.  Instead, sample+transpose
// PWGW_CHUNK_P rows at a time and accumulate into dW via Gemm:
//   dW = dY_chunk[F, chunk_size] @ X_chunk^T[chunk_size, C]  +  dW_prev
// The buffer size is fixed at PWGW_CHUNK_P * C_in floats regardless of P,
// at the cost of one extra RedMulE trigger per chunk (~200 cycles each).
//
// Stride is recovered from the input/output spatial ratios so the kernel
// signature stays compatible with the pulp-trainlib variant.
void PWConvGradW2d_fp32_fp32_fp32_CHW_Redmule(
    const float32_t *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float32_t *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, float32_t *__restrict__ pGradWeight,
    float32_t *__restrict__ pTransposeBuffer) {

  const int8_t core_id = pi_core_id();
  const uint32_t SP = (H_out > 0) ? (H_in / H_out) : 1;
  const uint32_t SQ = (W_out > 0) ? (W_in / W_out) : 1;
  const uint32_t P = H_out * W_out;

  // Initialise dW to zero so the first chunk's Gemm-with-bias starts from
  // a clean slate.  Done in parallel across cores.
  const uint32_t dw_total = C_out * C_in;
  const uint32_t dw_chunk = (dw_total + NUM_CORES - 1) / NUM_CORES;
  const uint32_t dw_lo = MIN((uint32_t)core_id * dw_chunk, dw_total);
  const uint32_t dw_hi = MIN(dw_lo + dw_chunk, dw_total);
  for (uint32_t i = dw_lo; i < dw_hi; ++i) {
    pGradWeight[i] = 0.0f;
  }
  pi_cl_team_barrier(0);

  for (uint32_t chunk_start = 0; chunk_start < P; chunk_start += PWGW_CHUNK_P) {
    const uint32_t this_chunk =
        ((P - chunk_start) < PWGW_CHUNK_P) ? (P - chunk_start) : PWGW_CHUNK_P;

    // ---- 1. Parallel sampled-transpose of this chunk's X positions -----
    //   pTransposeBuffer[k_local * C_in + c] = X[c, h_in, w_in]
    const uint32_t total = this_chunk * C_in;
    const uint32_t chunk_w = (total + NUM_CORES - 1) / NUM_CORES;
    const uint32_t lo = MIN((uint32_t)core_id * chunk_w, total);
    const uint32_t hi = MIN(lo + chunk_w, total);

    for (uint32_t idx = lo; idx < hi; ++idx) {
      const uint32_t k_local = idx / C_in;
      const uint32_t c = idx % C_in;
      const uint32_t k = chunk_start + k_local;
      const uint32_t h_out = k / W_out;
      const uint32_t w_out = k % W_out;
      const uint32_t h_in = h_out * SP;
      const uint32_t w_in = w_out * SQ;
      pTransposeBuffer[idx] = pInput[c * (H_in * W_in) + h_in * W_in + w_in];
    }

    pi_cl_team_barrier(0);

    // ---- 2. RedMulE Gemm: dW = dY_chunk @ X_chunk^T + dW_prev ---------
    //   dY_chunk is a contiguous stripe of dY along its inner spatial
    //   axis.  Since dY is laid out as [C_out, P] = [F, P] in CHW, the
    //   F-th row's slice [chunk_start : chunk_start + this_chunk] is at
    //   pGradOut + f * P + chunk_start -- NOT contiguous across F.
    //   For RedMulE to consume the chunk as [F, this_chunk] it needs to
    //   be contiguous, which here means we treat dY[F, P] as the left
    //   operand and pass chunk-stride math via N=this_chunk only when
    //   chunk_start == 0 AND this_chunk == P (i.e. P fits in one
    //   RedMulE call).  When chunks are smaller we must build a
    //   contiguous [F, this_chunk] view too -- skipped here because the
    //   transient already lives in a fixed slot; instead we pass the
    //   *full* dY [F, P] and X^T padded to P rows.  See follow-up note.
    //
    //   The simplest correct path used below is the single-chunk case
    //   (PWGW_CHUNK_P >= P), which holds whenever the tiler shrinks
    //   the spatial output to <= 16 positions per tile.  Otherwise we
    //   would need a dY scratch too; flagged for the next iteration.
    if (this_chunk == P) {
      if (core_id == 0) {
        Gemm_fp32_fp32_fp32_fp32_Redmule(pGradOut, pTransposeBuffer,
                                         pGradWeight, pGradWeight, C_out,
                                         this_chunk, C_in);
      }
    } else {
      // Multi-chunk path: gather a contiguous [F, this_chunk] view of dY
      // into the tail of pTransposeBuffer.  The template reserves enough
      // headroom (see RedmulePWConvGradWTemplate.computeTransientBuffersSize).
      float32_t *dY_view = pTransposeBuffer + (PWGW_CHUNK_P * C_in);
      const uint32_t dy_total = C_out * this_chunk;
      const uint32_t dy_chunk = (dy_total + NUM_CORES - 1) / NUM_CORES;
      const uint32_t dy_lo = MIN((uint32_t)core_id * dy_chunk, dy_total);
      const uint32_t dy_hi = MIN(dy_lo + dy_chunk, dy_total);
      for (uint32_t idx = dy_lo; idx < dy_hi; ++idx) {
        const uint32_t f = idx / this_chunk;
        const uint32_t k_local = idx % this_chunk;
        const uint32_t k = chunk_start + k_local;
        dY_view[idx] = pGradOut[f * P + k];
      }
      pi_cl_team_barrier(0);
      if (core_id == 0) {
        Gemm_fp32_fp32_fp32_fp32_Redmule(dY_view, pTransposeBuffer,
                                         pGradWeight, pGradWeight, C_out,
                                         this_chunk, C_in);
      }
    }

    pi_cl_team_barrier(0);
  }
}

// Pointwise (1x1) Conv backward input gradient, RedMulE-accelerated.
//
// Same shape relations as the forward path; stride > 1 means dX has more
// spatial positions than dY and only the strided samples are non-zero.
//
// Pipeline:
//   - Zero pGradIn.
//   - W^T transpose: pTransposeBuffer[0:C_in*C_out] = W^T.
//   - GEMM tmp[C_in, P] = W^T @ dY[C_out, P], P = H_out * W_out.
//     For stride 1 we write tmp directly into pGradIn (dX layout matches).
//     For stride > 1 we route the GEMM output to the tail of
//     pTransposeBuffer and scatter it into pGradIn at strided positions.
//
// Unlike the W kernel, X's GEMM dimensions don't scale with P alone --
// the K (inner) dim is C_out, which is bounded by the tile's
// channel-tile.  So the existing all-in-one-GEMM path remains feasible
// and we keep it; only the transient buffer changed shape (size cap
// reflected in RedmulePWConvGradXTemplate).
void PWConvGradX2d_fp32_fp32_fp32_CHW_Redmule(
    const float32_t *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float32_t *__restrict__ pWeight, uint32_t C_in,
    float32_t *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in,
    float32_t *__restrict__ pTransposeBuffer,
    uint32_t transposeBufferSize) {

  (void)transposeBufferSize;

  const int8_t core_id = pi_core_id();
  const uint32_t SP = (H_out > 0) ? (H_in / H_out) : 1;
  const uint32_t SQ = (W_out > 0) ? (W_in / W_out) : 1;
  const uint32_t P = H_out * W_out;
  const bool strided = (SP != 1) || (SQ != 1);

  // ---- 1. Zero pGradIn (parallel) ---------------------------------------
  const uint32_t dx_total = C_in * H_in * W_in;
  const uint32_t dx_chunk = (dx_total + NUM_CORES - 1) / NUM_CORES;
  const uint32_t dx_lo = MIN((uint32_t)core_id * dx_chunk, dx_total);
  const uint32_t dx_hi = MIN(dx_lo + dx_chunk, dx_total);
  for (uint32_t i = dx_lo; i < dx_hi; ++i) {
    pGradIn[i] = 0.0f;
  }

  // ---- 2. Parallel transpose W[C_out, C_in] -> W^T[C_in, C_out] --------
  const uint32_t wt_total = C_in * C_out;
  const uint32_t wt_chunk = (wt_total + NUM_CORES - 1) / NUM_CORES;
  const uint32_t wt_lo = MIN((uint32_t)core_id * wt_chunk, wt_total);
  const uint32_t wt_hi = MIN(wt_lo + wt_chunk, wt_total);
  for (uint32_t idx = wt_lo; idx < wt_hi; ++idx) {
    const uint32_t c_in = idx / C_out;
    const uint32_t c_out = idx % C_out;
    pTransposeBuffer[idx] = pWeight[c_out * C_in + c_in];
  }

  pi_cl_team_barrier(0);

  // ---- 3. RedMulE GEMM: dX_dense[C_in, P] = W^T[C_in, C_out] @ dY[C_out, P] -
  if (core_id == 0) {
    if (!strided) {
      MatMul_fp32_fp32_fp32_Redmule(pTransposeBuffer, pGradOut, pGradIn,
                                    C_in, C_out, P);
    } else {
      float32_t *tmp_gemm = pTransposeBuffer + (C_in * C_out);
      MatMul_fp32_fp32_fp32_Redmule(pTransposeBuffer, pGradOut, tmp_gemm,
                                    C_in, C_out, P);
    }
  }

  pi_cl_team_barrier(0);

  // ---- 4. Scatter (stride > 1 only) ------------------------------------
  if (strided) {
    float32_t *tmp_gemm = pTransposeBuffer + (C_in * C_out);
    const uint32_t scat_total = C_in * P;
    const uint32_t scat_chunk = (scat_total + NUM_CORES - 1) / NUM_CORES;
    const uint32_t scat_lo = MIN((uint32_t)core_id * scat_chunk, scat_total);
    const uint32_t scat_hi = MIN(scat_lo + scat_chunk, scat_total);
    for (uint32_t idx = scat_lo; idx < scat_hi; ++idx) {
      const uint32_t c = idx / P;
      const uint32_t k = idx % P;
      const uint32_t h_out = k / W_out;
      const uint32_t w_out = k % W_out;
      const uint32_t h_in = h_out * SP;
      const uint32_t w_in = w_out * SQ;
      pGradIn[c * (H_in * W_in) + h_in * W_in + w_in] = tmp_gemm[idx];
    }
    pi_cl_team_barrier(0);
  }
}
