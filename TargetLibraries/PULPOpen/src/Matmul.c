/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pmsis.h"

#include "DeeployPULPMath.h"

void PULP_MatMul_fp32_fp32_fp32_unroll1x7(const float32_t *__restrict__ pSrcA,
                                          const float32_t *__restrict__ pSrcB,
                                          float32_t *__restrict__ pDstY,
                                          uint32_t M, uint32_t N, uint32_t O) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t M_chunk = (M >> log2Core) + ((M & (NUM_CORES - 1)) != 0);
  uint32_t M_start = MIN(core_id * M_chunk, M);
  uint32_t M_end = MIN(M_start + M_chunk, M);
  uint32_t M_size = M_end - M_start;

  if (M_size == 0) {
    return;
  }

  const float32_t *local_pSrcA = pSrcA + M_start * N;
  float32_t *local_pDstY = pDstY + M_start * O;

  uint32_t O_block = O - (O % 7);

  for (uint32_t i = 0; i < M_size; i++) {

    for (uint32_t j = 0; j < O_block; j += 7) {
      float32_t sum0 = 0.0f;
      float32_t sum1 = 0.0f;
      float32_t sum2 = 0.0f;
      float32_t sum3 = 0.0f;
      float32_t sum4 = 0.0f;
      float32_t sum5 = 0.0f;
      float32_t sum6 = 0.0f;

      for (uint32_t k = 0; k < N; k++) {
        float32_t a0 = local_pSrcA[i * N + k];

        float32_t b0 = pSrcB[k * O + (j + 0)];
        float32_t b1 = pSrcB[k * O + (j + 1)];
        float32_t b2 = pSrcB[k * O + (j + 2)];
        float32_t b3 = pSrcB[k * O + (j + 3)];
        float32_t b4 = pSrcB[k * O + (j + 4)];
        float32_t b5 = pSrcB[k * O + (j + 5)];
        float32_t b6 = pSrcB[k * O + (j + 6)];

        sum0 += a0 * b0;
        sum1 += a0 * b1;
        sum2 += a0 * b2;
        sum3 += a0 * b3;
        sum4 += a0 * b4;
        sum5 += a0 * b5;
        sum6 += a0 * b6;
      }

      local_pDstY[i * O + (j + 0)] = sum0;
      local_pDstY[i * O + (j + 1)] = sum1;
      local_pDstY[i * O + (j + 2)] = sum2;
      local_pDstY[i * O + (j + 3)] = sum3;
      local_pDstY[i * O + (j + 4)] = sum4;
      local_pDstY[i * O + (j + 5)] = sum5;
      local_pDstY[i * O + (j + 6)] = sum6;
    }

    for (uint32_t j = O_block; j < O; j++) {
      float32_t sum = 0.0f;

      for (uint32_t k = 0; k < N; k++) {
        float32_t a_val = local_pSrcA[i * N + k];
        float32_t b_val = pSrcB[k * O + j];
        sum += a_val * b_val;
      }

      local_pDstY[i * O + j] = sum;
    }
  }
}
// Matmul against an int8 weight matrix, dequantising inside the loop.
//
// A weight-only-quantised model stores the frozen weight as int8 and needs it as
// float for the fp32 matmul. Emitting that as a separate Dequant node materialises
// the whole dequantised matrix -- on CCT-QLoRA, 1088 KB of fp32 tensors that exist
// only to be consumed by the very next node, which then has to be kept alive or
// recomputed. QLoRA itself never does this: it dequantises inside the matmul kernel
// and discards the value immediately.
//
// The dequantisation is affine and per-tensor, so it factors out of the inner loop
// entirely:
//
//   sum_k a_k * (b_k - zp) * scale  ==  scale * (sum_k a_k * b_k  -  zp * sum_k a_k)
//
// leaving one multiply per output element instead of one per multiply-accumulate.
// With zeroPoint 0 the second term vanishes and the loop is the fp32 loop with an
// int8 load. Numerically identical to Dequant followed by matmul, up to the order
// of the same floating-point operations.
void PULP_MatMul_fp32_i8_fp32_unroll1x7(const float32_t *__restrict__ pSrcA,
                                        const int8_t *__restrict__ pSrcB,
                                        float32_t *__restrict__ pDstY, uint32_t M,
                                        uint32_t N, uint32_t O, float32_t scale,
                                        int32_t zeroPoint) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  uint32_t M_chunk = (M >> log2Core) + ((M & (NUM_CORES - 1)) != 0);
  uint32_t M_start = MIN(core_id * M_chunk, M);
  uint32_t M_end = MIN(M_start + M_chunk, M);
  uint32_t M_size = M_end - M_start;

  if (M_size == 0) {
    return;
  }

  const float32_t *local_pSrcA = pSrcA + M_start * N;
  float32_t *local_pDstY = pDstY + M_start * O;
  uint32_t O_block = O - (O % 7);

  for (uint32_t i = 0; i < M_size; i++) {
    // sum_a is only needed when the zero point is non-zero; the compiler drops it
    // when zeroPoint is a compile-time 0 after inlining, and it costs one add per
    // element otherwise.
    float32_t sum_a = 0.0f;
    if (zeroPoint != 0) {
      for (uint32_t k = 0; k < N; k++) {
        sum_a += local_pSrcA[i * N + k];
      }
    }
    const float32_t correction = scale * (float32_t)zeroPoint * sum_a;

    for (uint32_t j = 0; j < O_block; j += 7) {
      float32_t sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      float32_t sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f;

      for (uint32_t k = 0; k < N; k++) {
        float32_t a0 = local_pSrcA[i * N + k];
        const int8_t *b_row = &pSrcB[k * O + j];
        sum0 += a0 * (float32_t)b_row[0];
        sum1 += a0 * (float32_t)b_row[1];
        sum2 += a0 * (float32_t)b_row[2];
        sum3 += a0 * (float32_t)b_row[3];
        sum4 += a0 * (float32_t)b_row[4];
        sum5 += a0 * (float32_t)b_row[5];
        sum6 += a0 * (float32_t)b_row[6];
      }

      local_pDstY[i * O + (j + 0)] = scale * sum0 - correction;
      local_pDstY[i * O + (j + 1)] = scale * sum1 - correction;
      local_pDstY[i * O + (j + 2)] = scale * sum2 - correction;
      local_pDstY[i * O + (j + 3)] = scale * sum3 - correction;
      local_pDstY[i * O + (j + 4)] = scale * sum4 - correction;
      local_pDstY[i * O + (j + 5)] = scale * sum5 - correction;
      local_pDstY[i * O + (j + 6)] = scale * sum6 - correction;
    }

    for (uint32_t j = O_block; j < O; j++) {
      float32_t sum = 0.0f;
      for (uint32_t k = 0; k < N; k++) {
        sum += local_pSrcA[i * N + k] * (float32_t)pSrcB[k * O + j];
      }
      local_pDstY[i * O + j] = scale * sum - correction;
    }
  }
}
