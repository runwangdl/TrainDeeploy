/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * RW 2026-10-13: ECO error-feedback update kernels for (quantized) ZO, see ZORuntime.h. The Rademacher z is
 * regenerated exactly as in ApplyPerturbQuantRademacher_CHW / _i32 / ApplyRademacherPerturbation (same seed,
 * one Xorshift32 word per 32 elements, LSB first), so it is the z of the step's +/- passes.
 */
#include "DeeployPULPMath.h"
#include "kernel/ZORuntime.h"
#include "pmsis.h"
#include <math.h>

static inline float32_t eco_round(float32_t q, uint32_t *sr_state) {
  if (!zo_eco_sr) return rintf(q);
  *sr_state = Xorshift32(*sr_state);
  const float32_t u = (float32_t)(*sr_state >> 8) * (1.0f / 16777216.0f);  // [0, 1)
  return floorf(q + u);
}

#define ECO_QUANT_BODY(LO, HI)                                                                                    \
  uint32_t rng = (seed * 1664525u) + 1013904223u, bits = 0;                                                   \
  uint32_t sr = (seed ^ 0x5bd1e995u) + zo_eco_step * 0x9E3779B1u + 1u;                                         \
  const float32_t sgn = (dir != 0u) ? 1.0f : -1.0f;                                                            \
  const float32_t b = zo_eco_beta, lr = zo_eco_lr, g1 = (1.0f - zo_eco_beta) * zo_eco_g;                        \
  const float32_t inj = (1.0f / lr) * (1.0f - 1.0f / b);                                                       \
  const float32_t unit = ldexpf(1.0f, -S) / perturb_eps_baked; /* M * unit = 1/s (LSB per real unit) */          \
  for (uint32_t i = 0; i < size; i++) {                                                                        \
    if ((i & 31u) == 0u) {                                                                                     \
      rng = Xorshift32(rng);                                                                                   \
      bits = rng;                                                                                              \
    }                                                                                                          \
    const float32_t z = sgn * ((bits & 1u) ? 1.0f : -1.0f);                                                    \
    bits >>= 1;                                                                                                \
    const float32_t inv_s = (float32_t)M[(start_offset + i) / channel_width] * unit;                           \
    const float32_t mt = b * mom[i] + g1 * z;                                                                  \
    const float32_t q = (float32_t)w[i] - lr * mt * inv_s;                                                     \
    float32_t wn = eco_round(q, &sr);                                                                          \
    wn = wn < (LO) ? (LO) : (wn > (HI) ? (HI) : wn);                                                           \
    mom[i] = mt + inj * ((q - wn) / inv_s);                                                                    \
    w_dest[i] = wn;                                                                                            \
  }

void ApplyECOQuantRademacher_i8(const int8_t *w, int8_t *w_dest, const int32_t *M, int32_t S, uint32_t channel_width,
                                uint32_t seed, uint32_t size, uint32_t start_offset, uint32_t dir, float32_t *mom) {
  ECO_QUANT_BODY(-127.0f, 127.0f)
}

void ApplyECOQuantRademacher_i32(const int32_t *w, int32_t *w_dest, const int32_t *M, int32_t S,
                                 uint32_t channel_width, uint32_t seed, uint32_t size, uint32_t start_offset,
                                 uint32_t dir, float32_t *mom) {
  ECO_QUANT_BODY(-2147483520.0f, 2147483520.0f)
}

void ApplyECORademacher_fp32(const float32_t *w, float32_t *w_dest, uint32_t seed, uint32_t dir, uint32_t size,
                             float32_t *mom) {
  // same small-tile overlap guard as ApplyRademacherPerturbation (data_out a few bytes ahead of data_in)
  const float32_t *src = w;
  long off = (long)((const char *)w_dest - (const char *)w);
  float32_t tmp[32];
  if (off > 0 && off < 128 && size <= 32u) {
    for (uint32_t k = 0; k < size; k++) tmp[k] = w[k];
    pi_cl_team_barrier(0);
    src = tmp;
  }
  uint32_t rng = (seed * 1664525u) + 1013904223u, bits = 0;
  const float32_t sgn = (dir != 0u) ? 1.0f : -1.0f;
  const float32_t b = zo_eco_beta, lr = zo_eco_lr, g1 = (1.0f - zo_eco_beta) * zo_eco_g;
  for (uint32_t i = 0; i < size; i++) {
    if ((i & 31u) == 0u) {
      rng = Xorshift32(rng);
      bits = rng;
    }
    const float32_t z = sgn * ((bits & 1u) ? 1.0f : -1.0f);
    bits >>= 1;
    const float32_t m = b * mom[i] + g1 * z;
    mom[i] = m;
    w_dest[i] = src[i] - lr * m;
  }
}
