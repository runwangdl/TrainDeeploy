/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp6): the "both signs from one draw" perturbation helpers, shared.
 *
 * Lifted VERBATIM out of DualConv.c (exp5) so the NE16 fused path (NE16DualPrep.c, GAP9) uses the
 * same code as the cluster fused kernel: same per-core chunking as the standalone RQSPerturbRademacher
 * node (`(size >> log2(cores)) + (size % cores != 0)`), same LCG warm-up + Xorshift32 stream (one
 * 32-bit draw per 32 elements, LSB first), same per-output-channel multiplier, same rounding, same
 * int8 clamp -- and the exp5 §7.3 rule: form `prod = z*mul` once and derive the -eps noise as
 * `(rounding - prod) >> shift`, never as `(-z)*mul` (mis-evaluated by the PULP clang-15 build). -- QW
 */
#ifndef __DEEPLOY_MATH_DUALPERTURB_KERNEL_HEADER_
#define __DEEPLOY_MATH_DUALPERTURB_KERNEL_HEADER_

#include <math.h>
#include <stdint.h>

#ifndef DC_CLAMP
#define DC_CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

static inline uint32_t dc_xorshift32(uint32_t s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

/* The perturb nodes' per-core split: chunk = (size >> log2(NUM_CORES)) + (size % NUM_CORES != 0). */
static inline void dc_chunk(uint32_t size, uint32_t num_cores, uint32_t core_id, uint32_t *start,
                            uint32_t *count) {
  uint32_t log2_cores = 0;
  while ((1u << (log2_cores + 1u)) <= num_cores)
    log2_cores++;
  uint32_t chunk = (size >> log2_cores) + ((size & (num_cores - 1u)) != 0u);
  uint32_t begin = chunk * core_id;
  if (begin > size)
    begin = size;
  uint32_t end = begin + chunk;
  if (end > size)
    end = size;
  *start = begin;
  *count = end - begin;
}

/* int8 weight: both signs from one stream, clamped to [-127, 127] like ApplyPerturbQuantRademacher_CHW. */
static inline void dc_perturb_i8(const int8_t *w, int8_t *wp, int8_t *wm, const int32_t *noise_mul,
                                 uint32_t channel_width, uint32_t seed, uint32_t start, uint32_t count,
                                 uint32_t shift, float eps_scale) {
  uint32_t rng = seed * 1664525u + 1013904223u;
  const int32_t rounding = (shift > 0u) ? (int32_t)(1u << (shift - 1u)) : 0;
  uint32_t i = start;
  const uint32_t end = start + count;
  while (i < end) {
    rng = dc_xorshift32(rng);
    uint32_t bits = rng;
    uint32_t run_len = end - i;
    if (run_len > 32u)
      run_len = 32u;
    for (uint32_t k = 0; k < run_len; k++, i++) {
      const int32_t z = (bits & 1u) ? 1 : -1;
      int32_t mul = noise_mul[i / channel_width];
      if (eps_scale != 1.0f)
        mul = (int32_t)lrintf((float)mul * eps_scale);
      const int32_t prod = z * mul;
      const int32_t noise_p = (prod + rounding) >> (int32_t)shift;
      const int32_t noise_m = (rounding - prod) >> (int32_t)shift;
      const int32_t w_val = (int32_t)w[i];
      wp[i] = (int8_t)DC_CLAMP(w_val + noise_p, -127, 127);
      wm[i] = (int8_t)DC_CLAMP(w_val + noise_m, -127, 127);
      bits >>= 1;
    }
  }
}

/* QW (dual_pass_fusion exp7, step 3): int8 weight, both signs, for Ci > 1. The source `w` is the RAW CHW
 * weight [co][ci][ky][kx] -- the layout the reference RQSPerturbRademacher node perturbs -- and this core walks
 * ITS CHW chunk [start, start+count) in memory order, so noise bit i lands on CHW element i exactly as in the
 * reference (same stream, chunking, seed, rounding, clamp as dc_perturb_i8). Each result is written to the
 * element's HWC position j = ((co*kh + ky)*kw + kx)*ci + c_i in wp / wm, the layout the conv loop reads.
 * dc_perturb_i8 on an already transposed weight is only correct for ci == 1 (block 0), where the two orders
 * coincide. The (co, ci, ky, kx) position is decoded once and then advanced incrementally (no division per
 * element); the multiplier is reloaded only when co changes. -- QW */
static inline void dc_perturb_i8_chw_v1(const int8_t *w, int8_t *wp, int8_t *wm, const int32_t *noise_mul, uint32_t ch_in,
                                        uint32_t kh, uint32_t kw, uint32_t seed, uint32_t start, uint32_t count,
                                        uint32_t shift, float eps_scale) {
  /* QW exp7 step 3 version (43-51 cycles per element for both signs); kept as the reference of the step-5b rewrite */
  const uint32_t K = ch_in * kh * kw;
  uint32_t co = start / K;
  uint32_t rem = start - co * K;          /* CHW: rem = (ci*kh + ky)*kw + kx */
  uint32_t kx = rem % kw;
  uint32_t t = rem / kw;
  uint32_t ky = t % kh;
  uint32_t ci = t / kh;
  uint32_t j = co * K + (ky * kw + kx) * ch_in + ci;
  int32_t mul = 0;
  if (count) {
    mul = noise_mul[co];
    if (eps_scale != 1.0f)
      mul = (int32_t)lrintf((float)mul * eps_scale);
  }
  uint32_t rng = seed * 1664525u + 1013904223u;
  const int32_t rounding = (shift > 0u) ? (int32_t)(1u << (shift - 1u)) : 0;
  uint32_t i = start;
  const uint32_t end = start + count;
  while (i < end) {
    rng = dc_xorshift32(rng);
    uint32_t bits = rng;
    uint32_t run_len = end - i;
    if (run_len > 32u)
      run_len = 32u;
    for (uint32_t k = 0; k < run_len; k++, i++) {
      const int32_t z = (bits & 1u) ? 1 : -1;
      const int32_t prod = z * mul;
      const int32_t noise_p = (prod + rounding) >> (int32_t)shift;
      const int32_t noise_m = (rounding - prod) >> (int32_t)shift;
      const int32_t w_val = (int32_t)w[i];
      wp[j] = (int8_t)DC_CLAMP(w_val + noise_p, -127, 127);
      wm[j] = (int8_t)DC_CLAMP(w_val + noise_m, -127, 127);
      bits >>= 1;
      /* advance (co, ci, ky, kx) in CHW order; j follows in HWC */
      if (++kx < kw) {
        j += ch_in;
        continue;
      }
      kx = 0;
      if (++ky == kh) {
        ky = 0;
        if (++ci == ch_in) {
          ci = 0;
          if (i + 1u < end) {
            mul = noise_mul[++co];
            if (eps_scale != 1.0f)
              mul = (int32_t)lrintf((float)mul * eps_scale);
          } else {
            ++co;
          }
        }
      }
      j = co * K + ky * kw * ch_in + ci;
    }
  }
}

/* QW (dual_pass_fusion exp7, step 5b): same contract and bit-identical result as dc_perturb_i8_chw_v1, cheaper.
 *  - z is +-1, so per output channel there are only two noise values: up = (mul + rounding) >> shift and
 *    dn = (rounding - mul) >> shift; z = +1 gives (+pass, -pass) = (up, dn), z = -1 gives (dn, up). Computed once per channel; the noise bit
 *    selects branch-free (xor mask) -- no multiply, no shift per element.
 *  - within one (co, ci) row the CHW chunk is kh*kw contiguous elements whose HWC positions advance by ch_in, for
 *    1xK and Kx1 kernels alike. The element loop runs over RUNS of min(bits left in the draw, row left, chunk left)
 *    elements, so its body has no branch (a hardware loop on RI5CY); row / channel / draw changes happen between runs.
 *  - out of line (noinline): inlined into the conv kernel it ran short of registers and reloaded its pointers from
 *    the stack every element (step-5b disassembly). -- QW */
static inline __attribute__((noinline)) void dc_perturb_i8_chw(const int8_t *w, int8_t *wp, int8_t *wm,
                                                        const int32_t *noise_mul, uint32_t ch_in, uint32_t kh,
                                                        uint32_t kw, uint32_t seed, uint32_t start, uint32_t count,
                                                        uint32_t shift, float eps_scale) {
  if (!count)
    return;
  const uint32_t taps = kh * kw;
  const uint32_t K = ch_in * taps;
  uint32_t co = start / K;
  uint32_t rem = start - co * K; /* CHW: rem = ci*taps + tap */
  uint32_t ci = rem / taps;
  uint32_t tap = rem - ci * taps;
  const int32_t rounding = (shift > 0u) ? (int32_t)(1u << (shift - 1u)) : 0;
  int32_t mul = noise_mul[co];
  if (eps_scale != 1.0f)
    mul = (int32_t)lrintf((float)mul * eps_scale);
  int32_t up = (mul + rounding) >> (int32_t)shift;
  int32_t dn = (rounding - mul) >> (int32_t)shift;
  const int8_t *src = w + start;
  int8_t *pWPlus = wp + co * K + tap * ch_in + ci;
  int8_t *pWMinus = wm + co * K + tap * ch_in + ci;
  uint32_t rng = seed * 1664525u + 1013904223u;
  uint32_t bits = 0, bits_left = 0, remaining = count;
  while (remaining) {
    if (bits_left == 0u) {
      rng = dc_xorshift32(rng);
      bits = rng;
      bits_left = 32u;
    }
    uint32_t run_len = taps - tap;
    if (run_len > bits_left)
      run_len = bits_left;
    if (run_len > remaining)
      run_len = remaining;
    const int32_t up_xor_dn = up ^ dn;
    for (uint32_t k = 0; k < run_len; k++) {
      const int32_t w_val = (int32_t)*src++;
      const int32_t noise_p = dn ^ (up_xor_dn & -(int32_t)(bits & 1u));
      const int32_t noise_m = noise_p ^ up_xor_dn;
      *pWPlus = (int8_t)DC_CLAMP(w_val + noise_p, -127, 127);
      *pWMinus = (int8_t)DC_CLAMP(w_val + noise_m, -127, 127);
      pWPlus += ch_in;
      pWMinus += ch_in;
      bits >>= 1;
    }
    remaining -= run_len;
    bits_left -= run_len;
    tap += run_len;
    if (tap == taps && remaining) { /* next (co, ci) row */
      tap = 0;
      if (++ci == ch_in) {
        ci = 0;
        ++co;
        mul = noise_mul[co];
        if (eps_scale != 1.0f)
          mul = (int32_t)lrintf((float)mul * eps_scale);
        up = (mul + rounding) >> (int32_t)shift;
        dn = (rounding - mul) >> (int32_t)shift;
      }
      pWPlus = wp + co * K + ci;
      pWMinus = wm + co * K + ci;
    }
  }
}

/* int32 requant-add: same stream shape, NO clamp, like ApplyPerturbQuantRademacher_i32. */
static inline void dc_perturb_i32(const int32_t *b, int32_t *bp, int32_t *bm, const int32_t *noise_mul,
                                  uint32_t channel_width, uint32_t seed, uint32_t start, uint32_t count,
                                  uint32_t shift, float eps_scale) {
  uint32_t rng = seed * 1664525u + 1013904223u;
  const int32_t rounding = (shift > 0u) ? (int32_t)(1u << (shift - 1u)) : 0;
  uint32_t i = start;
  const uint32_t end = start + count;
  while (i < end) {
    rng = dc_xorshift32(rng);
    uint32_t bits = rng;
    uint32_t run_len = end - i;
    if (run_len > 32u)
      run_len = 32u;
    for (uint32_t k = 0; k < run_len; k++, i++) {
      const int32_t z = (bits & 1u) ? 1 : -1;
      int32_t mul = noise_mul[i / channel_width];
      if (eps_scale != 1.0f)
        mul = (int32_t)lrintf((float)mul * eps_scale);
      const int32_t prod = z * mul;
      bp[i] = b[i] + ((prod + rounding) >> (int32_t)shift);
      bm[i] = b[i] + ((rounding - prod) >> (int32_t)shift);
      bits >>= 1;
    }
  }
}

/* QW (exp11, 2026-10-23): SINGLE-sign counterparts of dc_perturb_i8_chw / dc_perturb_i32 for the single-pass
 * in-kernel-perturbation conv (PerturbConv4x2.c, the unfused reference with every non-fusion optimisation). Same
 * stream, chunking, rounding and clamp; `sign` = perturbation_sign (1: +eps, 0: -eps). Per channel the noise is one of
 * two values (up for z = +1, dn for z = -1); for the - pass they swap. dc_perturb1_i8_chw is dc_perturb_i8_chw with
 * one output; bit-identical to ApplyPerturbQuantRademacher_CHW of that sign. -- QW */
static inline __attribute__((noinline)) void dc_perturb1_i8_chw(const int8_t *w, int8_t *wo, const int32_t *noise_mul,
                                                         uint32_t ch_in, uint32_t kh, uint32_t kw, uint32_t seed,
                                                         uint32_t start, uint32_t count, uint32_t shift,
                                                         float eps_scale, uint32_t sign) {
  if (!count)
    return;
  const uint32_t taps = kh * kw;
  const uint32_t K = ch_in * taps;
  uint32_t co = start / K;
  uint32_t rem = start - co * K; /* CHW: rem = ci*taps + tap */
  uint32_t ci = rem / taps;
  uint32_t tap = rem - ci * taps;
  const int32_t rounding = (shift > 0u) ? (int32_t)(1u << (shift - 1u)) : 0;
  int32_t mul = noise_mul[co];
  if (eps_scale != 1.0f)
    mul = (int32_t)lrintf((float)mul * eps_scale);
  int32_t up = (mul + rounding) >> (int32_t)shift;
  int32_t dn = (rounding - mul) >> (int32_t)shift;
  if (!sign) { /* - pass: z = +1 gets dn, z = -1 gets up */
    const int32_t t = up;
    up = dn;
    dn = t;
  }
  const int8_t *src = w + start;
  int8_t *pW = wo + co * K + tap * ch_in + ci;
  uint32_t rng = seed * 1664525u + 1013904223u;
  uint32_t bits = 0, bits_left = 0, remaining = count;
  while (remaining) {
    if (bits_left == 0u) {
      rng = dc_xorshift32(rng);
      bits = rng;
      bits_left = 32u;
    }
    uint32_t run_len = taps - tap;
    if (run_len > bits_left)
      run_len = bits_left;
    if (run_len > remaining)
      run_len = remaining;
    const int32_t up_xor_dn = up ^ dn;
    for (uint32_t k = 0; k < run_len; k++) {
      const int32_t w_val = (int32_t)*src++;
      const int32_t noise = dn ^ (up_xor_dn & -(int32_t)(bits & 1u));
      *pW = (int8_t)DC_CLAMP(w_val + noise, -127, 127);
      pW += ch_in;
      bits >>= 1;
    }
    remaining -= run_len;
    bits_left -= run_len;
    tap += run_len;
    if (tap == taps && remaining) { /* next (co, ci) row */
      tap = 0;
      if (++ci == ch_in) {
        ci = 0;
        ++co;
        mul = noise_mul[co];
        if (eps_scale != 1.0f)
          mul = (int32_t)lrintf((float)mul * eps_scale);
        up = (mul + rounding) >> (int32_t)shift;
        dn = (rounding - mul) >> (int32_t)shift;
        if (!sign) {
          const int32_t t = up;
          up = dn;
          dn = t;
        }
      }
      pW = wo + co * K + ci;
    }
  }
}
static inline void dc_perturb1_i32(const int32_t *b, int32_t *bo, const int32_t *noise_mul, uint32_t channel_width,
                                   uint32_t seed, uint32_t start, uint32_t count, uint32_t shift, float eps_scale,
                                   uint32_t sign) {
  uint32_t rng = seed * 1664525u + 1013904223u;
  const int32_t rounding = (shift > 0u) ? (int32_t)(1u << (shift - 1u)) : 0;
  uint32_t i = start;
  const uint32_t end = start + count;
  while (i < end) {
    rng = dc_xorshift32(rng);
    uint32_t bits = rng;
    uint32_t run_len = end - i;
    if (run_len > 32u)
      run_len = 32u;
    for (uint32_t k = 0; k < run_len; k++, i++) {
      const int32_t z = (bits & 1u) ? 1 : -1;
      int32_t mul = noise_mul[i / channel_width];
      if (eps_scale != 1.0f)
        mul = (int32_t)lrintf((float)mul * eps_scale);
      const int32_t prod = z * mul;
      bo[i] = b[i] + (sign ? ((prod + rounding) >> (int32_t)shift) : ((rounding - prod) >> (int32_t)shift));
      bits >>= 1;
    }
  }
}

#endif
