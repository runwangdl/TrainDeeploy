/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* QW (exp16c / blocker 1b): device-side NE16 bit-serial weight encoder.
 *
 * NE16 reads weights bit-plane transposed. exp16a/exp16b let the HOST do that, which only works
 * while the weight is a gs.Constant. In QZO the conv weight is produced on device by
 * RQSPerturbRademacher, twice per ZO step, so the encoding has to happen on device too.
 *
 * This is a C port of `_weightEncode`
 * (Deeploy/Targets/NE16/TopologyOptimizationPasses/Passes.py:24) for the case the 1xK -> K
 * pointwise decomposition needs: H*W == 1 per tap, bits == 8, dense (non-depthwise).
 *
 * The host function does two things; only the second survives here:
 *
 *   1. `weight_offset = values.min()` then `values -= weight_offset`.  DROPPED -- exp16a hardwires
 *      weight_offset = -128 (a data-dependent offset cannot work for a weight that changes every
 *      step), so the offset step degenerates to the unsigned reinterpretation w_u = w + 128,
 *      i.e. a single XOR with 0x80.
 *   2. The bit-plane transpose.  A fixed, data-INDEPENDENT permutation, reproduced below.
 *
 * Layouts
 * -------
 *   src: (cout, cin, H, W) int8, NCHW-contiguous, with H*W == taps. Tap j of output channel co,
 *        input channel ci is src[(co*cin + ci)*taps + j] -- which holds for BOTH a 1xK kernel
 *        (H=1, W=K) and a Kx1 one (H=K, W=1), since the tap is the last index either way.
 *   dst: (taps*cout, cinMajor, 16) uint8, matching
 *        `concat_j _weightEncode(w[:,:,:,j:j+1] + 128)` along axis 0 -- exactly what exp16a's
 *        build_fixtures.py produces on the host.
 *
 * For one (row, cinMajor) pair: gather the 16 input-channel lanes of that subtile (zero-padded
 * where cin is not a multiple of 16, matching numpy's `np.pad(..., constant_values=0)` which runs
 * AFTER the offset), then emit 16 bytes where output byte `b*2 + k` collects bit `b` of source
 * lanes 8k..8k+7, LSB-first (`bitorder="little"`).
 *
 * Plain C, no NE16 registers, no pulp-nnx dependency -- nothing here is an ISA change. -- QW
 */

#include "DeeployGAP9Math.h"

#define NE16_CIN_SUBTILE 16
#define NE16_WEIGHT_BITS 8

void NE16WeightEncode_i8_u8(const int8_t *__restrict__ src, uint8_t *__restrict__ dst,
                            const uint32_t cout, const uint32_t cin, const uint32_t taps,
                            const uint32_t row_start, const uint32_t row_count) {

  const uint32_t cinMajor = (cin + NE16_CIN_SUBTILE - 1) / NE16_CIN_SUBTILE;
  const uint32_t dstRowBytes = cinMajor * NE16_WEIGHT_BITS * (NE16_CIN_SUBTILE / 8);

  for (uint32_t row = row_start; row < row_start + row_count; row++) {
    // row == tap*cout + co  (taps-major, matching the host's concatenate along axis 0)
    const uint32_t tap = row / cout;
    const uint32_t co = row - tap * cout;
    const int8_t *srcCo = src + (uint32_t)co * cin * taps + tap;
    uint8_t *dstRow = dst + (uint32_t)row * dstRowBytes;

    for (uint32_t cm = 0; cm < cinMajor; cm++) {
      const uint32_t ciBase = cm * NE16_CIN_SUBTILE;

      for (uint32_t half = 0; half < NE16_CIN_SUBTILE / 8; half++) {
        // eight bit planes of this half-subtile: plane<b> collects bit b of its 8 lanes
        uint32_t plane0 = 0, plane1 = 0, plane2 = 0, plane3 = 0;
        uint32_t plane4 = 0, plane5 = 0, plane6 = 0, plane7 = 0;

        for (uint32_t lane = 0; lane < 8; lane++) {
          const uint32_t ci = ciBase + 8 * half + lane;
          // w_u = w + 128; lanes past cin are zero (NOT 128) -- the host pads after the offset
          const uint32_t w_u = (ci < cin) ? (uint32_t)((uint8_t)srcCo[ci * taps] ^ 0x80u) : 0u;
          plane0 |= ((w_u >> 0) & 1u) << lane;
          plane1 |= ((w_u >> 1) & 1u) << lane;
          plane2 |= ((w_u >> 2) & 1u) << lane;
          plane3 |= ((w_u >> 3) & 1u) << lane;
          plane4 |= ((w_u >> 4) & 1u) << lane;
          plane5 |= ((w_u >> 5) & 1u) << lane;
          plane6 |= ((w_u >> 6) & 1u) << lane;
          plane7 |= ((w_u >> 7) & 1u) << lane;
        }

        uint8_t *dstHalf = dstRow + cm * (NE16_WEIGHT_BITS * (NE16_CIN_SUBTILE / 8)) + half;
        dstHalf[0 * 2] = (uint8_t)plane0;
        dstHalf[1 * 2] = (uint8_t)plane1;
        dstHalf[2 * 2] = (uint8_t)plane2;
        dstHalf[3 * 2] = (uint8_t)plane3;
        dstHalf[4 * 2] = (uint8_t)plane4;
        dstHalf[5 * 2] = (uint8_t)plane5;
        dstHalf[6 * 2] = (uint8_t)plane6;
        dstHalf[7 * 2] = (uint8_t)plane7;
      }
    }
  }
}

/* -- QW exp21c (opt S): the same encoding for taps == 1 (the NE16 dual-prep case: one contiguous row of `cin` int8 per
 *    output row), word-wise instead of bit by bit. Per 16-lane subtile:
 *      1. four word loads (lanes 0-3, 4-7, 8-11, 12-15), XOR 0x80808080 (the +128 offset of every lane);
 *      2. one 8x8 bit-matrix transpose per 8 lanes (Hacker's Delight transpose8, 3 delta-swap stages on two words):
 *         with lo = lanes 0-3 and hi = lanes 4-7 loaded little-endian, the result holds plane b of the 8 lanes in byte
 *         b of (lo, hi), lane L at bit L -- exactly `plane<b>` of the loop above;
 *      3. interleave the two half-subtiles bytewise (dst byte b*2 + half) and store four words.
 *    About 7 instead of ~42 cycles per element per core (exp21c E_breakdown). Bit-identical: a fixed permutation, checked
 *    on the host against NE16WeightEncode_i8_u8 on 3.7 M random elements, every value in every lane, K in {1, 4, 15,
 *    16, 17, 20, 33, 112, 128, 224} (impl/scripts/enc_host_test). Partial subtiles (cin % 16 != 0, block 0: cin = 4)
 *    gather their lanes bytewise with the zero padding; unaligned src/dst/cin fall back to the bit loop. -- QW exp21c */
static inline void x21c_t8x8(uint32_t *lo, uint32_t *hi) {
  uint32_t x = *hi, y = *lo, t;
  t = (x ^ (x >> 7)) & 0x00AA00AAu;
  x = x ^ t ^ (t << 7);
  t = (y ^ (y >> 7)) & 0x00AA00AAu;
  y = y ^ t ^ (t << 7);
  t = (x ^ (x >> 14)) & 0x0000CCCCu;
  x = x ^ t ^ (t << 14);
  t = (y ^ (y >> 14)) & 0x0000CCCCu;
  y = y ^ t ^ (t << 14);
  t = (x & 0xF0F0F0F0u) | ((y >> 4) & 0x0F0F0F0Fu);
  y = ((x << 4) & 0xF0F0F0F0u) | (y & 0x0F0F0F0Fu);
  *hi = t;
  *lo = y;
}
static inline uint32_t x21c_zip(uint32_t a, uint32_t b) { /* bytes (a0, b0, a1, b1) from the low halves */
  a = (a & 0xFFu) | ((a & 0xFF00u) << 8);
  b = (b & 0xFFu) | ((b & 0xFF00u) << 8);
  return a | (b << 8);
}

void NE16WeightEncode_i8_u8_t1(const int8_t *__restrict__ src, uint8_t *__restrict__ dst, const uint32_t cout,
                               const uint32_t cin, const uint32_t row_start, const uint32_t row_count) {
  if (((((uintptr_t)src) | ((uintptr_t)dst) | cin) & 3u) != 0u) { /* word loads/stores need alignment */
    NE16WeightEncode_i8_u8(src, dst, cout, cin, 1u, row_start, row_count);
    return;
  }
  const uint32_t cinMajor = (cin + NE16_CIN_SUBTILE - 1) / NE16_CIN_SUBTILE;
  const uint32_t cinFull = cin / NE16_CIN_SUBTILE;
  for (uint32_t row = row_start; row < row_start + row_count; row++) {
    const uint32_t *s = (const uint32_t *)(src + row * cin);
    uint32_t *d = (uint32_t *)(dst + row * cinMajor * 16u);
    for (uint32_t cm = 0; cm < cinMajor; cm++) {
      uint32_t a0, a1, b0, b1;
      if (cm < cinFull) {
        a0 = s[0] ^ 0x80808080u;
        a1 = s[1] ^ 0x80808080u;
        b0 = s[2] ^ 0x80808080u;
        b1 = s[3] ^ 0x80808080u;
      } else { /* partial subtile: lanes past cin are 0 (the host pads after the offset) */
        const int8_t *sb = (const int8_t *)s;
        const uint32_t n = cin - cm * NE16_CIN_SUBTILE;
        uint32_t w[4] = {0u, 0u, 0u, 0u};
        for (uint32_t i = 0; i < n; i++)
          w[i >> 2] |= ((uint32_t)((uint8_t)sb[i] ^ 0x80u)) << (8u * (i & 3u));
        a0 = w[0], a1 = w[1], b0 = w[2], b1 = w[3];
      }
      x21c_t8x8(&a0, &a1); /* a0: planes 0-3 of lanes 0-7, a1: planes 4-7 */
      x21c_t8x8(&b0, &b1); /* lanes 8-15 */
      d[0] = x21c_zip(a0, b0);
      d[1] = x21c_zip(a0 >> 16, b0 >> 16);
      d[2] = x21c_zip(a1, b1);
      d[3] = x21c_zip(a1 >> 16, b1 >> 16);
      d += 4;
      s += 4;
    }
  }
}

/* QW (exp16c phase 4 / BLOCKER 3): signed-activation bias correction.
 *
 * NE16 reads input activations as UNSIGNED. SpeechNet block 0's activation is genuinely signed
 * (range [-66, 127]), and CONFIG0 bit 26 -- what PR #183 calls `input_signed` -- is undecoded by
 * the hardware (ne16_regfile.cpp:200-231), so there is no register to flip.
 *
 * The standard fix: feed x_u = x + 128 and remove the induced term afterwards. Exactly,
 *
 *     sum_k w_k * x_k  =  sum_k w_k * (x_u,k - 128)  =  sum_k w_k * x_u,k  -  128 * sum_k w_k
 *
 * and `128 * sum_k w_k` is a per-OUTPUT-CHANNEL constant. It cannot be folded on the host, because
 * in QZO the weight is produced on device by RQSPerturbRademacher and changes every ZO step -- so
 * the correction is recomputed here, on device, from the same perturbed weight the encoder sees.
 *
 * Where it is applied matters. The RequantShift computes
 *
 *     out = (acc * mul + add) >> log2(div)
 *
 * so `add` is applied AFTER the multiply. Correcting `acc` by `-128*sum_w` is therefore equivalent
 * to correcting `add` by `-128 * sum_w * mul`:
 *
 *     out = ((acc_meas - 128*sum_w) * mul + add) >> s  =  (acc_meas * mul + [add - 128*sum_w*mul]) >> s
 *
 * Range: |128 * sum_w * mul| <= 128 * 127 * cin*H*W * max(mul). For block 0 that is
 * 128 * 508 * 454 ~= 3.0e7, comfortably inside int32 (checked against the real tensors before this
 * was written). A layer with a large `mul` and a wide receptive field could overflow; the parser
 * therefore records the shapes and this bound is documented rather than assumed.
 *
 * Parallelised over output channels, like the encoder. -- QW
 */
void NE16SignedInputBias_i32(const int8_t *__restrict__ weight, const int32_t *__restrict__ mul,
                             const int32_t *__restrict__ add, int32_t *__restrict__ out,
                             const uint32_t cout, const uint32_t cinTaps, const int32_t offset,
                             const uint32_t co_start, const uint32_t co_count) {
  (void)cout;
  for (uint32_t co = co_start; co < co_start + co_count; co++) {
    const int8_t *w = weight + (uint32_t)co * cinTaps;
    int32_t w_sum = 0;
    for (uint32_t i = 0; i < cinTaps; i++) {
      w_sum += (int32_t)w[i];
    }
    out[co] = add[co] - offset * w_sum * mul[co];
  }
}
