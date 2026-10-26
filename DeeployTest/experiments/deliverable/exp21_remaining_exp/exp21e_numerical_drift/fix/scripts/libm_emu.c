// exp21e/fix: bit-exact host emulation of the device's libm expf / logf (picolibc, fdlibm-derived float
// versions, as linked into the Siracusa image). Transliterated INSTRUCTION BY INSTRUCTION from the device
// disassembly (results/picolibc_expf_logf.dis) incl. every fused multiply-add the compiler emitted
// (RISC-V fmadd/fmsub/fnmsub.s -> C fmaf, which glibc rounds correctly). Constants = the words read from the
// image (.data at 0x1c02c620.., .rodata at 0x1c001390..). Build WITHOUT contraction:
//   gcc -O2 -ffp-contract=off -fno-fast-math -shared -fPIC -o libm_emu.so libm_emu.c -lm      -- QW exp21e
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline uint32_t fb(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static inline float bf(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }

// expf constants
#define HALF_0   bf(0x3f000000u)  /* halF[0] = 0.5  */
#define HALF_1   bf(0xbf000000u)  /* halF[1] = -0.5 */
#define LN2LO_0  bf(0x3717f7d1u)
#define LN2LO_1  bf(0xb717f7d1u)
#define LN2HI_0  bf(0x3f317180u)
#define LN2HI_1  bf(0xbf317180u)
#define HUGE_    bf(0x7149f2cau)  /* 1e30 */
#define ONE      bf(0x3f800000u)
#define INVLN2   bf(0x3fb8aa3bu)
#define NLN2HI   bf(0xbf317180u)  /* -ln2HI (used by the fused hi = x - t*ln2HI) */
#define LN2LO    bf(0x3717f7d1u)
#define P4       bf(0xb5ddea0eu)
#define P5       bf(0x3331bb4cu)
#define P3       bf(0x388ab355u)
#define P2       bf(0xbb360b61u)
#define P1       bf(0x3e2aaaabu)
#define TWO      bf(0x40000000u)
#define TWOM100  bf(0x0d800000u)
#define MTWO     bf(0xc0000000u)

float emu_expf(float x) {
  const uint32_t b = fb(x), hx = b & 0x7fffffffu;
  if (hx >= 0x7f800001u) return x + x;                    // NaN
  if (hx == 0x7f800000u) return ((int32_t)b >= 0) ? x : 0.0f;
  if ((int32_t)b >= 0x42B17218) return HUGE_VALF;         // __math_oflowf
  if (!((int32_t)b > -1 || hx < 0x42CFF1B6u)) return 0.0f; // __math_uflowf
  float hi = 0.0f, lo = 0.0f, xr;
  int32_t k;
  if (hx < 0x3EB17219u) {                                  // |x| <= 0.5 ln2
    const float t = x + HUGE_;
    if (!(!(ONE < t) || ((hx >> 26) > 12u))) return x + ONE; // tiny |x|
    k = 0;
    xr = x;
  } else {
    const uint32_t xsb = b >> 31;
    if (hx <= 0x3F851591u) {                               // |x| < 1.5 ln2
      hi = x - (xsb ? LN2HI_1 : LN2HI_0);
      lo = xsb ? LN2LO_1 : LN2LO_0;
      k = (int32_t)(xsb ^ 1u) - (int32_t)xsb;
    } else {
      const float kf = fmaf(x, INVLN2, xsb ? HALF_1 : HALF_0);
      k = (int32_t)kf;                                     // fcvt.w.s rtz
      const float t = (float)k;
      hi = fmaf(t, NLN2HI, x);
      lo = t * LN2LO;
    }
    xr = hi - lo;
  }
  const float t = xr * xr;
  float c = fmaf(t, P5, P4);
  c = fmaf(t, c, P3);
  c = fmaf(t, c, P2);
  c = fmaf(t, c, P1);
  c = fmaf(-t, c, xr);                                     // fnmsub: xr - t*c
  const float xc = xr * c;
  if (k == 0) {
    const float d = c + MTWO;
    const float q = xc / d;
    return (xr - q) + ONE;
  }
  const float d = TWO - c;
  const float q = xc / d;
  const float y = (hi - (lo - q)) + ONE;
  const uint32_t yb = fb(y);
  if (k >= -125) return bf(yb + ((uint32_t)k << 23));
  return bf(yb + ((uint32_t)k << 23) + 0x32000000u) * TWOM100;
}

// logf constants
#define TWO25   bf(0x4c000000u)
#define MONE    bf(0xbf800000u)
#define LG4     bf(0x3e638e29u)
#define LG6     bf(0x3e1cd04fu)
#define LG2     bf(0x3ecccccdu)
#define LG5     bf(0x3e3a3325u)
#define LG7     bf(0x3e178897u)
#define LG3     bf(0x3e924925u)
#define LG1     bf(0x3f2aaaabu)
#define NLN2_LO bf(0xb717f7d1u)
#define LN2_HI  bf(0x3f317180u)
#define HALF    bf(0x3f000000u)
#define LN2_LO  bf(0x3717f7d1u)
#define MTHIRD  bf(0xbeaaaaabu)

float emu_logf(float x) {
  uint32_t b = fb(x);
  if ((b & 0x7fffffffu) == 0) return -HUGE_VALF;           // __math_divzerof
  if ((int32_t)b < 0) return NAN;                          // __math_invalidf
  if ((b >> 23) >= 255u) return x + x;
  int32_t k0 = -127;
  if ((b >> 23) == 0) { x = x * TWO25; b = fb(x); k0 = -152; }
  const int32_t a3 = ((int32_t)b) >> 23;
  const uint32_t ix = b & 0x7fffffu;
  const uint32_t a6 = ix + 0x4afb20u;
  const uint32_t i = a6 & 0x800000u;
  const float xn = bf((i | ix) ^ 0x3f800000u);
  const int32_t k = k0 + a3 + (int32_t)(a6 >> 23);
  const float f = xn + MONE;
  if (((b + 15u) & 0x7ffff0u) == 0) {                      // |f| < 2^-20
    if (f == 0.0f) {
      if (k == 0) return 0.0f;
      const float dk = (float)k;
      return fmaf(dk, LN2_HI, dk * LN2_LO);
    }
    const float R = (f * f) * fmaf(f, MTHIRD, HALF);
    if (k == 0) return f - R;
    const float dk = (float)k;
    const float t1 = fmaf(dk, NLN2_LO, R);
    return fmaf(dk, LN2_HI, -(t1 - f));
  }
  const float s = f / (f + TWO);
  const float z = s * s, w = z * z;
  float a = fmaf(w, LG6, LG4);
  const float t1in = fmaf(w, a, LG2);
  const float dk = (float)k;
  float c = fmaf(w, LG7, LG5);
  c = fmaf(w, c, LG3);
  const int32_t ii = (int32_t)ix - 0x30a3d0, jj = 0x35c288 - (int32_t)ix;
  const float t1 = w * t1in;
  const float t2in = fmaf(w, c, LG1);
  const float t2 = z * t2in;
  const float R = t1 + t2;
  if ((ii | jj) > 0) {
    const float hfsq = f * (f * HALF);
    const float hR = hfsq + R;
    if (k == 0) return f - fmaf(-s, hR, hfsq);
    const float l = dk * LN2_LO;
    const float u = hfsq - fmaf(s, hR, l);
    return fmaf(dk, LN2_HI, -(u - f));
  }
  const float fR = f - R;
  if (k == 0) return fmaf(-s, fR, f);
  const float l = dk * NLN2_LO;
  const float u = fmaf(s, fR, l);
  return fmaf(dk, LN2_HI, -(u - f));
}

// vectorised helpers for ctypes
void emu_expf_v(const float *in, float *out, int n) { for (int i = 0; i < n; i++) out[i] = emu_expf(in[i]); }
void emu_logf_v(const float *in, float *out, int n) { for (int i = 0; i < n; i++) out[i] = emu_logf(in[i]); }

// ---------------------------------------------------------------------------------------------------------------
// GAP9 (gap9-sdk gcc 7.1.1 + newlib, Zfinx): __ieee754_expf is the SAME instruction-level computation as picolibc's
// expf above (identical constants and fused sites; only operand order of exactly-negated subtractions differs), so
// emu_expf is reused. __ieee754_logf differs in its fused sites (results/gap9_newlib_expf_logf.dis):
//   R = fma(t1in, w, t2)  (t1 not rounded separately),  i>0/k==0: f - (hfsq - s*(hfsq+R)) unfused, ...
float emu_logf_gap9(float x) {
  uint32_t b = fb(x);
  if ((b << 1) == 0) return -HUGE_VALF;                    // -two25/0
  if ((int32_t)b < 0) return NAN;                          // (x-x)/0
  if ((int32_t)b >= 0x7f800000) return x + x;
  int32_t kadj = 0;
  if ((int32_t)b < 0x800000) { x = x * TWO25; b = fb(x); kadj = -25; }
  int32_t k = (((int32_t)b) >> 23) - 127 + kadj;
  const uint32_t ix = b & 0x7fffffu;
  const uint32_t i = (0x4afb20u + ix) & 0x800000u;
  k += (int32_t)(i >> 23);
  const float xn = bf((i ^ 0x3f800000u) | ix);
  const float f = xn - ONE;
  if (((ix + 15u) & 0x7fffffu) <= 15u) {
    if (f == 0.0f) {
      if (k == 0) return 0.0f;
      const float dk = (float)k;
      return fmaf(dk, LN2_HI, dk * LN2_LO);
    }
    const float R = fmaf(-f, bf(0x3eaaaaabu), HALF) * (f * f);
    if (k == 0) return f - R;
    const float dk = (float)k;
    const float u = fmaf(-dk, LN2_LO, R) - f;
    return fmaf(dk, LN2_HI, -u);
  }
  const int32_t ii = (int32_t)ix - 0x30a3d0;
  const float dk = (float)k;
  const float s = f / (f + TWO);
  const float z = s * s, w = z * z;
  float t1in = fmaf(w, LG6, LG4);
  t1in = fmaf(t1in, w, LG2);
  float t2in = fmaf(w, LG7, LG5);
  t2in = fmaf(t2in, w, LG3);
  t2in = fmaf(t2in, w, LG1);
  const float t2 = t2in * z;
  const float R = fmaf(t1in, w, t2);
  const int32_t jj = 0x35c288 - (int32_t)ix;
  if ((jj | ii) > 0) {
    const float hfsq = (f * HALF) * f;
    const float sr = (R + hfsq) * s;
    if (k == 0) return f - (hfsq - sr);
    const float u = (hfsq - fmaf(dk, LN2_LO, sr)) - f;
    return fmaf(dk, LN2_HI, -u);
  }
  const float sr = (f - R) * s;
  if (k == 0) return f - sr;
  const float u = fmaf(-dk, LN2_LO, sr) - f;
  return fmaf(dk, LN2_HI, -u);
}
void emu_logf_gap9_v(const float *in, float *out, int n) { for (int i = 0; i < n; i++) out[i] = emu_logf_gap9(in[i]); }
