/* -- QW exp21c: host check of the DEVICE source TargetLibraries/GAP9/src/NE16WeightEncode.c: the word-wise encoder
 *    (NE16WeightEncode_i8_u8_t1) must give the same bytes as the bit loop (NE16WeightEncode_i8_u8). Build:
 *    cc -O2 -Istub test_device_src.c ../../../../../../../../TargetLibraries/GAP9/src/NE16WeightEncode.c */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void NE16WeightEncode_i8_u8(const int8_t *src, uint8_t *dst, const uint32_t cout, const uint32_t cin,
                            const uint32_t taps, const uint32_t row_start, const uint32_t row_count);
void NE16WeightEncode_i8_u8_t1(const int8_t *src, uint8_t *dst, const uint32_t cout, const uint32_t cin,
                               const uint32_t row_start, const uint32_t row_count);
int main(void) {
  const uint32_t Ks[] = {1, 4, 15, 16, 17, 20, 33, 112, 128, 224};
  int fails = 0; long tot = 0;
  srand(1);
  for (int ki = 0; ki < 10; ki++)
    for (int rep = 0; rep < 300; rep++) {
      uint32_t K = Ks[ki], rows = 1 + rand() % 64, cm = (K + 15) / 16;
      int8_t *src = aligned_alloc(16, ((rows * K + 15) / 16) * 16 + 16);
      uint8_t *a = calloc(rows * cm * 16, 1), *b = calloc(rows * cm * 16, 1);
      for (uint32_t i = 0; i < rows * K; i++) {
        int m = rand() % 4;
        src[i] = m == 0 ? (int8_t)(rand() % 256 - 128) : m == 1 ? 127 : m == 2 ? -128 : -127;
        if (rep % 7 == 0) src[i] = (int8_t)(rand() % 256 - 128);
      }
      NE16WeightEncode_i8_u8(src, a, rows, K, 1, 0, rows);
      uint32_t r0 = rand() % rows;
      NE16WeightEncode_i8_u8_t1(src, b, rows, K, 0, r0);          /* split rows like the per-core shares */
      NE16WeightEncode_i8_u8_t1(src, b, rows, K, r0, rows - r0);
      if (memcmp(a, b, rows * cm * 16)) { fails++; if (fails < 5) printf("FAIL K=%u rows=%u\n", K, rows); }
      tot += rows * K; free(src); free(a); free(b);
    }
  for (int lane = 0; lane < 16; lane++)
    for (int v = -128; v < 128; v++) {
      int8_t s[16] __attribute__((aligned(4))); uint8_t a[16] __attribute__((aligned(4))), b[16] __attribute__((aligned(4)));
      for (int i = 0; i < 16; i++) s[i] = (int8_t)((i * 37 + v) & 0xff);
      s[lane] = (int8_t)v;
      NE16WeightEncode_i8_u8(s, a, 1, 16, 1, 0, 1); NE16WeightEncode_i8_u8_t1(s, b, 1, 16, 0, 1);
      if (memcmp(a, b, 16)) fails++;
    }
  printf("device-source check: random elements=%ld, exhaustive lane x value=4096, fails=%d -> %s\n", tot, fails,
         fails ? "MISMATCH" : "IDENTICAL");
  return fails != 0;
}
