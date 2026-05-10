/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Untiled-L3 baseline: redirect cluster-L1 allocations to a static L2 arena
 * — but ONLY for allocations that don't fit in the real L1 heap.
 *
 * Activated by `-DDEEPLOY_L1_AS_L2 -DDEEPLOY_FAKE_L1_SIZE=N` plus the linker
 * flags `-Wl,--wrap=pi_cl_l1_malloc -Wl,--wrap=pi_cl_l1_free`.
 *
 * Coexistence with the SDK
 * ------------------------
 * The wrap intercepts EVERY pi_cl_l1_malloc call site, including SDK-internal
 * ones (cluster init, driver scratch, etc.). A first version that always
 * served from the static arena broke the cluster on small models — likely
 * because SDK code received a pointer outside the real L1 region and either
 * its own bookkeeping went wrong or a downstream API rejected it.
 *
 * Mitigation: try the SDK's real L1 allocator first via the linker's
 * `__real_*` symbols. If that succeeds, hand the SDK pointer back. Only when
 * the request is too big for real L1 (the case we're here for: an oversized
 * MEMORYARENA_L1) fall through to the static FC-L2 arena. Free mirrors the
 * decision by checking whether the pointer falls inside our arena.
 *
 * Trade-off: kernels that were tuned for L1's 1-cycle access see L2 latency
 * (~7 cycles) for the buffers served from the fake arena. Cycles under this
 * mode are NOT silicon-representative — the mode exists to give a per-op
 * single-tile latency baseline against the existing tiled-L3 path.
 */

#include <stdint.h>

#ifdef DEEPLOY_L1_AS_L2

#ifndef DEEPLOY_FAKE_L1_SIZE
#error "DEEPLOY_L1_AS_L2 requires -DDEEPLOY_FAKE_L1_SIZE=<bytes>"
#endif

#include "pmsis.h"

/* Linker-provided originals (--wrap=foo exposes __real_foo). */
extern void *__real_pi_cl_l1_malloc(struct pi_device *device, uint32_t size);
extern void __real_pi_cl_l1_free(struct pi_device *device, void *chunk, int size);

/* Static arena in FC L2 — sized at compile time. PI_L2 maps to the SDK's L2
 * shared region (0x1C010000-0x1C200000 on Siracusa, ~1.94 MB). Aligned to 8
 * to satisfy any kernel that reinterprets float32_t / pulpv2 v2f16 buffers. */
__attribute__((aligned(8)))
PI_L2 static uint8_t deeploy_fake_l1_arena[DEEPLOY_FAKE_L1_SIZE];
static uint32_t deeploy_fake_l1_offset = 0;

static inline int in_fake_arena(const void *p) {
  return (const uint8_t *)p >= deeploy_fake_l1_arena
         && (const uint8_t *)p < deeploy_fake_l1_arena + DEEPLOY_FAKE_L1_SIZE;
}

void *__wrap_pi_cl_l1_malloc(struct pi_device *device, uint32_t size) {
  /* Try real L1 first — any small SDK / Deeploy alloc that fits stays in
   * real L1, so SDK bookkeeping and L1-tuned kernels are unaffected. */
  void *p = __real_pi_cl_l1_malloc(device, size);
  if (p != (void *)0) {
    return p;
  }
  /* Real L1 exhausted (or request bigger than L1 heap). Serve from FC-L2
   * arena: the only legitimate caller here is Deeploy's MEMORYARENA_L1
   * for a model whose peak L1 working set exceeds 256 KB. */
  uint32_t aligned = (size + 7u) & ~7u;
  if (deeploy_fake_l1_offset + aligned > DEEPLOY_FAKE_L1_SIZE) {
    return (void *)0;
  }
  void *q = (void *)&deeploy_fake_l1_arena[deeploy_fake_l1_offset];
  deeploy_fake_l1_offset += aligned;
  return q;
}

void __wrap_pi_cl_l1_free(struct pi_device *device, void *chunk, int size) {
  if (!in_fake_arena(chunk)) {
    __real_pi_cl_l1_free(device, chunk, size);
    return;
  }
  /* Bump-allocator rewind for arena pointers. Safe under LIFO free order
   * (the only pattern Deeploy's harness uses); a non-LIFO free is a no-op
   * and leaks until the next full reset. */
  uint32_t aligned = ((uint32_t)size + 7u) & ~7u;
  if (deeploy_fake_l1_offset >= aligned) {
    uint8_t *expected = deeploy_fake_l1_arena + deeploy_fake_l1_offset - aligned;
    if ((uint8_t *)chunk == expected) {
      deeploy_fake_l1_offset -= aligned;
    }
  }
}

#endif /* DEEPLOY_L1_AS_L2 */
