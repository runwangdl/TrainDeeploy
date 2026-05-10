/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Untiled-L3 baseline: redirect cluster-L1 allocations to a static L2 arena.
 *
 * Activated by `-DDEEPLOY_L1_AS_L2 -DDEEPLOY_FAKE_L1_SIZE=N` plus the linker
 * flags `-Wl,--wrap=pi_cl_l1_malloc -Wl,--wrap=pi_cl_l1_free`. The wrap pair
 * intercepts the SDK's strong symbols so the deeploy-generated code keeps
 * calling pmsis_l1_malloc / pi_cl_l1_malloc as if it were targeting cluster
 * L1 — physically the bytes live in the FC L2 region instead, which on
 * Siracusa has ~1.94 MB of headroom (vs. ~256 KB for real cluster L1).
 *
 * Trade-off: kernels that were tuned for L1's 1-cycle access see L2 latency
 * (~7 cycles), so cycle counts under this mode are NOT representative of
 * silicon. The mode exists to provide a single-tile-per-tensor untiled-L3
 * baseline for *correctness* comparison against the tiled L3 path; cycle
 * realism for the same workload still requires the tiled run.
 */

#include <stdint.h>

#ifdef DEEPLOY_L1_AS_L2

#ifndef DEEPLOY_FAKE_L1_SIZE
#error "DEEPLOY_L1_AS_L2 requires -DDEEPLOY_FAKE_L1_SIZE=<bytes>"
#endif

#include "pmsis.h"

/* Static arena in FC L2 — sized at compile time. PI_L2 maps to the SDK's L2
 * shared region (0x1C010000-0x1C200000 on Siracusa, ~1.94 MB). Aligned to 8
 * to satisfy any kernel that reinterprets float32_t / pulpv2 v2f16 buffers. */
__attribute__((aligned(8)))
PI_L2 static uint8_t deeploy_fake_l1_arena[DEEPLOY_FAKE_L1_SIZE];

/* Bump pointer + free-list. The deeploy-generated harness allocates the L1
 * arena once at InitTrainingNetwork() time and frees it once at teardown,
 * so we don't need a real heap — a bump allocator with a single rewind on
 * full-arena free is sufficient and cheap.
 *
 * If the harness pattern ever changes (e.g. fine-grained per-op alloc/free),
 * swap this for an extern_alloc_t pool the way dory_mem.c does for L3. */
static uint32_t deeploy_fake_l1_offset = 0;

void *__wrap_pi_cl_l1_malloc(struct pi_device *device, uint32_t size) {
  (void)device;
  /* 8-byte alignment for every allocation so consecutive callers stay
   * aligned even when `size` is not a multiple of 8. */
  uint32_t aligned = (size + 7u) & ~7u;
  if (deeploy_fake_l1_offset + aligned > DEEPLOY_FAKE_L1_SIZE) {
    return (void *)0;
  }
  void *p = (void *)&deeploy_fake_l1_arena[deeploy_fake_l1_offset];
  deeploy_fake_l1_offset += aligned;
  return p;
}

void __wrap_pi_cl_l1_free(struct pi_device *device, void *chunk, int size) {
  (void)device;
  (void)chunk;
  (void)size;
  /* Bump-allocator semantics: per-block free is a no-op. The harness
   * frees the whole arena at teardown; we rewind there. */
  if (deeploy_fake_l1_offset >= (uint32_t)size) {
    deeploy_fake_l1_offset -= ((uint32_t)size + 7u) & ~7u;
  }
}

#endif /* DEEPLOY_L1_AS_L2 */
