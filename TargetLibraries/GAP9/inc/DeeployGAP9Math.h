/*
 * SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_HEADER_
#define __DEEPLOY_MATH_HEADER_

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BEGIN_SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (pi_core_id() == 8 || pi_core_id() == 0)

#include "DeeployBasicMath.h"

// RW: Declare the GAP9-native float kernels (defined in GAP9/src/). GAP9 has
// its own kernel implementations with signatures that differ from PULPOpen's,
// so it cannot include PULPOpen's kernel headers. Without these prototypes the
// generated TrainingNetwork.c called the high-arity gradient kernels (>8 args,
// hence stack-passed) with implicit (int) declarations, mismatching the ABI and
// corrupting memory after training (a gvsoc "not indexable" abort).
#include "kernel/GAP9Kernels.h"

#include "dory_dma.h"
#include "dory_mem.h"

#include "pmsis.h"

// -- QW: the generated ZO training network references the perturb kernels + ZO runtime
//    globals (perturbation_sign / perturb_seed_base / perturb_eps_*). DeeployPULPMath.h
//    pulls these in for Siracusa; mirror it here so the GAP9-generated network compiles.
//    PULPOpen/inc is PUBLIC on deeploygap9, so these resolve. -- QW
#include "kernel/RandomNoise.h"
#include "kernel/ZORuntime.h"
#include "kernel/Im2ColU8.h" // -- QW (dual_pass_fusion exp6): NE16 im2col route, kernel in ../PULPOpen/src
#include "kernel/DualConv.h" // -- QW (exp6): exp5 fused cluster kernel on GAP9 (baseline C_cl); without this
                             //    prototype the call was implicit-int and crashed (float arg promoted) -- QW
#include "kernel/FlatConv.h" // -- QW (exp6)
#include "kernel/Conv4x2.h"  // -- QW (exp7a step 2a)
#include "kernel/DualConv2In.h" // -- QW (exp7 step 3)
#include "kernel/PerturbConv4x2.h" // -- QW (exp11): single-pass in-kernel-perturbation conv

/* -- QW (dual_pass_fusion exp6, task 2): cluster-side preparation for the FUSED dual-pass conv on NE16.
 * Implementation and derivation: TargetLibraries/GAP9/src/NE16DualPrep.c */
void PULP_NE16DualPerturbEncode(const int8_t *pW, const int32_t *pWMul, uint8_t *pWEnc, int8_t *pScratch,
                                uint32_t seed_w_base, uint32_t tile_seed_xor, uint32_t log2D_w,
                                float eps_scale, uint32_t num_cores, uint32_t ch_out, uint32_t K,
                                uint32_t sign_sel, /* exp7: 0 stacked [W+;W-], 1 W+, 2 W- */
                                uint32_t hwc, uint32_t ch_in, uint32_t kh, uint32_t kw); /* exp7: rows in (ky,kx,ci) order */
void PULP_NE16DualBias(const int8_t *pW, const int32_t *pWMul, const int32_t *pAdd, const int32_t *pBMul,
                       const int32_t *pMul, int32_t *pAddOut, int8_t *pScratch, uint32_t seed_w_base,
                       uint32_t seed_b_base, uint32_t tile_seed_xor, uint32_t log2D_w, uint32_t log2D_b,
                       float eps_scale, uint32_t num_cores, uint32_t ch_out, uint32_t K, int32_t input_offset,
                       uint32_t sign_sel); /* exp7: 0 stacked, 1 +, 2 - */

/* -- QW (dual_pass_fusion exp7 step 7): both ZO passes of a 1xK / Kx1 conv on NE16 in ONE node, SDK-style pipeline --
 * word-wise HWC column per output row, double-buffered against NE16 via its 2-deep job queue. Runs on the cluster
 * master core (NE16 engine), forks the workers for the columns. TargetLibraries/GAP9/src/NE16DualConv2in.c */
void PULP_NE16DualConv2in(const int8_t *pInP, const int8_t *pInM, const uint8_t *pWEnc2, const int32_t *pMul,
                          const int32_t *pAdd2, int8_t *pOut, uint8_t *pColBuf, uint32_t num_cores, uint32_t conf0,
                          uint16_t dim_in_y, uint16_t dim_in_x, uint16_t ch_in, uint16_t dim_out_y, uint16_t dim_out_x,
                          uint16_t ch_out, uint16_t dim_kernel_y, uint16_t dim_kernel_x, uint16_t pad_top,
                          uint16_t pad_bottom, uint16_t pad_left, uint16_t pad_right, uint16_t stride_y,
                          uint16_t stride_x, int32_t weight_offset, uint32_t pipelined, uint32_t job_w,
                          uint32_t in_stride, int32_t input_offset); /* exp8: ch_in, or 2*ch_in for a stacked [x+ | x-] input (pInM = pInP + ch_in) */

/* -- QW (exp16c / blocker 1b): device-side NE16 bit-serial weight encoder. Declared here (rather
 * than in a pulp-nnx header) so TargetLibraries/third_party/pulp-nnx stays pristine -- this is a
 * plain C kernel, not an ISA change. Implementation: TargetLibraries/GAP9/src/NE16WeightEncode.c */
void NE16WeightEncode_i8_u8(const int8_t *__restrict__ src, uint8_t *__restrict__ dst,
                            const uint32_t cout, const uint32_t cin, const uint32_t taps,
                            const uint32_t row_start, const uint32_t row_count);

/* -- QW (exp16c phase 4 / BLOCKER 3): per-output-channel bias correction for signed activations
 * fed to NE16 as x+128. See TargetLibraries/GAP9/src/NE16WeightEncode.c for the derivation. */
void NE16SignedInputBias_i32(const int8_t *__restrict__ weight, const int32_t *__restrict__ mul,
                             const int32_t *__restrict__ add, int32_t *__restrict__ out,
                             const uint32_t cout, const uint32_t cinTaps, const int32_t offset,
                             const uint32_t co_start, const uint32_t co_count);

#endif // __DEEPLOY_MATH_HEADER_
