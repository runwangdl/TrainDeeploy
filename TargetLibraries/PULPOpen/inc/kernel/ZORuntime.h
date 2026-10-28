/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * ZO (MeZO) runtime perturb controls. -- QW
 * Mutable globals the ZO runner (C main) sets before each perturbed forward / update,
 * so that the Perturb* kernel templates can drive +eps/-eps (perturbation_sign),
 * per-step seeds (perturb_seed_base), and the runtime update coefficient
 * (perturb_eps_override). Defaults are NEUTRAL: sign=+1, seed_base=0, override off ->
 * the templates fall back to the per-node baked seed/eps (inference / isolated-kernel
 * behavior is unchanged).
 */
#ifndef __DEEPLOY_ZO_RUNTIME_HEADER_
#define __DEEPLOY_ZO_RUNTIME_HEADER_

#include "DeeployPULPMath.h"

// +eps (dir=1) / -eps (dir=0), shared across all perturb nodes in a pass.
extern uint32_t perturbation_sign;
// Per-step seed base added onto each node's baked seed:  eff_seed = baked_seed + perturb_seed_base.
extern uint32_t perturb_seed_base;
// Runtime override for the perturb magnitude (used for the ZO update coefficient |lr*g_proj|).
extern float32_t perturb_eps_override;
// When non-zero, kernels use perturb_eps_override instead of the baked ${eps}.
extern uint32_t perturb_eps_use_override;
// The baked export-time eps (the magnitude encoded in the RQSPerturb integer `mul` vectors). The ZO
// runner sets it to ZO_EPS so integer perturb kernels can scale mul by (override/baked) when the
// override is active (the zo_update coefficient -lr*g_proj). Default 1.0 = no scaling. -- QW
extern float32_t perturb_eps_baked;
// -- QW exp21c (NE16 prep hoisting, opt H): bumped by the ZO harness every time the trainable weights change (after
//    each zo_update run, after a resume). Kernels that cache work derived from the weights (NE16DualPrep.c: the
//    perturbed+encoded weight and the perturbed bias of an update step) key their cache on it together with the
//    seeds and the eps scale. Default 0; a harness that never bumps it must not change the weights between two runs
//    of the training graph that use the same perturb_seed_base. -- QW exp21c
extern volatile uint32_t zo_weight_version;


/* RW 2026-10-13: ECO error feedback for (quantized) ZO, arXiv 2601.22101, memory-free form, as in Qiwen's host
 * simulation Onnx4Deeploy/QZO_exp/exp15_QZO_eco/run_eco.py. Active only in the update pass (override on) with
 * zo_eco_on != 0. Per trainable tensor k (= the update node's idx) zo_eco_mom[k] is an fp32 momentum buffer.
 *   int8 / int32 tensors (scale s):  m~ = b m + (1-b) g z;  q = w - lr m~ / s;  w' = clamp(round(q));
 *                                    e = (q - w') s;  m = m~ + (1/lr)(1 - 1/b) e
 *   fp32 tensors:                    m = b m + (1-b) g z;  w' = w - lr m
 * round = nearest-even, or floor(q + U) (stochastic, unbiased) when zo_eco_sr != 0. */
extern uint32_t zo_eco_on;
extern uint32_t zo_eco_sr;
extern uint32_t zo_eco_step;     // SR noise stream advances per update step
extern float32_t zo_eco_g;       // projected gradient g_proj of this step
extern float32_t zo_eco_lr;
extern float32_t zo_eco_beta;
#define ZO_ECO_MAX_TENSORS 64
extern float32_t *zo_eco_mom[ZO_ECO_MAX_TENSORS];

void ApplyECOQuantRademacher_i8(const int8_t *w, int8_t *w_dest, const int32_t *M, int32_t S, uint32_t channel_width,
                                uint32_t seed, uint32_t size, uint32_t start_offset, uint32_t dir, float32_t *mom);
void ApplyECOQuantRademacher_i32(const int32_t *w, int32_t *w_dest, const int32_t *M, int32_t S,
                                 uint32_t channel_width, uint32_t seed, uint32_t size, uint32_t start_offset,
                                 uint32_t dir, float32_t *mom);
void ApplyECORademacher_fp32(const float32_t *w, float32_t *w_dest, uint32_t seed, uint32_t dir, uint32_t size,
                             float32_t *mom);

#endif //__DEEPLOY_ZO_RUNTIME_HEADER_
