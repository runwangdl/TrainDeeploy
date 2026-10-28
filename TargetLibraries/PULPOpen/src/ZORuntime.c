/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * ZO (MeZO) runtime perturb controls — single definition, linked into every binary. -- QW
 * See ZORuntime.h. Neutral defaults reproduce the baked-attr (inference) behavior.
 */
#include "kernel/ZORuntime.h"

uint32_t  perturbation_sign        = 1;    // +eps
uint32_t  perturb_seed_base        = 0;    // no per-step seed offset
float32_t perturb_eps_override     = 0.0f; // unused until enabled
uint32_t  perturb_eps_use_override = 0;    // use baked ${eps}
float32_t perturb_eps_baked        = 1.0f; // export-time eps baked in the RQSPerturb mul (set by the ZO runner) -- QW
volatile uint32_t zo_weight_version = 0;   // -- QW exp21c: bumped by the harness whenever the weights change (opt H)

// RW 2026-10-13: ECO error feedback state (see ZORuntime.h)
uint32_t  zo_eco_on   = 0;
uint32_t  zo_eco_sr   = 0;
uint32_t  zo_eco_step = 0;
float32_t zo_eco_g    = 0.0f;
float32_t zo_eco_lr   = 0.0f;
float32_t zo_eco_beta = 0.9f;
float32_t *zo_eco_mom[ZO_ECO_MAX_TENSORS];
