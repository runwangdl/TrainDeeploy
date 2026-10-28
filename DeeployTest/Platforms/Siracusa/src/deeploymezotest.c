/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* -- QW: ZO (MeZO) on-device training harness for Siracusa. Whole file added by QW. -- QW
 *
 * Adapted from deeploytraintest.c (the BP skeleton). The BP grad-buffer /
 * optimizer-copy machinery is stripped: the ZO update is performed IN-PLACE on
 * the shared weight constants by RunOptimizerNetwork (zo_update, 22
 * PerturbRademacher nodes, ZERO inputs / ZERO outputs), so there are no grad
 * buffers to zero, no lazy_reset_grad, and no weight/grad copies around the
 * optimizer dispatch.
 *
 * The ZO loop (per RUNNER_PLAN.md):
 *
 *   for update_step in [0, N_TRAIN_STEPS):
 *     acc = 0
 *     seed_base = update_step * ZO_Q            (q_i = 0 for ZO_Q == 1)
 *     for accum_step in [0, N_ACCUM_STEPS):
 *       mb = update_step*N_ACCUM_STEPS + accum_step
 *       load mini-batch mb into DeeployNetwork_inputs[0..NUM_DATA-1]
 *       perturbation_sign=1; perturb_seed_base=seed_base; perturb_eps_use_override=0
 *       RunTrainingNetwork();  Lp = outputs[0];  stored_loss_plus[mb]  = Lp
 *       perturbation_sign=0
 *       RunTrainingNetwork();  Lm = outputs[0];  stored_loss_minus[mb] = Lm
 *       acc += (Lp - Lm)                          (FP -> on cluster)
 *     g_proj = acc / (2 * ZO_EPS * N_ACCUM_STEPS) (FP -> on cluster)
 *     perturb_eps_use_override=1; perturb_eps_override = -ZO_LR*g_proj
 *     perturbation_sign=1; perturb_seed_base=seed_base
 *     RunOptimizerNetwork();                       (in-place on shared weights)
 *     perturb_eps_use_override=0
 *   compare stored_loss_plus  vs testLossPlusRef
 *   compare stored_loss_minus vs testLossMinusRef
 *
 * FPU caveat: the FC has no FPU. The scalar loss DeeployNetwork_outputs[0] is
 * read FPU-free (memcpy / ram_read). All float arithmetic ((Lp-Lm), g_proj,
 * -lr*g_proj, loss comparisons) is done in on-cluster helpers. The extern ZO
 * runtime globals (ZORuntime.h) are set on the FC BEFORE the cluster dispatch;
 * the perturb kernel reads them on the cluster.
 *
 * Compile-time constants (from CMake -D):
 *   N_TRAIN_STEPS            number of optimizer (weight-update) steps
 *   N_ACCUM_STEPS            number of mini-batches accumulated per update
 *   TRAINING_NUM_DATA_INPUTS inputs that change each mini-batch (data + labels)
 *   ZO_EPS                   perturbation epsilon
 *   ZO_LR                    learning rate
 *   ZO_Q                     number of random directions per step (q_i offset)
 *   ZO_SEED                  base seed (informational; baked into the graph)
 *
 * Reference comparison constants (emitted into testoutputs.h by the ZO codegen):
 *   N_LOSS_REFS              number of reference loss values
 *   testLossPlusRef[]        reference L+ per mini-batch
 *   testLossMinusRef[]       reference L- per mini-batch
 *   ZO_TOLERANCE_ABS         absolute comparison tolerance
 */

#include <math.h>    // -- QW
#include <stdint.h>  // -- QW
#include <string.h>  // -- QW

#include "CycleCounter.h"       // -- QW
#include "OptimizerNetwork.h"   // -- QW  (zo_update, DeeployOptNetwork_ prefix)
#include "TrainingNetwork.h"    // -- QW  (zo_train,  DeeployNetwork_ prefix)
#include "dory_mem.h"           // -- QW
#include "kernel/ZORuntime.h"   // -- QW  (perturbation_sign / perturb_seed_base / perturb_eps_(use_)override)
#include "pmsis.h"              // -- QW
#include "testinputs.h"         // -- QW
#include "testoutputs.h"        // -- QW

/* Helper: true when ptr is in L2 (CPU-accessible); false when in L3. -- QW */
#define IS_L2(ptr) ((uint32_t)(ptr) >= 0x10000000u)  // -- QW

/* -------------------------------------------------------------------------
 * Compile-time defaults — override via CMake target_compile_definitions -- QW
 * ---------------------------------------------------------------------- */

#ifndef N_TRAIN_STEPS            // -- QW
#define N_TRAIN_STEPS 1          // -- QW
#endif                           // -- QW

#ifndef N_ACCUM_STEPS            // -- QW
#define N_ACCUM_STEPS 1          // -- QW
#endif                           // -- QW

#ifndef TRAINING_NUM_DATA_INPUTS // -- QW
#define TRAINING_NUM_DATA_INPUTS 2 // -- QW
#endif                           // -- QW

#ifndef ZO_EPS                   // -- QW
#define ZO_EPS 0.01f             // -- QW
#endif                           // -- QW

#ifndef ZO_LR                    // -- QW
#define ZO_LR 3e-6f              // -- QW
#endif                           // -- QW

#ifndef ZO_Q                     // -- QW
#define ZO_Q 1                   // -- QW
#endif                           // -- QW

#ifndef ZO_SEED                  // -- QW
#define ZO_SEED 0                // -- QW
#endif                           // -- QW

#define MAINSTACKSIZE 12000      // -- QW
#define SLAVESTACKSIZE 3800      // -- QW

/* -------------------------------------------------------------------------
 * Cluster device -- QW
 * ---------------------------------------------------------------------- */

struct pi_device cluster_dev;    // -- QW

/* -------------------------------------------------------------------------
 * Cycle accumulators -- QW
 * ---------------------------------------------------------------------- */

/* QW: 64-bit cycle accumulators — a uint32 sum wraps on long rounds (e.g. the 200-epoch ZO
 * round is ~385 G cycles, ~90x past 2^32). getCycles() per call (~17.8 M) is safe; only the
 * running sum needs 64-bit. -- QW */
static unsigned long long g_train_cycles_acc = 0;  // -- QW
static unsigned long long g_opt_cycles_acc = 0;    // -- QW

/* -------------------------------------------------------------------------
 * Loss storage — one L+ and one L- per forward-pass pair (per mini-batch). -- QW
 * ---------------------------------------------------------------------- */

#define TOTAL_FWD_PASSES (N_TRAIN_STEPS * N_ACCUM_STEPS)  // -- QW
static float stored_loss_plus[TOTAL_FWD_PASSES];          // -- QW
static float stored_loss_minus[TOTAL_FWD_PASSES];         // -- QW

/* -------------------------------------------------------------------------
 * On-cluster ZO update-coefficient helper (FC has no FPU). -- QW
 *
 * Given the scalar-accumulated (sum over accum steps of L+ - L-) and the number
 * of accum steps, compute g_proj = acc / (2*ZO_EPS*N_ACCUM_STEPS) and write the
 * runtime perturb magnitude perturb_eps_override = -ZO_LR * g_proj. Runs on the
 * cluster where the FPU is available, then the FC dispatches RunOptimizerNetwork.
 * ---------------------------------------------------------------------- */

typedef struct {                 // -- QW
  uint32_t acc_bits;             // -- QW  raw bits of the FP accumulator (FC never does FP)
  uint32_t n_accum;              // -- QW  N_ACCUM_STEPS
  float *coeff_out;              // -- QW  receives -ZO_LR*g_proj (written into perturb_eps_override)
} ZOUpdateCoeffArgs;             // -- QW

static void ComputeZOUpdateCoeffOnCluster(void *args) {  // -- QW
  if (pi_core_id() != 0)         // -- QW
    return;                      // -- QW
  ZOUpdateCoeffArgs *a = (ZOUpdateCoeffArgs *)args;       // -- QW
  float acc; memcpy(&acc, &a->acc_bits, sizeof(acc));     // -- QW  reinterpret bits->float on cluster (has FPU)
  float denom = 2.0f * (float)ZO_EPS * (float)a->n_accum; // -- QW
  float g_proj = acc / denom;    // -- QW
  *a->coeff_out = -(float)ZO_LR * g_proj;                 // -- QW
}                                // -- QW

/* -------------------------------------------------------------------------
 * On-cluster (L+ - L-) accumulation helper (FC has no FPU). -- QW
 * Adds (Lp - Lm) onto *acc_out. Called once per accum step after both passes.
 * ---------------------------------------------------------------------- */

typedef struct {                 // -- QW
  uint32_t lp_bits;              // -- QW  raw bits (FC never does FP)
  uint32_t lm_bits;              // -- QW
  float *acc_out;                // -- QW
} ZODiffArgs;                    // -- QW

static void AccumulateZODiffOnCluster(void *args) {  // -- QW
  if (pi_core_id() != 0)         // -- QW
    return;                      // -- QW
  ZODiffArgs *a = (ZODiffArgs *)args;                // -- QW
  float lp, lm;                                      // -- QW  reinterpret on cluster (has FPU)
  memcpy(&lp, &a->lp_bits, sizeof(lp));              // -- QW
  memcpy(&lm, &a->lm_bits, sizeof(lm));              // -- QW
  *a->acc_out += (lp - lm);                          // -- QW
}                                // -- QW

/* -------------------------------------------------------------------------
 * Numerical comparison helper — run on cluster (FC has no FPU). -- QW
 * Mirrors deeploytraintest.c CompareLossesOnCluster, but reads ZO_TOLERANCE_ABS.
 * ---------------------------------------------------------------------- */

typedef struct {                 // -- QW
  float *computed;               // -- QW
  float *reference;              // -- QW
  uint32_t n;                    // -- QW
  uint32_t *err_count;           // -- QW
  const char *tag;               // -- QW
} ZOLossCompareArgs;             // -- QW

static void CompareLossesOnCluster(void *args) {  // -- QW
  if (pi_core_id() != 0)         // -- QW
    return;                      // -- QW
  ZOLossCompareArgs *a = (ZOLossCompareArgs *)args;  // -- QW
  float tol = ZO_TOLERANCE_ABS;  // -- QW  read on cluster — has FPU
  uint32_t errors = 0;           // -- QW
  for (uint32_t i = 0; i < a->n; i++) {  // -- QW
    float diff = a->computed[i] - a->reference[i];  // -- QW
    if (diff < 0.0f)             // -- QW
      diff = -diff;              // -- QW
    printf("  [%s %u] computed=%.6f  ref=%.6f  diff=%.6f  TOL=%.6f\r\n", a->tag, i,  // -- QW
           (double)a->computed[i], (double)a->reference[i], (double)diff,            // -- QW
           (double)tol);         // -- QW
    if (diff > tol) {            // -- QW
      errors++;                  // -- QW
    }                            // -- QW
  }                              // -- QW
  *a->err_count = errors;        // -- QW
}                                // -- QW

/* -------------------------------------------------------------------------
 * main -- QW
 * ---------------------------------------------------------------------- */

/* QW: immediate-flush trace to locate on-device stalls (device stdout is
 * otherwise buffered until main() returns, so a killed run shows nothing) -- QW */
#define ZTRACE(...) do { printf(__VA_ARGS__); fflush(stdout); } while (0)

#ifdef DUMP_WEIGHTS
/* QW: on-device ZO weight extraction (mirror of deeploytraintest.c dump_weights) - whole fn added by QW -- QW
 * Dump the final ZO-updated trainable weights as raw 32-bit hex words (FPU-free, bit-exact). The updated
 * weights live in the training-input buffers DeeployNetwork_inputs[TRAINING_NUM_DATA_INPUTS + wi] (the
 * zo_update graph writes them in place, aliased onto zo_train's weight inputs). Prints one line per tensor:
 * "[WDUMP s=<step> wi=<i> n=<#floats>] <hex> <hex> ...", parsed off the runner log by extract_zo_weights.py. */
static void dump_zo_weights(uint32_t step) {  // -- QW
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)  // -- QW
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {   // -- QW
    uint32_t idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;                  // -- QW
    uint32_t bytes = DeeployNetwork_inputs_bytes[idx];                      // -- QW
    void *buf = DeeployNetwork_inputs[idx];                                 // -- QW
    uint32_t n = bytes / 4u;                                                // -- QW
    printf("[WDUMP s=%u wi=%u n=%u]", (unsigned)step, (unsigned)wi, (unsigned)n);  // -- QW
    for (uint32_t k = 0; k < n; k++) {                                      // -- QW
      uint32_t word;                                                        // -- QW
      if (IS_L2(buf)) {                                                     // -- QW
        word = ((const uint32_t *)buf)[k];                                  // -- QW
      } else {                                                             // -- QW
        ram_read(&word, (uint8_t *)buf + 4u * k, 4u);                       // -- QW
      }                                                                    // -- QW
      printf(" %08x", (unsigned)word);                                     // -- QW
    }                                                                      // -- QW
    printf("\r\n");                                                        // -- QW
  }                                                                        // -- QW
#endif                                                                     // -- QW
}                                                                          // -- QW
#endif /* DUMP_WEIGHTS -- QW */

int main(void) {                 // -- QW

  ZTRACE("=== Siracusa MeZO (ZO) Training Harness ===\r\n");  // -- QW
  printf("N_TRAIN_STEPS=%u  N_ACCUM_STEPS=%u  DATA_INPUTS=%u\r\n",  // -- QW
         (unsigned)N_TRAIN_STEPS, (unsigned)N_ACCUM_STEPS,          // -- QW
         (unsigned)TRAINING_NUM_DATA_INPUTS);                       // -- QW
  printf("ZO_EPS=%.6f  ZO_LR=%.9f  ZO_Q=%u  ZO_SEED=%u\r\n",        // -- QW
         (double)(float)ZO_EPS, (double)(float)ZO_LR,               // -- QW
         (unsigned)ZO_Q, (unsigned)ZO_SEED);                        // -- QW

  struct pi_cluster_conf conf;   // -- QW
  pi_cluster_conf_init(&conf);   // -- QW
  conf.id = 0;                   // -- QW
  pi_open_from_conf(&cluster_dev, &conf);  // -- QW
  if (pi_cluster_open(&cluster_dev))       // -- QW
    return -1;                   // -- QW

#ifndef NOFLASH                  // -- QW
  mem_init();                    // -- QW
  open_fs();                     // -- QW
#endif                           // -- QW

  struct pi_cluster_task cluster_task;  // -- QW

  /* ------------------------------------------------------------------
   * Init training network (zo_train). -- QW
   * ------------------------------------------------------------------ */

  ZTRACE("Initializing TrainingNetwork (zo_train)...\r\n");  // -- QW
  pi_cluster_task(&cluster_task, InitTrainingNetwork, NULL);  // -- QW
  cluster_task.stack_size = MAINSTACKSIZE;                    // -- QW
  cluster_task.slave_stack_size = SLAVESTACKSIZE;             // -- QW
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);    // -- QW
  ZTRACE("[PHASE] InitTrainingNetwork done\r\n");             // -- QW

  /* ------------------------------------------------------------------
   * Init optimizer network (zo_update). Shares the training weight
   * constants via name-matched buffer redirection (see codeGenerateTraining
   * _patch_shared_buffers) — the update is in-place, no copies needed. -- QW
   * ------------------------------------------------------------------ */

  ZTRACE("Initializing OptimizerNetwork (zo_update)...\r\n");  // -- QW
  pi_cluster_task(&cluster_task, InitOptimizerNetwork, NULL);  // -- QW
  cluster_task.stack_size = MAINSTACKSIZE;                     // -- QW
  cluster_task.slave_stack_size = SLAVESTACKSIZE;              // -- QW
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);     // -- QW
  ZTRACE("[PHASE] InitOptimizerNetwork done\r\n");             // -- QW

  /* ------------------------------------------------------------------
   * Copy initial weights into the training weight buffers, if the graph
   * exposes trainable weights as inputs (BP-style layout). For the ZO
   * SpeechNet fixture the weights are baked graph constants (loaded by
   * InitTrainingNetwork), so TRAINING_NUM_WEIGHT_INPUTS is undefined and
   * this block compiles out. -- QW
   * ------------------------------------------------------------------ */

#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)  // -- QW
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {   // -- QW
    uint32_t idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;                  // -- QW
    void *dst = DeeployNetwork_inputs[idx];                                 // -- QW
    const void *src = testInitWeights[wi];                                  // -- QW
    uint32_t bytes = DeeployNetwork_inputs_bytes[idx];                      // -- QW
    if (IS_L2(dst) && IS_L2(src)) {                                         // -- QW
      memcpy(dst, (void *)src, bytes);                                      // -- QW
    } else if (IS_L2(dst)) {                                                // -- QW
      ram_read(dst, (void *)src, bytes);                                    // -- QW
    } else if (IS_L2(src)) {                                                // -- QW
      ram_write(dst, (void *)src, bytes);                                   // -- QW
    }                                                                       // -- QW
  }                                                                         // -- QW
#endif                                                                      // -- QW

  ZTRACE("Starting ZO training (%u update steps x %u accum steps)...\r\n",  // -- QW
         (unsigned)N_TRAIN_STEPS, (unsigned)N_ACCUM_STEPS);                 // -- QW

  /* QW: publish the export-time eps so the integer RQSPerturb kernels can scale their baked `mul` by
   * (perturb_eps_override / perturb_eps_baked) during the zo_update pass. -- QW */
  { extern float32_t perturb_eps_baked; perturb_eps_baked = (float32_t)ZO_EPS; }  // -- QW

#ifdef BN_FROZEN_STATS  /* QW: normalize training BN with frozen pretrained running stats -- QW */
  extern uint32_t g_bn_frozen_stats;  // -- QW
  g_bn_frozen_stats = 1u;             // -- QW
  printf("[BN_FROZEN_STATS] training BN uses frozen running statistics "     // -- QW
         "(train==inference)\n");     // -- QW
#endif                                // -- QW

  for (uint32_t update_step = 0; update_step < N_TRAIN_STEPS; update_step++) {  // -- QW

    uint32_t acc_bits = 0u;  // -- QW  raw bits of the FP accumulator (0 == 0.0f); FC never does FP
    uint32_t seed_base = update_step * (uint32_t)ZO_Q + 0u;  // -- QW  q_i=0 for ZO_Q==1

    for (uint32_t accum_step = 0; accum_step < N_ACCUM_STEPS; accum_step++) {  // -- QW

      uint32_t mb = update_step * N_ACCUM_STEPS + accum_step;  // -- QW

      ZTRACE("  update %u/%u  accum %u/%u  (mini-batch %u)  seed_base=%u\r\n",  // -- QW
             update_step + 1, (unsigned)N_TRAIN_STEPS, accum_step + 1,          // -- QW
             (unsigned)N_ACCUM_STEPS, mb, (unsigned)seed_base);                 // -- QW

      /* ① Load this mini-batch's data + labels (cycle via modulo). -- QW */
      for (uint32_t buf = 0; buf < TRAINING_NUM_DATA_INPUTS; buf++) {  // -- QW
        void *dst = DeeployNetwork_inputs[buf];                        // -- QW
        const void *src = testDataVector[mb % TRAINING_DATA_SIZE][buf];// -- QW
        uint32_t bytes = DeeployNetwork_inputs_bytes[buf];            // -- QW
        if (IS_L2(dst) && IS_L2(src)) {                              // -- QW
          memcpy(dst, (void *)src, bytes);                           // -- QW
        } else if (IS_L2(dst)) {                                    // -- QW
          ram_read(dst, (void *)src, bytes);                         // -- QW
        } else if (IS_L2(src)) {                                    // -- QW
          ram_write(dst, (void *)src, bytes);                        // -- QW
        }                                                           // -- QW
      }                                                             // -- QW

      uint32_t lp_bits = 0u;  // -- QW  raw loss bits (FC never does FP)
      uint32_t lm_bits = 0u;  // -- QW

      /* ② +eps pass -> L+. Set ZO globals on FC before dispatch. -- QW */
      perturbation_sign = 1u;             // -- QW  +eps
      perturb_seed_base = seed_base;      // -- QW  per-step seed
      perturb_eps_use_override = 0u;      // -- QW  use baked ZO_EPS
      ZTRACE("[PHASE] +eps forward START\r\n");                  // -- QW
      pi_cluster_task(&cluster_task, RunTrainingNetwork, NULL);  // -- QW
      cluster_task.stack_size = MAINSTACKSIZE;                   // -- QW
      cluster_task.slave_stack_size = SLAVESTACKSIZE;            // -- QW
      ResetTimer();                                              // -- QW
      StartTimer();                                              // -- QW
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);   // -- QW
      StopTimer();                                               // -- QW
      ZTRACE("[PHASE] +eps forward DONE (%u cyc)\r\n", (unsigned)getCycles());  // -- QW
      g_train_cycles_acc += getCycles();                         // -- QW
      /* Read scalar loss FPU-free (FC has no FPU): raw bytes only, never a float op. -- QW */
      {                                                          // -- QW
        void *loss_ptr = DeeployNetwork_outputs[0];              // -- QW
        if (IS_L2(loss_ptr)) {                                   // -- QW
          memcpy(&lp_bits, loss_ptr, sizeof(lp_bits));           // -- QW
        } else {                                                 // -- QW
          ram_read(&lp_bits, loss_ptr, sizeof(lp_bits));         // -- QW
        }                                                        // -- QW
      }                                                          // -- QW
      memcpy(&stored_loss_plus[mb], &lp_bits, sizeof(lp_bits));  // -- QW  integer store into float[] (no FP)
      ZTRACE("[PHASE] +eps loss read OK: lp_bits=0x%08x\r\n", (unsigned)lp_bits);  // -- QW

#ifdef ZO_FUSED_DUAL  // -- QW exp8: the dual-pass graph computed both passes in the run above; L- = output 1
      {                                                          // -- QW
        void *loss_ptr = DeeployNetwork_outputs[1];              // -- QW
        if (IS_L2(loss_ptr)) {                                   // -- QW
          memcpy(&lm_bits, loss_ptr, sizeof(lm_bits));           // -- QW
        } else {                                                 // -- QW
          ram_read(&lm_bits, loss_ptr, sizeof(lm_bits));         // -- QW
        }                                                        // -- QW
      }                                                          // -- QW
      memcpy(&stored_loss_minus[mb], &lm_bits, sizeof(lm_bits)); // -- QW
      ZTRACE("[PHASE] fused dual pass: lm_bits=0x%08x\r\n", (unsigned)lm_bits);  // -- QW
#else
      /* ③ -eps pass -> L-. Same seed_base, flip sign only. -- QW */
      perturbation_sign = 0u;             // -- QW  -eps
      /* perturb_seed_base / perturb_eps_use_override unchanged -- QW */
      ZTRACE("[PHASE] -eps forward START\r\n");                  // -- QW
      pi_cluster_task(&cluster_task, RunTrainingNetwork, NULL);  // -- QW
      cluster_task.stack_size = MAINSTACKSIZE;                   // -- QW
      cluster_task.slave_stack_size = SLAVESTACKSIZE;            // -- QW
      ResetTimer();                                              // -- QW
      StartTimer();                                              // -- QW
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);   // -- QW
      StopTimer();                                               // -- QW
      ZTRACE("[PHASE] -eps forward DONE (%u cyc)\r\n", (unsigned)getCycles());  // -- QW
      g_train_cycles_acc += getCycles();                         // -- QW
      {                                                          // -- QW
        void *loss_ptr = DeeployNetwork_outputs[0];              // -- QW
        if (IS_L2(loss_ptr)) {                                   // -- QW
          memcpy(&lm_bits, loss_ptr, sizeof(lm_bits));           // -- QW
        } else {                                                 // -- QW
          ram_read(&lm_bits, loss_ptr, sizeof(lm_bits));         // -- QW
        }                                                        // -- QW
      }                                                          // -- QW
      memcpy(&stored_loss_minus[mb], &lm_bits, sizeof(lm_bits)); // -- QW  integer store into float[] (no FP)
      ZTRACE("[PHASE] -eps loss read OK: lm_bits=0x%08x\r\n", (unsigned)lm_bits);  // -- QW
#endif  // ZO_FUSED_DUAL -- QW

      /* ④ acc += (L+ - L-)  — FP done on cluster (FC has no FPU). -- QW */
      {                                                          // -- QW
        ZODiffArgs diff_args = {.lp_bits = lp_bits, .lm_bits = lm_bits, .acc_out = (float *)&acc_bits};  // -- QW
        pi_cluster_task(&cluster_task, AccumulateZODiffOnCluster, &diff_args);  // -- QW
        cluster_task.stack_size = MAINSTACKSIZE;                 // -- QW
        cluster_task.slave_stack_size = SLAVESTACKSIZE;          // -- QW
        pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task); // -- QW
      }                                                          // -- QW

    } /* end accum_step loop -- QW */

    /* ⑤ Compute g_proj and the runtime update coefficient on cluster, then
     * write perturb_eps_override = -ZO_LR*g_proj. -- QW */
    {                                                            // -- QW
      ZOUpdateCoeffArgs coeff_args = {                           // -- QW
          .acc_bits = acc_bits,                                  // -- QW  raw bits (FC never does FP)
          .n_accum = (uint32_t)N_ACCUM_STEPS,                    // -- QW
          .coeff_out = &perturb_eps_override,                    // -- QW  cluster writes the float global directly
      };                                                         // -- QW
      pi_cluster_task(&cluster_task, ComputeZOUpdateCoeffOnCluster, &coeff_args);  // -- QW
      cluster_task.stack_size = MAINSTACKSIZE;                   // -- QW
      cluster_task.slave_stack_size = SLAVESTACKSIZE;            // -- QW
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);   // -- QW
    }                                                            // -- QW

    /* ⑥ In-place ZO weight update via zo_update (RunOptimizerNetwork).
     * Same seed_base so z is identical to the +/-eps passes; +eps direction
     * (sign carried by the coefficient's own sign). -- QW */
    perturb_eps_use_override = 1u;   // -- QW  use perturb_eps_override (= -ZO_LR*g_proj)
    perturbation_sign = 1u;          // -- QW
    perturb_seed_base = seed_base;   // -- QW  same z as the train passes
    ZTRACE("[PHASE] update (zo_update) START\r\n");            // -- QW
    pi_cluster_task(&cluster_task, RunOptimizerNetwork, NULL);  // -- QW
    cluster_task.stack_size = MAINSTACKSIZE;                    // -- QW
    cluster_task.slave_stack_size = SLAVESTACKSIZE;             // -- QW
    ResetTimer();                                               // -- QW
    StartTimer();                                               // -- QW
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);    // -- QW
    StopTimer();                                                // -- QW
    ZTRACE("[PHASE] update (zo_update) DONE (%u cyc)\r\n", (unsigned)getCycles());  // -- QW
    g_opt_cycles_acc += getCycles();                            // -- QW
    perturb_eps_use_override = 0u;   // -- QW  restore neutral default

#if defined(DUMP_WEIGHTS) && defined(DUMP_STEP_LO) && defined(DUMP_STEP_HI)
    /* QW exp11_bitexact_SCE_verif: dump weights AFTER each update in the window
     * [DUMP_STEP_LO, DUMP_STEP_HI] (weights[s]); the host trace dumps the same. Then
     * increment[s] = weights[s]-weights[s-1], and weights-INTO-step-s = weights[s-1].
     * Default-off (defines absent) => existing final-only dump behaviour unchanged. -- QW */
    if (update_step >= (uint32_t)DUMP_STEP_LO && update_step <= (uint32_t)DUMP_STEP_HI) {
      dump_zo_weights(update_step);
    }
#endif

  } /* end update_step loop -- QW */

#ifdef DUMP_WEIGHTS
  /* QW: dump the final on-device ZO-updated weights (after the last update step) so
   * extract_zo_weights.py can rebuild the carry checkpoint. FPU-free. -- QW */
  dump_zo_weights((uint32_t)(N_TRAIN_STEPS - 1));  // -- QW
#endif

  /* ------------------------------------------------------------------
   * Numerical verification — run on cluster (FC has no FPU). -- QW
   * ------------------------------------------------------------------ */

  uint32_t total_loss_checks =                                       // -- QW
      (TOTAL_FWD_PASSES < N_LOSS_REFS) ? TOTAL_FWD_PASSES : N_LOSS_REFS;  // -- QW

  uint32_t err_plus = 0;   // -- QW
  uint32_t err_minus = 0;  // -- QW

  ZOLossCompareArgs cmp_plus = {          // -- QW
      .computed = stored_loss_plus,       // -- QW
      .reference = (float *)testLossPlusRef,  // -- QW
      .n = total_loss_checks,             // -- QW
      .err_count = &err_plus,             // -- QW
      .tag = "loss+",                     // -- QW
  };                                      // -- QW
  pi_cluster_task(&cluster_task, CompareLossesOnCluster, &cmp_plus);  // -- QW
  cluster_task.stack_size = MAINSTACKSIZE;                            // -- QW
  cluster_task.slave_stack_size = SLAVESTACKSIZE;                     // -- QW
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);            // -- QW

  ZOLossCompareArgs cmp_minus = {         // -- QW
      .computed = stored_loss_minus,      // -- QW
      .reference = (float *)testLossMinusRef,  // -- QW
      .n = total_loss_checks,             // -- QW
      .err_count = &err_minus,            // -- QW
      .tag = "loss-",                     // -- QW
  };                                      // -- QW
  pi_cluster_task(&cluster_task, CompareLossesOnCluster, &cmp_minus);  // -- QW
  cluster_task.stack_size = MAINSTACKSIZE;                             // -- QW
  cluster_task.slave_stack_size = SLAVESTACKSIZE;                      // -- QW
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);             // -- QW

  uint32_t loss_err_count = err_plus + err_minus;          // -- QW
  uint32_t total_checks = 2u * total_loss_checks;          // -- QW
  printf("Errors: %u out of %u\r\n", (unsigned)loss_err_count,  // -- QW
         (unsigned)total_checks);                          // -- QW

  /* QW: print 64-bit totals as hi/lo 32-bit halves (FC-safe integer, no %llu dependency).
   * Host reassembles: cycles = hi*4294967296 + lo. Faithful even when the round exceeds 2^32. -- QW */
  printf("BENCH train_cycles_hi=%u train_cycles_lo=%u opt_cycles_hi=%u opt_cycles_lo=%u\r\n",  // -- QW
         (unsigned)(g_train_cycles_acc >> 32), (unsigned)(g_train_cycles_acc & 0xffffffffu),   // -- QW
         (unsigned)(g_opt_cycles_acc >> 32), (unsigned)(g_opt_cycles_acc & 0xffffffffu));      // -- QW

  return loss_err_count == 0 ? 0 : 1;  // -- QW
}                                      // -- QW
