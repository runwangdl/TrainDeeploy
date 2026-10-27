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

#ifdef POWER_MEASUREMENT
/* RW 2026-10-15: PPK2 trigger for the power rig (AI_AGENT/Deeploy_Basics/GAP9_Power_Measurement.md). GPIO 89 is
 * high for the whole update loop (all forward passes + updates of the round), low for boot / init / verification. */
static unsigned int GPIOs = 89;
#endif

// RW 2026-10-13: ZO_QUIET (see ZTRACE below) -- print only mismatching / every 500th loss comparison
#if defined(ZO_ECO)
// RW 2026-10-13: the fixture's reference losses are the plain-ZO trajectory, so with ECO nearly every comparison
// "fails"; printing them all cost ~85 s of 115200-baud UART per 120-epoch board run. Print every 500th + the last.
#define ZO_PRINT_CMP(i, n, bad) ((i) % 500u == 0u || (i) + 1u == (n))
#elif defined(ZO_QUIET)
#define ZO_PRINT_CMP(i, n, bad) ((bad) || (i) % 500u == 0u || (i) + 1u == (n))
#else
#define ZO_PRINT_CMP(i, n, bad) 1
#endif

// -- QW: GAP9 pi_cluster_task expects a void(void*) entry; the generated network Init/Run
//    fns are void(uint32_t,uint32_t). Wrap them, supplying core_id / n_cores from pmsis
//    (mirrors the stock GAP9 deeploytest.c InitNetworkWrapper pattern). -- QW
static void InitTrainingNetworkWrapper(void *args)  { (void)args; InitTrainingNetwork(pi_core_id(), pi_cl_cluster_nb_cores()); }
// RW: start the cluster core's own perf counter so --profileTiling's per-node getCycles() (read on the
//     cluster) is not 0 on GAP9; the FC-side ResetTimer/StartTimer around the dispatch is unaffected.
#ifdef TRAINING_TESTDATA_L3
// RW merge 2026-10-13: devel's generator moves the baked windows/labels/initial weights to L3
// (testinputs.h: LoadTestDataL3, cl_ram_malloc + load_file_to_ram on the cluster). Without this call
// every testDataVector / testInitWeights pointer stays 0 and the copies below silently skip.
static void LoadTestDataL3Wrapper(void *args) { (void)args; LoadTestDataL3(); }
#endif
static void RunTrainingNetworkWrapper(void *args)   { (void)args; ResetTimer(); StartTimer(); RunTrainingNetwork(pi_core_id(), pi_cl_cluster_nb_cores()); }
static void InitOptimizerNetworkWrapper(void *args) { (void)args; InitOptimizerNetwork(pi_core_id(), pi_cl_cluster_nb_cores()); }
static void RunOptimizerNetworkWrapper(void *args)  { (void)args; RunOptimizerNetwork(pi_core_id(), pi_cl_cluster_nb_cores()); }

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

#ifdef ZO_ECO  // RW 2026-10-13
#ifndef ZO_ECO_BETA
#define ZO_ECO_BETA 0.9f
#endif
#ifndef ZO_ECO_SR
#define ZO_ECO_SR 0u
#endif
#define ZO_ECO_STR_(x) #x
#define ZO_ECO_STR(x) ZO_ECO_STR_(x)
#define ZO_ECO_BETA_STR ZO_ECO_STR(ZO_ECO_BETA)
#endif

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
#ifndef SLAVESTACKSIZE           // RW: overridable (-D SLAVESTACKSIZE=N), like deeploytraintest.c
// RW 2026-10-13: 768 (was 3800 -- QW). Worst-case slave call chain after the stack-diet kernels: ZO 352 B
// (conv fork -> conv dispatcher -> conv1ch_rows), QZO 560 B (prebuilt pulp_nn_conv -> pulp_nn_matmul);
// measured with scripts/stack_depth.py. 640 also runs but leaves only 80 B of margin.
#define SLAVESTACKSIZE 768
#endif

// -- QW: relocate the cluster slave stacks from L1 to L2. GAP9's SDK only pi_cl_l1_malloc's
//    the stacks when task->stacks == NULL (cluster.c:440-444), consuming ~30KB of the 128KB
//    L1 that the Deeploy tiling arena also needs (usable L1 -> ~93KB). Handing it our own L2
//    buffer via pi_cluster_task_stacks() makes the SDK skip that L1 malloc, freeing ~30KB back
//    to L1 (usable -> ~123KB). Size = nb_cores(=pi_cl_cluster_nb_cores(), 9 on GAP9) *
//    slave_stack_size; over-provision to 12 cores for safety. Lives in L2 (plain .bss). -- QW
// RW 2026-10-13: slave stacks live in L1 (SDK pi_cl_l1_malloc) by default; with SLAVESTACKSIZE=768 they cost
// ~7 KB of L1 instead of the ~30 KB that made Qiwen move them to L2. In L2 every spill pays L2 latency (the
// prebuilt -Os pulp-nn conv was 4.6x slower). -D MEZO_STACKS_L2=ON restores the L2 buffer.
#ifndef MEZO_STACKS_L2
#define MEZO_SET_STACKS(t) ((t).slave_stack_size = SLAVESTACKSIZE)
#else
#define MEZO_SET_STACKS(t) pi_cluster_task_stacks(&(t), cluster_slave_stacks, SLAVESTACKSIZE)
#endif
#define N_STACK_CORES 12                                                  // -- QW
#ifdef MEZO_STACKS_L2
static unsigned char cluster_slave_stacks[N_STACK_CORES * SLAVESTACKSIZE] // -- QW
    __attribute__((aligned(16)));                                         // -- QW
#endif

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
  // RW 2026-10-13: guard removed (kept as comment). pi_cluster_send_task_to_cl() runs this entry once, on the
  // cluster controller, whose pi_core_id() is 8 on GAP9: the guard always fired, so (L+ - L-) was never
  // accumulated (ZO update coefficient 0, weights never changed) and the loss comparison never ran (vacuous
  // "Errors: 0"). Same bug as TrainDeeploy PR #71 / AI_AGENT result_check.md appendix B.
  // if (pi_core_id() != 0)         // -- QW
  //   return;                      // -- QW
  ZOUpdateCoeffArgs *a = (ZOUpdateCoeffArgs *)args;       // -- QW
  float acc; memcpy(&acc, &a->acc_bits, sizeof(acc));     // -- QW  reinterpret bits->float on cluster (has FPU)
  float denom = 2.0f * (float)ZO_EPS * (float)a->n_accum; // -- QW
  float g_proj = acc / denom;    // -- QW
  *a->coeff_out = -(float)ZO_LR * g_proj;                 // -- QW
#ifdef ZO_ECO
  zo_eco_g = g_proj;  // RW 2026-10-13: ECO needs g and lr separately
#endif
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
  // RW 2026-10-13: guard removed (kept as comment). pi_cluster_send_task_to_cl() runs this entry once, on the
  // cluster controller, whose pi_core_id() is 8 on GAP9: the guard always fired, so (L+ - L-) was never
  // accumulated (ZO update coefficient 0, weights never changed) and the loss comparison never ran (vacuous
  // "Errors: 0"). Same bug as TrainDeeploy PR #71 / AI_AGENT result_check.md appendix B.
  // if (pi_core_id() != 0)         // -- QW
  //   return;                      // -- QW
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
  // RW 2026-10-13: guard removed (kept as comment). pi_cluster_send_task_to_cl() runs this entry once, on the
  // cluster controller, whose pi_core_id() is 8 on GAP9: the guard always fired, so (L+ - L-) was never
  // accumulated (ZO update coefficient 0, weights never changed) and the loss comparison never ran (vacuous
  // "Errors: 0"). Same bug as TrainDeeploy PR #71 / AI_AGENT result_check.md appendix B.
  // if (pi_core_id() != 0)         // -- QW
  //   return;                      // -- QW
  ZOLossCompareArgs *a = (ZOLossCompareArgs *)args;  // -- QW
  float tol = ZO_TOLERANCE_ABS;  // -- QW  read on cluster — has FPU
  uint32_t errors = 0;           // -- QW
  for (uint32_t i = 0; i < a->n; i++) {  // -- QW
    float diff = a->computed[i] - a->reference[i];  // -- QW
    if (diff < 0.0f)             // -- QW
      diff = -diff;              // -- QW
    if (ZO_PRINT_CMP(i, a->n, diff > tol))  // RW: ZO_QUIET prints mismatches + every 500th only
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
// -- QW: GAP9 links the lightweight pmsis printf (libpmsis_printf) — no newlib stdio, so
//    fflush/stdout (_impure_ptr) are unavailable and not needed (pmsis printf flushes itself). -- QW
// RW 2026-10-13: -D ZO_QUIET=ON silences the per-phase trace and prints only mismatching / every 500th loss
// comparison. On the EVK the UART (115200 baud, ~11.5 KB/s) dominated a full round: ~8 trace lines per
// forward pair plus 2 x 10,800 comparison lines is several MB; the 2700-step compute itself is ~5 min.
#ifdef ZO_QUIET
#define ZTRACE(...) do { } while (0)
#else
#define ZTRACE(...) do { printf(__VA_ARGS__); } while (0)
#endif

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

/* RW 2026-10-13: device-resident weights across fine-tuning rounds.
 * ZO_PERSIST: after the last update the trainable tensors are written to a reserved region at the END of the
 *   external flash (the readfs image sits at its start), behind a header with magic, tensor count, a hash of the
 *   tensor sizes, the accumulated step count and the round number, and a CRC-32 of the payload; the write is read
 *   back and checked.
 * ZO_RESUME: before training, a valid region (magic, same tensor layout, CRC) replaces the initial weights from
 *   readfs; otherwise the readfs weights are used and the reason is printed. A full reflash may erase the region,
 *   which then simply fails the check. The UART weight dump is no longer needed to carry weights between rounds. */
#if (defined(ZO_PERSIST) || defined(ZO_RESUME)) && defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
#define ZOP_MAGIC 0x5A4F5754u /* "ZOWT" */
#define ZOP_REGION (256u * 1024u)
#ifndef ZOP_FLASH_SIZE
#define ZOP_FLASH_SIZE (64u * 1024u * 1024u)
#endif
typedef struct {
  uint32_t magic, version, n_tensors, payload_bytes, layout_hash, steps_total, round, crc;
} zop_hdr_t;

static uint32_t zop_crc32(const uint8_t *p, uint32_t n, uint32_t crc) {
  crc = ~crc;
  for (uint32_t i = 0; i < n; i++) {
    crc ^= p[i];
    for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

static uint32_t zop_layout(uint32_t *payload) {
  uint32_t h = 2166136261u, tot = 0;
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t b = DeeployNetwork_inputs_bytes[(uint32_t)TRAINING_NUM_DATA_INPUTS + wi];
    h = (h ^ b) * 16777619u;
    tot += (b + 3u) & ~3u;
  }
  *payload = tot;
  return h;
}

static uint32_t zop_addr(struct pi_device *fl) {
  struct pi_flash_info info;
  pi_flash_ioctl(fl, PI_FLASH_IOCTL_INFO, &info);
  uint32_t region = (ZOP_REGION + info.sector_size - 1u) / info.sector_size * info.sector_size;
  /* the driver reports size 0 for the EVK octo-SPI flash; the device (and gapy --flash-size) has 64 MiB */
  uint32_t size = (info.size != 0u && info.size < ZOP_FLASH_SIZE) ? info.size : ZOP_FLASH_SIZE;
  return size - region;
}

/* copy one weight tensor between its network buffer (L2 or L3) and an L2 staging pointer */
static void zop_xfer(uint32_t wi, uint8_t *stage, int to_net) {
  uint32_t idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
  void *buf = DeeployNetwork_inputs[idx];
  uint32_t bytes = DeeployNetwork_inputs_bytes[idx];
  if (IS_L2(buf)) {
    if (to_net) memcpy(buf, stage, bytes); else memcpy(stage, buf, bytes);
  } else {
    if (to_net) ram_write(buf, stage, bytes); else ram_read(stage, buf, bytes);
  }
}
#endif

#if defined(ZO_PERSIST) && defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
static uint32_t g_zop_steps_before = 0, g_zop_round = 1, g_zop_fail = 0;
static int zop_save(uint32_t steps_done) {
  struct pi_device *fl = get_flash_ptr();
  uint32_t payload; zop_hdr_t h = {ZOP_MAGIC, 1u, TRAINING_NUM_WEIGHT_INPUTS, 0, 0, 0, 0, 0};
  h.layout_hash = zop_layout(&payload); h.payload_bytes = payload;
  h.steps_total = g_zop_steps_before + steps_done; h.round = g_zop_round;
  uint32_t tot = sizeof(h) + payload;
  uint8_t *img = (uint8_t *)pi_l2_malloc(tot);
  if (!img) { printf("[ZOP] save: no L2 for %u B\r\n", (unsigned)tot); g_zop_fail = 1; return -1; }
  memset(img, 0, tot);
  uint32_t off = sizeof(h);
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    zop_xfer(wi, img + off, 0);
    off += (DeeployNetwork_inputs_bytes[(uint32_t)TRAINING_NUM_DATA_INPUTS + wi] + 3u) & ~3u;
  }
  h.crc = zop_crc32(img + sizeof(h), payload, 0);
  memcpy(img, &h, sizeof(h));
  uint32_t a = zop_addr(fl);
  pi_flash_erase(fl, a, (int)tot);
  pi_flash_program(fl, a, img, tot);
  memset(img, 0, tot);
  pi_flash_read(fl, a, img, tot);  /* read back and verify */
  zop_hdr_t r; memcpy(&r, img, sizeof(r));
  int ok = r.magic == ZOP_MAGIC && r.crc == zop_crc32(img + sizeof(r), payload, 0);
  printf("[ZOP] saved round %u (%u steps total) %u B at flash 0x%08x crc %08x: %s\r\n", (unsigned)h.round,
         (unsigned)h.steps_total, (unsigned)tot, (unsigned)a, (unsigned)h.crc, ok ? "verified" : "VERIFY FAILED");
  pi_l2_free(img, tot);
  g_zop_fail = !ok;
  return ok ? 0 : -1;
}
#endif

#if defined(ZO_RESUME) && defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
static void zop_resume(void) {
  struct pi_device *fl = get_flash_ptr();
  uint32_t payload, layout = zop_layout(&payload), a = zop_addr(fl);
  zop_hdr_t h; pi_flash_read(fl, a, &h, sizeof(h));
  const char *why = NULL;
  if (h.magic != ZOP_MAGIC) why = "no saved weights";
  else if (h.n_tensors != TRAINING_NUM_WEIGHT_INPUTS || h.layout_hash != layout || h.payload_bytes != payload)
    why = "saved weights have a different tensor layout";
  uint8_t *img = NULL;
  if (!why) {
    img = (uint8_t *)pi_l2_malloc(payload);
    if (!img) why = "no L2 for the payload";
    else {
      pi_flash_read(fl, a + sizeof(h), img, payload);
      if (zop_crc32(img, payload, 0) != h.crc) why = "CRC mismatch";
    }
  }
  if (why) {
    printf("[ZOP] resume: %s -> initial weights from readfs\r\n", why);
  } else {
    uint32_t off = 0;
    for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
      zop_xfer(wi, img + off, 1);
      off += (DeeployNetwork_inputs_bytes[(uint32_t)TRAINING_NUM_DATA_INPUTS + wi] + 3u) & ~3u;
    }
    printf("[ZOP] resumed round %u weights (%u steps so far, crc %08x)\r\n", (unsigned)h.round,
           (unsigned)h.steps_total, (unsigned)h.crc);
#ifdef ZO_PERSIST
    g_zop_steps_before = h.steps_total; g_zop_round = h.round + 1u;
#endif
  }
  if (img) pi_l2_free(img, payload);
}
#endif

/* RW 2026-10-13: weight-update self-check (FPU-free, FC only). A ZO round once reported "Errors: 0 out of
 * 21600" with every weight bit-identical to the pretrained one (the loss tolerance cannot see a missing
 * update: lr 3e-6 moves the losses by < 1e-3). wchk_hash() hashes the raw bits of one trainable tensor;
 * main() records the hashes before training, prints "[WSUM s=..]" every ZO_WCHK_EVERY updates and
 * "[WCHK] changed k/N" at the end, and counts a run in which no tensor changed as an error.
 * (QZO int8 conv weights may legitimately stay put below 1 LSB, so only "none changed" fails.) */
#ifndef ZO_WCHK_EVERY
#define ZO_WCHK_EVERY 100
#endif
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
static uint32_t g_wchk0[TRAINING_NUM_WEIGHT_INPUTS];
static uint32_t wchk_hash(uint32_t wi) {
  uint32_t idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
  uint32_t n = DeeployNetwork_inputs_bytes[idx] / 4u;
  void *buf = DeeployNetwork_inputs[idx];
  uint32_t h = 2166136261u;
  for (uint32_t k = 0; k < n; k++) {
    uint32_t word;
    if (IS_L2(buf)) {
      word = ((const uint32_t *)buf)[k];
    } else {
      ram_read(&word, (uint8_t *)buf + 4u * k, 4u);
    }
    h = (h ^ word) * 16777619u;
  }
  return h;
}
static uint32_t wchk_all(void) {
  uint32_t h = 0;
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) h = h * 31u + wchk_hash(wi);
  return h;
}
#endif

int main(void) {                 // -- QW
#ifdef POWER_MEASUREMENT         // RW 2026-10-15: configure the trigger first, as in deeploytraintest.c
  pi_pad_function_set(GPIOs, 1);
  pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
  pi_gpio_pin_write(GPIOs, 0);
#endif

  ZTRACE("=== GAP9 MeZO (ZO) Training Harness ===\r\n");  // -- QW (ported from Siracusa)
  printf("N_TRAIN_STEPS=%u  N_ACCUM_STEPS=%u  DATA_INPUTS=%u\r\n",  // -- QW
         (unsigned)N_TRAIN_STEPS, (unsigned)N_ACCUM_STEPS,          // -- QW
         (unsigned)TRAINING_NUM_DATA_INPUTS);                       // -- QW
  printf("ZO_EPS=%.6f  ZO_LR=%.9f  ZO_Q=%u  ZO_SEED=%u\r\n",        // -- QW
         (double)(float)ZO_EPS, (double)(float)ZO_LR,               // -- QW
         (unsigned)ZO_Q, (unsigned)ZO_SEED);                        // -- QW

  struct pi_cluster_conf conf;   // -- QW
  pi_cluster_conf_init(&conf);   // -- QW
  conf.id = 0;                   // -- QW
  // RW 2026-10-13 (from deeploytraintest.c): size the cluster-controller stack explicitly. It is carved from
  // L1; the slave stacks already live in L2 here, so with CC_STACK_SIZE=4096 SLAVESTACKSIZE=512 (the GAP9 CI
  // configuration) the tiler gets ~122 KB of L1 instead of ~110 KB.
#ifndef CC_STACK_SIZE
#define CC_STACK_SIZE 4096
#endif
  conf.cc_stack_size = CC_STACK_SIZE;
  pi_open_from_conf(&cluster_dev, &conf);  // -- QW
  if (pi_cluster_open(&cluster_dev))       // -- QW
    return -1;                   // -- QW

#ifndef NOFLASH                  // -- QW
// #if !defined(POWER_MEASUREMENT) && !defined(GVSOC_DEFAULT_FREQ)
#if !defined(GVSOC_DEFAULT_FREQ)  // RW 2026-10-15: power runs use the same 370 MHz / 0.8 V point
  /* RW 2026-10-13 (from the Mamba harness, AI_AGENT/Mamba/board_bringup_2026-10-06.md 2.1): run gvsoc at the
   * EVK operating point. At the SDK default (cluster 50 MHz) every PSRAM wait is ~7x cheaper in cycles
   * (FEMBA: 65.0 M gvsoc-default vs 107.8 M gvsoc@370 vs 108.8 M board). -DGVSOC_DEFAULT_FREQ = old clocks. */
  pi_freq_set(PI_FREQ_DOMAIN_FC, 370 * 1000 * 1000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 370 * 1000 * 1000);
  pi_freq_set(PI_FREQ_DOMAIN_PERIPH, 370 * 1000 * 1000);
  pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, 800);
  printf("[FREQ] FC=%d CL=%d PE=%d Hz\r\n", pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL),
         pi_freq_get(PI_FREQ_DOMAIN_PERIPH));
#endif
#ifdef POWER_MEASUREMENT         // RW 2026-10-15: re-assert the pad after the PERIPH retune
  pi_pad_function_set(GPIOs, 1);
  pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
  pi_gpio_pin_write(GPIOs, 0);
#endif
  mem_init();                    // -- QW
  open_fs();                     // -- QW
#endif                           // -- QW

  struct pi_cluster_task cluster_task;  // -- QW

  /* ------------------------------------------------------------------
   * Init training network (zo_train). -- QW
   * ------------------------------------------------------------------ */

  ZTRACE("Initializing TrainingNetwork (zo_train)...\r\n");  // -- QW
  pi_cluster_task(&cluster_task, InitTrainingNetworkWrapper, NULL);  // -- QW
  MEZO_SET_STACKS(cluster_task);             // -- QW
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);    // -- QW
  ZTRACE("[PHASE] InitTrainingNetwork done\r\n");             // -- QW
#ifdef TRAINING_TESTDATA_L3
  pi_cluster_task(&cluster_task, LoadTestDataL3Wrapper, NULL);                            // RW merge
  MEZO_SET_STACKS(cluster_task);            // RW merge
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);                                // RW merge
  ZTRACE("[PHASE] LoadTestDataL3 done\r\n");                                             // RW merge
#endif

  /* ------------------------------------------------------------------
   * Init optimizer network (zo_update). Shares the training weight
   * constants via name-matched buffer redirection (see codeGenerateTraining
   * _patch_shared_buffers) — the update is in-place, no copies needed. -- QW
   * ------------------------------------------------------------------ */

  ZTRACE("Initializing OptimizerNetwork (zo_update)...\r\n");  // -- QW
  pi_cluster_task(&cluster_task, InitOptimizerNetworkWrapper, NULL);  // -- QW
  MEZO_SET_STACKS(cluster_task);              // -- QW
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

#if defined(ZO_RESUME) && defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  zop_resume();  // RW 2026-10-13: previous round's device weights from flash, if present
  zo_weight_version++;  // -- QW exp21c: weights replaced -> invalidate the NE16 prep cache (opt H)
#endif
  ZTRACE("Starting ZO training (%u update steps x %u accum steps)...\r\n",  // -- QW
         (unsigned)N_TRAIN_STEPS, (unsigned)N_ACCUM_STEPS);                 // -- QW

  /* QW: publish the export-time eps so the integer RQSPerturb kernels can scale their baked `mul` by
   * (perturb_eps_override / perturb_eps_baked) during the zo_update pass. -- QW */
  { extern float32_t perturb_eps_baked; perturb_eps_baked = (float32_t)ZO_EPS; }  // -- QW

#ifdef ZO_ECO
  /* RW 2026-10-13: ECO error feedback (ZORuntime.h). One zero-initialised fp32 momentum buffer per update-graph
   * input (= update node idx); inputs_bytes floats is an upper bound on the element count of every dtype. The
   * momentum restarts every run, as in the host simulation (one run = one round). */
  {
    uint32_t tot = 0;
    for (uint32_t k = 0; k < DeeployOptNetwork_num_inputs && k < ZO_ECO_MAX_TENSORS; k++) {
      // uint32_t bytes = DeeployOptNetwork_inputs_bytes[k] * 4u;
      /* QW (exp8): one fp32 per ELEMENT -- bytes*4 over-allocated the int32 / fp32 tensors 4x (69,264 B instead of
       * 61,956 B for SpeechNet), and the fused dual-pass graph's larger L2 arena left too little heap for that. The
       * update node idx k is the optimizer input k (checked for the SpeechNet update graph). -- QW */
      uint32_t bytes = DeeployOptNetwork_inputs_elems[k] * 4u;
      zo_eco_mom[k] = (float32_t *)pi_l2_malloc(bytes);
      if (zo_eco_mom[k] == NULL) { printf("[ECO] momentum alloc failed (k=%u)\r\n", (unsigned)k); return -1; }
      memset(zo_eco_mom[k], 0, bytes);
      tot += bytes;
    }
    zo_eco_on = 1u;
    zo_eco_lr = (float32_t)ZO_LR;
    zo_eco_beta = (float32_t)ZO_ECO_BETA;
    zo_eco_sr = ZO_ECO_SR;
    printf("[ECO] memory-free error feedback, beta=%s sr=%u, %u B momentum\r\n", ZO_ECO_BETA_STR, (unsigned)ZO_ECO_SR,
           (unsigned)tot);
  }
#endif

#ifdef BN_FROZEN_STATS  /* QW: normalize training BN with frozen pretrained running stats -- QW */
  extern uint32_t g_bn_frozen_stats;  // -- QW
  g_bn_frozen_stats = 1u;             // -- QW
  printf("[BN_FROZEN_STATS] training BN uses frozen running statistics "     // -- QW
         "(train==inference)\n");     // -- QW
#endif                                // -- QW

#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) g_wchk0[wi] = wchk_hash(wi);  // RW
  printf("[WSUM s=init] 0x%08x\r\n", (unsigned)wchk_all());                                            // RW
#endif
#ifdef POWER_MEASUREMENT
  pi_gpio_pin_write(GPIOs, 1);   // RW 2026-10-15: power window = the update loop
#endif
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
      pi_cluster_task(&cluster_task, RunTrainingNetworkWrapper, NULL);  // -- QW
      MEZO_SET_STACKS(cluster_task);            // -- QW
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
      pi_cluster_task(&cluster_task, RunTrainingNetworkWrapper, NULL);  // -- QW
      MEZO_SET_STACKS(cluster_task);            // -- QW
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
        MEZO_SET_STACKS(cluster_task);          // -- QW
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
      MEZO_SET_STACKS(cluster_task);            // -- QW
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);   // -- QW
    }                                                            // -- QW

    /* ⑥ In-place ZO weight update via zo_update (RunOptimizerNetwork).
     * Same seed_base so z is identical to the +/-eps passes; +eps direction
     * (sign carried by the coefficient's own sign). -- QW */
    perturb_eps_use_override = 1u;   // -- QW  use perturb_eps_override (= -ZO_LR*g_proj)
    perturbation_sign = 1u;          // -- QW
    perturb_seed_base = seed_base;   // -- QW  same z as the train passes
#ifdef ZO_ECO
    zo_eco_step = update_step;       // RW: stochastic-rounding stream
    { uint32_t gb; memcpy(&gb, (const void *)&zo_eco_g, 4); ZTRACE("[ECO g s=%u] 0x%08x\r\n", (unsigned)update_step, (unsigned)gb); }
#endif
    ZTRACE("[PHASE] update (zo_update) START\r\n");            // -- QW
    pi_cluster_task(&cluster_task, RunOptimizerNetworkWrapper, NULL);  // -- QW
    MEZO_SET_STACKS(cluster_task);             // -- QW
    ResetTimer();                                               // -- QW
    StartTimer();                                               // -- QW
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);    // -- QW
    StopTimer();                                                // -- QW
    ZTRACE("[PHASE] update (zo_update) DONE (%u cyc)\r\n", (unsigned)getCycles());  // -- QW
    g_opt_cycles_acc += getCycles();                            // -- QW
    zo_weight_version++;             // -- QW exp21c: the weights changed -> invalidate the NE16 prep cache (opt H)
    perturb_eps_use_override = 0u;   // -- QW  restore neutral default
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
    if ((update_step + 1u) % ZO_WCHK_EVERY == 0u || update_step + 1u == (uint32_t)N_TRAIN_STEPS)        // RW
      printf("[WSUM s=%u] 0x%08x\r\n", (unsigned)update_step, (unsigned)wchk_all());                   // RW
#endif

#if defined(DUMP_WEIGHTS) && defined(DUMP_STEP_LO) && defined(DUMP_STEP_HI)
    /* QW exp11_bitexact_SCE_verif: dump weights AFTER each update in the window
     * [DUMP_STEP_LO, DUMP_STEP_HI] (weights[s]); the host trace dumps the same. Then
     * increment[s] = weights[s]-weights[s-1], and weights-INTO-step-s = weights[s-1].
     * Default-off (defines absent) => existing final-only dump behaviour unchanged. -- QW */
    if (update_step + 1u > (uint32_t)DUMP_STEP_LO && update_step <= (uint32_t)DUMP_STEP_HI) {  // RW: "+1 >" so LO=0 compiles (-Wtype-limits)
      dump_zo_weights(update_step);
    }
#endif

  } /* end update_step loop -- QW */
#ifdef POWER_MEASUREMENT
  pi_gpio_pin_write(GPIOs, 0);   // RW 2026-10-15
#endif
#if defined(ZO_PERSIST) && defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  zop_save((uint32_t)N_TRAIN_STEPS);  // RW 2026-10-13: keep the weights on the device for the next round
#endif

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
  MEZO_SET_STACKS(cluster_task);                     // -- QW
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);            // -- QW

  ZOLossCompareArgs cmp_minus = {         // -- QW
      .computed = stored_loss_minus,      // -- QW
      .reference = (float *)testLossMinusRef,  // -- QW
      .n = total_loss_checks,             // -- QW
      .err_count = &err_minus,            // -- QW
      .tag = "loss-",                     // -- QW
  };                                      // -- QW
  pi_cluster_task(&cluster_task, CompareLossesOnCluster, &cmp_minus);  // -- QW
  MEZO_SET_STACKS(cluster_task);                      // -- QW
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);             // -- QW

  uint32_t loss_err_count = err_plus + err_minus;          // -- QW
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  {                                                                                            // RW
    uint32_t changed = 0;                                                                      // RW
    for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) changed += (wchk_hash(wi) != g_wchk0[wi]);
    printf("[WCHK] changed %u/%u trainable tensors after %u updates\r\n", (unsigned)changed,      // RW
           (unsigned)TRAINING_NUM_WEIGHT_INPUTS, (unsigned)N_TRAIN_STEPS);                     // RW
    if (N_TRAIN_STEPS > 0 && changed == 0) {                                                   // RW
      printf("[WCHK] FAIL: no trainable tensor changed -- the update did not run\r\n");        // RW
      loss_err_count += 1u;                                                                    // RW
    }                                                                                          // RW
  }                                                                                            // RW
#endif
#if defined(ZO_EXPECT_WSUM) && defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  {  // RW 2026-10-13: CI gate on the trained weights (hash of all trainable tensors after the last update)
    uint32_t w = wchk_all();
    printf("[WSUM] final 0x%08x expected 0x%08x: %s\r\n", (unsigned)w, (unsigned)(ZO_EXPECT_WSUM),
           w == (uint32_t)(ZO_EXPECT_WSUM) ? "ok" : "MISMATCH");
    if (w != (uint32_t)(ZO_EXPECT_WSUM)) loss_err_count += 1u;
  }
#endif
#if defined(ZO_PERSIST) && defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  if (g_zop_fail) {  // RW: the flash write-back did not verify
    printf("[ZOP] FAIL: saved weights did not verify\r\n");
    loss_err_count += 1u;
  }
#endif
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
