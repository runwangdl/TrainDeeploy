/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Training harness for GAP9 — Phase 2 (with Deeploy-compiled OptimizerNetwork)
 *
 * Loop structure:
 *
 *   InitTrainingNetwork()
 *   InitOptimizerNetwork()
 *   Connect optimizer buffers → training network's weight/grad buffers
 *
 *   for update_step in [0, N_TRAIN_STEPS):          // optimizer steps
 *       for accum_step in [0, N_ACCUM_STEPS):        // mini-batches per update
 *           lazy_reset_grad = (accum_step == 0)      // reset on first,
 * accumulate on rest load data for this mini-batch RunTrainingNetwork() // fwd
 * + bwd + InPlaceAccumulatorV2 store loss value
 *       // SGD weight update via Deeploy-compiled optimizer kernel:
 *       copy weights + grad_acc → optimizer input buffers
 *       RunOptimizerNetwork()
 *       copy weight_updated ← optimizer output buffers → training weight
 * buffers
 *
 *   Numerical verification:
 *     - Compare stored loss values against testLossRef[] (from testoutputs.h)
 *
 * Note: GAP9 generated Init/Run functions take (uint32_t core_id, uint32_t
 * numThreads) arguments, so we use static wrapper functions to call them from
 * pi_cluster_send_task_to_cl.
 *
 * Compile-time constants (emitted by code generator into testinputs.h):
 *   N_TRAIN_STEPS              number of optimizer (weight-update) steps
 *   N_ACCUM_STEPS              number of mini-batches accumulated per update
 *   TRAINING_NUM_DATA_INPUTS   inputs that change each mini-batch (data +
 * labels) TRAINING_GRAD_BUF_START_IDX  first grad acc buffer index in
 * DeeployNetwork_inputs[] TRAINING_NUM_GRAD_INPUTS   number of grad
 * accumulation buffers (== number of weights) TRAINING_NUM_WEIGHT_INPUTS number
 * of trainable weight buffers TRAINING_LEARNING_RATE     SGD learning rate (for
 * reference — embedded in optimizer ONNX)
 *
 * Reference comparison constants (emitted into testoutputs.h):
 *   N_LOSS_REFS                number of reference loss values
 *   NUM_WEIGHT_REFS            number of reference weight tensors
 *   TRAINING_TOLERANCE_ABS     absolute comparison tolerance
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "CycleCounter.h"
#include "OptimizerNetwork.h"
#include "TrainingNetwork.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "testinputs.h"
#include "testoutputs.h"

/* Helper: true when ptr is CPU-directly-accessible (L1 TCDM or L2), false when
 * in L3 (external HyperRAM).
 *
 * GAP9 address map:
 *   L1  TCDM     : 0x1000_0000 .. 0x1002_0000
 *   L2  (priv+shared): 0x1C00_0000 .. 0x1C19_0000
 *   L3  HyperRAM : external alloc space, observed 0x8000_0000 .. 0xBFFF_FFFF
 *
 * The previous test `ptr >= 0x10000000` wrongly classified HyperRAM/L3
 * addresses (0x8.._0xb..) as CPU-accessible, so l3_aware_copy() did a raw
 * memcpy on an L3 pointer → FC "Invalid fetch" crash. Test the real on-chip
 * windows instead. */
#define IS_L1(ptr)                                                             \
  ((uint32_t)(ptr) >= 0x10000000u && (uint32_t)(ptr) < 0x10040000u)
#define IS_L2(ptr)                                                             \
  (((uint32_t)(ptr) >= 0x1C000000u && (uint32_t)(ptr) < 0x1C200000u) ||        \
   IS_L1(ptr))

/* -------------------------------------------------------------------------
 * Compile-time defaults — override via CMake target_compile_definitions
 * ---------------------------------------------------------------------- */

#ifndef N_TRAIN_STEPS
#define N_TRAIN_STEPS 1
#endif

#ifndef N_ACCUM_STEPS
#define N_ACCUM_STEPS 1
#endif

#ifndef TRAINING_NUM_DATA_INPUTS
#define TRAINING_NUM_DATA_INPUTS 2
#endif

/* -------------------------------------------------------------------------
 * Power measurement (PPK2) — only compiled when -DPOWER_MEASUREMENT=ON
 *
 * Mirrors the inference harness (deeploytest.c): GPIO 89 is driven as a digital
 * trigger that brackets the region-of-interest, so the external power monitor
 * can isolate the training compute from boot / data-load / UART overhead.
 * Additionally pins the operating point (FC / cluster / periph frequency, and
 * optionally chip voltage) so the measured power is taken at a known, fixed
 * frequency/voltage rather than the arbitrary boot default. Units: MHz.
 * Override on the cmake line, e.g. -DFREQ_FC=370 -DFREQ_CL=370 -DVOLTAGE=800.
 * ---------------------------------------------------------------------- */
#ifdef POWER_MEASUREMENT
unsigned int GPIOs = 89;
#define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs, x)

#ifndef FREQ_FC
#define FREQ_FC 240
#endif
#ifndef FREQ_CL
#define FREQ_CL 240
#endif
#ifndef FREQ_PE
#define FREQ_PE 240
#endif

#ifdef POWER_BISECT
/* Debug-only execution bisect: emit `n` short GPIO pulses (1ms hi / 1ms lo) as
 * a burst so the PPK2 trace shows how far main() progressed before any hang.
 * The highest burst count seen in the trace = last checkpoint reached. Build
 * with -DPOWER_MEASUREMENT=ON -DPOWER_BISECT. Not part of the normal ROI
 * harness. */
/* Busy-wait, NOT pi_time_wait_us: under openocd load_and_start_binary there is
 * no debugger attached and we must not depend on the OS tick/timer being live.
 * ~100k nops @ 240MHz ≈ a few ms — comfortably visible at 100kHz PPK2 sampling.
 */
static void bisect_delay(void) {
  for (volatile uint32_t i = 0; i < 100000u; i++) {
    __asm__ volatile("nop");
  }
}
static void bisect_mark(int n) {
  for (int i = 0; i < n; i++) {
    pi_gpio_pin_write(GPIOs, 1);
    bisect_delay();
    pi_gpio_pin_write(GPIOs, 0);
    bisect_delay();
  }
  bisect_delay(); /* inter-burst gap */
  bisect_delay();
  bisect_delay();
}
#define BISECT(n) bisect_mark(n)
#else
#define BISECT(n)                                                              \
  do {                                                                         \
  } while (0)
#endif
#endif /* POWER_MEASUREMENT */

/* Fallback when not building for power measurement at all. */
#ifndef BISECT
#define BISECT(n)                                                              \
  do {                                                                         \
  } while (0)
#endif

/* RW: GAP9 SDK does not use MAINSTACKSIZE for pi_cluster_task.
 * SLAVESTACKSIZE is the per-core cluster slave-stack size. Overridable per model
 * (-D SLAVESTACKSIZE=N): the FP32 trainlib kernels keep their working sets in
 * explicit L1 buffers, not on the stack, so the real per-core stack need is
 * <256B; the historical 3800 was ~15x over-provisioned. */
#ifndef SLAVESTACKSIZE
#define SLAVESTACKSIZE 3800
#endif

/* RW: Cluster slave-stack PLACEMENT. The GAP9 SDK only pi_cl_l1_malloc's the
 * slave stacks when task->stacks == NULL (see __pi_cluster_task_set_stack in the
 * SDK cluster driver).
 *
 *   default (L2):  hand the SDK our own static L2 buffer -> the per-core stacks
 *                  live in L2, freeing the L1 they would take (8 cores x
 *                  SLAVESTACKSIZE) for the tile arena. ALWAYS fits (L2 is 1MB+),
 *                  so it is the safe default for any net. Cost: every stack
 *                  access goes to slow L2 (measured +50..63% train cycles).
 *   -D SLAVE_STACK_L1: stacks=NULL -> SDK allocates the per-core stacks in fast
 *                  L1 TCDM (1-cycle). With a small SLAVESTACKSIZE this costs only
 *                  a few KB of L1 yet removes the L2 penalty (measured -22..-38%
 *                  vs L2). Use whenever the L1 budget has room for the stacks.
 *
 * Best config is per-experiment (see GAP9_TRAINING_MODEL_OVERRIDES): keep the
 * stack small and in L1 (SLAVE_STACK_L1) whenever arena+cc_stack+stacks fit the
 * 128KB TCDM; fall back to L2 only when the arena needs every byte (e.g. CCT). */
#define CLUSTER_MAX_CORES 9
#ifndef SLAVE_STACK_L1
static uint8_t cluster_slave_stacks[SLAVESTACKSIZE * CLUSTER_MAX_CORES]
    __attribute__((aligned(16)));
#define SET_SLAVE_STACK(task)                                                  \
  do {                                                                         \
    (task).slave_stack_size = SLAVESTACKSIZE;                                  \
    (task).stacks = cluster_slave_stacks;                                      \
  } while (0)
#else
#define SET_SLAVE_STACK(task)                                                  \
  do {                                                                         \
    (task).slave_stack_size = SLAVESTACKSIZE;                                  \
    (task).stacks = NULL; /* SDK allocates per-core stacks in fast L1 TCDM */  \
  } while (0)
#endif

/* -------------------------------------------------------------------------
 * Cluster device
 * ---------------------------------------------------------------------- */

struct pi_device cluster_dev;

/* -------------------------------------------------------------------------
 * Cycle accumulators — incremented after every RunTrainingNetwork /
 * RunOptimizerNetwork cluster dispatch.
 * ---------------------------------------------------------------------- */

static unsigned int g_train_cycles_acc = 0;
static unsigned int g_opt_cycles_acc = 0;

/* -------------------------------------------------------------------------
 * Loss storage (one value per forward pass)
 * ---------------------------------------------------------------------- */

#define TOTAL_FWD_PASSES (N_TRAIN_STEPS * N_ACCUM_STEPS)
static float stored_losses[TOTAL_FWD_PASSES];

/* -------------------------------------------------------------------------
 * Cluster wrappers — GAP9 generated functions take (core_id, numThreads)
 * ---------------------------------------------------------------------- */

static void InitTrainingNetworkWrapper(void *args) {
  (void)args;
  InitTrainingNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
}

#ifdef TRAINING_TESTDATA_L3
/* Large baked test inputs were moved out of L2 to L3; LoadTestDataL3 (generated
 * in testinputs.h) cl_ram_malloc's + load_file_to_ram's them on the cluster
 * controller (the file/ram APIs are cluster-delegated). */
static void LoadTestDataL3Wrapper(void *args) {
  (void)args;
  LoadTestDataL3();
}
#endif

static void RunTrainingNetworkWrapper(void *args) {
  (void)args;
  ResetTimer();
  StartTimer();
  RunTrainingNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
  g_train_cycles_acc += getCycles();
  StopTimer();
}

static void InitOptimizerNetworkWrapper(void *args) {
  (void)args;
  InitOptimizerNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
}

static void RunOptimizerNetworkWrapper(void *args) {
  (void)args;
  ResetTimer();
  StartTimer();
  RunOptimizerNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
  g_opt_cycles_acc += getCycles();
  StopTimer();
}

/* -------------------------------------------------------------------------
 * L3-aware memory transfer: handles all combinations of L2/L3 src and dst
 * ---------------------------------------------------------------------- */

static void l3_aware_copy(void *dst, const void *src, uint32_t bytes) {
  if (IS_L2(dst) && IS_L2(src)) {
    memcpy(dst, src, bytes);
  } else if (IS_L2(dst)) {
    /* L3 → L2 */
    ram_read(dst, (void *)src, bytes);
  } else if (IS_L2(src)) {
    /* L2 → L3 */
    ram_write(dst, (void *)src, bytes);
  } else {
    /* L3 → L3: stage through a temporary L2 buffer */
    void *tmp = pi_l2_malloc(bytes);
    ram_read(tmp, (void *)src, bytes);
    ram_write(dst, tmp, bytes);
    pi_l2_free(tmp, bytes);
  }
}

static void run_optimizer_step(void) {
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  /* --- Step A: copy current weights + grad acc → optimizer input buffers ---
   * Skipped when codegen has shared the buffers (pointer equality test).
   * The optimizer only has slots for weights that have an SGD node, which can
   * be fewer than the training graph's weight count (e.g. frozen params); cap
   * the loop by the optimizer input array (2 inputs — weight+grad — per slot)
   * so we never index past DeeployOptNetwork_inputs[]. */
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS &&
                        (2u * wi + 1u) < DeeployOptNetwork_num_inputs;
       wi++) {
    uint32_t train_w_idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    uint32_t train_g_idx = (uint32_t)TRAINING_GRAD_BUF_START_IDX + wi;
    uint32_t opt_w_in = 2u * wi;
    uint32_t opt_g_in = 2u * wi + 1u;

    if (DeeployOptNetwork_inputs[opt_w_in] !=
        DeeployNetwork_inputs[train_w_idx]) {
      l3_aware_copy(DeeployOptNetwork_inputs[opt_w_in],
                    DeeployNetwork_inputs[train_w_idx],
                    DeeployOptNetwork_inputs_bytes[opt_w_in]);
    }
    if (DeeployOptNetwork_inputs[opt_g_in] !=
        DeeployNetwork_inputs[train_g_idx]) {
      l3_aware_copy(DeeployOptNetwork_inputs[opt_g_in],
                    DeeployNetwork_inputs[train_g_idx],
                    DeeployOptNetwork_inputs_bytes[opt_g_in]);
    }
  }

  /* --- Step B: run optimizer kernel on cluster --- */
  struct pi_cluster_task opt_task;
  pi_cluster_task(&opt_task, RunOptimizerNetworkWrapper, NULL);
  SET_SLAVE_STACK(opt_task);
#ifdef POWER_MEASUREMENT
  WRITE_GPIO(1); /* OptimizerNetwork (SGD) dispatch — its own power peak */
#endif
  pi_cluster_send_task_to_cl(&cluster_dev, &opt_task);
#ifdef POWER_MEASUREMENT
  WRITE_GPIO(0);
#endif

  /* --- Step C: copy weight_updated back to training network's weight buffers
   * --- Skipped when codegen has shared the output buffer with the training
   * input. */
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS &&
                        wi < DeeployOptNetwork_num_outputs;
       wi++) {
    uint32_t train_w_idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    uint32_t opt_w_out = wi;

    if (DeeployOptNetwork_outputs[opt_w_out] ==
        DeeployNetwork_inputs[train_w_idx]) {
      continue; /* in-place: training buffer already updated */
    }

    uint32_t opt_bytes = DeeployOptNetwork_outputs_bytes[opt_w_out];
    uint32_t train_bytes = DeeployNetwork_inputs_bytes[train_w_idx];
    if (opt_bytes == train_bytes) {
      l3_aware_copy(DeeployNetwork_inputs[train_w_idx],
                    DeeployOptNetwork_outputs[opt_w_out], opt_bytes);
    } else {
      /* Broadcasted bias: fill every tile with updated value. */
      for (uint32_t off = 0; off < train_bytes; off += opt_bytes) {
        uint32_t chunk =
            (off + opt_bytes <= train_bytes) ? opt_bytes : (train_bytes - off);
        l3_aware_copy((char *)DeeployNetwork_inputs[train_w_idx] + off,
                      DeeployOptNetwork_outputs[opt_w_out], chunk);
      }
    }
  }
#endif /* TRAINING_NUM_WEIGHT_INPUTS */
}

/* -------------------------------------------------------------------------
 * Numerical comparison helpers — run on cluster (FC has no FPU)
 * ---------------------------------------------------------------------- */

typedef struct {
  float *computed;
  float *reference;
  uint32_t n;
  uint32_t *err_count;
  uint32_t *computed_bits_out; /* cluster's view of each computed loss (hex) */
} LossCompareArgs;

static void CompareLossesOnCluster(void *args) {
  if (pi_core_id() != 0)
    return;
  LossCompareArgs *a = (LossCompareArgs *)args;
  float tol = TRAINING_TOLERANCE_ABS;
  uint32_t errors = 0;
  for (uint32_t i = 0; i < a->n; i++) {
    if (a->computed_bits_out) {
      uint32_t b;
      memcpy(&b, &a->computed[i], sizeof(uint32_t));
      a->computed_bits_out[i] = b;
    }
    float diff = a->computed[i] - a->reference[i];
    if (diff < 0.0f)
      diff = -diff;
    printf("  [loss %u] computed=%.6f  ref=%.6f  diff=%.6f  TOL=%.6f\r\n", i,
           (double)a->computed[i], (double)a->reference[i], (double)diff,
           (double)tol);
    if (diff > tol) {
      errors++;
    }
  }
  *a->err_count = errors;
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(void) {

#ifdef POWER_MEASUREMENT
  /* Configure GPIO 89 as a digital trigger FIRST, at the very top of main —
   * exactly like the SDK's known-good GPIO example (helloworld.c) and the
   * inference harness: pad function -> output -> drive low, before any cluster
   * open / frequency change touches the IO subsystem. */
  pi_pad_function_set(GPIOs, 1);
  pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
  pi_gpio_pin_write(GPIOs, 0);
  WRITE_GPIO(0);
#endif

  BISECT(1); /* reached main, GPIO works */

  /* printf is safe here: the board build uses UART printf (sdk_board.config
   * CONFIG_IO_TYPE_UART), not semihosting — so it does not need an attached
   * debugger and does not hang when started via openocd load_and_start_binary.
   */
  printf("=== GAP9 Training Harness (Phase 2 — with OptimizerNetwork) ===\r\n");
  printf("N_TRAIN_STEPS=%u  N_ACCUM_STEPS=%u  DATA_INPUTS=%u\r\n",
         (unsigned)N_TRAIN_STEPS, (unsigned)N_ACCUM_STEPS,
         (unsigned)TRAINING_NUM_DATA_INPUTS);

  BISECT(9); /* banner region passed (before cluster open) */

  struct pi_cluster_conf conf;
  pi_cluster_conf_init(&conf);
  conf.id = 0;
  /* RW: The cluster-controller (CC / master core) stack lives at the bottom of
   * L1 (gap9 cluster.c carves cc_stack = l1_base + cc_stack_size, growing DOWN
   * toward the L1 base). The SDK default PI_CL_CC_STACK_SIZE is only 0x800
   * (2048B) -- far too small for Deeploy's nested L3->L2->L1 tiling-closure
   * call chain plus the forward-conv wrapper. With the default it overflows
   * below the L1 base: before the per-tile DMA tables were moved to L2 the
   * overflow landed in the L1 preload .data, silently clobbering an
   * OptimizerNetwork DMA `cmd` table entry -> garbage mchan command ->
   * mchan_transfer_wait hangs; after the tables moved to L2 it ran off the
   * bottom of L1 -> pe8 LSU invalid write at ~L1_base-0x20 in
   * pulp_conv2d_fp32_fw_cl. The pmsis pi_cluster_task path takes the size from
   * conf.cc_stack_size (NOT the AutoTiler-only CONFIG_CL_MASTER_CORE_STACK_SIZE
   * kconfig). Measured need ~3KB; give generous headroom -- L1 has room now
   * that the slave stacks and the tile control tables both live in L2. */
#ifndef CC_STACK_SIZE
#define CC_STACK_SIZE 8192
#endif
  conf.cc_stack_size = CC_STACK_SIZE;
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return -1;

  BISECT(2); /* cluster open ok */

#ifdef POWER_MEASUREMENT
  /* Pin the operating point (cluster domain requires the cluster powered on,
   * hence after pi_cluster_open). GPIO was already configured at the top of
   * main. Re-assert the pad function after the frequency change in case the
   * PERIPH FLL retune perturbed the IO mux. */
  pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC * 1000 * 1000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL * 1000 * 1000);
  pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE * 1000 * 1000);
#ifdef VOLTAGE
  pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
#endif
  printf("[POWER] FC=%dMHz CL=%dMHz PE=%dMHz (get FC=%d CL=%d)\r\n", FREQ_FC,
         FREQ_CL, FREQ_PE, pi_freq_get(PI_FREQ_DOMAIN_FC),
         pi_freq_get(PI_FREQ_DOMAIN_CL));

  pi_pad_function_set(GPIOs, 1);
  pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
  pi_gpio_pin_write(GPIOs, 0);
#endif

  BISECT(3); /* freq/voltage set */

  mem_init();
#ifndef NOFLASH
  open_fs();
#endif

  BISECT(4); /* mem_init + open_fs */

  struct pi_cluster_task cluster_task;

  /* ------------------------------------------------------------------
   * Init training network
   * ------------------------------------------------------------------ */

  printf("Initializing TrainingNetwork...\r\n");
  pi_cluster_task(&cluster_task, InitTrainingNetworkWrapper, NULL);
  SET_SLAVE_STACK(cluster_task);
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  BISECT(5); /* InitTrainingNetwork done */

#ifdef TRAINING_TESTDATA_L3
  /* Load the L3-resident test inputs (separate cluster task; runs after
   * InitTrainingNetwork so the FC is idle -> no alloc-path contention). */
  pi_cluster_task(&cluster_task, LoadTestDataL3Wrapper, NULL);
  SET_SLAVE_STACK(cluster_task);
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
#endif

  /* ------------------------------------------------------------------
   * Zero-initialise gradient accumulation buffers.
   * ------------------------------------------------------------------ */

  for (uint32_t _gi = 0; _gi < (uint32_t)TRAINING_NUM_GRAD_INPUTS; _gi++) {
    uint32_t _idx = (uint32_t)TRAINING_GRAD_BUF_START_IDX + _gi;
    uint32_t bytes = DeeployNetwork_inputs_bytes[_idx];
    void *buf = DeeployNetwork_inputs[_idx];
    if (IS_L2(buf)) {
      memset(buf, 0, bytes);
    } else {
      /* Write zeros into L3 via DMA using a temporary L2 zero page */
      uint8_t *zero_page = pi_l2_malloc(512);
      memset(zero_page, 0, 512);
      for (uint32_t off = 0; off < bytes; off += 512) {
        uint32_t chunk = (off + 512 <= bytes) ? 512 : (bytes - off);
        ram_write((char *)buf + off, zero_page, chunk);
      }
      pi_l2_free(zero_page, 512);
    }
  }

  BISECT(6); /* grad buffers zeroed (+ LoadTestDataL3 if L3) */

  /* ------------------------------------------------------------------
   * Init optimizer network
   * ------------------------------------------------------------------ */

  printf("Initializing OptimizerNetwork...\r\n");
  pi_cluster_task(&cluster_task, InitOptimizerNetworkWrapper, NULL);
  SET_SLAVE_STACK(cluster_task);
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  BISECT(7); /* InitOptimizerNetwork done */

  uint32_t reset_idx = DeeployNetwork_num_inputs - 1;

  /* ------------------------------------------------------------------
   * Copy initial weights into network input buffers.
   * (InitTrainingNetwork only malloc's them; testInitWeights[] holds
   *  the actual starting values from inputs.npz.)
   *
   * Skipped when TRAINING_SKIP_INITWEIGHT_COPY is defined: on GAP9 the
   * generated InitTrainingNetwork already loads every L3-resident weight from
   * its N.hex file at boot, so this copy (and the testInitWeights[] arrays it
   * reads) is redundant and would needlessly double the on-chip footprint.
   * ------------------------------------------------------------------ */

#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0) && \
    !defined(TRAINING_SKIP_INITWEIGHT_COPY)
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    l3_aware_copy(DeeployNetwork_inputs[idx], testInitWeights[wi],
                  DeeployNetwork_inputs_bytes[idx]);
  }
#endif

  BISECT(8); /* weight copy done — about to enter ROI */

  printf("Starting training (%u optimizer steps x %u accum steps)...\r\n",
         (unsigned)N_TRAIN_STEPS, (unsigned)N_ACCUM_STEPS);

  /* Power measurement: GPIO is driven per-dispatch (high during each
   * TrainingNetwork and OptimizerNetwork cluster run, low during host-side data
   * loading) rather than once around the whole loop — so the PPK2 trace shows a
   * separate peak for every fwd/bwd mini-batch and every optimizer step,
   * letting you attribute power AND time to each phase. See WRITE_GPIO in the
   * loop body and in run_optimizer_step(). */

  for (uint32_t update_step = 0; update_step < N_TRAIN_STEPS; update_step++) {

    for (uint32_t accum_step = 0; accum_step < N_ACCUM_STEPS; accum_step++) {

      uint32_t mb = update_step * N_ACCUM_STEPS + accum_step;

#ifndef POWER_MEASUREMENT
      /* Suppress per-mini-batch UART chatter inside the ROI — printing over the
       * UART during the measured window would inflate the power trace. */
      printf("  update %u/%u  accum %u/%u  (mini-batch %u)\r\n",
             update_step + 1, (unsigned)N_TRAIN_STEPS, accum_step + 1,
             (unsigned)N_ACCUM_STEPS, mb);
#endif

      /* ① Set lazy_reset_grad. */
      {
        void *reset_ptr = DeeployNetwork_inputs[reset_idx];
        uint8_t reset_val = (accum_step == 0) ? 1u : 0u;
        if (IS_L2(reset_ptr)) {
          *((uint8_t *)reset_ptr) = reset_val;
        } else {
          ram_write(reset_ptr, &reset_val, sizeof(uint8_t));
        }
      }

      /* ② Load this mini-batch's data + labels (cycle through unique samples).
       */
      for (uint32_t buf = 0; buf < TRAINING_NUM_DATA_INPUTS; buf++) {
        l3_aware_copy(DeeployNetwork_inputs[buf],
                      testDataVector[mb % TRAINING_DATA_SIZE][buf],
                      DeeployNetwork_inputs_bytes[buf]);
      }

      /* ③ Forward + backward + InPlaceAccumulatorV2. */
      pi_cluster_task(&cluster_task, RunTrainingNetworkWrapper, NULL);
      SET_SLAVE_STACK(cluster_task);
#ifdef POWER_MEASUREMENT
      /* One power peak per mini-batch = TrainingNetwork (fwd+bwd+accumulate,
       * fused in the compiled graph — they cannot be split host-side). GPIO is
       * low during the surrounding host-side L3 data loading, so each dispatch
       * shows as a cleanly separated peak in the PPK2 trace. */
      WRITE_GPIO(1);
#endif
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
#ifdef POWER_MEASUREMENT
      WRITE_GPIO(0);
#endif

      /* ④ Store loss — use memcpy to avoid float registers on FC (no FPU). */
      {
        void *loss_ptr = DeeployNetwork_outputs[0];
        if (IS_L2(loss_ptr)) {
          memcpy(&stored_losses[mb], loss_ptr, sizeof(float));
        } else {
          ram_read(&stored_losses[mb], loss_ptr, sizeof(float));
        }
        /* UART printf (not semihost) — safe even with openocd detached. */
        uint32_t _lbits;
        memcpy(&_lbits, &stored_losses[mb], sizeof(uint32_t));
        printf("LOSSLIVE %u hex=%08x\r\n", (unsigned)mb, (unsigned)_lbits);
      }

    } /* end accum_step loop */

    /* ⑤ SGD weight update via Deeploy-compiled OptimizerNetwork. */
    run_optimizer_step();

  } /* end update_step loop */

  /* ------------------------------------------------------------------
   * Numerical verification — run on cluster (FC has no FPU)
   * ------------------------------------------------------------------ */

  uint32_t loss_err_count = 0;
  uint32_t total_loss_checks =
      (TOTAL_FWD_PASSES < N_LOSS_REFS) ? TOTAL_FWD_PASSES : N_LOSS_REFS;
  static uint32_t cluster_computed_bits[TOTAL_FWD_PASSES];
  LossCompareArgs loss_cmp_args = {
      .computed = stored_losses,
      .reference = (float *)testLossRef,
      .n = total_loss_checks,
      .err_count = &loss_err_count,
      .computed_bits_out = cluster_computed_bits,
  };
  pi_cluster_task(&cluster_task, CompareLossesOnCluster, &loss_cmp_args);
  SET_SLAVE_STACK(cluster_task);
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  printf("Errors: %u out of %u\r\n", (unsigned)loss_err_count,
         (unsigned)total_loss_checks);

  /* ------------------------------------------------------------------
   * FC-side raw loss dump (no FPU needed): emit each computed/reference
   * loss as its 32-bit IEEE-754 hex so the host can decode the full
   * per-step trajectory. Parsed by panel_a loss-extraction script.
   * computed_hex = FC view, cluster_hex = cluster's view (post-compare).
   * ------------------------------------------------------------------ */
  for (uint32_t _li = 0; _li < total_loss_checks; _li++) {
    uint32_t _cbits, _rbits;
    memcpy(&_cbits, &stored_losses[_li], sizeof(uint32_t));
    memcpy(&_rbits, &((float *)testLossRef)[_li], sizeof(uint32_t));
    printf("LOSSDUMP %u computed_hex=%08x ref_hex=%08x cluster_hex=%08x\r\n",
           (unsigned)_li, (unsigned)_cbits, (unsigned)_rbits,
           (unsigned)cluster_computed_bits[_li]);
  }

  /* ------------------------------------------------------------------
   * Benchmark summary — parsed by benchmark_training.py
   * ------------------------------------------------------------------ */

  uint32_t weight_sram_bytes = 0;
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  for (uint32_t _wi = 0; _wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; _wi++) {
    weight_sram_bytes +=
        DeeployNetwork_inputs_bytes[(uint32_t)TRAINING_NUM_DATA_INPUTS + _wi];
  }
#endif

  printf("BENCH train_cycles=%u opt_cycles=%u weight_sram=%u\r\n",
         g_train_cycles_acc, g_opt_cycles_acc, weight_sram_bytes);

  return loss_err_count == 0 ? 0 : 1;
}
