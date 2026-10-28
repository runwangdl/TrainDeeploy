/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * QW (dual_pass_fusion exp4): does INTERLEAVING two independent, mirror-image streams reduce
 * cycles on a single-issue in-order core?
 *
 * A ZO step runs the forward graph twice (theta+eps*z, theta-eps*z). The two passes are
 * independent and have identical instruction structure. The question here is NOT the arithmetic
 * sharing of worklog Step 4 (one X*W base + X*Delta correction) -- this variant reduces no
 * arithmetic at all. It only reorders: instead of
 *
 *     for(i) A_i ;  for(i) B_i             (SEQ -- stream A to completion, then stream B)
 *
 * issue
 *
 *     for(i) { A_i ; B_i }                 (ILV -- alternating, one loop)
 *
 * so that B's instructions can occupy issue slots that A's load-use hazards would otherwise
 * waste. On RI5CY/CV32E40P (4-stage, in-order, single-issue) a load consumed by the next
 * instruction stalls; an independent instruction placed between them is free.
 *
 * WHY THIS MIGHT LOSE, and why the sweep below exists: interleaving doubles the live inner-loop
 * body, and Siracusa gives each core only a 512 B private L0 I-cache. exp3 measured the block-0
 * QZO conv at 22.8 % of its cycles in instruction refill already, so a bigger body can cost more
 * than the stalls it fills. The experiment therefore sweeps two axes that decide this:
 *
 *   TAPS  -- inner length. Block 0 is the extreme short case (num_col_im2col_w>>2 == 1, i.e. ONE
 *            SIMD iteration per matmul call), where loop and call overhead dominate. Longer taps
 *            move the loop toward being load-use bound, which is where interleaving should pay.
 *   ACC   -- accumulators per stream, i.e. the ILP the stream ALREADY has. pulp_nn_matmul_i8_i8_i8
 *            keeps 8 live accumulators (4 weight rows x 2 columns); with that much independent
 *            work in flight there may be no stall left for a second stream to fill. ACC=1 is the
 *            dependent, stall-prone extreme.
 *
 * Both variants compute the same sums from the same data, and the device asserts they agree, so a
 * cycle difference cannot come from doing less work.
 *
 * Arithmetic is the real kernel's: v4s SIMD int8 dot-product accumulate (`SumDotps4` ->
 * __builtin_pulp_sdotsp4), the primitive pulp_nn_matmul_i8_i8_i8 uses.
 */

#include "perf_utils.h"
#include "pmsis.h"
#include <stdio.h>

#define MAINSTACKSIZE 8000
#define SLAVESTACKSIZE 3800

typedef signed char v4s __attribute__((vector_size(4)));
#define DOTP(a, b, c) __builtin_pulp_sdotsp4((a), (b), (c))

/* Inner lengths swept (in v4s words, so TAPS=1 is block 0's single SIMD iteration). */
#define N_LEN 4
static const int LENGTHS[N_LEN] = {1, 8, 64, 256};
#define MAX_LEN 256
#define REPS 512 /* outer repeats, to lift each measurement clear of timer noise */

/* Operands live in L1, as the real kernel's tiles do. Two streams => two weight sets; the
 * activation is shared, exactly as in a ZO dual pass (same window, mirrored weights). */
static PI_CL_L1 v4s g_act[MAX_LEN];
static PI_CL_L1 v4s g_wA[MAX_LEN];
static PI_CL_L1 v4s g_wB[MAX_LEN];
static PI_CL_L1 int g_out[4];

struct pi_device cluster_dev;

/* ---------------------------------------------------------------------------
 * ACC = 1 : one accumulator per stream. Every DOTP depends on the previous one,
 *           so the stream has no internal ILP -- the best case for interleaving.
 * ------------------------------------------------------------------------ */
static void __attribute__((noinline))
seq_acc1(const v4s *act, const v4s *wA, const v4s *wB, int len, int *oA, int *oB) {
  int a = 0;
  for (int i = 0; i < len; i++)
    a = DOTP(act[i], wA[i], a);
  int b = 0;
  for (int i = 0; i < len; i++)
    b = DOTP(act[i], wB[i], b);
  *oA = a;
  *oB = b;
}

/* ILV-NS ("no share"): interleaved, but each stream loads the activation for ITSELF. This is the
 * control that isolates PURE SCHEDULING. `ilv_acc1` below additionally shares the activation load
 * -- legitimate for a ZO dual pass (same window, mirrored weights), but a different mechanism, and
 * without this control the two are indistinguishable in the cycle count. */
static void __attribute__((noinline))
ilv_acc1_ns(const v4s *act, const v4s *wA, const v4s *wB, int len, int *oA, int *oB) {
  int a = 0, b = 0;
  for (int i = 0; i < len; i++) {
    a = DOTP(act[i], wA[i], a);
    b = DOTP(act[i], wB[i], b);   /* deliberately a second load of act[i] */
  }
  *oA = a;
  *oB = b;
}

static void __attribute__((noinline))
ilv_acc1(const v4s *act, const v4s *wA, const v4s *wB, int len, int *oA, int *oB) {
  int a = 0, b = 0;
  for (int i = 0; i < len; i++) {
    v4s x = act[i]; /* the shared operand is loaded once, as in a fused pass */
    a = DOTP(x, wA[i], a);
    b = DOTP(x, wB[i], b);
  }
  *oA = a;
  *oB = b;
}

/* ---------------------------------------------------------------------------
 * ACC = 8 : eight accumulators per stream -- the ILP pulp_nn_matmul already has.
 *           If interleaving still helps here, it helps the real kernel.
 * ------------------------------------------------------------------------ */
#define BODY8(x, w, i, s)                                                      \
  do {                                                                         \
    s##0 = DOTP((x), (w)[(i)], s##0);                                          \
    s##1 = DOTP((x), (w)[(i)], s##1);                                          \
    s##2 = DOTP((x), (w)[(i)], s##2);                                          \
    s##3 = DOTP((x), (w)[(i)], s##3);                                          \
    s##4 = DOTP((x), (w)[(i)], s##4);                                          \
    s##5 = DOTP((x), (w)[(i)], s##5);                                          \
    s##6 = DOTP((x), (w)[(i)], s##6);                                          \
    s##7 = DOTP((x), (w)[(i)], s##7);                                          \
  } while (0)

#define DECL8(s) int s##0 = 0, s##1 = 0, s##2 = 0, s##3 = 0, s##4 = 0, s##5 = 0, s##6 = 0, s##7 = 0
#define RED8(s) (s##0 + s##1 + s##2 + s##3 + s##4 + s##5 + s##6 + s##7)

static void __attribute__((noinline))
ilv_acc8_ns(const v4s *act, const v4s *wA, const v4s *wB, int len, int *oA, int *oB) {
  DECL8(a);
  DECL8(b);
  for (int i = 0; i < len; i++) {
    BODY8(act[i], wA, i, a);
    BODY8(act[i], wB, i, b);      /* no shared temporary: pure reordering */
  }
  *oA = RED8(a);
  *oB = RED8(b);
}

static void __attribute__((noinline))
seq_acc8(const v4s *act, const v4s *wA, const v4s *wB, int len, int *oA, int *oB) {
  DECL8(a);
  for (int i = 0; i < len; i++) {
    v4s x = act[i];
    BODY8(x, wA, i, a);
  }
  DECL8(b);
  for (int i = 0; i < len; i++) {
    v4s x = act[i];
    BODY8(x, wB, i, b);
  }
  *oA = RED8(a);
  *oB = RED8(b);
}

static void __attribute__((noinline))
ilv_acc8(const v4s *act, const v4s *wA, const v4s *wB, int len, int *oA, int *oB) {
  DECL8(a);
  DECL8(b);
  for (int i = 0; i < len; i++) {
    v4s x = act[i];
    BODY8(x, wA, i, a);
    BODY8(x, wB, i, b);
  }
  *oA = RED8(a);
  *oB = RED8(b);
}

/* ------------------------------------------------------------------------ */

typedef void (*kern_t)(const v4s *, const v4s *, const v4s *, int, int *, int *);

static void measure(const char *tag, int acc, int len, kern_t fn, perf_stats_t *st, int *oA,
                    int *oB) {
  fn(g_act, g_wA, g_wB, len, oA, oB); /* warm the I-cache: we are measuring steady state */
  perf_bench_start();
  for (int r = 0; r < REPS; r++)
    fn(g_act, g_wA, g_wB, len, oA, oB);
  perf_bench_stop();
  perf_bench_read(st);
  printf("RESULT %s acc=%d len=%3d cycles=%10u instr=%10u ld=%9u ld_stall=%8u imiss=%8u "
         "jmp_stall=%8u sumA=%d sumB=%d\n",
         tag, acc, len, st->cycles, st->instr, st->ld, st->ld_stall, st->imiss, st->jmp_stall, *oA,
         *oB);
}

static void bench_body(void *arg) {
  (void)arg;
  if (pi_core_id() != 0)
    return; /* single-issue scheduling question: one core, no parallel confound */

  perf_bench_init();

  for (int k = 0; k < N_LEN; k++) {
    int len = LENGTHS[k];
    perf_stats_t s_seq1, s_ilv1, s_seq8, s_ilv8;
    int a1, b1, a2, b2, a3, b3, a4, b4;

    perf_stats_t s_ns1, s_ns8;
    int a5, b5, a6, b6;
    measure("SEQ   ", 1, len, seq_acc1, &s_seq1, &a1, &b1);
    measure("ILV_NS", 1, len, ilv_acc1_ns, &s_ns1, &a5, &b5);   /* pure scheduling */
    measure("ILV_SH", 1, len, ilv_acc1, &s_ilv1, &a2, &b2);     /* + shared activation load */
    measure("SEQ   ", 8, len, seq_acc8, &s_seq8, &a3, &b3);
    measure("ILV_NS", 8, len, ilv_acc8_ns, &s_ns8, &a6, &b6);
    measure("ILV_SH", 8, len, ilv_acc8, &s_ilv8, &a4, &b4);

    /* Same arithmetic, so a cycle win cannot be a work reduction. Asserted on device. */
    printf("CHECK len=%3d acc1_equal=%d acc8_equal=%d\n", len, (a1 == a2 && b1 == b2),
           (a3 == a4 && b3 == b4 && a3 == a6 && b3 == b6 && a1 == a5 && b1 == b5));
    printf("SPEEDUP len=%3d acc1: sched=%.4f sched+share=%.4f | acc8: sched=%.4f sched+share=%.4f\n",
           len, s_ns1.cycles ? (double)s_seq1.cycles / s_ns1.cycles : 0.0,
           s_ilv1.cycles ? (double)s_seq1.cycles / s_ilv1.cycles : 0.0,
           s_ns8.cycles ? (double)s_seq8.cycles / s_ns8.cycles : 0.0,
           s_ilv8.cycles ? (double)s_seq8.cycles / s_ilv8.cycles : 0.0);
  }
  printf("DUALPASS_BENCH_DONE\n");
}

int main(void) {
  struct pi_cluster_conf conf;
  pi_cluster_conf_init(&conf);
  conf.id = 0;
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return -1;

  /* Deterministic, non-trivial operands; wB is the mirror of wA, as theta-eps*z is of theta+eps*z. */
  for (int i = 0; i < MAX_LEN; i++) {
    signed char *a = (signed char *)&g_act[i];
    signed char *wa = (signed char *)&g_wA[i];
    signed char *wb = (signed char *)&g_wB[i];
    for (int j = 0; j < 4; j++) {
      a[j] = (signed char)(((i * 7 + j * 13) % 61) - 30);
      wa[j] = (signed char)(((i * 11 + j * 5) % 41) - 20);
      wb[j] = (signed char)(-wa[j]);
    }
  }
  g_out[0] = 0;

  struct pi_cluster_task task;
  pi_cluster_task(&task, bench_body, NULL);
  task.stack_size = MAINSTACKSIZE;
  task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &task);
  pi_cluster_close(&cluster_dev);
  return 0;
}
