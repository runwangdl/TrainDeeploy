#include "DeeployPULPMath.h"
#include "bsp/ram.h"
#include "dory_mem.h"
#include "mchan_siracusa.h"
#include "perf_utils.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "stdint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Network.h"

int8_t *DeeployNetwork_MEMORYARENA_L1;
int8_t *DeeployNetwork_MEMORYARENA_L2;
int8_t *DeeployNetwork_input_0;
int8_t *DeeployNetwork_input_1;
int32_t *DeeployNetwork_input_2;
int8_t *DeeployNetwork_output_0;

static PI_L2 int32_t DeeployNetwork_blocks1convweight_int8_pmul_tensor[16] = {234330, 242824, 241010, 274843, 272868, 260272, 252869, 248192,
                                                                              219675, 282055, 236004, 249942, 272232, 237316, 271684, 279170};

static PI_L2 int32_t DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor[16] = {1973910, 2045467, 2030184, 2315178, 2298543, 2192441, 2130083, 2090680,
                                                                              1850469, 2375935, 1988014, 2105422, 2293184, 1999064, 2288570, 2351634};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor[16] = {65, 63, 64, 56, 56, 59, 61, 62,
                                                                                                                 70, 54, 65, 61, 56, 65, 56, 55};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[3];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref;
} _rqsp_blocks1convweight_int8_tiling_closure_args_t;

static void _rqsp_blocks1convweight_int8_tiling_closure(void *_rqsp_blocks1convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convweight_int8_tiling_closure_args_t *args =
      (_rqsp_blocks1convweight_int8_tiling_closure_args_t *)_rqsp_blocks1convweight_int8_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_blocks1convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks1convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks1convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks1convweight_int8_chunk = (2048 >> rqsp_blocks1convweight_int8_log2Core) + ((2048 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks1convweight_int8_chunk_start = (uint32_t)MIN(rqsp_blocks1convweight_int8_chunk * rqsp_blocks1convweight_int8_core_id, (uint32_t)2048);
  uint32_t rqsp_blocks1convweight_int8_chunk_stop = (uint32_t)MIN(rqsp_blocks1convweight_int8_chunk_start + rqsp_blocks1convweight_int8_chunk, (uint32_t)2048);
  uint32_t rqsp_blocks1convweight_int8_local_size = rqsp_blocks1convweight_int8_chunk_stop - rqsp_blocks1convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks1convweight_int8_channel_start_offset = rqsp_blocks1convweight_int8_chunk_start % 128;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 2 + rqsp_blocks1convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks1convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref[rqsp_blocks1convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref[rqsp_blocks1convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref, 15, 128, chunk_seed, rqsp_blocks1convweight_int8_local_size,
        rqsp_blocks1convweight_int8_chunk_start, perturbation_sign, zo_eco_mom[2] + rqsp_blocks1convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref[rqsp_blocks1convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref[rqsp_blocks1convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref, 15, 128, chunk_seed, rqsp_blocks1convweight_int8_local_size,
        rqsp_blocks1convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks1convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref;
} _rqsp_blocks1convweight_int8_cluster_fork_args_t;

static void _rqsp_blocks1convweight_int8_cluster_fork(void *_rqsp_blocks1convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convweight_int8_cluster_fork_args_t *args = (_rqsp_blocks1convweight_int8_cluster_fork_args_t *)_rqsp_blocks1convweight_int8_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks1convweight_int8_tiling_closure_args_t DeeployNetwork__rqsp_blocks1convweight_int8_tiling_closure_args =
      (_rqsp_blocks1convweight_int8_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref};

  // _rqsp_blocks1convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_blocks1convweight_int8_tiling_closure(&DeeployNetwork__rqsp_blocks1convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr;
} _rqsp_blocks1convweight_int8_closure_args_t;

static void _rqsp_blocks1convweight_int8_closure(void *_rqsp_blocks1convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convweight_int8_closure_args_t *args = (_rqsp_blocks1convweight_int8_closure_args_t *)_rqsp_blocks1convweight_int8_closure_args;

  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor = args->DeeployNetwork_blocks1convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4096);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_blocks1convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_blocks1convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_input_1_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_blocks1convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_blocks1convweight_int8_cluster_fork_args_t DeeployNetwork__rqsp_blocks1convweight_int8_cluster_fork_args =
        (_rqsp_blocks1convweight_int8_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks1convweight_int8_cluster_fork, &DeeployNetwork__rqsp_blocks1convweight_int8_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312768, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_blocks1convweight_int8_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr;
} _rqsp_blocks1convweight_int8_closure_L3_args_t;

static void _rqsp_blocks1convweight_int8_closure_L3(void *_rqsp_blocks1convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convweight_int8_closure_L3_args_t *args = (_rqsp_blocks1convweight_int8_closure_L3_args_t *)_rqsp_blocks1convweight_int8_closure_L3_args;

  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor = args->DeeployNetwork_blocks1convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks1convweight_int8_closure_args_t DeeployNetwork__rqsp_blocks1convweight_int8_closure_args = (_rqsp_blocks1convweight_int8_closure_args_t){
      .DeeployNetwork_blocks1convweight_int8_pert_tensor = DeeployNetwork_blocks1convweight_int8_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr};

  // _rqsp_blocks1convweight_int8_closure CLOSURE CALL
  _rqsp_blocks1convweight_int8_closure(&DeeployNetwork__rqsp_blocks1convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref;
} _rqsp_blocks1convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_blocks1convbias_rqsadd_tiling_closure(void *_rqsp_blocks1convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_blocks1convbias_rqsadd_tiling_closure_args_t *)_rqsp_blocks1convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_blocks1convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks1convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks1convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks1convbias_rqsadd_chunk = (16 >> rqsp_blocks1convbias_rqsadd_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks1convbias_rqsadd_chunk_start = (uint32_t)MIN(rqsp_blocks1convbias_rqsadd_chunk * rqsp_blocks1convbias_rqsadd_core_id, (uint32_t)16);
  uint32_t rqsp_blocks1convbias_rqsadd_chunk_stop = (uint32_t)MIN(rqsp_blocks1convbias_rqsadd_chunk_start + rqsp_blocks1convbias_rqsadd_chunk, (uint32_t)16);
  uint32_t rqsp_blocks1convbias_rqsadd_local_size = rqsp_blocks1convbias_rqsadd_chunk_stop - rqsp_blocks1convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks1convbias_rqsadd_channel_start_offset = rqsp_blocks1convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 3 + rqsp_blocks1convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks1convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref[rqsp_blocks1convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref[rqsp_blocks1convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks1convbias_rqsadd_local_size,
        rqsp_blocks1convbias_rqsadd_chunk_start, perturbation_sign, zo_eco_mom[3] + rqsp_blocks1convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref[rqsp_blocks1convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref[rqsp_blocks1convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks1convbias_rqsadd_local_size,
        rqsp_blocks1convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks1convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref;
} _rqsp_blocks1convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_blocks1convbias_rqsadd_cluster_fork(void *_rqsp_blocks1convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convbias_rqsadd_cluster_fork_args_t *args = (_rqsp_blocks1convbias_rqsadd_cluster_fork_args_t *)_rqsp_blocks1convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks1convbias_rqsadd_tiling_closure_args_t DeeployNetwork__rqsp_blocks1convbias_rqsadd_tiling_closure_args =
      (_rqsp_blocks1convbias_rqsadd_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref};

  // _rqsp_blocks1convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_blocks1convbias_rqsadd_tiling_closure(&DeeployNetwork__rqsp_blocks1convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks1convbias_rqsadd_closure_args_t;

static void _rqsp_blocks1convbias_rqsadd_closure(void *_rqsp_blocks1convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convbias_rqsadd_closure_args_t *args = (_rqsp_blocks1convbias_rqsadd_closure_args_t *)_rqsp_blocks1convbias_rqsadd_closure_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_blocks1convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_blocks1convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_input_2_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_blocks1convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_blocks1convbias_rqsadd_cluster_fork_args_t DeeployNetwork__rqsp_blocks1convbias_rqsadd_cluster_fork_args =
        (_rqsp_blocks1convbias_rqsadd_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks1convbias_rqsadd_cluster_fork, &DeeployNetwork__rqsp_blocks1convbias_rqsadd_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_blocks1convbias_rqsadd_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks1convbias_rqsadd_closure_L3_args_t;

static void _rqsp_blocks1convbias_rqsadd_closure_L3(void *_rqsp_blocks1convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks1convbias_rqsadd_closure_L3_args_t *args = (_rqsp_blocks1convbias_rqsadd_closure_L3_args_t *)_rqsp_blocks1convbias_rqsadd_closure_L3_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks1convbias_rqsadd_closure_args_t DeeployNetwork__rqsp_blocks1convbias_rqsadd_closure_args = (_rqsp_blocks1convbias_rqsadd_closure_args_t){
      .DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = DeeployNetwork_blocks1convbias_rqsadd_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr};

  // _rqsp_blocks1convbias_rqsadd_closure CLOSURE CALL
  _rqsp_blocks1convbias_rqsadd_closure(&DeeployNetwork__rqsp_blocks1convbias_rqsadd_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure(void *__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 8, 14, 87] -> [1, 14, 87, 8] (Name: _MERGE_CONVRQ_PASS_0_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 87;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
  const uint32_t offS_m1 = 0;
  const uint32_t offD_m1 = 0;
  const uint32_t sIn_3 = 1;
  const uint32_t sIn_2 = sIn_3 * dimLen_3;
  const uint32_t sIn_1 = sIn_2 * dimLen_2;
  const uint32_t sIn_0 = sIn_1 * dimLen_1;
  const uint32_t sOut_3 = 1;
  const uint32_t sOut_2 = sOut_3 * dimLen_1;
  const uint32_t sOut_1 = sOut_2 * dimLen_3;
  const uint32_t sOut_0 = sOut_1 * dimLen_2;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {
    const uint32_t offS_0 = offS_m1 + i_0 * sIn_0;
    const uint32_t offD_0 = offD_m1 + i_0 * sOut_0;

    for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {
      const uint32_t offS_1 = offS_0 + i_2 * sIn_2;
      const uint32_t offD_1 = offD_0 + i_2 * sOut_1;

      const uint32_t baseChunk_2 = dimLen_3 / NUM_CORES;
      const uint32_t leftover_2 = dimLen_3 - baseChunk_2 * NUM_CORES;
      const uint32_t offset_2 = baseChunk_2 * coreId + (coreId < leftover_2 ? coreId : leftover_2);
      const uint32_t chunk_2 = coreId < leftover_2 ? baseChunk_2 + 1 : baseChunk_2;
      for (uint32_t i_3 = offset_2; i_3 < offset_2 + chunk_2; i_3++) {
        const uint32_t offS_2 = offS_1 + i_3 * sIn_3;
        const uint32_t offD_2 = offD_1 + i_3 * sOut_2;

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {
          const uint32_t offS_3 = offS_2 + i_1 * sIn_1;
          const uint32_t offD_3 = offD_2 + i_1 * sOut_3;

          dst[offD_3] = src[offS_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork(void *__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *)__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args =
      (__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
                                                                          DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                                                                      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
                                                                          DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref};

  // __MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_input_0_transpose_closure(void *__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *)__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args;

  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 9744);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 + DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1451536, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    __MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args =
        (__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
                                                                          DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                                                                      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
                                                                          DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1320464, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(void *__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *)__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args;

  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args =
      (__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t){.DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed =
                                                                   DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed,
                                                               .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                                                                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_input_0_transpose_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_input_0_transpose_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure(
    void *__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [16, 8, 1, 16] -> [16, 1, 16, 8] (Name: _MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 16;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 1;

  uint16_t dimLen_3 = 16;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref;
  const uint32_t offS_m1 = 0;
  const uint32_t offD_m1 = 0;
  const uint32_t sIn_3 = 1;
  const uint32_t sIn_2 = sIn_3 * dimLen_3;
  const uint32_t sIn_1 = sIn_2 * dimLen_2;
  const uint32_t sIn_0 = sIn_1 * dimLen_1;
  const uint32_t sOut_3 = 1;
  const uint32_t sOut_2 = sOut_3 * dimLen_1;
  const uint32_t sOut_1 = sOut_2 * dimLen_3;
  const uint32_t sOut_0 = sOut_1 * dimLen_2;

  const uint32_t baseChunk_0 = dimLen_0 / NUM_CORES;
  const uint32_t leftover_0 = dimLen_0 - baseChunk_0 * NUM_CORES;
  const uint32_t offset_0 = baseChunk_0 * coreId + (coreId < leftover_0 ? coreId : leftover_0);
  const uint32_t chunk_0 = coreId < leftover_0 ? baseChunk_0 + 1 : baseChunk_0;
  for (uint32_t i_0 = offset_0; i_0 < offset_0 + chunk_0; i_0++) {
    const uint32_t offS_0 = offS_m1 + i_0 * sIn_0;
    const uint32_t offD_0 = offD_m1 + i_0 * sOut_0;

    for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {
      const uint32_t offS_1 = offS_0 + i_2 * sIn_2;
      const uint32_t offD_1 = offD_0 + i_2 * sOut_1;

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {
        const uint32_t offS_2 = offS_1 + i_3 * sIn_3;
        const uint32_t offD_2 = offD_1 + i_3 * sOut_2;

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {
          const uint32_t offS_3 = offS_2 + i_1 * sIn_1;
          const uint32_t offD_3 = offD_2 + i_1 * sOut_3;

          dst[offD_3] = src[offS_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_cluster_fork(
    void *__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args =
          (__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref};

  // __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure(
    void *__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_args;

  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor = args->DeeployNetwork_blocks1convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_blocks1convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1443840, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_blocks1convweight_int8_pert_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args =
            (__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1312768, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3(
    void *__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args;

  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor = args->DeeployNetwork_blocks1convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_args =
          (__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_args_t){
              .DeeployNetwork_blocks1convweight_int8_pert_tensor = DeeployNetwork_blocks1convweight_int8_pert_tensor,
              .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed =
                  DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} __MERGE_CONVRQ_PASS_0_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_tiling_closure(void *__MERGE_CONVRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_tiling_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_0_tiling_closure_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP CONV 4x2 (single-pass control kernel for exp7a step 2a; Name: _MERGE_CONVRQ_PASS_0, Op: RequantizedConv)  -- QW
  PULP_Conv4x2_i8_i8_i8_HWC(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref, (int8_t *)DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer,
                            DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
                            DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                            DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref, NUM_CORES, 16, 87, 14, 8, 88, 14, 16, 16, 1, 0, 0, 8, 8, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} __MERGE_CONVRQ_PASS_0_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_0_cluster_fork(void *__MERGE_CONVRQ_PASS_0_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_cluster_fork_args_t *args = (__MERGE_CONVRQ_PASS_0_cluster_fork_args_t *)__MERGE_CONVRQ_PASS_0_cluster_fork_args;

  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_tiling_closure_args = (__MERGE_CONVRQ_PASS_0_tiling_closure_args_t){
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

  // __MERGE_CONVRQ_PASS_0_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_closure(void *__MERGE_CONVRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_args_t *)__MERGE_CONVRQ_PASS_0_closure_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19712);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 29456);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 33616);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 33552);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 31504);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1451536, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convbias_rqsadd_pert_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    __MERGE_CONVRQ_PASS_0_cluster_fork_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_cluster_fork_args = (__MERGE_CONVRQ_PASS_0_cluster_fork_args_t){
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_cluster_fork, &DeeployNetwork___MERGE_CONVRQ_PASS_0_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1330432, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_closure_L3(void *__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)__MERGE_CONVRQ_PASS_0_closure_L3_args;

  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args = (__MERGE_CONVRQ_PASS_0_closure_args_t){
      .DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = DeeployNetwork_blocks1convbias_rqsadd_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure(void *__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 14, 88, 16] -> [1, 16, 14, 88] (Name: _MERGE_CONVRQ_PASS_0_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 14;

  uint16_t dimLen_2 = 88;

  uint16_t dimLen_3 = 16;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_3][dimLen_1][dimLen_2])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;
  const uint32_t offS_m1 = 0;
  const uint32_t offD_m1 = 0;
  const uint32_t sIn_3 = 1;
  const uint32_t sIn_2 = sIn_3 * dimLen_3;
  const uint32_t sIn_1 = sIn_2 * dimLen_2;
  const uint32_t sIn_0 = sIn_1 * dimLen_1;
  const uint32_t sOut_3 = 1;
  const uint32_t sOut_2 = sOut_3 * dimLen_2;
  const uint32_t sOut_1 = sOut_2 * dimLen_1;
  const uint32_t sOut_0 = sOut_1 * dimLen_3;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {
    const uint32_t offS_0 = offS_m1 + i_0 * sIn_0;
    const uint32_t offD_0 = offD_m1 + i_0 * sOut_0;

    for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {
      const uint32_t offS_1 = offS_0 + i_3 * sIn_3;
      const uint32_t offD_1 = offD_0 + i_3 * sOut_1;

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {
        const uint32_t offS_2 = offS_1 + i_1 * sIn_1;
        const uint32_t offD_2 = offD_1 + i_1 * sOut_2;

        const uint32_t baseChunk_3 = dimLen_2 / NUM_CORES;
        const uint32_t leftover_3 = dimLen_2 - baseChunk_3 * NUM_CORES;
        const uint32_t offset_3 = baseChunk_3 * coreId + (coreId < leftover_3 ? coreId : leftover_3);
        const uint32_t chunk_3 = coreId < leftover_3 ? baseChunk_3 + 1 : baseChunk_3;
        for (uint32_t i_2 = offset_3; i_2 < offset_3 + chunk_3; i_2++) {
          const uint32_t offS_3 = offS_2 + i_2 * sIn_2;
          const uint32_t offD_3 = offD_2 + i_2 * sOut_3;

          dst[offD_3] = src[offS_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork(void *__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t *)__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args =
      (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref};

  // __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure(void *__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t *)__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args;

  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19712);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 + DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1461504, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args =
        (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1330432, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3(void *__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t *)__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args;

  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args =
      (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args_t){
          .DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork() {
  int8_t *DeeployNetwork_blocks1convweight_int8_pert_tensor;
  int32_t *DeeployNetwork_blocks1convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr;
    DeeployNetwork_blocks1convweight_int8_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _rqsp_blocks1convweight_int8_closure_L3_args_t DeeployNetwork__rqsp_blocks1convweight_int8_closure_L3_args =
        (_rqsp_blocks1convweight_int8_closure_L3_args_t){.DeeployNetwork_blocks1convweight_int8_pert_tensor = DeeployNetwork_blocks1convweight_int8_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convweight_int8_tileIdxPtr};

    // _rqsp_blocks1convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_blocks1convweight_int8_closure_L3(&DeeployNetwork__rqsp_blocks1convweight_int8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr;
    DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 19712);
    _rqsp_blocks1convbias_rqsadd_closure_L3_args_t DeeployNetwork__rqsp_blocks1convbias_rqsadd_closure_L3_args =
        (_rqsp_blocks1convbias_rqsadd_closure_L3_args_t){.DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = DeeployNetwork_blocks1convbias_rqsadd_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks1convbias_rqsadd_tileIdxPtr};

    // _rqsp_blocks1convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_blocks1convbias_rqsadd_closure_L3(&DeeployNetwork__rqsp_blocks1convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 19776);
    __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args =
        (__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t){.DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed =
                                                                        DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                                                                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 29520);
    __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args =
            (__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args_t){
                .DeeployNetwork_blocks1convweight_int8_pert_tensor = DeeployNetwork_blocks1convweight_int8_pert_tensor,
                .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed =
                    DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3(
        &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    __MERGE_CONVRQ_PASS_0_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
        .DeeployNetwork_blocks1convbias_rqsadd_pert_tensor = DeeployNetwork_blocks1convbias_rqsadd_pert_tensor,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks1convweight_int8_pert_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr;
    __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args =
        (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args_t){
            .DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_output_0_pre_transpose_closure_L3_args);
  }
}

void InitNetwork() {

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 39424);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 39424);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 2048);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 11856);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 11792);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 19712);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
