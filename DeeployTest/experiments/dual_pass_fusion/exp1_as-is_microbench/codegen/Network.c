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
float32_t *DeeployNetwork_input_0;
float32_t *DeeployNetwork_input_1;
float32_t *DeeployNetwork_input_2;
float32_t *DeeployNetwork_output_0;

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_numTiles[2] = {0, 4};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_cumByteOffset[4] = {0, 32, 64, 96};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_cumByteOffset[4] = {0, 8, 16, 24};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_cumByteOffset[4] = {0, 8, 16, 24};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_numTiles[2] = {0, 8};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_cumByteOffset[8] = {0,  4,  8,  12,
                                                                                                                                         16, 20, 24, 28};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_cumByteOffset[8] = {
    0, 39256, 78512, 117768, 157024, 196280, 235536, 274792};

void *DeeployNetwork_inputs[3];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref;
} _perturbrademacher_0_blocks_0_0_weight_tiling_closure_args_t;

static void _perturbrademacher_0_blocks_0_0_weight_tiling_closure(void *_perturbrademacher_0_blocks_0_0_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_0_blocks_0_0_weight_tiling_closure_args_t *args =
      (_perturbrademacher_0_blocks_0_0_weight_tiling_closure_args_t *)_perturbrademacher_0_blocks_0_0_weight_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_0_blocks_0_0_weight, Op: PerturbRademacher)
  uint8_t perturbrademacher_0_blocks_0_0_weight_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_0_blocks_0_0_weight_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_0_blocks_0_0_weight_chunk = (32 >> perturbrademacher_0_blocks_0_0_weight_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_0_blocks_0_0_weight_chunk_start =
      (uint32_t)MIN(perturbrademacher_0_blocks_0_0_weight_chunk * perturbrademacher_0_blocks_0_0_weight_core_id, (uint32_t)32);
  uint32_t perturbrademacher_0_blocks_0_0_weight_chunk_stop =
      (uint32_t)MIN(perturbrademacher_0_blocks_0_0_weight_chunk_start + perturbrademacher_0_blocks_0_0_weight_chunk, (uint32_t)32);
  uint32_t perturbrademacher_0_blocks_0_0_weight_local_size =
      perturbrademacher_0_blocks_0_0_weight_chunk_stop - perturbrademacher_0_blocks_0_0_weight_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 0 + perturbrademacher_0_blocks_0_0_weight_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_0_blocks_0_0_weight_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
#ifdef USE_SEQUENTIAL_RADEMACHER
  ApplySequentialRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref[perturbrademacher_0_blocks_0_0_weight_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref[perturbrademacher_0_blocks_0_0_weight_chunk_start],
      chunk_seed,
      perturbation_sign, // ZO runtime control (ZORuntime.h)
      perturbrademacher_0_blocks_0_0_weight_local_size, perturbrademacher_0_blocks_0_0_weight_eps);
#else
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref[perturbrademacher_0_blocks_0_0_weight_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref[perturbrademacher_0_blocks_0_0_weight_chunk_start],
      chunk_seed,
      perturbation_sign, // ZO runtime control (ZORuntime.h)
      perturbrademacher_0_blocks_0_0_weight_local_size, perturbrademacher_0_blocks_0_0_weight_eps);
#endif

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref;
} _perturbrademacher_0_blocks_0_0_weight_cluster_fork_args_t;

static void _perturbrademacher_0_blocks_0_0_weight_cluster_fork(void *_perturbrademacher_0_blocks_0_0_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_0_blocks_0_0_weight_cluster_fork_args_t *args =
      (_perturbrademacher_0_blocks_0_0_weight_cluster_fork_args_t *)_perturbrademacher_0_blocks_0_0_weight_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_0_blocks_0_0_weight_tiling_closure_args_t DeeployNetwork__perturbrademacher_0_blocks_0_0_weight_tiling_closure_args =
      (_perturbrademacher_0_blocks_0_0_weight_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref};

  // _perturbrademacher_0_blocks_0_0_weight_tiling_closure CLOSURE CALL
  _perturbrademacher_0_blocks_0_0_weight_tiling_closure(&DeeployNetwork__perturbrademacher_0_blocks_0_0_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_0_blocks_0_0_weight_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr;
} _perturbrademacher_0_blocks_0_0_weight_closure_args_t;

static void _perturbrademacher_0_blocks_0_0_weight_closure(void *_perturbrademacher_0_blocks_0_0_weight_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_0_blocks_0_0_weight_closure_args_t *args =
      (_perturbrademacher_0_blocks_0_0_weight_closure_args_t *)_perturbrademacher_0_blocks_0_0_weight_closure_args;

  float32_t *DeeployNetwork_0_blocks_0_0_weight_tensor = args->DeeployNetwork_0_blocks_0_0_weight_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 + DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_0_blocks_0_0_weight_tensor_ref =
      (void *)((char *)DeeployNetwork_0_blocks_0_0_weight_tensor + DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_cumByteOffset
                                                                       [*DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_input_1_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_0_blocks_0_0_weight_cluster_fork_args_t DeeployNetwork__perturbrademacher_0_blocks_0_0_weight_cluster_fork_args =
        (_perturbrademacher_0_blocks_0_0_weight_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_in_ref,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_0_blocks_0_0_weight_cluster_fork,
                    &DeeployNetwork__perturbrademacher_0_blocks_0_0_weight_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_0_blocks_0_0_weight_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_0_blocks_0_0_weight_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr;
} _perturbrademacher_0_blocks_0_0_weight_closure_L3_args_t;

static void _perturbrademacher_0_blocks_0_0_weight_closure_L3(void *_perturbrademacher_0_blocks_0_0_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_0_blocks_0_0_weight_closure_L3_args_t *args =
      (_perturbrademacher_0_blocks_0_0_weight_closure_L3_args_t *)_perturbrademacher_0_blocks_0_0_weight_closure_L3_args;

  float32_t *DeeployNetwork_0_blocks_0_0_weight_tensor = args->DeeployNetwork_0_blocks_0_0_weight_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_0_blocks_0_0_weight_closure_args_t DeeployNetwork__perturbrademacher_0_blocks_0_0_weight_closure_args =
      (_perturbrademacher_0_blocks_0_0_weight_closure_args_t){.DeeployNetwork_0_blocks_0_0_weight_tensor = DeeployNetwork_0_blocks_0_0_weight_tensor,
                                                              .DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr =
                                                                  DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr};

  // _perturbrademacher_0_blocks_0_0_weight_closure CLOSURE CALL
  _perturbrademacher_0_blocks_0_0_weight_closure(&DeeployNetwork__perturbrademacher_0_blocks_0_0_weight_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref;
} _perturbrademacher_1_blocks_0_0_bias_tiling_closure_args_t;

static void _perturbrademacher_1_blocks_0_0_bias_tiling_closure(void *_perturbrademacher_1_blocks_0_0_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_1_blocks_0_0_bias_tiling_closure_args_t *args =
      (_perturbrademacher_1_blocks_0_0_bias_tiling_closure_args_t *)_perturbrademacher_1_blocks_0_0_bias_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_1_blocks_0_0_bias, Op: PerturbRademacher)
  uint8_t perturbrademacher_1_blocks_0_0_bias_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_1_blocks_0_0_bias_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_1_blocks_0_0_bias_chunk = (8 >> perturbrademacher_1_blocks_0_0_bias_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_1_blocks_0_0_bias_chunk_start =
      (uint32_t)MIN(perturbrademacher_1_blocks_0_0_bias_chunk * perturbrademacher_1_blocks_0_0_bias_core_id, (uint32_t)8);
  uint32_t perturbrademacher_1_blocks_0_0_bias_chunk_stop =
      (uint32_t)MIN(perturbrademacher_1_blocks_0_0_bias_chunk_start + perturbrademacher_1_blocks_0_0_bias_chunk, (uint32_t)8);
  uint32_t perturbrademacher_1_blocks_0_0_bias_local_size = perturbrademacher_1_blocks_0_0_bias_chunk_stop - perturbrademacher_1_blocks_0_0_bias_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 1 + perturbrademacher_1_blocks_0_0_bias_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_1_blocks_0_0_bias_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
#ifdef USE_SEQUENTIAL_RADEMACHER
  ApplySequentialRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref[perturbrademacher_1_blocks_0_0_bias_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref[perturbrademacher_1_blocks_0_0_bias_chunk_start],
      chunk_seed,
      perturbation_sign, // ZO runtime control (ZORuntime.h)
      perturbrademacher_1_blocks_0_0_bias_local_size, perturbrademacher_1_blocks_0_0_bias_eps);
#else
  ApplyRademacherPerturbation(
      (const float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref[perturbrademacher_1_blocks_0_0_bias_chunk_start],
      (float32_t *)&DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref[perturbrademacher_1_blocks_0_0_bias_chunk_start],
      chunk_seed,
      perturbation_sign, // ZO runtime control (ZORuntime.h)
      perturbrademacher_1_blocks_0_0_bias_local_size, perturbrademacher_1_blocks_0_0_bias_eps);
#endif

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref;
} _perturbrademacher_1_blocks_0_0_bias_cluster_fork_args_t;

static void _perturbrademacher_1_blocks_0_0_bias_cluster_fork(void *_perturbrademacher_1_blocks_0_0_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_1_blocks_0_0_bias_cluster_fork_args_t *args =
      (_perturbrademacher_1_blocks_0_0_bias_cluster_fork_args_t *)_perturbrademacher_1_blocks_0_0_bias_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_1_blocks_0_0_bias_tiling_closure_args_t DeeployNetwork__perturbrademacher_1_blocks_0_0_bias_tiling_closure_args =
      (_perturbrademacher_1_blocks_0_0_bias_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref =
                                                                       DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref,
                                                                   .DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref =
                                                                       DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref};

  // _perturbrademacher_1_blocks_0_0_bias_tiling_closure CLOSURE CALL
  _perturbrademacher_1_blocks_0_0_bias_tiling_closure(&DeeployNetwork__perturbrademacher_1_blocks_0_0_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_1_blocks_0_0_bias_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr;
} _perturbrademacher_1_blocks_0_0_bias_closure_args_t;

static void _perturbrademacher_1_blocks_0_0_bias_closure(void *_perturbrademacher_1_blocks_0_0_bias_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_1_blocks_0_0_bias_closure_args_t *args =
      (_perturbrademacher_1_blocks_0_0_bias_closure_args_t *)_perturbrademacher_1_blocks_0_0_bias_closure_args;

  float32_t *DeeployNetwork_1_blocks_0_0_bias_tensor = args->DeeployNetwork_1_blocks_0_0_bias_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 + DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_1_blocks_0_0_bias_tensor_ref =
      (void *)((char *)DeeployNetwork_1_blocks_0_0_bias_tensor + DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_cumByteOffset
                                                                     [*DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_input_2_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_1_blocks_0_0_bias_cluster_fork_args_t DeeployNetwork__perturbrademacher_1_blocks_0_0_bias_cluster_fork_args =
        (_perturbrademacher_1_blocks_0_0_bias_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref =
                                                                       DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_in_ref,
                                                                   .DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref =
                                                                       DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_1_blocks_0_0_bias_cluster_fork,
                    &DeeployNetwork__perturbrademacher_1_blocks_0_0_bias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_1_blocks_0_0_bias_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_1_blocks_0_0_bias_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr;
} _perturbrademacher_1_blocks_0_0_bias_closure_L3_args_t;

static void _perturbrademacher_1_blocks_0_0_bias_closure_L3(void *_perturbrademacher_1_blocks_0_0_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_1_blocks_0_0_bias_closure_L3_args_t *args =
      (_perturbrademacher_1_blocks_0_0_bias_closure_L3_args_t *)_perturbrademacher_1_blocks_0_0_bias_closure_L3_args;

  float32_t *DeeployNetwork_1_blocks_0_0_bias_tensor = args->DeeployNetwork_1_blocks_0_0_bias_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_1_blocks_0_0_bias_closure_args_t DeeployNetwork__perturbrademacher_1_blocks_0_0_bias_closure_args =
      (_perturbrademacher_1_blocks_0_0_bias_closure_args_t){.DeeployNetwork_1_blocks_0_0_bias_tensor = DeeployNetwork_1_blocks_0_0_bias_tensor,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr =
                                                                DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr};

  // _perturbrademacher_1_blocks_0_0_bias_closure CLOSURE CALL
  _perturbrademacher_1_blocks_0_0_bias_closure(&DeeployNetwork__perturbrademacher_1_blocks_0_0_bias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref;
} _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure(void *_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure_args_t *)_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 1, 14, 700] -> [1, 14, 700, 1] (Name: node_1_blocks_0_blocks_0_0_Conv_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 700;

  // RW: GCC Segmentation fault
  float32_t(*src)[dimLen_1][dimLen_2][dimLen_3] =
      (float32_t(*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref;
  float32_t(*dst)[dimLen_2][dimLen_3][dimLen_1] =
      (float32_t(*)[dimLen_2][dimLen_3][dimLen_1])DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_2 / NUM_CORES;
    const uint32_t leftover = dimLen_2 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_2 = offset; i_2 < offset + chunk; i_2++) {

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

          dst[i_0][i_2][i_3][i_1] = src[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref;
} _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_cluster_fork_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_cluster_fork(void *_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_cluster_fork_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_cluster_fork_args_t *)_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure_args_t
      DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure_args =
          (_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref};

  // _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure CLOSURE CALL
  _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure(&DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr;
} _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure(void *_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_args_t *)_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_args;

  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed = args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39200);
  void *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 + DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1480992, DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_cluster_fork_args_t DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_cluster_fork_args =
        (_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_cluster_fork,
                    &DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1349920, DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr;
} _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3(void *_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3_args_t *)_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3_args;

  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed = args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_args_t DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_args =
      (_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_args_t){
          .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed = DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr};

  // _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure CLOSURE CALL
  _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure(&DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref;
} _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure(
    void *_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure_args_t *)
          _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [8, 1, 1, 4] -> [8, 1, 4, 1] (Name: node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 8;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 1;

  uint16_t dimLen_3 = 4;

  // RW: GCC Segmentation fault
  float32_t(*src)[dimLen_1][dimLen_2][dimLen_3] = (float32_t(*)[dimLen_1][dimLen_2][dimLen_3])
      DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref;
  float32_t(*dst)[dimLen_2][dimLen_3][dimLen_1] = (float32_t(*)[dimLen_2][dimLen_3][dimLen_1])
      DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref;

  const uint32_t baseChunk = dimLen_0 / NUM_CORES;
  const uint32_t leftover = dimLen_0 - baseChunk * NUM_CORES;
  const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
  const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
  for (uint32_t i_0 = offset; i_0 < offset + chunk; i_0++) {

    for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

          dst[i_0][i_2][i_3][i_1] = src[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref;
} _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_cluster_fork_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_cluster_fork(
    void *_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_cluster_fork_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_cluster_fork_args_t *)
          _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure_args_t
      DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure_args =
          (_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref};

  // _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure CLOSURE CALL
  _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure(
      &DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_0_blocks_0_0_weight_tensor;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr;
} _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure(
    void *_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_args_t *)
          _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_args;

  float32_t *DeeployNetwork_0_blocks_0_0_weight_tensor = args->DeeployNetwork_0_blocks_0_0_weight_tensor;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed =
      args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_0_blocks_0_0_weight_tensor_ref =
      (void *)((char *)DeeployNetwork_0_blocks_0_0_weight_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_0_blocks_0_0_weight_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_cluster_fork_args_t
        DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_cluster_fork_args =
            (_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_cluster_fork,
                    &DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1310848, DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_0_blocks_0_0_weight_tensor;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr;
} _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3(
    void *_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3_args_t *)
          _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3_args;

  float32_t *DeeployNetwork_0_blocks_0_0_weight_tensor = args->DeeployNetwork_0_blocks_0_0_weight_tensor;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed =
      args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_args_t
      DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_args =
          (_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_args_t){
              .DeeployNetwork_0_blocks_0_0_weight_tensor = DeeployNetwork_0_blocks_0_0_weight_tensor,
              .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed =
                  DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr};

  // _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure CLOSURE CALL
  _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure(
      &DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref;
} _node_1_blocks_0_blocks_0_0_Conv_tiling_closure_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_tiling_closure(void *_node_1_blocks_0_blocks_0_0_Conv_tiling_closure_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_tiling_closure_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_tiling_closure_args_t *)_node_1_blocks_0_blocks_0_0_Conv_tiling_closure_args;

  void *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer = args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref;

  // CLOSURE FUNCTION CALL

  // 2D FP Conv HWC with Im2Col and ChannelOout parallelism (Name: node_1_blocks_0_blocks_0_0_Conv, Op: Conv)

  float32_t *
      ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref =
          DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref;
  float32_t *
      ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref =
          DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref;

  for (uint32_t n = 0; n < 1; ++n) {
    PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
        ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref,
        14, 700, 1, DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref, 2, 1, 4, 1, 1,
        DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref, true,
        ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref,
        0, 0, 2, 2, DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer);

    ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref +=
        1 * 14 * 700;
    ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref +=
        2 * 14 * 701;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref;
} _node_1_blocks_0_blocks_0_0_Conv_cluster_fork_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_cluster_fork(void *_node_1_blocks_0_blocks_0_0_Conv_cluster_fork_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_cluster_fork_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_cluster_fork_args_t *)_node_1_blocks_0_blocks_0_0_Conv_cluster_fork_args;

  void *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer = args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref;

  // CLOSURE FUNCTION CALL
  _node_1_blocks_0_blocks_0_0_Conv_tiling_closure_args_t DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_tiling_closure_args =
      (_node_1_blocks_0_blocks_0_0_Conv_tiling_closure_args_t){.DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer =
                                                                   DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer,
                                                               .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref,
                                                               .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref,
                                                               .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref,
                                                               .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref};

  // _node_1_blocks_0_blocks_0_0_Conv_tiling_closure CLOSURE CALL
  _node_1_blocks_0_blocks_0_0_Conv_tiling_closure(&DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_1_blocks_0_0_bias_tensor;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr;
} _node_1_blocks_0_blocks_0_0_Conv_closure_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_closure(void *_node_1_blocks_0_blocks_0_0_Conv_closure_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_closure_args_t *args = (_node_1_blocks_0_blocks_0_0_Conv_closure_args_t *)_node_1_blocks_0_blocks_0_0_Conv_closure_args;

  float32_t *DeeployNetwork_1_blocks_0_0_bias_tensor = args->DeeployNetwork_1_blocks_0_0_bias_tensor;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed = args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed =
      args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed =
      args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 78512);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 117840);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 117872);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_1_blocks_0_0_bias_tensor_ref =
      (void *)((char *)DeeployNetwork_1_blocks_0_0_bias_tensor + DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_cumByteOffset
                                                                     [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr]);
  void *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 117712);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1480992, DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed_ref);
    mchan_transfer_1d(
        1441824, DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref,
        DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed_ref);

    // UPDATE VARIABLE
    // DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed_ref
    DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed_ref =
        (void
             *)((char
                     *)(DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed_ref) +
                32);

    mchan_transfer_1d(1441800, DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_1_blocks_0_0_bias_tensor_ref);

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_1_blocks_0_0_bias_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_1_blocks_0_0_bias_tensor_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_1_blocks_0_0_bias_tensor_ref) + 8);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _node_1_blocks_0_blocks_0_0_Conv_cluster_fork_args_t DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_cluster_fork_args =
        (_node_1_blocks_0_blocks_0_0_Conv_cluster_fork_args_t){.DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer =
                                                                   DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer,
                                                               .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_in_ref,
                                                               .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_weight_ref,
                                                               .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_bias_ref,
                                                               .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref =
                                                                   DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_node_1_blocks_0_blocks_0_0_Conv_cluster_fork, &DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(1913520, DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_data_out_ref,
                                  DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed_ref,
                                  8, 32);

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed_ref
    DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed_ref) + 8);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_1_blocks_0_0_bias_tensor;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr;
} _node_1_blocks_0_blocks_0_0_Conv_closure_L3_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_closure_L3(void *_node_1_blocks_0_blocks_0_0_Conv_closure_L3_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_closure_L3_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_closure_L3_args_t *)_node_1_blocks_0_blocks_0_0_Conv_closure_L3_args;

  float32_t *DeeployNetwork_1_blocks_0_0_bias_tensor = args->DeeployNetwork_1_blocks_0_0_bias_tensor;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed = args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed =
      args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed =
      args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _node_1_blocks_0_blocks_0_0_Conv_closure_args_t DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_closure_args =
      (_node_1_blocks_0_blocks_0_0_Conv_closure_args_t){
          .DeeployNetwork_1_blocks_0_0_bias_tensor = DeeployNetwork_1_blocks_0_0_bias_tensor,
          .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed = DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed,
          .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed =
              DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed,
          .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed = DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr};

  // _node_1_blocks_0_blocks_0_0_Conv_closure CLOSURE CALL
  _node_1_blocks_0_blocks_0_0_Conv_closure(&DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref;
} _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure_args_t;

static void
_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure(void *_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure_args_t *)
          _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 14, 701, 8] -> [1, 8, 14, 701] (Name: node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 14;

  uint16_t dimLen_2 = 701;

  uint16_t dimLen_3 = 1;

  // RW: GCC Segmentation fault
  float32_t(*src)[dimLen_1][dimLen_2][dimLen_3] =
      (float32_t(*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref;
  float32_t(*dst)[dimLen_3][dimLen_1][dimLen_2] =
      (float32_t(*)[dimLen_3][dimLen_1][dimLen_2])DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    const uint32_t baseChunk = dimLen_3 / NUM_CORES;
    const uint32_t leftover = dimLen_3 - baseChunk * NUM_CORES;
    const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
    const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
    for (uint32_t i_3 = offset; i_3 < offset + chunk; i_3++) {

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

          dst[i_0][i_3][i_1][i_2] = src[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref;
} _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_cluster_fork_args_t;

static void
_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_cluster_fork(void *_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_cluster_fork_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_cluster_fork_args_t *)_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_cluster_fork_args;

  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref;
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure_args_t
      DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure_args =
          (_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref};

  // _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure CLOSURE CALL
  _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure(
      &DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr;
} _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure(void *_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_args_t *)_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_args;

  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed =
      args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39256);
  float32_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref =
      (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 + DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(
        2005336, DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed_ref, 4,
        32);

    // UPDATE VARIABLE
    // DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed_ref
    DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed_ref =
        (void
             *)((char
                     *)(DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed_ref) +
                4);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_cluster_fork_args_t
        DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_cluster_fork_args =
            (_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_cluster_fork,
                    &DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1349976, DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_output_0_ref);

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_output_0_ref
    DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_output_0_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_output_0_ref) + 39256);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr;
} _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3_args_t;

static void _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3(void *_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3_args_t *args =
      (_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3_args_t *)_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3_args;

  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed =
      args->DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_args_t DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_args =
      (_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_args_t){
          .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed = DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed,
          .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr =
              DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr};

  // _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure CLOSURE CALL
  _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure(&DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork() {
  float32_t *DeeployNetwork_0_blocks_0_0_weight_tensor;
  float32_t *DeeployNetwork_1_blocks_0_0_bias_tensor;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed;
  float32_t *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed;
  void *DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_buffer;
  {

    perf_stats_t perturbrademacher_0_blocks_0_0_weight_perf_start, perturbrademacher_0_blocks_0_0_weight_perf_end,
        perturbrademacher_0_blocks_0_0_weight_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&perturbrademacher_0_blocks_0_0_weight_perf_start);
    }

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr;
    DeeployNetwork_0_blocks_0_0_weight_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _perturbrademacher_0_blocks_0_0_weight_closure_L3_args_t DeeployNetwork__perturbrademacher_0_blocks_0_0_weight_closure_L3_args =
        (_perturbrademacher_0_blocks_0_0_weight_closure_L3_args_t){.DeeployNetwork_0_blocks_0_0_weight_tensor = DeeployNetwork_0_blocks_0_0_weight_tensor,
                                                                   .DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr =
                                                                       DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_0_blocks_0_0_weight_tileIdxPtr};

    // _perturbrademacher_0_blocks_0_0_weight_closure_L3 CLOSURE CALL
    _perturbrademacher_0_blocks_0_0_weight_closure_L3(&DeeployNetwork__perturbrademacher_0_blocks_0_0_weight_closure_L3_args);

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&perturbrademacher_0_blocks_0_0_weight_perf_end);
      perf_bench_diff(&perturbrademacher_0_blocks_0_0_weight_perf_total, &perturbrademacher_0_blocks_0_0_weight_perf_end,
                      &perturbrademacher_0_blocks_0_0_weight_perf_start);
      perf_bench_print("perturbrademacher_0_blocks_0_0_weight", &perturbrademacher_0_blocks_0_0_weight_perf_total);
    }
  }
  {

    perf_stats_t perturbrademacher_1_blocks_0_0_bias_perf_start, perturbrademacher_1_blocks_0_0_bias_perf_end, perturbrademacher_1_blocks_0_0_bias_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&perturbrademacher_1_blocks_0_0_bias_perf_start);
    }

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr;
    DeeployNetwork_1_blocks_0_0_bias_tensor = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 314048);
    _perturbrademacher_1_blocks_0_0_bias_closure_L3_args_t DeeployNetwork__perturbrademacher_1_blocks_0_0_bias_closure_L3_args =
        (_perturbrademacher_1_blocks_0_0_bias_closure_L3_args_t){.DeeployNetwork_1_blocks_0_0_bias_tensor = DeeployNetwork_1_blocks_0_0_bias_tensor,
                                                                 .DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr =
                                                                     DeeployNetwork_TILING_CODEGEN_L1_perturbrademacher_1_blocks_0_0_bias_tileIdxPtr};

    // _perturbrademacher_1_blocks_0_0_bias_closure_L3 CLOSURE CALL
    _perturbrademacher_1_blocks_0_0_bias_closure_L3(&DeeployNetwork__perturbrademacher_1_blocks_0_0_bias_closure_L3_args);

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&perturbrademacher_1_blocks_0_0_bias_perf_end);
      perf_bench_diff(&perturbrademacher_1_blocks_0_0_bias_perf_total, &perturbrademacher_1_blocks_0_0_bias_perf_end,
                      &perturbrademacher_1_blocks_0_0_bias_perf_start);
      perf_bench_print("perturbrademacher_1_blocks_0_0_bias", &perturbrademacher_1_blocks_0_0_bias_perf_total);
    }
  }
  {

    perf_stats_t node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_perf_start, node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_perf_end,
        node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_perf_start);
    }

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr;
    DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 314080);
    _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3_args_t DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3_args =
        (_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3_args_t){
            .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed = DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_tileIdxPtr};

    // _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3 CLOSURE CALL
    _node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3(&DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_closure_L3_args);

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_perf_end);
      perf_bench_diff(&node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_perf_total, &node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_perf_end,
                      &node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_perf_start);
      perf_bench_print("node_1_blocks_0_blocks_0_0_Conv_input_0_transpose", &node_1_blocks_0_blocks_0_0_Conv_input_0_transpose_perf_total);
    }
  }
  {

    perf_stats_t node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_perf_start,
        node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_perf_end,
        node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_perf_start);
    }

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr;
    DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 353280);
    _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3_args_t
        DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3_args =
            (_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3_args_t){
                .DeeployNetwork_0_blocks_0_0_weight_tensor = DeeployNetwork_0_blocks_0_0_weight_tensor,
                .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed =
                    DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_tileIdxPtr};

    // _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3 CLOSURE CALL
    _node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3(
        &DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_closure_L3_args);

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_perf_end);
      perf_bench_diff(&node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_perf_total,
                      &node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_perf_end,
                      &node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_perf_start);
      perf_bench_print("node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose",
                       &node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transpose_perf_total);
    }
  }
  {

    perf_stats_t node_1_blocks_0_blocks_0_0_Conv_perf_start, node_1_blocks_0_blocks_0_0_Conv_perf_end, node_1_blocks_0_blocks_0_0_Conv_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&node_1_blocks_0_blocks_0_0_Conv_perf_start);
    }

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr;
    DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _node_1_blocks_0_blocks_0_0_Conv_closure_L3_args_t DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_closure_L3_args =
        (_node_1_blocks_0_blocks_0_0_Conv_closure_L3_args_t){
            .DeeployNetwork_1_blocks_0_0_bias_tensor = DeeployNetwork_1_blocks_0_0_bias_tensor,
            .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed = DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_input_0_transposed,
            .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed =
                DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_0_blocks_0_0_weight_tensor_transposed,
            .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed = DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_tileIdxPtr};

    // _node_1_blocks_0_blocks_0_0_Conv_closure_L3 CLOSURE CALL
    _node_1_blocks_0_blocks_0_0_Conv_closure_L3(&DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_closure_L3_args);

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&node_1_blocks_0_blocks_0_0_Conv_perf_end);
      perf_bench_diff(&node_1_blocks_0_blocks_0_0_Conv_perf_total, &node_1_blocks_0_blocks_0_0_Conv_perf_end, &node_1_blocks_0_blocks_0_0_Conv_perf_start);
      perf_bench_print("node_1_blocks_0_blocks_0_0_Conv", &node_1_blocks_0_blocks_0_0_Conv_perf_total);
    }
  }
  {

    perf_stats_t node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_perf_start, node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_perf_end,
        node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_perf_start);
    }

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr;
    _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3_args_t
        DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3_args =
            (_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3_args_t){
                .DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed =
                    DeeployNetwork_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1_node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_tileIdxPtr};

    // _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3 CLOSURE CALL
    _node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3(&DeeployNetwork__node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_closure_L3_args);

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_perf_end);
      perf_bench_diff(&node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_perf_total, &node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_perf_end,
                      &node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_perf_start);
      perf_bench_print("node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose", &node_1_blocks_0_blocks_0_0_Conv_output_0_pre_transpose_perf_total);
    }
  }
}

void InitNetwork() {

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 117880);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 628096);

  DeeployNetwork_input_0 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 128);
  DeeployNetwork_input_1 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 39360);
  DeeployNetwork_input_2 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 39328);
  DeeployNetwork_output_0 = (float32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 314048);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
