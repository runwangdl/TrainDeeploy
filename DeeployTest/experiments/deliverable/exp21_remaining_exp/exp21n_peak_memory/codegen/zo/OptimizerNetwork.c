#include "OptimizerNetwork.h"
#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "TrainingNetwork.h"
#include "perf_utils.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int8_t *DeeployOptNetwork_MEMORYARENA_L1;
int8_t *DeeployOptNetwork_MEMORYARENA_L2;
float32_t *DeeployOptNetwork_input_0;
float32_t *DeeployOptNetwork_input_1;
float32_t *DeeployOptNetwork_input_2;
float32_t *DeeployOptNetwork_input_3;
float32_t *DeeployOptNetwork_input_4;
float32_t *DeeployOptNetwork_input_5;
float32_t *DeeployOptNetwork_input_6;
float32_t *DeeployOptNetwork_input_7;
float32_t *DeeployOptNetwork_input_8;
float32_t *DeeployOptNetwork_input_9;
float32_t *DeeployOptNetwork_input_10;
float32_t *DeeployOptNetwork_input_11;
float32_t *DeeployOptNetwork_input_12;
float32_t *DeeployOptNetwork_input_13;
float32_t *DeeployOptNetwork_input_14;
float32_t *DeeployOptNetwork_input_15;
float32_t *DeeployOptNetwork_input_16;
float32_t *DeeployOptNetwork_input_17;
float32_t *DeeployOptNetwork_input_18;
float32_t *DeeployOptNetwork_input_19;
float32_t *DeeployOptNetwork_input_20;
float32_t *DeeployOptNetwork_input_21;
float32_t *DeeployOptNetwork_output_0;
float32_t *DeeployOptNetwork_output_1;
float32_t *DeeployOptNetwork_output_2;
float32_t *DeeployOptNetwork_output_3;
float32_t *DeeployOptNetwork_output_4;
float32_t *DeeployOptNetwork_output_5;
float32_t *DeeployOptNetwork_output_6;
float32_t *DeeployOptNetwork_output_7;
float32_t *DeeployOptNetwork_output_8;
float32_t *DeeployOptNetwork_output_9;
float32_t *DeeployOptNetwork_output_10;
float32_t *DeeployOptNetwork_output_11;
float32_t *DeeployOptNetwork_output_12;
float32_t *DeeployOptNetwork_output_13;
float32_t *DeeployOptNetwork_output_14;
float32_t *DeeployOptNetwork_output_15;
float32_t *DeeployOptNetwork_output_16;
float32_t *DeeployOptNetwork_output_17;
float32_t *DeeployOptNetwork_output_18;
float32_t *DeeployOptNetwork_output_19;
float32_t *DeeployOptNetwork_output_20;
float32_t *DeeployOptNetwork_output_21;

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_cumByteOffset[1] = {0};

void *DeeployOptNetwork_inputs[22];
void *DeeployOptNetwork_outputs[22];
extern struct pi_device cluster_dev;
typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref;
} _perturbrademacher_blocks_0_0_bias_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_0_0_bias_updated_tiling_closure(void *_perturbrademacher_blocks_0_0_bias_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_0_bias_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_0_0_bias_updated_tiling_closure_args_t *)_perturbrademacher_blocks_0_0_bias_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_0_0_bias_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_0_0_bias_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_0_0_bias_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_0_0_bias_updated_chunk = (8 >> perturbrademacher_blocks_0_0_bias_updated_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_0_0_bias_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_0_0_bias_updated_chunk * perturbrademacher_blocks_0_0_bias_updated_core_id, (uint32_t)8);
  uint32_t perturbrademacher_blocks_0_0_bias_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_0_0_bias_updated_chunk_start + perturbrademacher_blocks_0_0_bias_updated_chunk, (uint32_t)8);
  uint32_t perturbrademacher_blocks_0_0_bias_updated_local_size =
      perturbrademacher_blocks_0_0_bias_updated_chunk_stop - perturbrademacher_blocks_0_0_bias_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 1 + perturbrademacher_blocks_0_0_bias_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_0_0_bias_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref
                                [perturbrademacher_blocks_0_0_bias_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref
                                [perturbrademacher_blocks_0_0_bias_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_0_0_bias_updated_local_size,
                            zo_eco_mom[1] + perturbrademacher_blocks_0_0_bias_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref
                                              [perturbrademacher_blocks_0_0_bias_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref
                                              [perturbrademacher_blocks_0_0_bias_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_0_0_bias_updated_local_size, perturbrademacher_blocks_0_0_bias_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref
                                    [perturbrademacher_blocks_0_0_bias_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref
                                    [perturbrademacher_blocks_0_0_bias_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_0_0_bias_updated_local_size, perturbrademacher_blocks_0_0_bias_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref;
} _perturbrademacher_blocks_0_0_bias_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_0_0_bias_updated_cluster_fork(void *_perturbrademacher_blocks_0_0_bias_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_0_bias_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_0_0_bias_updated_cluster_fork_args_t *)_perturbrademacher_blocks_0_0_bias_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_0_0_bias_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_0_0_bias_updated_tiling_closure_args =
      (_perturbrademacher_blocks_0_0_bias_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref};

  // _perturbrademacher_blocks_0_0_bias_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_0_0_bias_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_0_0_bias_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_0_0_bias_updated_closure_args_t;

static void _perturbrademacher_blocks_0_0_bias_updated_closure(void *_perturbrademacher_blocks_0_0_bias_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_0_bias_updated_closure_args_t *args =
      (_perturbrademacher_blocks_0_0_bias_updated_closure_args_t *)_perturbrademacher_blocks_0_0_bias_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 32);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_input_1_ref =
      (void *)((char *)DeeployOptNetwork_input_1 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_output_1_ref =
      (void *)((char *)DeeployOptNetwork_output_1 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_input_1_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_0_0_bias_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_0_0_bias_updated_cluster_fork_args =
        (_perturbrademacher_blocks_0_0_bias_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_0_0_bias_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_0_0_bias_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_output_1_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_0_0_bias_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_0_0_bias_updated_closure_L3(void *_perturbrademacher_blocks_0_0_bias_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_0_bias_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_0_0_bias_updated_closure_L3_args_t *)_perturbrademacher_blocks_0_0_bias_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_0_0_bias_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_0_0_bias_updated_closure_args =
      (_perturbrademacher_blocks_0_0_bias_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr =
                                                                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr};

  // _perturbrademacher_blocks_0_0_bias_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_0_0_bias_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_0_0_bias_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref;
} _perturbrademacher_blocks_0_1_weight_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_0_1_weight_updated_tiling_closure(void *_perturbrademacher_blocks_0_1_weight_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_1_weight_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_0_1_weight_updated_tiling_closure_args_t *)_perturbrademacher_blocks_0_1_weight_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_0_1_weight_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_0_1_weight_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_0_1_weight_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_0_1_weight_updated_chunk = (8 >> perturbrademacher_blocks_0_1_weight_updated_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_0_1_weight_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_0_1_weight_updated_chunk * perturbrademacher_blocks_0_1_weight_updated_core_id, (uint32_t)8);
  uint32_t perturbrademacher_blocks_0_1_weight_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_0_1_weight_updated_chunk_start + perturbrademacher_blocks_0_1_weight_updated_chunk, (uint32_t)8);
  uint32_t perturbrademacher_blocks_0_1_weight_updated_local_size =
      perturbrademacher_blocks_0_1_weight_updated_chunk_stop - perturbrademacher_blocks_0_1_weight_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 2 + perturbrademacher_blocks_0_1_weight_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_0_1_weight_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref
                                [perturbrademacher_blocks_0_1_weight_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref
                                [perturbrademacher_blocks_0_1_weight_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_0_1_weight_updated_local_size,
                            zo_eco_mom[2] + perturbrademacher_blocks_0_1_weight_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref
                                              [perturbrademacher_blocks_0_1_weight_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref
                                              [perturbrademacher_blocks_0_1_weight_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_0_1_weight_updated_local_size, perturbrademacher_blocks_0_1_weight_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref
                                    [perturbrademacher_blocks_0_1_weight_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref
                                    [perturbrademacher_blocks_0_1_weight_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_0_1_weight_updated_local_size, perturbrademacher_blocks_0_1_weight_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref;
} _perturbrademacher_blocks_0_1_weight_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_0_1_weight_updated_cluster_fork(void *_perturbrademacher_blocks_0_1_weight_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_1_weight_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_0_1_weight_updated_cluster_fork_args_t *)_perturbrademacher_blocks_0_1_weight_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_0_1_weight_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_0_1_weight_updated_tiling_closure_args =
      (_perturbrademacher_blocks_0_1_weight_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref};

  // _perturbrademacher_blocks_0_1_weight_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_0_1_weight_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_0_1_weight_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_0_1_weight_updated_closure_args_t;

static void _perturbrademacher_blocks_0_1_weight_updated_closure(void *_perturbrademacher_blocks_0_1_weight_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_1_weight_updated_closure_args_t *args =
      (_perturbrademacher_blocks_0_1_weight_updated_closure_args_t *)_perturbrademacher_blocks_0_1_weight_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 32);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_input_2_ref =
      (void *)((char *)DeeployOptNetwork_input_2 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_output_2_ref =
      (void *)((char *)DeeployOptNetwork_output_2 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_input_2_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_0_1_weight_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_0_1_weight_updated_cluster_fork_args =
        (_perturbrademacher_blocks_0_1_weight_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_0_1_weight_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_0_1_weight_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_output_2_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_0_1_weight_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_0_1_weight_updated_closure_L3(void *_perturbrademacher_blocks_0_1_weight_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_1_weight_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_0_1_weight_updated_closure_L3_args_t *)_perturbrademacher_blocks_0_1_weight_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_0_1_weight_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_0_1_weight_updated_closure_args =
      (_perturbrademacher_blocks_0_1_weight_updated_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr};

  // _perturbrademacher_blocks_0_1_weight_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_0_1_weight_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_0_1_weight_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref;
} _perturbrademacher_blocks_0_1_bias_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_0_1_bias_updated_tiling_closure(void *_perturbrademacher_blocks_0_1_bias_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_1_bias_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_0_1_bias_updated_tiling_closure_args_t *)_perturbrademacher_blocks_0_1_bias_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_0_1_bias_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_0_1_bias_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_0_1_bias_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_0_1_bias_updated_chunk = (8 >> perturbrademacher_blocks_0_1_bias_updated_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_0_1_bias_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_0_1_bias_updated_chunk * perturbrademacher_blocks_0_1_bias_updated_core_id, (uint32_t)8);
  uint32_t perturbrademacher_blocks_0_1_bias_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_0_1_bias_updated_chunk_start + perturbrademacher_blocks_0_1_bias_updated_chunk, (uint32_t)8);
  uint32_t perturbrademacher_blocks_0_1_bias_updated_local_size =
      perturbrademacher_blocks_0_1_bias_updated_chunk_stop - perturbrademacher_blocks_0_1_bias_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 3 + perturbrademacher_blocks_0_1_bias_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_0_1_bias_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref
                                [perturbrademacher_blocks_0_1_bias_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref
                                [perturbrademacher_blocks_0_1_bias_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_0_1_bias_updated_local_size,
                            zo_eco_mom[3] + perturbrademacher_blocks_0_1_bias_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref
                                              [perturbrademacher_blocks_0_1_bias_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref
                                              [perturbrademacher_blocks_0_1_bias_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_0_1_bias_updated_local_size, perturbrademacher_blocks_0_1_bias_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref
                                    [perturbrademacher_blocks_0_1_bias_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref
                                    [perturbrademacher_blocks_0_1_bias_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_0_1_bias_updated_local_size, perturbrademacher_blocks_0_1_bias_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref;
} _perturbrademacher_blocks_0_1_bias_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_0_1_bias_updated_cluster_fork(void *_perturbrademacher_blocks_0_1_bias_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_1_bias_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_0_1_bias_updated_cluster_fork_args_t *)_perturbrademacher_blocks_0_1_bias_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_0_1_bias_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_0_1_bias_updated_tiling_closure_args =
      (_perturbrademacher_blocks_0_1_bias_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref};

  // _perturbrademacher_blocks_0_1_bias_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_0_1_bias_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_0_1_bias_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_0_1_bias_updated_closure_args_t;

static void _perturbrademacher_blocks_0_1_bias_updated_closure(void *_perturbrademacher_blocks_0_1_bias_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_1_bias_updated_closure_args_t *args =
      (_perturbrademacher_blocks_0_1_bias_updated_closure_args_t *)_perturbrademacher_blocks_0_1_bias_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 32);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_input_3_ref =
      (void *)((char *)DeeployOptNetwork_input_3 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_output_3_ref =
      (void *)((char *)DeeployOptNetwork_output_3 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_input_3_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_0_1_bias_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_0_1_bias_updated_cluster_fork_args =
        (_perturbrademacher_blocks_0_1_bias_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_0_1_bias_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_0_1_bias_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_output_3_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_0_1_bias_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_0_1_bias_updated_closure_L3(void *_perturbrademacher_blocks_0_1_bias_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_1_bias_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_0_1_bias_updated_closure_L3_args_t *)_perturbrademacher_blocks_0_1_bias_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_0_1_bias_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_0_1_bias_updated_closure_args =
      (_perturbrademacher_blocks_0_1_bias_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr =
                                                                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr};

  // _perturbrademacher_blocks_0_1_bias_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_0_1_bias_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_0_1_bias_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref;
} _perturbrademacher_fc_bias_updated_tiling_closure_args_t;

static void _perturbrademacher_fc_bias_updated_tiling_closure(void *_perturbrademacher_fc_bias_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_fc_bias_updated_tiling_closure_args_t *args =
      (_perturbrademacher_fc_bias_updated_tiling_closure_args_t *)_perturbrademacher_fc_bias_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_fc_bias_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_fc_bias_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_fc_bias_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_fc_bias_updated_chunk = (9 >> perturbrademacher_fc_bias_updated_log2Core) + ((9 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_fc_bias_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_fc_bias_updated_chunk * perturbrademacher_fc_bias_updated_core_id, (uint32_t)9);
  uint32_t perturbrademacher_fc_bias_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_fc_bias_updated_chunk_start + perturbrademacher_fc_bias_updated_chunk, (uint32_t)9);
  uint32_t perturbrademacher_fc_bias_updated_local_size = perturbrademacher_fc_bias_updated_chunk_stop - perturbrademacher_fc_bias_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 31 + perturbrademacher_fc_bias_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_fc_bias_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref[perturbrademacher_fc_bias_updated_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref[perturbrademacher_fc_bias_updated_chunk_start],
        chunk_seed, perturbation_sign, perturbrademacher_fc_bias_updated_local_size, zo_eco_mom[31] + perturbrademacher_fc_bias_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref[perturbrademacher_fc_bias_updated_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref[perturbrademacher_fc_bias_updated_chunk_start],
        chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        perturbrademacher_fc_bias_updated_local_size, perturbrademacher_fc_bias_updated_eps);
#else
    ApplyRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref[perturbrademacher_fc_bias_updated_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref[perturbrademacher_fc_bias_updated_chunk_start],
        chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        perturbrademacher_fc_bias_updated_local_size, perturbrademacher_fc_bias_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref;
} _perturbrademacher_fc_bias_updated_cluster_fork_args_t;

static void _perturbrademacher_fc_bias_updated_cluster_fork(void *_perturbrademacher_fc_bias_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_fc_bias_updated_cluster_fork_args_t *args =
      (_perturbrademacher_fc_bias_updated_cluster_fork_args_t *)_perturbrademacher_fc_bias_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_fc_bias_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_fc_bias_updated_tiling_closure_args =
      (_perturbrademacher_fc_bias_updated_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref =
                                                                     DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref,
                                                                 .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref =
                                                                     DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref};

  // _perturbrademacher_fc_bias_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_fc_bias_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_fc_bias_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr;
} _perturbrademacher_fc_bias_updated_closure_args_t;

static void _perturbrademacher_fc_bias_updated_closure(void *_perturbrademacher_fc_bias_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_fc_bias_updated_closure_args_t *args =
      (_perturbrademacher_fc_bias_updated_closure_args_t *)_perturbrademacher_fc_bias_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 36);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_input_21_ref =
      (void *)((char *)DeeployOptNetwork_input_21 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_output_21_ref =
      (void *)((char *)DeeployOptNetwork_output_21 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441828, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_input_21_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_fc_bias_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_fc_bias_updated_cluster_fork_args =
        (_perturbrademacher_fc_bias_updated_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref =
                                                                     DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_in_ref,
                                                                 .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref =
                                                                     DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_fc_bias_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_fc_bias_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310756, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_output_21_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr;
} _perturbrademacher_fc_bias_updated_closure_L3_args_t;

static void _perturbrademacher_fc_bias_updated_closure_L3(void *_perturbrademacher_fc_bias_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_fc_bias_updated_closure_L3_args_t *args =
      (_perturbrademacher_fc_bias_updated_closure_L3_args_t *)_perturbrademacher_fc_bias_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_fc_bias_updated_closure_args_t DeeployOptNetwork__perturbrademacher_fc_bias_updated_closure_args =
      (_perturbrademacher_fc_bias_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr =
                                                              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr};

  // _perturbrademacher_fc_bias_updated_closure CLOSURE CALL
  _perturbrademacher_fc_bias_updated_closure(&DeeployOptNetwork__perturbrademacher_fc_bias_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref;
} _perturbrademacher_blocks_1_0_bias_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_1_0_bias_updated_tiling_closure(void *_perturbrademacher_blocks_1_0_bias_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_0_bias_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_1_0_bias_updated_tiling_closure_args_t *)_perturbrademacher_blocks_1_0_bias_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_1_0_bias_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_1_0_bias_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_1_0_bias_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_1_0_bias_updated_chunk = (16 >> perturbrademacher_blocks_1_0_bias_updated_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_1_0_bias_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_1_0_bias_updated_chunk * perturbrademacher_blocks_1_0_bias_updated_core_id, (uint32_t)16);
  uint32_t perturbrademacher_blocks_1_0_bias_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_1_0_bias_updated_chunk_start + perturbrademacher_blocks_1_0_bias_updated_chunk, (uint32_t)16);
  uint32_t perturbrademacher_blocks_1_0_bias_updated_local_size =
      perturbrademacher_blocks_1_0_bias_updated_chunk_stop - perturbrademacher_blocks_1_0_bias_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 7 + perturbrademacher_blocks_1_0_bias_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_1_0_bias_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref
                                [perturbrademacher_blocks_1_0_bias_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref
                                [perturbrademacher_blocks_1_0_bias_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_1_0_bias_updated_local_size,
                            zo_eco_mom[7] + perturbrademacher_blocks_1_0_bias_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref
                                              [perturbrademacher_blocks_1_0_bias_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref
                                              [perturbrademacher_blocks_1_0_bias_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_1_0_bias_updated_local_size, perturbrademacher_blocks_1_0_bias_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref
                                    [perturbrademacher_blocks_1_0_bias_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref
                                    [perturbrademacher_blocks_1_0_bias_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_1_0_bias_updated_local_size, perturbrademacher_blocks_1_0_bias_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref;
} _perturbrademacher_blocks_1_0_bias_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_1_0_bias_updated_cluster_fork(void *_perturbrademacher_blocks_1_0_bias_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_0_bias_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_1_0_bias_updated_cluster_fork_args_t *)_perturbrademacher_blocks_1_0_bias_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_1_0_bias_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_1_0_bias_updated_tiling_closure_args =
      (_perturbrademacher_blocks_1_0_bias_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref};

  // _perturbrademacher_blocks_1_0_bias_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_1_0_bias_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_1_0_bias_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_1_0_bias_updated_closure_args_t;

static void _perturbrademacher_blocks_1_0_bias_updated_closure(void *_perturbrademacher_blocks_1_0_bias_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_0_bias_updated_closure_args_t *args =
      (_perturbrademacher_blocks_1_0_bias_updated_closure_args_t *)_perturbrademacher_blocks_1_0_bias_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_input_5_ref =
      (void *)((char *)DeeployOptNetwork_input_5 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_output_5_ref =
      (void *)((char *)DeeployOptNetwork_output_5 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_input_5_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_1_0_bias_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_1_0_bias_updated_cluster_fork_args =
        (_perturbrademacher_blocks_1_0_bias_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_1_0_bias_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_1_0_bias_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_output_5_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_1_0_bias_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_1_0_bias_updated_closure_L3(void *_perturbrademacher_blocks_1_0_bias_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_0_bias_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_1_0_bias_updated_closure_L3_args_t *)_perturbrademacher_blocks_1_0_bias_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_1_0_bias_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_1_0_bias_updated_closure_args =
      (_perturbrademacher_blocks_1_0_bias_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr =
                                                                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr};

  // _perturbrademacher_blocks_1_0_bias_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_1_0_bias_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_1_0_bias_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref;
} _perturbrademacher_blocks_1_1_weight_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_1_1_weight_updated_tiling_closure(void *_perturbrademacher_blocks_1_1_weight_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_1_weight_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_1_1_weight_updated_tiling_closure_args_t *)_perturbrademacher_blocks_1_1_weight_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_1_1_weight_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_1_1_weight_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_1_1_weight_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_1_1_weight_updated_chunk = (16 >> perturbrademacher_blocks_1_1_weight_updated_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_1_1_weight_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_1_1_weight_updated_chunk * perturbrademacher_blocks_1_1_weight_updated_core_id, (uint32_t)16);
  uint32_t perturbrademacher_blocks_1_1_weight_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_1_1_weight_updated_chunk_start + perturbrademacher_blocks_1_1_weight_updated_chunk, (uint32_t)16);
  uint32_t perturbrademacher_blocks_1_1_weight_updated_local_size =
      perturbrademacher_blocks_1_1_weight_updated_chunk_stop - perturbrademacher_blocks_1_1_weight_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 8 + perturbrademacher_blocks_1_1_weight_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_1_1_weight_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref
                                [perturbrademacher_blocks_1_1_weight_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref
                                [perturbrademacher_blocks_1_1_weight_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_1_1_weight_updated_local_size,
                            zo_eco_mom[8] + perturbrademacher_blocks_1_1_weight_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref
                                              [perturbrademacher_blocks_1_1_weight_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref
                                              [perturbrademacher_blocks_1_1_weight_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_1_1_weight_updated_local_size, perturbrademacher_blocks_1_1_weight_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref
                                    [perturbrademacher_blocks_1_1_weight_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref
                                    [perturbrademacher_blocks_1_1_weight_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_1_1_weight_updated_local_size, perturbrademacher_blocks_1_1_weight_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref;
} _perturbrademacher_blocks_1_1_weight_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_1_1_weight_updated_cluster_fork(void *_perturbrademacher_blocks_1_1_weight_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_1_weight_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_1_1_weight_updated_cluster_fork_args_t *)_perturbrademacher_blocks_1_1_weight_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_1_1_weight_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_1_1_weight_updated_tiling_closure_args =
      (_perturbrademacher_blocks_1_1_weight_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref};

  // _perturbrademacher_blocks_1_1_weight_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_1_1_weight_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_1_1_weight_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_1_1_weight_updated_closure_args_t;

static void _perturbrademacher_blocks_1_1_weight_updated_closure(void *_perturbrademacher_blocks_1_1_weight_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_1_weight_updated_closure_args_t *args =
      (_perturbrademacher_blocks_1_1_weight_updated_closure_args_t *)_perturbrademacher_blocks_1_1_weight_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_input_6_ref =
      (void *)((char *)DeeployOptNetwork_input_6 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_output_6_ref =
      (void *)((char *)DeeployOptNetwork_output_6 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_input_6_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_1_1_weight_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_1_1_weight_updated_cluster_fork_args =
        (_perturbrademacher_blocks_1_1_weight_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_1_1_weight_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_1_1_weight_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_output_6_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_1_1_weight_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_1_1_weight_updated_closure_L3(void *_perturbrademacher_blocks_1_1_weight_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_1_weight_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_1_1_weight_updated_closure_L3_args_t *)_perturbrademacher_blocks_1_1_weight_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_1_1_weight_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_1_1_weight_updated_closure_args =
      (_perturbrademacher_blocks_1_1_weight_updated_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr};

  // _perturbrademacher_blocks_1_1_weight_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_1_1_weight_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_1_1_weight_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref;
} _perturbrademacher_blocks_1_1_bias_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_1_1_bias_updated_tiling_closure(void *_perturbrademacher_blocks_1_1_bias_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_1_bias_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_1_1_bias_updated_tiling_closure_args_t *)_perturbrademacher_blocks_1_1_bias_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_1_1_bias_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_1_1_bias_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_1_1_bias_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_1_1_bias_updated_chunk = (16 >> perturbrademacher_blocks_1_1_bias_updated_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_1_1_bias_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_1_1_bias_updated_chunk * perturbrademacher_blocks_1_1_bias_updated_core_id, (uint32_t)16);
  uint32_t perturbrademacher_blocks_1_1_bias_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_1_1_bias_updated_chunk_start + perturbrademacher_blocks_1_1_bias_updated_chunk, (uint32_t)16);
  uint32_t perturbrademacher_blocks_1_1_bias_updated_local_size =
      perturbrademacher_blocks_1_1_bias_updated_chunk_stop - perturbrademacher_blocks_1_1_bias_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 9 + perturbrademacher_blocks_1_1_bias_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_1_1_bias_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref
                                [perturbrademacher_blocks_1_1_bias_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref
                                [perturbrademacher_blocks_1_1_bias_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_1_1_bias_updated_local_size,
                            zo_eco_mom[9] + perturbrademacher_blocks_1_1_bias_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref
                                              [perturbrademacher_blocks_1_1_bias_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref
                                              [perturbrademacher_blocks_1_1_bias_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_1_1_bias_updated_local_size, perturbrademacher_blocks_1_1_bias_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref
                                    [perturbrademacher_blocks_1_1_bias_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref
                                    [perturbrademacher_blocks_1_1_bias_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_1_1_bias_updated_local_size, perturbrademacher_blocks_1_1_bias_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref;
} _perturbrademacher_blocks_1_1_bias_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_1_1_bias_updated_cluster_fork(void *_perturbrademacher_blocks_1_1_bias_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_1_bias_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_1_1_bias_updated_cluster_fork_args_t *)_perturbrademacher_blocks_1_1_bias_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_1_1_bias_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_1_1_bias_updated_tiling_closure_args =
      (_perturbrademacher_blocks_1_1_bias_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref};

  // _perturbrademacher_blocks_1_1_bias_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_1_1_bias_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_1_1_bias_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_1_1_bias_updated_closure_args_t;

static void _perturbrademacher_blocks_1_1_bias_updated_closure(void *_perturbrademacher_blocks_1_1_bias_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_1_bias_updated_closure_args_t *args =
      (_perturbrademacher_blocks_1_1_bias_updated_closure_args_t *)_perturbrademacher_blocks_1_1_bias_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_input_7_ref =
      (void *)((char *)DeeployOptNetwork_input_7 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_output_7_ref =
      (void *)((char *)DeeployOptNetwork_output_7 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_input_7_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_1_1_bias_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_1_1_bias_updated_cluster_fork_args =
        (_perturbrademacher_blocks_1_1_bias_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_1_1_bias_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_1_1_bias_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_output_7_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_1_1_bias_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_1_1_bias_updated_closure_L3(void *_perturbrademacher_blocks_1_1_bias_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_1_bias_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_1_1_bias_updated_closure_L3_args_t *)_perturbrademacher_blocks_1_1_bias_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_1_1_bias_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_1_1_bias_updated_closure_args =
      (_perturbrademacher_blocks_1_1_bias_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr =
                                                                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr};

  // _perturbrademacher_blocks_1_1_bias_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_1_1_bias_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_1_1_bias_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref;
} _perturbrademacher_blocks_2_0_bias_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_2_0_bias_updated_tiling_closure(void *_perturbrademacher_blocks_2_0_bias_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_0_bias_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_2_0_bias_updated_tiling_closure_args_t *)_perturbrademacher_blocks_2_0_bias_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_2_0_bias_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_2_0_bias_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_2_0_bias_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_2_0_bias_updated_chunk = (16 >> perturbrademacher_blocks_2_0_bias_updated_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_2_0_bias_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_2_0_bias_updated_chunk * perturbrademacher_blocks_2_0_bias_updated_core_id, (uint32_t)16);
  uint32_t perturbrademacher_blocks_2_0_bias_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_2_0_bias_updated_chunk_start + perturbrademacher_blocks_2_0_bias_updated_chunk, (uint32_t)16);
  uint32_t perturbrademacher_blocks_2_0_bias_updated_local_size =
      perturbrademacher_blocks_2_0_bias_updated_chunk_stop - perturbrademacher_blocks_2_0_bias_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 13 + perturbrademacher_blocks_2_0_bias_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_2_0_bias_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref
                                [perturbrademacher_blocks_2_0_bias_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref
                                [perturbrademacher_blocks_2_0_bias_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_2_0_bias_updated_local_size,
                            zo_eco_mom[13] + perturbrademacher_blocks_2_0_bias_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref
                                              [perturbrademacher_blocks_2_0_bias_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref
                                              [perturbrademacher_blocks_2_0_bias_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_2_0_bias_updated_local_size, perturbrademacher_blocks_2_0_bias_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref
                                    [perturbrademacher_blocks_2_0_bias_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref
                                    [perturbrademacher_blocks_2_0_bias_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_2_0_bias_updated_local_size, perturbrademacher_blocks_2_0_bias_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref;
} _perturbrademacher_blocks_2_0_bias_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_2_0_bias_updated_cluster_fork(void *_perturbrademacher_blocks_2_0_bias_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_0_bias_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_2_0_bias_updated_cluster_fork_args_t *)_perturbrademacher_blocks_2_0_bias_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_2_0_bias_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_2_0_bias_updated_tiling_closure_args =
      (_perturbrademacher_blocks_2_0_bias_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref};

  // _perturbrademacher_blocks_2_0_bias_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_2_0_bias_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_2_0_bias_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_2_0_bias_updated_closure_args_t;

static void _perturbrademacher_blocks_2_0_bias_updated_closure(void *_perturbrademacher_blocks_2_0_bias_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_0_bias_updated_closure_args_t *args =
      (_perturbrademacher_blocks_2_0_bias_updated_closure_args_t *)_perturbrademacher_blocks_2_0_bias_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_input_9_ref =
      (void *)((char *)DeeployOptNetwork_input_9 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_output_9_ref =
      (void *)((char *)DeeployOptNetwork_output_9 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_input_9_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_2_0_bias_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_2_0_bias_updated_cluster_fork_args =
        (_perturbrademacher_blocks_2_0_bias_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_2_0_bias_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_2_0_bias_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_output_9_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_2_0_bias_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_2_0_bias_updated_closure_L3(void *_perturbrademacher_blocks_2_0_bias_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_0_bias_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_2_0_bias_updated_closure_L3_args_t *)_perturbrademacher_blocks_2_0_bias_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_2_0_bias_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_2_0_bias_updated_closure_args =
      (_perturbrademacher_blocks_2_0_bias_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr =
                                                                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr};

  // _perturbrademacher_blocks_2_0_bias_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_2_0_bias_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_2_0_bias_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref;
} _perturbrademacher_blocks_2_1_weight_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_2_1_weight_updated_tiling_closure(void *_perturbrademacher_blocks_2_1_weight_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_1_weight_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_2_1_weight_updated_tiling_closure_args_t *)_perturbrademacher_blocks_2_1_weight_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_2_1_weight_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_2_1_weight_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_2_1_weight_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_2_1_weight_updated_chunk = (16 >> perturbrademacher_blocks_2_1_weight_updated_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_2_1_weight_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_2_1_weight_updated_chunk * perturbrademacher_blocks_2_1_weight_updated_core_id, (uint32_t)16);
  uint32_t perturbrademacher_blocks_2_1_weight_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_2_1_weight_updated_chunk_start + perturbrademacher_blocks_2_1_weight_updated_chunk, (uint32_t)16);
  uint32_t perturbrademacher_blocks_2_1_weight_updated_local_size =
      perturbrademacher_blocks_2_1_weight_updated_chunk_stop - perturbrademacher_blocks_2_1_weight_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 14 + perturbrademacher_blocks_2_1_weight_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_2_1_weight_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref
                                [perturbrademacher_blocks_2_1_weight_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref
                                [perturbrademacher_blocks_2_1_weight_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_2_1_weight_updated_local_size,
                            zo_eco_mom[14] + perturbrademacher_blocks_2_1_weight_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref
                                              [perturbrademacher_blocks_2_1_weight_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref
                                              [perturbrademacher_blocks_2_1_weight_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_2_1_weight_updated_local_size, perturbrademacher_blocks_2_1_weight_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref
                                    [perturbrademacher_blocks_2_1_weight_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref
                                    [perturbrademacher_blocks_2_1_weight_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_2_1_weight_updated_local_size, perturbrademacher_blocks_2_1_weight_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref;
} _perturbrademacher_blocks_2_1_weight_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_2_1_weight_updated_cluster_fork(void *_perturbrademacher_blocks_2_1_weight_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_1_weight_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_2_1_weight_updated_cluster_fork_args_t *)_perturbrademacher_blocks_2_1_weight_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_2_1_weight_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_2_1_weight_updated_tiling_closure_args =
      (_perturbrademacher_blocks_2_1_weight_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref};

  // _perturbrademacher_blocks_2_1_weight_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_2_1_weight_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_2_1_weight_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_2_1_weight_updated_closure_args_t;

static void _perturbrademacher_blocks_2_1_weight_updated_closure(void *_perturbrademacher_blocks_2_1_weight_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_1_weight_updated_closure_args_t *args =
      (_perturbrademacher_blocks_2_1_weight_updated_closure_args_t *)_perturbrademacher_blocks_2_1_weight_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_input_10_ref =
      (void *)((char *)DeeployOptNetwork_input_10 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_output_10_ref =
      (void *)((char *)DeeployOptNetwork_output_10 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_input_10_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_2_1_weight_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_2_1_weight_updated_cluster_fork_args =
        (_perturbrademacher_blocks_2_1_weight_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_2_1_weight_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_2_1_weight_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_output_10_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_2_1_weight_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_2_1_weight_updated_closure_L3(void *_perturbrademacher_blocks_2_1_weight_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_1_weight_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_2_1_weight_updated_closure_L3_args_t *)_perturbrademacher_blocks_2_1_weight_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_2_1_weight_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_2_1_weight_updated_closure_args =
      (_perturbrademacher_blocks_2_1_weight_updated_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr};

  // _perturbrademacher_blocks_2_1_weight_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_2_1_weight_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_2_1_weight_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref;
} _perturbrademacher_blocks_2_1_bias_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_2_1_bias_updated_tiling_closure(void *_perturbrademacher_blocks_2_1_bias_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_1_bias_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_2_1_bias_updated_tiling_closure_args_t *)_perturbrademacher_blocks_2_1_bias_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_2_1_bias_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_2_1_bias_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_2_1_bias_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_2_1_bias_updated_chunk = (16 >> perturbrademacher_blocks_2_1_bias_updated_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_2_1_bias_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_2_1_bias_updated_chunk * perturbrademacher_blocks_2_1_bias_updated_core_id, (uint32_t)16);
  uint32_t perturbrademacher_blocks_2_1_bias_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_2_1_bias_updated_chunk_start + perturbrademacher_blocks_2_1_bias_updated_chunk, (uint32_t)16);
  uint32_t perturbrademacher_blocks_2_1_bias_updated_local_size =
      perturbrademacher_blocks_2_1_bias_updated_chunk_stop - perturbrademacher_blocks_2_1_bias_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 15 + perturbrademacher_blocks_2_1_bias_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_2_1_bias_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref
                                [perturbrademacher_blocks_2_1_bias_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref
                                [perturbrademacher_blocks_2_1_bias_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_2_1_bias_updated_local_size,
                            zo_eco_mom[15] + perturbrademacher_blocks_2_1_bias_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref
                                              [perturbrademacher_blocks_2_1_bias_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref
                                              [perturbrademacher_blocks_2_1_bias_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_2_1_bias_updated_local_size, perturbrademacher_blocks_2_1_bias_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref
                                    [perturbrademacher_blocks_2_1_bias_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref
                                    [perturbrademacher_blocks_2_1_bias_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_2_1_bias_updated_local_size, perturbrademacher_blocks_2_1_bias_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref;
} _perturbrademacher_blocks_2_1_bias_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_2_1_bias_updated_cluster_fork(void *_perturbrademacher_blocks_2_1_bias_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_1_bias_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_2_1_bias_updated_cluster_fork_args_t *)_perturbrademacher_blocks_2_1_bias_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_2_1_bias_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_2_1_bias_updated_tiling_closure_args =
      (_perturbrademacher_blocks_2_1_bias_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref};

  // _perturbrademacher_blocks_2_1_bias_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_2_1_bias_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_2_1_bias_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_2_1_bias_updated_closure_args_t;

static void _perturbrademacher_blocks_2_1_bias_updated_closure(void *_perturbrademacher_blocks_2_1_bias_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_1_bias_updated_closure_args_t *args =
      (_perturbrademacher_blocks_2_1_bias_updated_closure_args_t *)_perturbrademacher_blocks_2_1_bias_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_input_11_ref =
      (void *)((char *)DeeployOptNetwork_input_11 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_output_11_ref =
      (void *)((char *)DeeployOptNetwork_output_11 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_input_11_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_2_1_bias_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_2_1_bias_updated_cluster_fork_args =
        (_perturbrademacher_blocks_2_1_bias_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_2_1_bias_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_2_1_bias_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_output_11_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_2_1_bias_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_2_1_bias_updated_closure_L3(void *_perturbrademacher_blocks_2_1_bias_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_1_bias_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_2_1_bias_updated_closure_L3_args_t *)_perturbrademacher_blocks_2_1_bias_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_2_1_bias_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_2_1_bias_updated_closure_args =
      (_perturbrademacher_blocks_2_1_bias_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr =
                                                                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr};

  // _perturbrademacher_blocks_2_1_bias_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_2_1_bias_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_2_1_bias_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref;
} _perturbrademacher_blocks_0_0_weight_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_0_0_weight_updated_tiling_closure(void *_perturbrademacher_blocks_0_0_weight_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_0_weight_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_0_0_weight_updated_tiling_closure_args_t *)_perturbrademacher_blocks_0_0_weight_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_0_0_weight_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_0_0_weight_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_0_0_weight_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_0_0_weight_updated_chunk = (32 >> perturbrademacher_blocks_0_0_weight_updated_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_0_0_weight_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_0_0_weight_updated_chunk * perturbrademacher_blocks_0_0_weight_updated_core_id, (uint32_t)32);
  uint32_t perturbrademacher_blocks_0_0_weight_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_0_0_weight_updated_chunk_start + perturbrademacher_blocks_0_0_weight_updated_chunk, (uint32_t)32);
  uint32_t perturbrademacher_blocks_0_0_weight_updated_local_size =
      perturbrademacher_blocks_0_0_weight_updated_chunk_stop - perturbrademacher_blocks_0_0_weight_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 0 + perturbrademacher_blocks_0_0_weight_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_0_0_weight_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref
                                [perturbrademacher_blocks_0_0_weight_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref
                                [perturbrademacher_blocks_0_0_weight_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_0_0_weight_updated_local_size,
                            zo_eco_mom[0] + perturbrademacher_blocks_0_0_weight_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref
                                              [perturbrademacher_blocks_0_0_weight_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref
                                              [perturbrademacher_blocks_0_0_weight_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_0_0_weight_updated_local_size, perturbrademacher_blocks_0_0_weight_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref
                                    [perturbrademacher_blocks_0_0_weight_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref
                                    [perturbrademacher_blocks_0_0_weight_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_0_0_weight_updated_local_size, perturbrademacher_blocks_0_0_weight_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref;
} _perturbrademacher_blocks_0_0_weight_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_0_0_weight_updated_cluster_fork(void *_perturbrademacher_blocks_0_0_weight_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_0_weight_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_0_0_weight_updated_cluster_fork_args_t *)_perturbrademacher_blocks_0_0_weight_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_0_0_weight_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_0_0_weight_updated_tiling_closure_args =
      (_perturbrademacher_blocks_0_0_weight_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref};

  // _perturbrademacher_blocks_0_0_weight_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_0_0_weight_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_0_0_weight_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_0_0_weight_updated_closure_args_t;

static void _perturbrademacher_blocks_0_0_weight_updated_closure(void *_perturbrademacher_blocks_0_0_weight_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_0_weight_updated_closure_args_t *args =
      (_perturbrademacher_blocks_0_0_weight_updated_closure_args_t *)_perturbrademacher_blocks_0_0_weight_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_input_0_ref =
      (void *)((char *)DeeployOptNetwork_input_0 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_output_0_ref =
      (void *)((char *)DeeployOptNetwork_output_0 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_0_0_weight_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_0_0_weight_updated_cluster_fork_args =
        (_perturbrademacher_blocks_0_0_weight_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_0_0_weight_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_0_0_weight_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_output_0_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_0_0_weight_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_0_0_weight_updated_closure_L3(void *_perturbrademacher_blocks_0_0_weight_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_0_0_weight_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_0_0_weight_updated_closure_L3_args_t *)_perturbrademacher_blocks_0_0_weight_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_0_0_weight_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_0_0_weight_updated_closure_args =
      (_perturbrademacher_blocks_0_0_weight_updated_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr};

  // _perturbrademacher_blocks_0_0_weight_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_0_0_weight_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_0_0_weight_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref;
} _perturbrademacher_blocks_3_0_bias_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_3_0_bias_updated_tiling_closure(void *_perturbrademacher_blocks_3_0_bias_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_0_bias_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_3_0_bias_updated_tiling_closure_args_t *)_perturbrademacher_blocks_3_0_bias_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_3_0_bias_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_3_0_bias_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_3_0_bias_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_3_0_bias_updated_chunk = (32 >> perturbrademacher_blocks_3_0_bias_updated_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_3_0_bias_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_3_0_bias_updated_chunk * perturbrademacher_blocks_3_0_bias_updated_core_id, (uint32_t)32);
  uint32_t perturbrademacher_blocks_3_0_bias_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_3_0_bias_updated_chunk_start + perturbrademacher_blocks_3_0_bias_updated_chunk, (uint32_t)32);
  uint32_t perturbrademacher_blocks_3_0_bias_updated_local_size =
      perturbrademacher_blocks_3_0_bias_updated_chunk_stop - perturbrademacher_blocks_3_0_bias_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 19 + perturbrademacher_blocks_3_0_bias_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_3_0_bias_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref
                                [perturbrademacher_blocks_3_0_bias_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref
                                [perturbrademacher_blocks_3_0_bias_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_3_0_bias_updated_local_size,
                            zo_eco_mom[19] + perturbrademacher_blocks_3_0_bias_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref
                                              [perturbrademacher_blocks_3_0_bias_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref
                                              [perturbrademacher_blocks_3_0_bias_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_3_0_bias_updated_local_size, perturbrademacher_blocks_3_0_bias_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref
                                    [perturbrademacher_blocks_3_0_bias_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref
                                    [perturbrademacher_blocks_3_0_bias_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_3_0_bias_updated_local_size, perturbrademacher_blocks_3_0_bias_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref;
} _perturbrademacher_blocks_3_0_bias_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_3_0_bias_updated_cluster_fork(void *_perturbrademacher_blocks_3_0_bias_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_0_bias_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_3_0_bias_updated_cluster_fork_args_t *)_perturbrademacher_blocks_3_0_bias_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_3_0_bias_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_3_0_bias_updated_tiling_closure_args =
      (_perturbrademacher_blocks_3_0_bias_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref};

  // _perturbrademacher_blocks_3_0_bias_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_3_0_bias_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_3_0_bias_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_3_0_bias_updated_closure_args_t;

static void _perturbrademacher_blocks_3_0_bias_updated_closure(void *_perturbrademacher_blocks_3_0_bias_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_0_bias_updated_closure_args_t *args =
      (_perturbrademacher_blocks_3_0_bias_updated_closure_args_t *)_perturbrademacher_blocks_3_0_bias_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_input_13_ref =
      (void *)((char *)DeeployOptNetwork_input_13 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_output_13_ref =
      (void *)((char *)DeeployOptNetwork_output_13 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_input_13_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_3_0_bias_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_3_0_bias_updated_cluster_fork_args =
        (_perturbrademacher_blocks_3_0_bias_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_3_0_bias_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_3_0_bias_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_output_13_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_3_0_bias_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_3_0_bias_updated_closure_L3(void *_perturbrademacher_blocks_3_0_bias_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_0_bias_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_3_0_bias_updated_closure_L3_args_t *)_perturbrademacher_blocks_3_0_bias_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_3_0_bias_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_3_0_bias_updated_closure_args =
      (_perturbrademacher_blocks_3_0_bias_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr =
                                                                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr};

  // _perturbrademacher_blocks_3_0_bias_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_3_0_bias_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_3_0_bias_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref;
} _perturbrademacher_blocks_3_1_weight_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_3_1_weight_updated_tiling_closure(void *_perturbrademacher_blocks_3_1_weight_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_1_weight_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_3_1_weight_updated_tiling_closure_args_t *)_perturbrademacher_blocks_3_1_weight_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_3_1_weight_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_3_1_weight_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_3_1_weight_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_3_1_weight_updated_chunk = (32 >> perturbrademacher_blocks_3_1_weight_updated_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_3_1_weight_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_3_1_weight_updated_chunk * perturbrademacher_blocks_3_1_weight_updated_core_id, (uint32_t)32);
  uint32_t perturbrademacher_blocks_3_1_weight_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_3_1_weight_updated_chunk_start + perturbrademacher_blocks_3_1_weight_updated_chunk, (uint32_t)32);
  uint32_t perturbrademacher_blocks_3_1_weight_updated_local_size =
      perturbrademacher_blocks_3_1_weight_updated_chunk_stop - perturbrademacher_blocks_3_1_weight_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 20 + perturbrademacher_blocks_3_1_weight_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_3_1_weight_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref
                                [perturbrademacher_blocks_3_1_weight_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref
                                [perturbrademacher_blocks_3_1_weight_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_3_1_weight_updated_local_size,
                            zo_eco_mom[20] + perturbrademacher_blocks_3_1_weight_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref
                                              [perturbrademacher_blocks_3_1_weight_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref
                                              [perturbrademacher_blocks_3_1_weight_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_3_1_weight_updated_local_size, perturbrademacher_blocks_3_1_weight_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref
                                    [perturbrademacher_blocks_3_1_weight_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref
                                    [perturbrademacher_blocks_3_1_weight_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_3_1_weight_updated_local_size, perturbrademacher_blocks_3_1_weight_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref;
} _perturbrademacher_blocks_3_1_weight_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_3_1_weight_updated_cluster_fork(void *_perturbrademacher_blocks_3_1_weight_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_1_weight_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_3_1_weight_updated_cluster_fork_args_t *)_perturbrademacher_blocks_3_1_weight_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_3_1_weight_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_3_1_weight_updated_tiling_closure_args =
      (_perturbrademacher_blocks_3_1_weight_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref};

  // _perturbrademacher_blocks_3_1_weight_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_3_1_weight_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_3_1_weight_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_3_1_weight_updated_closure_args_t;

static void _perturbrademacher_blocks_3_1_weight_updated_closure(void *_perturbrademacher_blocks_3_1_weight_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_1_weight_updated_closure_args_t *args =
      (_perturbrademacher_blocks_3_1_weight_updated_closure_args_t *)_perturbrademacher_blocks_3_1_weight_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_input_14_ref =
      (void *)((char *)DeeployOptNetwork_input_14 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_output_14_ref =
      (void *)((char *)DeeployOptNetwork_output_14 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_input_14_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_3_1_weight_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_3_1_weight_updated_cluster_fork_args =
        (_perturbrademacher_blocks_3_1_weight_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_3_1_weight_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_3_1_weight_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_output_14_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_3_1_weight_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_3_1_weight_updated_closure_L3(void *_perturbrademacher_blocks_3_1_weight_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_1_weight_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_3_1_weight_updated_closure_L3_args_t *)_perturbrademacher_blocks_3_1_weight_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_3_1_weight_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_3_1_weight_updated_closure_args =
      (_perturbrademacher_blocks_3_1_weight_updated_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr};

  // _perturbrademacher_blocks_3_1_weight_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_3_1_weight_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_3_1_weight_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref;
} _perturbrademacher_blocks_3_1_bias_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_3_1_bias_updated_tiling_closure(void *_perturbrademacher_blocks_3_1_bias_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_1_bias_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_3_1_bias_updated_tiling_closure_args_t *)_perturbrademacher_blocks_3_1_bias_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_3_1_bias_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_3_1_bias_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_3_1_bias_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_3_1_bias_updated_chunk = (32 >> perturbrademacher_blocks_3_1_bias_updated_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_3_1_bias_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_3_1_bias_updated_chunk * perturbrademacher_blocks_3_1_bias_updated_core_id, (uint32_t)32);
  uint32_t perturbrademacher_blocks_3_1_bias_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_3_1_bias_updated_chunk_start + perturbrademacher_blocks_3_1_bias_updated_chunk, (uint32_t)32);
  uint32_t perturbrademacher_blocks_3_1_bias_updated_local_size =
      perturbrademacher_blocks_3_1_bias_updated_chunk_stop - perturbrademacher_blocks_3_1_bias_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 21 + perturbrademacher_blocks_3_1_bias_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_3_1_bias_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref
                                [perturbrademacher_blocks_3_1_bias_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref
                                [perturbrademacher_blocks_3_1_bias_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_3_1_bias_updated_local_size,
                            zo_eco_mom[21] + perturbrademacher_blocks_3_1_bias_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref
                                              [perturbrademacher_blocks_3_1_bias_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref
                                              [perturbrademacher_blocks_3_1_bias_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_3_1_bias_updated_local_size, perturbrademacher_blocks_3_1_bias_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref
                                    [perturbrademacher_blocks_3_1_bias_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref
                                    [perturbrademacher_blocks_3_1_bias_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_3_1_bias_updated_local_size, perturbrademacher_blocks_3_1_bias_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref;
} _perturbrademacher_blocks_3_1_bias_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_3_1_bias_updated_cluster_fork(void *_perturbrademacher_blocks_3_1_bias_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_1_bias_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_3_1_bias_updated_cluster_fork_args_t *)_perturbrademacher_blocks_3_1_bias_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_3_1_bias_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_3_1_bias_updated_tiling_closure_args =
      (_perturbrademacher_blocks_3_1_bias_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref};

  // _perturbrademacher_blocks_3_1_bias_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_3_1_bias_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_3_1_bias_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_3_1_bias_updated_closure_args_t;

static void _perturbrademacher_blocks_3_1_bias_updated_closure(void *_perturbrademacher_blocks_3_1_bias_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_1_bias_updated_closure_args_t *args =
      (_perturbrademacher_blocks_3_1_bias_updated_closure_args_t *)_perturbrademacher_blocks_3_1_bias_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_input_15_ref =
      (void *)((char *)DeeployOptNetwork_input_15 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_output_15_ref =
      (void *)((char *)DeeployOptNetwork_output_15 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_input_15_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_3_1_bias_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_3_1_bias_updated_cluster_fork_args =
        (_perturbrademacher_blocks_3_1_bias_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_3_1_bias_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_3_1_bias_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_output_15_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_3_1_bias_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_3_1_bias_updated_closure_L3(void *_perturbrademacher_blocks_3_1_bias_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_1_bias_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_3_1_bias_updated_closure_L3_args_t *)_perturbrademacher_blocks_3_1_bias_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_3_1_bias_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_3_1_bias_updated_closure_args =
      (_perturbrademacher_blocks_3_1_bias_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr =
                                                                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr};

  // _perturbrademacher_blocks_3_1_bias_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_3_1_bias_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_3_1_bias_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref;
} _perturbrademacher_blocks_4_0_bias_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_4_0_bias_updated_tiling_closure(void *_perturbrademacher_blocks_4_0_bias_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_0_bias_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_4_0_bias_updated_tiling_closure_args_t *)_perturbrademacher_blocks_4_0_bias_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_4_0_bias_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_4_0_bias_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_4_0_bias_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_4_0_bias_updated_chunk = (32 >> perturbrademacher_blocks_4_0_bias_updated_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_4_0_bias_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_4_0_bias_updated_chunk * perturbrademacher_blocks_4_0_bias_updated_core_id, (uint32_t)32);
  uint32_t perturbrademacher_blocks_4_0_bias_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_4_0_bias_updated_chunk_start + perturbrademacher_blocks_4_0_bias_updated_chunk, (uint32_t)32);
  uint32_t perturbrademacher_blocks_4_0_bias_updated_local_size =
      perturbrademacher_blocks_4_0_bias_updated_chunk_stop - perturbrademacher_blocks_4_0_bias_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 25 + perturbrademacher_blocks_4_0_bias_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_4_0_bias_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref
                                [perturbrademacher_blocks_4_0_bias_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref
                                [perturbrademacher_blocks_4_0_bias_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_4_0_bias_updated_local_size,
                            zo_eco_mom[25] + perturbrademacher_blocks_4_0_bias_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref
                                              [perturbrademacher_blocks_4_0_bias_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref
                                              [perturbrademacher_blocks_4_0_bias_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_4_0_bias_updated_local_size, perturbrademacher_blocks_4_0_bias_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref
                                    [perturbrademacher_blocks_4_0_bias_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref
                                    [perturbrademacher_blocks_4_0_bias_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_4_0_bias_updated_local_size, perturbrademacher_blocks_4_0_bias_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref;
} _perturbrademacher_blocks_4_0_bias_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_4_0_bias_updated_cluster_fork(void *_perturbrademacher_blocks_4_0_bias_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_0_bias_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_4_0_bias_updated_cluster_fork_args_t *)_perturbrademacher_blocks_4_0_bias_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_4_0_bias_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_4_0_bias_updated_tiling_closure_args =
      (_perturbrademacher_blocks_4_0_bias_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref};

  // _perturbrademacher_blocks_4_0_bias_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_4_0_bias_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_4_0_bias_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_4_0_bias_updated_closure_args_t;

static void _perturbrademacher_blocks_4_0_bias_updated_closure(void *_perturbrademacher_blocks_4_0_bias_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_0_bias_updated_closure_args_t *args =
      (_perturbrademacher_blocks_4_0_bias_updated_closure_args_t *)_perturbrademacher_blocks_4_0_bias_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_input_17_ref =
      (void *)((char *)DeeployOptNetwork_input_17 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_output_17_ref =
      (void *)((char *)DeeployOptNetwork_output_17 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_input_17_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_4_0_bias_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_4_0_bias_updated_cluster_fork_args =
        (_perturbrademacher_blocks_4_0_bias_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_4_0_bias_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_4_0_bias_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_output_17_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_4_0_bias_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_4_0_bias_updated_closure_L3(void *_perturbrademacher_blocks_4_0_bias_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_0_bias_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_4_0_bias_updated_closure_L3_args_t *)_perturbrademacher_blocks_4_0_bias_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_4_0_bias_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_4_0_bias_updated_closure_args =
      (_perturbrademacher_blocks_4_0_bias_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr =
                                                                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr};

  // _perturbrademacher_blocks_4_0_bias_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_4_0_bias_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_4_0_bias_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref;
} _perturbrademacher_blocks_4_1_weight_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_4_1_weight_updated_tiling_closure(void *_perturbrademacher_blocks_4_1_weight_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_1_weight_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_4_1_weight_updated_tiling_closure_args_t *)_perturbrademacher_blocks_4_1_weight_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_4_1_weight_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_4_1_weight_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_4_1_weight_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_4_1_weight_updated_chunk = (32 >> perturbrademacher_blocks_4_1_weight_updated_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_4_1_weight_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_4_1_weight_updated_chunk * perturbrademacher_blocks_4_1_weight_updated_core_id, (uint32_t)32);
  uint32_t perturbrademacher_blocks_4_1_weight_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_4_1_weight_updated_chunk_start + perturbrademacher_blocks_4_1_weight_updated_chunk, (uint32_t)32);
  uint32_t perturbrademacher_blocks_4_1_weight_updated_local_size =
      perturbrademacher_blocks_4_1_weight_updated_chunk_stop - perturbrademacher_blocks_4_1_weight_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 26 + perturbrademacher_blocks_4_1_weight_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_4_1_weight_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref
                                [perturbrademacher_blocks_4_1_weight_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref
                                [perturbrademacher_blocks_4_1_weight_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_4_1_weight_updated_local_size,
                            zo_eco_mom[26] + perturbrademacher_blocks_4_1_weight_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref
                                              [perturbrademacher_blocks_4_1_weight_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref
                                              [perturbrademacher_blocks_4_1_weight_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_4_1_weight_updated_local_size, perturbrademacher_blocks_4_1_weight_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref
                                    [perturbrademacher_blocks_4_1_weight_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref
                                    [perturbrademacher_blocks_4_1_weight_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_4_1_weight_updated_local_size, perturbrademacher_blocks_4_1_weight_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref;
} _perturbrademacher_blocks_4_1_weight_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_4_1_weight_updated_cluster_fork(void *_perturbrademacher_blocks_4_1_weight_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_1_weight_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_4_1_weight_updated_cluster_fork_args_t *)_perturbrademacher_blocks_4_1_weight_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_4_1_weight_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_4_1_weight_updated_tiling_closure_args =
      (_perturbrademacher_blocks_4_1_weight_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref};

  // _perturbrademacher_blocks_4_1_weight_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_4_1_weight_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_4_1_weight_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_4_1_weight_updated_closure_args_t;

static void _perturbrademacher_blocks_4_1_weight_updated_closure(void *_perturbrademacher_blocks_4_1_weight_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_1_weight_updated_closure_args_t *args =
      (_perturbrademacher_blocks_4_1_weight_updated_closure_args_t *)_perturbrademacher_blocks_4_1_weight_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_input_18_ref =
      (void *)((char *)DeeployOptNetwork_input_18 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_output_18_ref =
      (void *)((char *)DeeployOptNetwork_output_18 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_input_18_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_4_1_weight_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_4_1_weight_updated_cluster_fork_args =
        (_perturbrademacher_blocks_4_1_weight_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_4_1_weight_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_4_1_weight_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_output_18_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_4_1_weight_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_4_1_weight_updated_closure_L3(void *_perturbrademacher_blocks_4_1_weight_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_1_weight_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_4_1_weight_updated_closure_L3_args_t *)_perturbrademacher_blocks_4_1_weight_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_4_1_weight_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_4_1_weight_updated_closure_args =
      (_perturbrademacher_blocks_4_1_weight_updated_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr};

  // _perturbrademacher_blocks_4_1_weight_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_4_1_weight_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_4_1_weight_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref;
} _perturbrademacher_blocks_4_1_bias_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_4_1_bias_updated_tiling_closure(void *_perturbrademacher_blocks_4_1_bias_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_1_bias_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_4_1_bias_updated_tiling_closure_args_t *)_perturbrademacher_blocks_4_1_bias_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_4_1_bias_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_4_1_bias_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_4_1_bias_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_4_1_bias_updated_chunk = (32 >> perturbrademacher_blocks_4_1_bias_updated_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_4_1_bias_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_4_1_bias_updated_chunk * perturbrademacher_blocks_4_1_bias_updated_core_id, (uint32_t)32);
  uint32_t perturbrademacher_blocks_4_1_bias_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_4_1_bias_updated_chunk_start + perturbrademacher_blocks_4_1_bias_updated_chunk, (uint32_t)32);
  uint32_t perturbrademacher_blocks_4_1_bias_updated_local_size =
      perturbrademacher_blocks_4_1_bias_updated_chunk_stop - perturbrademacher_blocks_4_1_bias_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 27 + perturbrademacher_blocks_4_1_bias_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_4_1_bias_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref
                                [perturbrademacher_blocks_4_1_bias_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref
                                [perturbrademacher_blocks_4_1_bias_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_4_1_bias_updated_local_size,
                            zo_eco_mom[27] + perturbrademacher_blocks_4_1_bias_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref
                                              [perturbrademacher_blocks_4_1_bias_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref
                                              [perturbrademacher_blocks_4_1_bias_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_4_1_bias_updated_local_size, perturbrademacher_blocks_4_1_bias_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref
                                    [perturbrademacher_blocks_4_1_bias_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref
                                    [perturbrademacher_blocks_4_1_bias_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_4_1_bias_updated_local_size, perturbrademacher_blocks_4_1_bias_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref;
} _perturbrademacher_blocks_4_1_bias_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_4_1_bias_updated_cluster_fork(void *_perturbrademacher_blocks_4_1_bias_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_1_bias_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_4_1_bias_updated_cluster_fork_args_t *)_perturbrademacher_blocks_4_1_bias_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_4_1_bias_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_4_1_bias_updated_tiling_closure_args =
      (_perturbrademacher_blocks_4_1_bias_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref};

  // _perturbrademacher_blocks_4_1_bias_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_4_1_bias_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_4_1_bias_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_4_1_bias_updated_closure_args_t;

static void _perturbrademacher_blocks_4_1_bias_updated_closure(void *_perturbrademacher_blocks_4_1_bias_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_1_bias_updated_closure_args_t *args =
      (_perturbrademacher_blocks_4_1_bias_updated_closure_args_t *)_perturbrademacher_blocks_4_1_bias_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_input_19_ref =
      (void *)((char *)DeeployOptNetwork_input_19 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_output_19_ref =
      (void *)((char *)DeeployOptNetwork_output_19 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_input_19_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_4_1_bias_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_4_1_bias_updated_cluster_fork_args =
        (_perturbrademacher_blocks_4_1_bias_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_4_1_bias_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_4_1_bias_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_output_19_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr;
} _perturbrademacher_blocks_4_1_bias_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_4_1_bias_updated_closure_L3(void *_perturbrademacher_blocks_4_1_bias_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_1_bias_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_4_1_bias_updated_closure_L3_args_t *)_perturbrademacher_blocks_4_1_bias_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_4_1_bias_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_4_1_bias_updated_closure_args =
      (_perturbrademacher_blocks_4_1_bias_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr =
                                                                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr};

  // _perturbrademacher_blocks_4_1_bias_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_4_1_bias_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_4_1_bias_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref;
} _perturbrademacher_fc_weight_updated_tiling_closure_args_t;

static void _perturbrademacher_fc_weight_updated_tiling_closure(void *_perturbrademacher_fc_weight_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_fc_weight_updated_tiling_closure_args_t *args =
      (_perturbrademacher_fc_weight_updated_tiling_closure_args_t *)_perturbrademacher_fc_weight_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_fc_weight_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_fc_weight_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_fc_weight_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_fc_weight_updated_chunk = (288 >> perturbrademacher_fc_weight_updated_log2Core) + ((288 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_fc_weight_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_fc_weight_updated_chunk * perturbrademacher_fc_weight_updated_core_id, (uint32_t)288);
  uint32_t perturbrademacher_fc_weight_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_fc_weight_updated_chunk_start + perturbrademacher_fc_weight_updated_chunk, (uint32_t)288);
  uint32_t perturbrademacher_fc_weight_updated_local_size = perturbrademacher_fc_weight_updated_chunk_stop - perturbrademacher_fc_weight_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 30 + perturbrademacher_fc_weight_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_fc_weight_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32(
        (const float32_t
             *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref[perturbrademacher_fc_weight_updated_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref[perturbrademacher_fc_weight_updated_chunk_start],
        chunk_seed, perturbation_sign, perturbrademacher_fc_weight_updated_local_size, zo_eco_mom[30] + perturbrademacher_fc_weight_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t
             *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref[perturbrademacher_fc_weight_updated_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref[perturbrademacher_fc_weight_updated_chunk_start],
        chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        perturbrademacher_fc_weight_updated_local_size, perturbrademacher_fc_weight_updated_eps);
#else
    ApplyRademacherPerturbation(
        (const float32_t
             *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref[perturbrademacher_fc_weight_updated_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref[perturbrademacher_fc_weight_updated_chunk_start],
        chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        perturbrademacher_fc_weight_updated_local_size, perturbrademacher_fc_weight_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref;
} _perturbrademacher_fc_weight_updated_cluster_fork_args_t;

static void _perturbrademacher_fc_weight_updated_cluster_fork(void *_perturbrademacher_fc_weight_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_fc_weight_updated_cluster_fork_args_t *args =
      (_perturbrademacher_fc_weight_updated_cluster_fork_args_t *)_perturbrademacher_fc_weight_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_fc_weight_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_fc_weight_updated_tiling_closure_args =
      (_perturbrademacher_fc_weight_updated_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref =
                                                                       DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref,
                                                                   .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref =
                                                                       DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref};

  // _perturbrademacher_fc_weight_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_fc_weight_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_fc_weight_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr;
} _perturbrademacher_fc_weight_updated_closure_args_t;

static void _perturbrademacher_fc_weight_updated_closure(void *_perturbrademacher_fc_weight_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_fc_weight_updated_closure_args_t *args =
      (_perturbrademacher_fc_weight_updated_closure_args_t *)_perturbrademacher_fc_weight_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 1152);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_input_20_ref =
      (void *)((char *)DeeployOptNetwork_input_20 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_output_20_ref =
      (void *)((char *)DeeployOptNetwork_output_20 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442944, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_input_20_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_fc_weight_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_fc_weight_updated_cluster_fork_args =
        (_perturbrademacher_fc_weight_updated_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref =
                                                                       DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_in_ref,
                                                                   .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref =
                                                                       DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_fc_weight_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_fc_weight_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311872, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_output_20_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr;
} _perturbrademacher_fc_weight_updated_closure_L3_args_t;

static void _perturbrademacher_fc_weight_updated_closure_L3(void *_perturbrademacher_fc_weight_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_fc_weight_updated_closure_L3_args_t *args =
      (_perturbrademacher_fc_weight_updated_closure_L3_args_t *)_perturbrademacher_fc_weight_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_fc_weight_updated_closure_args_t DeeployOptNetwork__perturbrademacher_fc_weight_updated_closure_args =
      (_perturbrademacher_fc_weight_updated_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr =
                                                                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr};

  // _perturbrademacher_fc_weight_updated_closure CLOSURE CALL
  _perturbrademacher_fc_weight_updated_closure(&DeeployOptNetwork__perturbrademacher_fc_weight_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref;
} _perturbrademacher_blocks_1_0_weight_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_1_0_weight_updated_tiling_closure(void *_perturbrademacher_blocks_1_0_weight_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_0_weight_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_1_0_weight_updated_tiling_closure_args_t *)_perturbrademacher_blocks_1_0_weight_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_1_0_weight_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_1_0_weight_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_1_0_weight_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_1_0_weight_updated_chunk = (2048 >> perturbrademacher_blocks_1_0_weight_updated_log2Core) + ((2048 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_1_0_weight_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_1_0_weight_updated_chunk * perturbrademacher_blocks_1_0_weight_updated_core_id, (uint32_t)2048);
  uint32_t perturbrademacher_blocks_1_0_weight_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_1_0_weight_updated_chunk_start + perturbrademacher_blocks_1_0_weight_updated_chunk, (uint32_t)2048);
  uint32_t perturbrademacher_blocks_1_0_weight_updated_local_size =
      perturbrademacher_blocks_1_0_weight_updated_chunk_stop - perturbrademacher_blocks_1_0_weight_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 6 + perturbrademacher_blocks_1_0_weight_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_1_0_weight_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref
                                [perturbrademacher_blocks_1_0_weight_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref
                                [perturbrademacher_blocks_1_0_weight_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_1_0_weight_updated_local_size,
                            zo_eco_mom[6] + perturbrademacher_blocks_1_0_weight_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref
                                              [perturbrademacher_blocks_1_0_weight_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref
                                              [perturbrademacher_blocks_1_0_weight_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_1_0_weight_updated_local_size, perturbrademacher_blocks_1_0_weight_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref
                                    [perturbrademacher_blocks_1_0_weight_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref
                                    [perturbrademacher_blocks_1_0_weight_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_1_0_weight_updated_local_size, perturbrademacher_blocks_1_0_weight_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref;
} _perturbrademacher_blocks_1_0_weight_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_1_0_weight_updated_cluster_fork(void *_perturbrademacher_blocks_1_0_weight_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_0_weight_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_1_0_weight_updated_cluster_fork_args_t *)_perturbrademacher_blocks_1_0_weight_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_1_0_weight_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_1_0_weight_updated_tiling_closure_args =
      (_perturbrademacher_blocks_1_0_weight_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref};

  // _perturbrademacher_blocks_1_0_weight_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_1_0_weight_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_1_0_weight_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_1_0_weight_updated_closure_args_t;

static void _perturbrademacher_blocks_1_0_weight_updated_closure(void *_perturbrademacher_blocks_1_0_weight_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_0_weight_updated_closure_args_t *args =
      (_perturbrademacher_blocks_1_0_weight_updated_closure_args_t *)_perturbrademacher_blocks_1_0_weight_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 8192);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_input_4_ref =
      (void *)((char *)DeeployOptNetwork_input_4 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_output_4_ref =
      (void *)((char *)DeeployOptNetwork_output_4 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1449984, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_input_4_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_1_0_weight_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_1_0_weight_updated_cluster_fork_args =
        (_perturbrademacher_blocks_1_0_weight_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_1_0_weight_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_1_0_weight_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1318912, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_output_4_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_1_0_weight_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_1_0_weight_updated_closure_L3(void *_perturbrademacher_blocks_1_0_weight_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_1_0_weight_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_1_0_weight_updated_closure_L3_args_t *)_perturbrademacher_blocks_1_0_weight_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_1_0_weight_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_1_0_weight_updated_closure_args =
      (_perturbrademacher_blocks_1_0_weight_updated_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr};

  // _perturbrademacher_blocks_1_0_weight_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_1_0_weight_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_1_0_weight_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref;
} _perturbrademacher_blocks_2_0_weight_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_2_0_weight_updated_tiling_closure(void *_perturbrademacher_blocks_2_0_weight_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_0_weight_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_2_0_weight_updated_tiling_closure_args_t *)_perturbrademacher_blocks_2_0_weight_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_2_0_weight_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_2_0_weight_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_2_0_weight_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_2_0_weight_updated_chunk = (2048 >> perturbrademacher_blocks_2_0_weight_updated_log2Core) + ((2048 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_2_0_weight_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_2_0_weight_updated_chunk * perturbrademacher_blocks_2_0_weight_updated_core_id, (uint32_t)2048);
  uint32_t perturbrademacher_blocks_2_0_weight_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_2_0_weight_updated_chunk_start + perturbrademacher_blocks_2_0_weight_updated_chunk, (uint32_t)2048);
  uint32_t perturbrademacher_blocks_2_0_weight_updated_local_size =
      perturbrademacher_blocks_2_0_weight_updated_chunk_stop - perturbrademacher_blocks_2_0_weight_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 12 + perturbrademacher_blocks_2_0_weight_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_2_0_weight_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref
                                [perturbrademacher_blocks_2_0_weight_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref
                                [perturbrademacher_blocks_2_0_weight_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_2_0_weight_updated_local_size,
                            zo_eco_mom[12] + perturbrademacher_blocks_2_0_weight_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref
                                              [perturbrademacher_blocks_2_0_weight_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref
                                              [perturbrademacher_blocks_2_0_weight_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_2_0_weight_updated_local_size, perturbrademacher_blocks_2_0_weight_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref
                                    [perturbrademacher_blocks_2_0_weight_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref
                                    [perturbrademacher_blocks_2_0_weight_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_2_0_weight_updated_local_size, perturbrademacher_blocks_2_0_weight_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref;
} _perturbrademacher_blocks_2_0_weight_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_2_0_weight_updated_cluster_fork(void *_perturbrademacher_blocks_2_0_weight_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_0_weight_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_2_0_weight_updated_cluster_fork_args_t *)_perturbrademacher_blocks_2_0_weight_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_2_0_weight_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_2_0_weight_updated_tiling_closure_args =
      (_perturbrademacher_blocks_2_0_weight_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref};

  // _perturbrademacher_blocks_2_0_weight_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_2_0_weight_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_2_0_weight_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_2_0_weight_updated_closure_args_t;

static void _perturbrademacher_blocks_2_0_weight_updated_closure(void *_perturbrademacher_blocks_2_0_weight_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_0_weight_updated_closure_args_t *args =
      (_perturbrademacher_blocks_2_0_weight_updated_closure_args_t *)_perturbrademacher_blocks_2_0_weight_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 8192);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_input_8_ref =
      (void *)((char *)DeeployOptNetwork_input_8 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_output_8_ref =
      (void *)((char *)DeeployOptNetwork_output_8 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1449984, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_input_8_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_2_0_weight_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_2_0_weight_updated_cluster_fork_args =
        (_perturbrademacher_blocks_2_0_weight_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_2_0_weight_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_2_0_weight_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1318912, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_output_8_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_2_0_weight_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_2_0_weight_updated_closure_L3(void *_perturbrademacher_blocks_2_0_weight_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_2_0_weight_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_2_0_weight_updated_closure_L3_args_t *)_perturbrademacher_blocks_2_0_weight_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_2_0_weight_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_2_0_weight_updated_closure_args =
      (_perturbrademacher_blocks_2_0_weight_updated_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr};

  // _perturbrademacher_blocks_2_0_weight_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_2_0_weight_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_2_0_weight_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref;
} _perturbrademacher_blocks_3_0_weight_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_3_0_weight_updated_tiling_closure(void *_perturbrademacher_blocks_3_0_weight_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_0_weight_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_3_0_weight_updated_tiling_closure_args_t *)_perturbrademacher_blocks_3_0_weight_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_3_0_weight_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_3_0_weight_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_3_0_weight_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_3_0_weight_updated_chunk = (3584 >> perturbrademacher_blocks_3_0_weight_updated_log2Core) + ((3584 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_3_0_weight_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_3_0_weight_updated_chunk * perturbrademacher_blocks_3_0_weight_updated_core_id, (uint32_t)3584);
  uint32_t perturbrademacher_blocks_3_0_weight_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_3_0_weight_updated_chunk_start + perturbrademacher_blocks_3_0_weight_updated_chunk, (uint32_t)3584);
  uint32_t perturbrademacher_blocks_3_0_weight_updated_local_size =
      perturbrademacher_blocks_3_0_weight_updated_chunk_stop - perturbrademacher_blocks_3_0_weight_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 18 + perturbrademacher_blocks_3_0_weight_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_3_0_weight_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref
                                [perturbrademacher_blocks_3_0_weight_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref
                                [perturbrademacher_blocks_3_0_weight_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_3_0_weight_updated_local_size,
                            zo_eco_mom[18] + perturbrademacher_blocks_3_0_weight_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref
                                              [perturbrademacher_blocks_3_0_weight_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref
                                              [perturbrademacher_blocks_3_0_weight_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_3_0_weight_updated_local_size, perturbrademacher_blocks_3_0_weight_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref
                                    [perturbrademacher_blocks_3_0_weight_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref
                                    [perturbrademacher_blocks_3_0_weight_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_3_0_weight_updated_local_size, perturbrademacher_blocks_3_0_weight_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref;
} _perturbrademacher_blocks_3_0_weight_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_3_0_weight_updated_cluster_fork(void *_perturbrademacher_blocks_3_0_weight_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_0_weight_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_3_0_weight_updated_cluster_fork_args_t *)_perturbrademacher_blocks_3_0_weight_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_3_0_weight_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_3_0_weight_updated_tiling_closure_args =
      (_perturbrademacher_blocks_3_0_weight_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref};

  // _perturbrademacher_blocks_3_0_weight_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_3_0_weight_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_3_0_weight_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_3_0_weight_updated_closure_args_t;

static void _perturbrademacher_blocks_3_0_weight_updated_closure(void *_perturbrademacher_blocks_3_0_weight_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_0_weight_updated_closure_args_t *args =
      (_perturbrademacher_blocks_3_0_weight_updated_closure_args_t *)_perturbrademacher_blocks_3_0_weight_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 14336);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_input_12_ref =
      (void *)((char *)DeeployOptNetwork_input_12 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_output_12_ref =
      (void *)((char *)DeeployOptNetwork_output_12 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1456128, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_input_12_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_3_0_weight_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_3_0_weight_updated_cluster_fork_args =
        (_perturbrademacher_blocks_3_0_weight_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_3_0_weight_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_3_0_weight_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1325056, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_output_12_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_3_0_weight_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_3_0_weight_updated_closure_L3(void *_perturbrademacher_blocks_3_0_weight_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_3_0_weight_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_3_0_weight_updated_closure_L3_args_t *)_perturbrademacher_blocks_3_0_weight_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_3_0_weight_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_3_0_weight_updated_closure_args =
      (_perturbrademacher_blocks_3_0_weight_updated_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr};

  // _perturbrademacher_blocks_3_0_weight_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_3_0_weight_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_3_0_weight_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref;
} _perturbrademacher_blocks_4_0_weight_updated_tiling_closure_args_t;

static void _perturbrademacher_blocks_4_0_weight_updated_tiling_closure(void *_perturbrademacher_blocks_4_0_weight_updated_tiling_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_0_weight_updated_tiling_closure_args_t *args =
      (_perturbrademacher_blocks_4_0_weight_updated_tiling_closure_args_t *)_perturbrademacher_blocks_4_0_weight_updated_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: perturbrademacher_blocks_4_0_weight_updated, Op: PerturbRademacher)
  uint8_t perturbrademacher_blocks_4_0_weight_updated_core_id = (uint8_t)pi_core_id();
  uint8_t perturbrademacher_blocks_4_0_weight_updated_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t perturbrademacher_blocks_4_0_weight_updated_chunk = (7168 >> perturbrademacher_blocks_4_0_weight_updated_log2Core) + ((7168 & (NUM_CORES - 1)) != 0);
  uint32_t perturbrademacher_blocks_4_0_weight_updated_chunk_start =
      (uint32_t)MIN(perturbrademacher_blocks_4_0_weight_updated_chunk * perturbrademacher_blocks_4_0_weight_updated_core_id, (uint32_t)7168);
  uint32_t perturbrademacher_blocks_4_0_weight_updated_chunk_stop =
      (uint32_t)MIN(perturbrademacher_blocks_4_0_weight_updated_chunk_start + perturbrademacher_blocks_4_0_weight_updated_chunk, (uint32_t)7168);
  uint32_t perturbrademacher_blocks_4_0_weight_updated_local_size =
      perturbrademacher_blocks_4_0_weight_updated_chunk_stop - perturbrademacher_blocks_4_0_weight_updated_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 24 + perturbrademacher_blocks_4_0_weight_updated_core_id) ^ (0 * 0x9E3779B1u);
  float32_t perturbrademacher_blocks_4_0_weight_updated_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref
                                [perturbrademacher_blocks_4_0_weight_updated_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref
                                [perturbrademacher_blocks_4_0_weight_updated_chunk_start],
                            chunk_seed, perturbation_sign, perturbrademacher_blocks_4_0_weight_updated_local_size,
                            zo_eco_mom[24] + perturbrademacher_blocks_4_0_weight_updated_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref
                                              [perturbrademacher_blocks_4_0_weight_updated_chunk_start],
                                          (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref
                                              [perturbrademacher_blocks_4_0_weight_updated_chunk_start],
                                          chunk_seed,
                                          perturbation_sign, // ZO runtime control (ZORuntime.h)
                                          perturbrademacher_blocks_4_0_weight_updated_local_size, perturbrademacher_blocks_4_0_weight_updated_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref
                                    [perturbrademacher_blocks_4_0_weight_updated_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref
                                    [perturbrademacher_blocks_4_0_weight_updated_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                perturbrademacher_blocks_4_0_weight_updated_local_size, perturbrademacher_blocks_4_0_weight_updated_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref;
} _perturbrademacher_blocks_4_0_weight_updated_cluster_fork_args_t;

static void _perturbrademacher_blocks_4_0_weight_updated_cluster_fork(void *_perturbrademacher_blocks_4_0_weight_updated_cluster_fork_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_0_weight_updated_cluster_fork_args_t *args =
      (_perturbrademacher_blocks_4_0_weight_updated_cluster_fork_args_t *)_perturbrademacher_blocks_4_0_weight_updated_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_4_0_weight_updated_tiling_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_4_0_weight_updated_tiling_closure_args =
      (_perturbrademacher_blocks_4_0_weight_updated_tiling_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref,
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref};

  // _perturbrademacher_blocks_4_0_weight_updated_tiling_closure CLOSURE CALL
  _perturbrademacher_blocks_4_0_weight_updated_tiling_closure(&DeeployOptNetwork__perturbrademacher_blocks_4_0_weight_updated_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_4_0_weight_updated_closure_args_t;

static void _perturbrademacher_blocks_4_0_weight_updated_closure(void *_perturbrademacher_blocks_4_0_weight_updated_closure_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_0_weight_updated_closure_args_t *args =
      (_perturbrademacher_blocks_4_0_weight_updated_closure_args_t *)_perturbrademacher_blocks_4_0_weight_updated_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref =
      (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 28672);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_input_16_ref =
      (void *)((char *)DeeployOptNetwork_input_16 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_output_16_ref =
      (void *)((char *)DeeployOptNetwork_output_16 + DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1470464, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_input_16_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _perturbrademacher_blocks_4_0_weight_updated_cluster_fork_args_t DeeployOptNetwork__perturbrademacher_blocks_4_0_weight_updated_cluster_fork_args =
        (_perturbrademacher_blocks_4_0_weight_updated_cluster_fork_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_in_ref,
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_perturbrademacher_blocks_4_0_weight_updated_cluster_fork,
                    &DeeployOptNetwork__perturbrademacher_blocks_4_0_weight_updated_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1339392, DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_output_16_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr;
} _perturbrademacher_blocks_4_0_weight_updated_closure_L3_args_t;

static void _perturbrademacher_blocks_4_0_weight_updated_closure_L3(void *_perturbrademacher_blocks_4_0_weight_updated_closure_L3_args) {
  // CLOSURE ARG CAST
  _perturbrademacher_blocks_4_0_weight_updated_closure_L3_args_t *args =
      (_perturbrademacher_blocks_4_0_weight_updated_closure_L3_args_t *)_perturbrademacher_blocks_4_0_weight_updated_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _perturbrademacher_blocks_4_0_weight_updated_closure_args_t DeeployOptNetwork__perturbrademacher_blocks_4_0_weight_updated_closure_args =
      (_perturbrademacher_blocks_4_0_weight_updated_closure_args_t){
          .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr =
              DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr};

  // _perturbrademacher_blocks_4_0_weight_updated_closure CLOSURE CALL
  _perturbrademacher_blocks_4_0_weight_updated_closure(&DeeployOptNetwork__perturbrademacher_blocks_4_0_weight_updated_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunOptimizerNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr;
    _perturbrademacher_blocks_0_0_bias_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_0_0_bias_updated_closure_L3_args =
        (_perturbrademacher_blocks_0_0_bias_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_bias_updated_tileIdxPtr};

    // _perturbrademacher_blocks_0_0_bias_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_0_0_bias_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_0_0_bias_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr;
    _perturbrademacher_blocks_0_1_weight_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_0_1_weight_updated_closure_L3_args =
        (_perturbrademacher_blocks_0_1_weight_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_weight_updated_tileIdxPtr};

    // _perturbrademacher_blocks_0_1_weight_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_0_1_weight_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_0_1_weight_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr;
    _perturbrademacher_blocks_0_1_bias_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_0_1_bias_updated_closure_L3_args =
        (_perturbrademacher_blocks_0_1_bias_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_1_bias_updated_tileIdxPtr};

    // _perturbrademacher_blocks_0_1_bias_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_0_1_bias_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_0_1_bias_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr;
    _perturbrademacher_fc_bias_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_fc_bias_updated_closure_L3_args =
        (_perturbrademacher_fc_bias_updated_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_bias_updated_tileIdxPtr};

    // _perturbrademacher_fc_bias_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_fc_bias_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_fc_bias_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr;
    _perturbrademacher_blocks_1_0_bias_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_1_0_bias_updated_closure_L3_args =
        (_perturbrademacher_blocks_1_0_bias_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_bias_updated_tileIdxPtr};

    // _perturbrademacher_blocks_1_0_bias_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_1_0_bias_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_1_0_bias_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr;
    _perturbrademacher_blocks_1_1_weight_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_1_1_weight_updated_closure_L3_args =
        (_perturbrademacher_blocks_1_1_weight_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_weight_updated_tileIdxPtr};

    // _perturbrademacher_blocks_1_1_weight_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_1_1_weight_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_1_1_weight_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr;
    _perturbrademacher_blocks_1_1_bias_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_1_1_bias_updated_closure_L3_args =
        (_perturbrademacher_blocks_1_1_bias_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_1_bias_updated_tileIdxPtr};

    // _perturbrademacher_blocks_1_1_bias_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_1_1_bias_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_1_1_bias_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr;
    _perturbrademacher_blocks_2_0_bias_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_2_0_bias_updated_closure_L3_args =
        (_perturbrademacher_blocks_2_0_bias_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_bias_updated_tileIdxPtr};

    // _perturbrademacher_blocks_2_0_bias_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_2_0_bias_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_2_0_bias_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr;
    _perturbrademacher_blocks_2_1_weight_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_2_1_weight_updated_closure_L3_args =
        (_perturbrademacher_blocks_2_1_weight_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_weight_updated_tileIdxPtr};

    // _perturbrademacher_blocks_2_1_weight_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_2_1_weight_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_2_1_weight_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr;
    _perturbrademacher_blocks_2_1_bias_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_2_1_bias_updated_closure_L3_args =
        (_perturbrademacher_blocks_2_1_bias_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_1_bias_updated_tileIdxPtr};

    // _perturbrademacher_blocks_2_1_bias_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_2_1_bias_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_2_1_bias_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr;
    _perturbrademacher_blocks_0_0_weight_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_0_0_weight_updated_closure_L3_args =
        (_perturbrademacher_blocks_0_0_weight_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_0_0_weight_updated_tileIdxPtr};

    // _perturbrademacher_blocks_0_0_weight_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_0_0_weight_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_0_0_weight_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr;
    _perturbrademacher_blocks_3_0_bias_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_3_0_bias_updated_closure_L3_args =
        (_perturbrademacher_blocks_3_0_bias_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_bias_updated_tileIdxPtr};

    // _perturbrademacher_blocks_3_0_bias_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_3_0_bias_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_3_0_bias_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr;
    _perturbrademacher_blocks_3_1_weight_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_3_1_weight_updated_closure_L3_args =
        (_perturbrademacher_blocks_3_1_weight_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_weight_updated_tileIdxPtr};

    // _perturbrademacher_blocks_3_1_weight_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_3_1_weight_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_3_1_weight_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr;
    _perturbrademacher_blocks_3_1_bias_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_3_1_bias_updated_closure_L3_args =
        (_perturbrademacher_blocks_3_1_bias_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_1_bias_updated_tileIdxPtr};

    // _perturbrademacher_blocks_3_1_bias_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_3_1_bias_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_3_1_bias_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr;
    _perturbrademacher_blocks_4_0_bias_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_4_0_bias_updated_closure_L3_args =
        (_perturbrademacher_blocks_4_0_bias_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_bias_updated_tileIdxPtr};

    // _perturbrademacher_blocks_4_0_bias_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_4_0_bias_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_4_0_bias_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr;
    _perturbrademacher_blocks_4_1_weight_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_4_1_weight_updated_closure_L3_args =
        (_perturbrademacher_blocks_4_1_weight_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_weight_updated_tileIdxPtr};

    // _perturbrademacher_blocks_4_1_weight_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_4_1_weight_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_4_1_weight_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr;
    _perturbrademacher_blocks_4_1_bias_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_4_1_bias_updated_closure_L3_args =
        (_perturbrademacher_blocks_4_1_bias_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_1_bias_updated_tileIdxPtr};

    // _perturbrademacher_blocks_4_1_bias_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_4_1_bias_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_4_1_bias_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr;
    _perturbrademacher_fc_weight_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_fc_weight_updated_closure_L3_args =
        (_perturbrademacher_fc_weight_updated_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr =
                                                                     DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_fc_weight_updated_tileIdxPtr};

    // _perturbrademacher_fc_weight_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_fc_weight_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_fc_weight_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr;
    _perturbrademacher_blocks_1_0_weight_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_1_0_weight_updated_closure_L3_args =
        (_perturbrademacher_blocks_1_0_weight_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_1_0_weight_updated_tileIdxPtr};

    // _perturbrademacher_blocks_1_0_weight_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_1_0_weight_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_1_0_weight_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr;
    _perturbrademacher_blocks_2_0_weight_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_2_0_weight_updated_closure_L3_args =
        (_perturbrademacher_blocks_2_0_weight_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_2_0_weight_updated_tileIdxPtr};

    // _perturbrademacher_blocks_2_0_weight_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_2_0_weight_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_2_0_weight_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr;
    _perturbrademacher_blocks_3_0_weight_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_3_0_weight_updated_closure_L3_args =
        (_perturbrademacher_blocks_3_0_weight_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_3_0_weight_updated_tileIdxPtr};

    // _perturbrademacher_blocks_3_0_weight_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_3_0_weight_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_3_0_weight_updated_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr;
    _perturbrademacher_blocks_4_0_weight_updated_closure_L3_args_t DeeployOptNetwork__perturbrademacher_blocks_4_0_weight_updated_closure_L3_args =
        (_perturbrademacher_blocks_4_0_weight_updated_closure_L3_args_t){
            .DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr =
                DeeployOptNetwork_TILING_CODEGEN_L1_perturbrademacher_blocks_4_0_weight_updated_tileIdxPtr};

    // _perturbrademacher_blocks_4_0_weight_updated_closure_L3 CLOSURE CALL
    _perturbrademacher_blocks_4_0_weight_updated_closure_L3(&DeeployOptNetwork__perturbrademacher_blocks_4_0_weight_updated_closure_L3_args);
  }
}

void InitOptimizerNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  DeeployOptNetwork_MEMORYARENA_L1 = (int8_t *)DeeployNetwork_MEMORYARENA_L1; /* shared with TrainingNetwork */
  if (DeeployOptNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployOptNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 57344));
    return;
  }

  DeeployOptNetwork_input_0 = (float32_t *)DeeployNetwork_input_2;    /* shared with TrainingNetwork */
  DeeployOptNetwork_input_1 = (float32_t *)DeeployNetwork_input_3;    /* shared with TrainingNetwork */
  DeeployOptNetwork_input_2 = (float32_t *)DeeployNetwork_input_4;    /* shared with TrainingNetwork */
  DeeployOptNetwork_input_3 = (float32_t *)DeeployNetwork_input_5;    /* shared with TrainingNetwork */
  DeeployOptNetwork_input_4 = (float32_t *)DeeployNetwork_input_6;    /* shared with TrainingNetwork */
  DeeployOptNetwork_input_5 = (float32_t *)DeeployNetwork_input_7;    /* shared with TrainingNetwork */
  DeeployOptNetwork_input_6 = (float32_t *)DeeployNetwork_input_8;    /* shared with TrainingNetwork */
  DeeployOptNetwork_input_7 = (float32_t *)DeeployNetwork_input_9;    /* shared with TrainingNetwork */
  DeeployOptNetwork_input_8 = (float32_t *)DeeployNetwork_input_10;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_9 = (float32_t *)DeeployNetwork_input_11;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_10 = (float32_t *)DeeployNetwork_input_12;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_11 = (float32_t *)DeeployNetwork_input_13;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_12 = (float32_t *)DeeployNetwork_input_14;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_13 = (float32_t *)DeeployNetwork_input_15;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_14 = (float32_t *)DeeployNetwork_input_16;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_15 = (float32_t *)DeeployNetwork_input_17;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_16 = (float32_t *)DeeployNetwork_input_18;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_17 = (float32_t *)DeeployNetwork_input_19;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_18 = (float32_t *)DeeployNetwork_input_20;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_19 = (float32_t *)DeeployNetwork_input_21;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_20 = (float32_t *)DeeployNetwork_input_22;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_21 = (float32_t *)DeeployNetwork_input_23;  /* shared with TrainingNetwork */
  DeeployOptNetwork_output_0 = (float32_t *)DeeployNetwork_input_2;   /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_1 = (float32_t *)DeeployNetwork_input_3;   /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_2 = (float32_t *)DeeployNetwork_input_4;   /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_3 = (float32_t *)DeeployNetwork_input_5;   /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_4 = (float32_t *)DeeployNetwork_input_6;   /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_5 = (float32_t *)DeeployNetwork_input_7;   /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_6 = (float32_t *)DeeployNetwork_input_8;   /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_7 = (float32_t *)DeeployNetwork_input_9;   /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_8 = (float32_t *)DeeployNetwork_input_10;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_9 = (float32_t *)DeeployNetwork_input_11;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_10 = (float32_t *)DeeployNetwork_input_12; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_11 = (float32_t *)DeeployNetwork_input_13; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_12 = (float32_t *)DeeployNetwork_input_14; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_13 = (float32_t *)DeeployNetwork_input_15; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_14 = (float32_t *)DeeployNetwork_input_16; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_15 = (float32_t *)DeeployNetwork_input_17; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_16 = (float32_t *)DeeployNetwork_input_18; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_17 = (float32_t *)DeeployNetwork_input_19; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_18 = (float32_t *)DeeployNetwork_input_20; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_19 = (float32_t *)DeeployNetwork_input_21; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_20 = (float32_t *)DeeployNetwork_input_22; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_21 = (float32_t *)DeeployNetwork_input_23; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_inputs[0] = (void *)DeeployOptNetwork_input_0;
  DeeployOptNetwork_inputs[1] = (void *)DeeployOptNetwork_input_1;
  DeeployOptNetwork_inputs[2] = (void *)DeeployOptNetwork_input_2;
  DeeployOptNetwork_inputs[3] = (void *)DeeployOptNetwork_input_3;
  DeeployOptNetwork_inputs[4] = (void *)DeeployOptNetwork_input_4;
  DeeployOptNetwork_inputs[5] = (void *)DeeployOptNetwork_input_5;
  DeeployOptNetwork_inputs[6] = (void *)DeeployOptNetwork_input_6;
  DeeployOptNetwork_inputs[7] = (void *)DeeployOptNetwork_input_7;
  DeeployOptNetwork_inputs[8] = (void *)DeeployOptNetwork_input_8;
  DeeployOptNetwork_inputs[9] = (void *)DeeployOptNetwork_input_9;
  DeeployOptNetwork_inputs[10] = (void *)DeeployOptNetwork_input_10;
  DeeployOptNetwork_inputs[11] = (void *)DeeployOptNetwork_input_11;
  DeeployOptNetwork_inputs[12] = (void *)DeeployOptNetwork_input_12;
  DeeployOptNetwork_inputs[13] = (void *)DeeployOptNetwork_input_13;
  DeeployOptNetwork_inputs[14] = (void *)DeeployOptNetwork_input_14;
  DeeployOptNetwork_inputs[15] = (void *)DeeployOptNetwork_input_15;
  DeeployOptNetwork_inputs[16] = (void *)DeeployOptNetwork_input_16;
  DeeployOptNetwork_inputs[17] = (void *)DeeployOptNetwork_input_17;
  DeeployOptNetwork_inputs[18] = (void *)DeeployOptNetwork_input_18;
  DeeployOptNetwork_inputs[19] = (void *)DeeployOptNetwork_input_19;
  DeeployOptNetwork_inputs[20] = (void *)DeeployOptNetwork_input_20;
  DeeployOptNetwork_inputs[21] = (void *)DeeployOptNetwork_input_21;
  DeeployOptNetwork_outputs[0] = (void *)DeeployOptNetwork_output_0;
  DeeployOptNetwork_outputs[1] = (void *)DeeployOptNetwork_output_1;
  DeeployOptNetwork_outputs[2] = (void *)DeeployOptNetwork_output_2;
  DeeployOptNetwork_outputs[3] = (void *)DeeployOptNetwork_output_3;
  DeeployOptNetwork_outputs[4] = (void *)DeeployOptNetwork_output_4;
  DeeployOptNetwork_outputs[5] = (void *)DeeployOptNetwork_output_5;
  DeeployOptNetwork_outputs[6] = (void *)DeeployOptNetwork_output_6;
  DeeployOptNetwork_outputs[7] = (void *)DeeployOptNetwork_output_7;
  DeeployOptNetwork_outputs[8] = (void *)DeeployOptNetwork_output_8;
  DeeployOptNetwork_outputs[9] = (void *)DeeployOptNetwork_output_9;
  DeeployOptNetwork_outputs[10] = (void *)DeeployOptNetwork_output_10;
  DeeployOptNetwork_outputs[11] = (void *)DeeployOptNetwork_output_11;
  DeeployOptNetwork_outputs[12] = (void *)DeeployOptNetwork_output_12;
  DeeployOptNetwork_outputs[13] = (void *)DeeployOptNetwork_output_13;
  DeeployOptNetwork_outputs[14] = (void *)DeeployOptNetwork_output_14;
  DeeployOptNetwork_outputs[15] = (void *)DeeployOptNetwork_output_15;
  DeeployOptNetwork_outputs[16] = (void *)DeeployOptNetwork_output_16;
  DeeployOptNetwork_outputs[17] = (void *)DeeployOptNetwork_output_17;
  DeeployOptNetwork_outputs[18] = (void *)DeeployOptNetwork_output_18;
  DeeployOptNetwork_outputs[19] = (void *)DeeployOptNetwork_output_19;
  DeeployOptNetwork_outputs[20] = (void *)DeeployOptNetwork_output_20;
  DeeployOptNetwork_outputs[21] = (void *)DeeployOptNetwork_output_21;
}
