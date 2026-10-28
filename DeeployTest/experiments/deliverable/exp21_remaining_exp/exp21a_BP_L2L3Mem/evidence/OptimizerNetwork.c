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
int8_t *DeeployOptNetwork_MEMORYARENA_L3;
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
float32_t *DeeployOptNetwork_input_22;
float32_t *DeeployOptNetwork_input_23;
float32_t *DeeployOptNetwork_input_24;
float32_t *DeeployOptNetwork_input_25;
float32_t *DeeployOptNetwork_input_26;
float32_t *DeeployOptNetwork_input_27;
float32_t *DeeployOptNetwork_input_28;
float32_t *DeeployOptNetwork_input_29;
float32_t *DeeployOptNetwork_input_30;
float32_t *DeeployOptNetwork_input_31;
float32_t *DeeployOptNetwork_input_32;
float32_t *DeeployOptNetwork_input_33;
float32_t *DeeployOptNetwork_input_34;
float32_t *DeeployOptNetwork_input_35;
float32_t *DeeployOptNetwork_input_36;
float32_t *DeeployOptNetwork_input_37;
float32_t *DeeployOptNetwork_input_38;
float32_t *DeeployOptNetwork_input_39;
float32_t *DeeployOptNetwork_input_40;
float32_t *DeeployOptNetwork_input_41;
float32_t *DeeployOptNetwork_input_42;
float32_t *DeeployOptNetwork_input_43;
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

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_cumByteOffset[1] = {0};

void *DeeployOptNetwork_inputs[44];
void *DeeployOptNetwork_outputs[22];
extern struct pi_device cluster_dev;
typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref;
} _sgd_blocks_0_0_bias_tiling_closure_args_t;

static void _sgd_blocks_0_0_bias_tiling_closure(void *_sgd_blocks_0_0_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_0_bias_tiling_closure_args_t *args = (_sgd_blocks_0_0_bias_tiling_closure_args_t *)_sgd_blocks_0_0_bias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_0_0_bias, Op: SGD)
  int8_t sgd_blocks_0_0_bias_core_id = pi_core_id();
  int8_t sgd_blocks_0_0_bias_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_0_0_bias_chunk = (8 >> sgd_blocks_0_0_bias_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_0_0_bias_chunk_start = MIN(sgd_blocks_0_0_bias_chunk * sgd_blocks_0_0_bias_core_id, 8);
  int32_t sgd_blocks_0_0_bias_chunk_stop = MIN(sgd_blocks_0_0_bias_chunk_start + sgd_blocks_0_0_bias_chunk, 8);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_0_0_bias_chunk_start;
  for (; i + 5 < sgd_blocks_0_0_bias_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_0_0_bias_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref;
} _sgd_blocks_0_0_bias_cluster_fork_args_t;

static void _sgd_blocks_0_0_bias_cluster_fork(void *_sgd_blocks_0_0_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_0_bias_cluster_fork_args_t *args = (_sgd_blocks_0_0_bias_cluster_fork_args_t *)_sgd_blocks_0_0_bias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_0_0_bias_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_0_0_bias_tiling_closure_args = (_sgd_blocks_0_0_bias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref};

  // _sgd_blocks_0_0_bias_tiling_closure CLOSURE CALL
  _sgd_blocks_0_0_bias_tiling_closure(&DeeployOptNetwork__sgd_blocks_0_0_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref;
} _sgd_blocks_0_0_bias_closure_args_t;

static void _sgd_blocks_0_0_bias_closure(void *_sgd_blocks_0_0_bias_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_0_bias_closure_args_t *args = (_sgd_blocks_0_0_bias_closure_args_t *)_sgd_blocks_0_0_bias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 32);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_input_2_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_input_3_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_output_1_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441824,
                                      .size = 32,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_input_2_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441824,
                                      .size = 32,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_input_3_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_0_0_bias_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_0_0_bias_cluster_fork_args = (_sgd_blocks_0_0_bias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_0_0_bias_cluster_fork, &DeeployOptNetwork__sgd_blocks_0_0_bias_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310752,
                                      .size = 32,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_output_1_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_2_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_3_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_output_1_ref;
} _sgd_blocks_0_0_bias_closure_L3_args_t;

static void _sgd_blocks_0_0_bias_closure_L3(void *_sgd_blocks_0_0_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_0_bias_closure_L3_args_t *args = (_sgd_blocks_0_0_bias_closure_L3_args_t *)_sgd_blocks_0_0_bias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_2_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_2_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_3_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_3_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_output_1_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_output_1_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_2_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref, (uint32_t)32, (uint32_t)32, (uint32_t)32, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_3_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref, (uint32_t)32, (uint32_t)32, (uint32_t)32, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_0_0_bias_closure_args_t DeeployOptNetwork__sgd_blocks_0_0_bias_closure_args = (_sgd_blocks_0_0_bias_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref};

    // _sgd_blocks_0_0_bias_closure CLOSURE CALL
    _sgd_blocks_0_0_bias_closure(&DeeployOptNetwork__sgd_blocks_0_0_bias_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_output_1_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref, (uint32_t)32, (uint32_t)32, (uint32_t)32, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref;
} _sgd_blocks_0_1_weight_tiling_closure_args_t;

static void _sgd_blocks_0_1_weight_tiling_closure(void *_sgd_blocks_0_1_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_1_weight_tiling_closure_args_t *args = (_sgd_blocks_0_1_weight_tiling_closure_args_t *)_sgd_blocks_0_1_weight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_0_1_weight, Op: SGD)
  int8_t sgd_blocks_0_1_weight_core_id = pi_core_id();
  int8_t sgd_blocks_0_1_weight_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_0_1_weight_chunk = (8 >> sgd_blocks_0_1_weight_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_0_1_weight_chunk_start = MIN(sgd_blocks_0_1_weight_chunk * sgd_blocks_0_1_weight_core_id, 8);
  int32_t sgd_blocks_0_1_weight_chunk_stop = MIN(sgd_blocks_0_1_weight_chunk_start + sgd_blocks_0_1_weight_chunk, 8);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_0_1_weight_chunk_start;
  for (; i + 5 < sgd_blocks_0_1_weight_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_0_1_weight_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref;
} _sgd_blocks_0_1_weight_cluster_fork_args_t;

static void _sgd_blocks_0_1_weight_cluster_fork(void *_sgd_blocks_0_1_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_1_weight_cluster_fork_args_t *args = (_sgd_blocks_0_1_weight_cluster_fork_args_t *)_sgd_blocks_0_1_weight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_0_1_weight_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_0_1_weight_tiling_closure_args = (_sgd_blocks_0_1_weight_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref =
          DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref};

  // _sgd_blocks_0_1_weight_tiling_closure CLOSURE CALL
  _sgd_blocks_0_1_weight_tiling_closure(&DeeployOptNetwork__sgd_blocks_0_1_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref;
} _sgd_blocks_0_1_weight_closure_args_t;

static void _sgd_blocks_0_1_weight_closure(void *_sgd_blocks_0_1_weight_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_1_weight_closure_args_t *args = (_sgd_blocks_0_1_weight_closure_args_t *)_sgd_blocks_0_1_weight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 32);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 32);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_input_4_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_input_5_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_output_2_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441824,
                                      .size = 32,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_input_4_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441824,
                                      .size = 32,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_input_5_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_0_1_weight_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_0_1_weight_cluster_fork_args = (_sgd_blocks_0_1_weight_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_0_1_weight_cluster_fork, &DeeployOptNetwork__sgd_blocks_0_1_weight_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310752,
                                      .size = 32,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_output_2_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_4_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_5_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_output_2_ref;
} _sgd_blocks_0_1_weight_closure_L3_args_t;

static void _sgd_blocks_0_1_weight_closure_L3(void *_sgd_blocks_0_1_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_1_weight_closure_L3_args_t *args = (_sgd_blocks_0_1_weight_closure_L3_args_t *)_sgd_blocks_0_1_weight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_4_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_4_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_5_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_5_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_output_2_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_output_2_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_4_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref, (uint32_t)32, (uint32_t)32, (uint32_t)32, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_5_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref, (uint32_t)32, (uint32_t)32, (uint32_t)32, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_0_1_weight_closure_args_t DeeployOptNetwork__sgd_blocks_0_1_weight_closure_args = (_sgd_blocks_0_1_weight_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref};

    // _sgd_blocks_0_1_weight_closure CLOSURE CALL
    _sgd_blocks_0_1_weight_closure(&DeeployOptNetwork__sgd_blocks_0_1_weight_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_output_2_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref, (uint32_t)32, (uint32_t)32, (uint32_t)32, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref;
} _sgd_blocks_0_1_bias_tiling_closure_args_t;

static void _sgd_blocks_0_1_bias_tiling_closure(void *_sgd_blocks_0_1_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_1_bias_tiling_closure_args_t *args = (_sgd_blocks_0_1_bias_tiling_closure_args_t *)_sgd_blocks_0_1_bias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_0_1_bias, Op: SGD)
  int8_t sgd_blocks_0_1_bias_core_id = pi_core_id();
  int8_t sgd_blocks_0_1_bias_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_0_1_bias_chunk = (8 >> sgd_blocks_0_1_bias_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_0_1_bias_chunk_start = MIN(sgd_blocks_0_1_bias_chunk * sgd_blocks_0_1_bias_core_id, 8);
  int32_t sgd_blocks_0_1_bias_chunk_stop = MIN(sgd_blocks_0_1_bias_chunk_start + sgd_blocks_0_1_bias_chunk, 8);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_0_1_bias_chunk_start;
  for (; i + 5 < sgd_blocks_0_1_bias_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_0_1_bias_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref;
} _sgd_blocks_0_1_bias_cluster_fork_args_t;

static void _sgd_blocks_0_1_bias_cluster_fork(void *_sgd_blocks_0_1_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_1_bias_cluster_fork_args_t *args = (_sgd_blocks_0_1_bias_cluster_fork_args_t *)_sgd_blocks_0_1_bias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_0_1_bias_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_0_1_bias_tiling_closure_args = (_sgd_blocks_0_1_bias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref};

  // _sgd_blocks_0_1_bias_tiling_closure CLOSURE CALL
  _sgd_blocks_0_1_bias_tiling_closure(&DeeployOptNetwork__sgd_blocks_0_1_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref;
} _sgd_blocks_0_1_bias_closure_args_t;

static void _sgd_blocks_0_1_bias_closure(void *_sgd_blocks_0_1_bias_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_1_bias_closure_args_t *args = (_sgd_blocks_0_1_bias_closure_args_t *)_sgd_blocks_0_1_bias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 32);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_input_6_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_input_7_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_output_3_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441824,
                                      .size = 32,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_input_6_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441824,
                                      .size = 32,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_input_7_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_0_1_bias_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_0_1_bias_cluster_fork_args = (_sgd_blocks_0_1_bias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_0_1_bias_cluster_fork, &DeeployOptNetwork__sgd_blocks_0_1_bias_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310752,
                                      .size = 32,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_output_3_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_6_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_7_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_output_3_ref;
} _sgd_blocks_0_1_bias_closure_L3_args_t;

static void _sgd_blocks_0_1_bias_closure_L3(void *_sgd_blocks_0_1_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_1_bias_closure_L3_args_t *args = (_sgd_blocks_0_1_bias_closure_L3_args_t *)_sgd_blocks_0_1_bias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_6_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_6_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_7_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_7_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_output_3_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_output_3_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_6_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref, (uint32_t)32, (uint32_t)32, (uint32_t)32, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_7_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref, (uint32_t)32, (uint32_t)32, (uint32_t)32, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_0_1_bias_closure_args_t DeeployOptNetwork__sgd_blocks_0_1_bias_closure_args = (_sgd_blocks_0_1_bias_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref};

    // _sgd_blocks_0_1_bias_closure CLOSURE CALL
    _sgd_blocks_0_1_bias_closure(&DeeployOptNetwork__sgd_blocks_0_1_bias_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_output_3_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref, (uint32_t)32, (uint32_t)32, (uint32_t)32, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref;
} _sgd_fc_bias_tiling_closure_args_t;

static void _sgd_fc_bias_tiling_closure(void *_sgd_fc_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_fc_bias_tiling_closure_args_t *args = (_sgd_fc_bias_tiling_closure_args_t *)_sgd_fc_bias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_fc_bias, Op: SGD)
  int8_t sgd_fc_bias_core_id = pi_core_id();
  int8_t sgd_fc_bias_log2Core = log2(NUM_CORES);
  int32_t sgd_fc_bias_chunk = (9 >> sgd_fc_bias_log2Core) + ((9 & (NUM_CORES - 1)) != 0);
  int32_t sgd_fc_bias_chunk_start = MIN(sgd_fc_bias_chunk * sgd_fc_bias_core_id, 9);
  int32_t sgd_fc_bias_chunk_stop = MIN(sgd_fc_bias_chunk_start + sgd_fc_bias_chunk, 9);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_fc_bias_chunk_start;
  for (; i + 5 < sgd_fc_bias_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref[i] = ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_fc_bias_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref[i] = ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref;
} _sgd_fc_bias_cluster_fork_args_t;

static void _sgd_fc_bias_cluster_fork(void *_sgd_fc_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_fc_bias_cluster_fork_args_t *args = (_sgd_fc_bias_cluster_fork_args_t *)_sgd_fc_bias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_fc_bias_tiling_closure_args_t DeeployOptNetwork__sgd_fc_bias_tiling_closure_args = (_sgd_fc_bias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref};

  // _sgd_fc_bias_tiling_closure CLOSURE CALL
  _sgd_fc_bias_tiling_closure(&DeeployOptNetwork__sgd_fc_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref;
} _sgd_fc_bias_closure_args_t;

static void _sgd_fc_bias_closure(void *_sgd_fc_bias_closure_args) {
  // CLOSURE ARG CAST
  _sgd_fc_bias_closure_args_t *args = (_sgd_fc_bias_closure_args_t *)_sgd_fc_bias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 36);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_input_42_ref = (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_input_43_ref = (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_output_21_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441828,
                                      .size = 36,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_input_42_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441828,
                                      .size = 36,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_input_43_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_fc_bias_cluster_fork_args_t DeeployOptNetwork__sgd_fc_bias_cluster_fork_args = (_sgd_fc_bias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_fc_bias_cluster_fork, &DeeployOptNetwork__sgd_fc_bias_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310756,
                                      .size = 36,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_output_21_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_42_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_43_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_output_21_ref;
} _sgd_fc_bias_closure_L3_args_t;

static void _sgd_fc_bias_closure_L3(void *_sgd_fc_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_fc_bias_closure_L3_args_t *args = (_sgd_fc_bias_closure_L3_args_t *)_sgd_fc_bias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_42_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_42_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_43_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_43_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_output_21_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_output_21_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_42_ref, (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref,
                     (uint32_t)36, (uint32_t)36, (uint32_t)36, 1, &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_43_ref, (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref,
                     (uint32_t)36, (uint32_t)36, (uint32_t)36, 1, &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_fc_bias_closure_args_t DeeployOptNetwork__sgd_fc_bias_closure_args = (_sgd_fc_bias_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref};

    // _sgd_fc_bias_closure CLOSURE CALL
    _sgd_fc_bias_closure(&DeeployOptNetwork__sgd_fc_bias_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_output_21_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref, (uint32_t)36, (uint32_t)36, (uint32_t)36, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref;
} _sgd_blocks_1_0_bias_tiling_closure_args_t;

static void _sgd_blocks_1_0_bias_tiling_closure(void *_sgd_blocks_1_0_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_0_bias_tiling_closure_args_t *args = (_sgd_blocks_1_0_bias_tiling_closure_args_t *)_sgd_blocks_1_0_bias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_1_0_bias, Op: SGD)
  int8_t sgd_blocks_1_0_bias_core_id = pi_core_id();
  int8_t sgd_blocks_1_0_bias_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_1_0_bias_chunk = (16 >> sgd_blocks_1_0_bias_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_1_0_bias_chunk_start = MIN(sgd_blocks_1_0_bias_chunk * sgd_blocks_1_0_bias_core_id, 16);
  int32_t sgd_blocks_1_0_bias_chunk_stop = MIN(sgd_blocks_1_0_bias_chunk_start + sgd_blocks_1_0_bias_chunk, 16);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_1_0_bias_chunk_start;
  for (; i + 5 < sgd_blocks_1_0_bias_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_1_0_bias_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref;
} _sgd_blocks_1_0_bias_cluster_fork_args_t;

static void _sgd_blocks_1_0_bias_cluster_fork(void *_sgd_blocks_1_0_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_0_bias_cluster_fork_args_t *args = (_sgd_blocks_1_0_bias_cluster_fork_args_t *)_sgd_blocks_1_0_bias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_1_0_bias_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_1_0_bias_tiling_closure_args = (_sgd_blocks_1_0_bias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref};

  // _sgd_blocks_1_0_bias_tiling_closure CLOSURE CALL
  _sgd_blocks_1_0_bias_tiling_closure(&DeeployOptNetwork__sgd_blocks_1_0_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref;
} _sgd_blocks_1_0_bias_closure_args_t;

static void _sgd_blocks_1_0_bias_closure(void *_sgd_blocks_1_0_bias_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_0_bias_closure_args_t *args = (_sgd_blocks_1_0_bias_closure_args_t *)_sgd_blocks_1_0_bias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_input_10_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_input_11_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_output_5_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_input_10_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_input_11_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_1_0_bias_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_1_0_bias_cluster_fork_args = (_sgd_blocks_1_0_bias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_1_0_bias_cluster_fork, &DeeployOptNetwork__sgd_blocks_1_0_bias_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310784,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_output_5_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_10_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_11_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_output_5_ref;
} _sgd_blocks_1_0_bias_closure_L3_args_t;

static void _sgd_blocks_1_0_bias_closure_L3(void *_sgd_blocks_1_0_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_0_bias_closure_L3_args_t *args = (_sgd_blocks_1_0_bias_closure_L3_args_t *)_sgd_blocks_1_0_bias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_10_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_10_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_11_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_11_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_output_5_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_output_5_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_10_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_11_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_1_0_bias_closure_args_t DeeployOptNetwork__sgd_blocks_1_0_bias_closure_args = (_sgd_blocks_1_0_bias_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref};

    // _sgd_blocks_1_0_bias_closure CLOSURE CALL
    _sgd_blocks_1_0_bias_closure(&DeeployOptNetwork__sgd_blocks_1_0_bias_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_output_5_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref;
} _sgd_blocks_1_1_weight_tiling_closure_args_t;

static void _sgd_blocks_1_1_weight_tiling_closure(void *_sgd_blocks_1_1_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_1_weight_tiling_closure_args_t *args = (_sgd_blocks_1_1_weight_tiling_closure_args_t *)_sgd_blocks_1_1_weight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_1_1_weight, Op: SGD)
  int8_t sgd_blocks_1_1_weight_core_id = pi_core_id();
  int8_t sgd_blocks_1_1_weight_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_1_1_weight_chunk = (16 >> sgd_blocks_1_1_weight_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_1_1_weight_chunk_start = MIN(sgd_blocks_1_1_weight_chunk * sgd_blocks_1_1_weight_core_id, 16);
  int32_t sgd_blocks_1_1_weight_chunk_stop = MIN(sgd_blocks_1_1_weight_chunk_start + sgd_blocks_1_1_weight_chunk, 16);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_1_1_weight_chunk_start;
  for (; i + 5 < sgd_blocks_1_1_weight_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_1_1_weight_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref;
} _sgd_blocks_1_1_weight_cluster_fork_args_t;

static void _sgd_blocks_1_1_weight_cluster_fork(void *_sgd_blocks_1_1_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_1_weight_cluster_fork_args_t *args = (_sgd_blocks_1_1_weight_cluster_fork_args_t *)_sgd_blocks_1_1_weight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_1_1_weight_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_1_1_weight_tiling_closure_args = (_sgd_blocks_1_1_weight_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref =
          DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref};

  // _sgd_blocks_1_1_weight_tiling_closure CLOSURE CALL
  _sgd_blocks_1_1_weight_tiling_closure(&DeeployOptNetwork__sgd_blocks_1_1_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref;
} _sgd_blocks_1_1_weight_closure_args_t;

static void _sgd_blocks_1_1_weight_closure(void *_sgd_blocks_1_1_weight_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_1_weight_closure_args_t *args = (_sgd_blocks_1_1_weight_closure_args_t *)_sgd_blocks_1_1_weight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_input_12_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_input_13_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_output_6_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_input_12_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_input_13_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_1_1_weight_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_1_1_weight_cluster_fork_args = (_sgd_blocks_1_1_weight_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_1_1_weight_cluster_fork, &DeeployOptNetwork__sgd_blocks_1_1_weight_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310784,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_output_6_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_12_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_13_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_output_6_ref;
} _sgd_blocks_1_1_weight_closure_L3_args_t;

static void _sgd_blocks_1_1_weight_closure_L3(void *_sgd_blocks_1_1_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_1_weight_closure_L3_args_t *args = (_sgd_blocks_1_1_weight_closure_L3_args_t *)_sgd_blocks_1_1_weight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_12_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_12_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_13_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_13_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_output_6_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_output_6_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_12_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_13_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_1_1_weight_closure_args_t DeeployOptNetwork__sgd_blocks_1_1_weight_closure_args = (_sgd_blocks_1_1_weight_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref};

    // _sgd_blocks_1_1_weight_closure CLOSURE CALL
    _sgd_blocks_1_1_weight_closure(&DeeployOptNetwork__sgd_blocks_1_1_weight_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_output_6_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref;
} _sgd_blocks_1_1_bias_tiling_closure_args_t;

static void _sgd_blocks_1_1_bias_tiling_closure(void *_sgd_blocks_1_1_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_1_bias_tiling_closure_args_t *args = (_sgd_blocks_1_1_bias_tiling_closure_args_t *)_sgd_blocks_1_1_bias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_1_1_bias, Op: SGD)
  int8_t sgd_blocks_1_1_bias_core_id = pi_core_id();
  int8_t sgd_blocks_1_1_bias_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_1_1_bias_chunk = (16 >> sgd_blocks_1_1_bias_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_1_1_bias_chunk_start = MIN(sgd_blocks_1_1_bias_chunk * sgd_blocks_1_1_bias_core_id, 16);
  int32_t sgd_blocks_1_1_bias_chunk_stop = MIN(sgd_blocks_1_1_bias_chunk_start + sgd_blocks_1_1_bias_chunk, 16);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_1_1_bias_chunk_start;
  for (; i + 5 < sgd_blocks_1_1_bias_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_1_1_bias_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref;
} _sgd_blocks_1_1_bias_cluster_fork_args_t;

static void _sgd_blocks_1_1_bias_cluster_fork(void *_sgd_blocks_1_1_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_1_bias_cluster_fork_args_t *args = (_sgd_blocks_1_1_bias_cluster_fork_args_t *)_sgd_blocks_1_1_bias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_1_1_bias_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_1_1_bias_tiling_closure_args = (_sgd_blocks_1_1_bias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref};

  // _sgd_blocks_1_1_bias_tiling_closure CLOSURE CALL
  _sgd_blocks_1_1_bias_tiling_closure(&DeeployOptNetwork__sgd_blocks_1_1_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref;
} _sgd_blocks_1_1_bias_closure_args_t;

static void _sgd_blocks_1_1_bias_closure(void *_sgd_blocks_1_1_bias_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_1_bias_closure_args_t *args = (_sgd_blocks_1_1_bias_closure_args_t *)_sgd_blocks_1_1_bias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_input_14_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_input_15_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_output_7_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_input_14_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_input_15_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_1_1_bias_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_1_1_bias_cluster_fork_args = (_sgd_blocks_1_1_bias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_1_1_bias_cluster_fork, &DeeployOptNetwork__sgd_blocks_1_1_bias_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310784,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_output_7_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_14_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_15_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_output_7_ref;
} _sgd_blocks_1_1_bias_closure_L3_args_t;

static void _sgd_blocks_1_1_bias_closure_L3(void *_sgd_blocks_1_1_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_1_bias_closure_L3_args_t *args = (_sgd_blocks_1_1_bias_closure_L3_args_t *)_sgd_blocks_1_1_bias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_14_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_14_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_15_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_15_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_output_7_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_output_7_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_14_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_15_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_1_1_bias_closure_args_t DeeployOptNetwork__sgd_blocks_1_1_bias_closure_args = (_sgd_blocks_1_1_bias_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref};

    // _sgd_blocks_1_1_bias_closure CLOSURE CALL
    _sgd_blocks_1_1_bias_closure(&DeeployOptNetwork__sgd_blocks_1_1_bias_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_output_7_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref;
} _sgd_blocks_2_0_bias_tiling_closure_args_t;

static void _sgd_blocks_2_0_bias_tiling_closure(void *_sgd_blocks_2_0_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_0_bias_tiling_closure_args_t *args = (_sgd_blocks_2_0_bias_tiling_closure_args_t *)_sgd_blocks_2_0_bias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_2_0_bias, Op: SGD)
  int8_t sgd_blocks_2_0_bias_core_id = pi_core_id();
  int8_t sgd_blocks_2_0_bias_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_2_0_bias_chunk = (16 >> sgd_blocks_2_0_bias_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_2_0_bias_chunk_start = MIN(sgd_blocks_2_0_bias_chunk * sgd_blocks_2_0_bias_core_id, 16);
  int32_t sgd_blocks_2_0_bias_chunk_stop = MIN(sgd_blocks_2_0_bias_chunk_start + sgd_blocks_2_0_bias_chunk, 16);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_2_0_bias_chunk_start;
  for (; i + 5 < sgd_blocks_2_0_bias_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_2_0_bias_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref;
} _sgd_blocks_2_0_bias_cluster_fork_args_t;

static void _sgd_blocks_2_0_bias_cluster_fork(void *_sgd_blocks_2_0_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_0_bias_cluster_fork_args_t *args = (_sgd_blocks_2_0_bias_cluster_fork_args_t *)_sgd_blocks_2_0_bias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_2_0_bias_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_2_0_bias_tiling_closure_args = (_sgd_blocks_2_0_bias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref};

  // _sgd_blocks_2_0_bias_tiling_closure CLOSURE CALL
  _sgd_blocks_2_0_bias_tiling_closure(&DeeployOptNetwork__sgd_blocks_2_0_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref;
} _sgd_blocks_2_0_bias_closure_args_t;

static void _sgd_blocks_2_0_bias_closure(void *_sgd_blocks_2_0_bias_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_0_bias_closure_args_t *args = (_sgd_blocks_2_0_bias_closure_args_t *)_sgd_blocks_2_0_bias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_input_18_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_input_19_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_output_9_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_input_18_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_input_19_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_2_0_bias_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_2_0_bias_cluster_fork_args = (_sgd_blocks_2_0_bias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_2_0_bias_cluster_fork, &DeeployOptNetwork__sgd_blocks_2_0_bias_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310784,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_output_9_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_18_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_19_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_output_9_ref;
} _sgd_blocks_2_0_bias_closure_L3_args_t;

static void _sgd_blocks_2_0_bias_closure_L3(void *_sgd_blocks_2_0_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_0_bias_closure_L3_args_t *args = (_sgd_blocks_2_0_bias_closure_L3_args_t *)_sgd_blocks_2_0_bias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_18_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_18_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_19_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_19_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_output_9_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_output_9_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_18_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_19_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_2_0_bias_closure_args_t DeeployOptNetwork__sgd_blocks_2_0_bias_closure_args = (_sgd_blocks_2_0_bias_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref};

    // _sgd_blocks_2_0_bias_closure CLOSURE CALL
    _sgd_blocks_2_0_bias_closure(&DeeployOptNetwork__sgd_blocks_2_0_bias_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_output_9_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref;
} _sgd_blocks_2_1_weight_tiling_closure_args_t;

static void _sgd_blocks_2_1_weight_tiling_closure(void *_sgd_blocks_2_1_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_1_weight_tiling_closure_args_t *args = (_sgd_blocks_2_1_weight_tiling_closure_args_t *)_sgd_blocks_2_1_weight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_2_1_weight, Op: SGD)
  int8_t sgd_blocks_2_1_weight_core_id = pi_core_id();
  int8_t sgd_blocks_2_1_weight_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_2_1_weight_chunk = (16 >> sgd_blocks_2_1_weight_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_2_1_weight_chunk_start = MIN(sgd_blocks_2_1_weight_chunk * sgd_blocks_2_1_weight_core_id, 16);
  int32_t sgd_blocks_2_1_weight_chunk_stop = MIN(sgd_blocks_2_1_weight_chunk_start + sgd_blocks_2_1_weight_chunk, 16);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_2_1_weight_chunk_start;
  for (; i + 5 < sgd_blocks_2_1_weight_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_2_1_weight_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref;
} _sgd_blocks_2_1_weight_cluster_fork_args_t;

static void _sgd_blocks_2_1_weight_cluster_fork(void *_sgd_blocks_2_1_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_1_weight_cluster_fork_args_t *args = (_sgd_blocks_2_1_weight_cluster_fork_args_t *)_sgd_blocks_2_1_weight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_2_1_weight_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_2_1_weight_tiling_closure_args = (_sgd_blocks_2_1_weight_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref =
          DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref};

  // _sgd_blocks_2_1_weight_tiling_closure CLOSURE CALL
  _sgd_blocks_2_1_weight_tiling_closure(&DeeployOptNetwork__sgd_blocks_2_1_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref;
} _sgd_blocks_2_1_weight_closure_args_t;

static void _sgd_blocks_2_1_weight_closure(void *_sgd_blocks_2_1_weight_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_1_weight_closure_args_t *args = (_sgd_blocks_2_1_weight_closure_args_t *)_sgd_blocks_2_1_weight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_input_20_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_input_21_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_output_10_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_input_20_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_input_21_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_2_1_weight_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_2_1_weight_cluster_fork_args = (_sgd_blocks_2_1_weight_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_2_1_weight_cluster_fork, &DeeployOptNetwork__sgd_blocks_2_1_weight_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310784,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_output_10_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_20_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_21_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_output_10_ref;
} _sgd_blocks_2_1_weight_closure_L3_args_t;

static void _sgd_blocks_2_1_weight_closure_L3(void *_sgd_blocks_2_1_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_1_weight_closure_L3_args_t *args = (_sgd_blocks_2_1_weight_closure_L3_args_t *)_sgd_blocks_2_1_weight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_20_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_20_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_21_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_21_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_output_10_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_output_10_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_20_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_21_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_2_1_weight_closure_args_t DeeployOptNetwork__sgd_blocks_2_1_weight_closure_args = (_sgd_blocks_2_1_weight_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref};

    // _sgd_blocks_2_1_weight_closure CLOSURE CALL
    _sgd_blocks_2_1_weight_closure(&DeeployOptNetwork__sgd_blocks_2_1_weight_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_output_10_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref;
} _sgd_blocks_2_1_bias_tiling_closure_args_t;

static void _sgd_blocks_2_1_bias_tiling_closure(void *_sgd_blocks_2_1_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_1_bias_tiling_closure_args_t *args = (_sgd_blocks_2_1_bias_tiling_closure_args_t *)_sgd_blocks_2_1_bias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_2_1_bias, Op: SGD)
  int8_t sgd_blocks_2_1_bias_core_id = pi_core_id();
  int8_t sgd_blocks_2_1_bias_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_2_1_bias_chunk = (16 >> sgd_blocks_2_1_bias_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_2_1_bias_chunk_start = MIN(sgd_blocks_2_1_bias_chunk * sgd_blocks_2_1_bias_core_id, 16);
  int32_t sgd_blocks_2_1_bias_chunk_stop = MIN(sgd_blocks_2_1_bias_chunk_start + sgd_blocks_2_1_bias_chunk, 16);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_2_1_bias_chunk_start;
  for (; i + 5 < sgd_blocks_2_1_bias_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_2_1_bias_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref;
} _sgd_blocks_2_1_bias_cluster_fork_args_t;

static void _sgd_blocks_2_1_bias_cluster_fork(void *_sgd_blocks_2_1_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_1_bias_cluster_fork_args_t *args = (_sgd_blocks_2_1_bias_cluster_fork_args_t *)_sgd_blocks_2_1_bias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_2_1_bias_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_2_1_bias_tiling_closure_args = (_sgd_blocks_2_1_bias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref};

  // _sgd_blocks_2_1_bias_tiling_closure CLOSURE CALL
  _sgd_blocks_2_1_bias_tiling_closure(&DeeployOptNetwork__sgd_blocks_2_1_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref;
} _sgd_blocks_2_1_bias_closure_args_t;

static void _sgd_blocks_2_1_bias_closure(void *_sgd_blocks_2_1_bias_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_1_bias_closure_args_t *args = (_sgd_blocks_2_1_bias_closure_args_t *)_sgd_blocks_2_1_bias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_input_22_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_input_23_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_output_11_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_input_22_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_input_23_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_2_1_bias_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_2_1_bias_cluster_fork_args = (_sgd_blocks_2_1_bias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_2_1_bias_cluster_fork, &DeeployOptNetwork__sgd_blocks_2_1_bias_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310784,
                                      .size = 64,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_output_11_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_22_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_23_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_output_11_ref;
} _sgd_blocks_2_1_bias_closure_L3_args_t;

static void _sgd_blocks_2_1_bias_closure_L3(void *_sgd_blocks_2_1_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_1_bias_closure_L3_args_t *args = (_sgd_blocks_2_1_bias_closure_L3_args_t *)_sgd_blocks_2_1_bias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_22_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_22_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_23_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_23_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_output_11_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_output_11_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_22_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_23_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_2_1_bias_closure_args_t DeeployOptNetwork__sgd_blocks_2_1_bias_closure_args = (_sgd_blocks_2_1_bias_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref};

    // _sgd_blocks_2_1_bias_closure CLOSURE CALL
    _sgd_blocks_2_1_bias_closure(&DeeployOptNetwork__sgd_blocks_2_1_bias_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_output_11_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref, (uint32_t)64, (uint32_t)64, (uint32_t)64, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref;
} _sgd_blocks_0_0_weight_tiling_closure_args_t;

static void _sgd_blocks_0_0_weight_tiling_closure(void *_sgd_blocks_0_0_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_0_weight_tiling_closure_args_t *args = (_sgd_blocks_0_0_weight_tiling_closure_args_t *)_sgd_blocks_0_0_weight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_0_0_weight, Op: SGD)
  int8_t sgd_blocks_0_0_weight_core_id = pi_core_id();
  int8_t sgd_blocks_0_0_weight_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_0_0_weight_chunk = (32 >> sgd_blocks_0_0_weight_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_0_0_weight_chunk_start = MIN(sgd_blocks_0_0_weight_chunk * sgd_blocks_0_0_weight_core_id, 32);
  int32_t sgd_blocks_0_0_weight_chunk_stop = MIN(sgd_blocks_0_0_weight_chunk_start + sgd_blocks_0_0_weight_chunk, 32);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_0_0_weight_chunk_start;
  for (; i + 5 < sgd_blocks_0_0_weight_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_0_0_weight_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref;
} _sgd_blocks_0_0_weight_cluster_fork_args_t;

static void _sgd_blocks_0_0_weight_cluster_fork(void *_sgd_blocks_0_0_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_0_weight_cluster_fork_args_t *args = (_sgd_blocks_0_0_weight_cluster_fork_args_t *)_sgd_blocks_0_0_weight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_0_0_weight_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_0_0_weight_tiling_closure_args = (_sgd_blocks_0_0_weight_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref =
          DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref};

  // _sgd_blocks_0_0_weight_tiling_closure CLOSURE CALL
  _sgd_blocks_0_0_weight_tiling_closure(&DeeployOptNetwork__sgd_blocks_0_0_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref;
} _sgd_blocks_0_0_weight_closure_args_t;

static void _sgd_blocks_0_0_weight_closure(void *_sgd_blocks_0_0_weight_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_0_weight_closure_args_t *args = (_sgd_blocks_0_0_weight_closure_args_t *)_sgd_blocks_0_0_weight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_input_0_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_input_1_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_output_0_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_input_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_input_1_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_0_0_weight_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_0_0_weight_cluster_fork_args = (_sgd_blocks_0_0_weight_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_0_0_weight_cluster_fork, &DeeployOptNetwork__sgd_blocks_0_0_weight_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310848,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_0_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_1_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_output_0_ref;
} _sgd_blocks_0_0_weight_closure_L3_args_t;

static void _sgd_blocks_0_0_weight_closure_L3(void *_sgd_blocks_0_0_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_0_0_weight_closure_L3_args_t *args = (_sgd_blocks_0_0_weight_closure_L3_args_t *)_sgd_blocks_0_0_weight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_0_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_0_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_1_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_1_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_output_0_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_output_0_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_0_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_1_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_0_0_weight_closure_args_t DeeployOptNetwork__sgd_blocks_0_0_weight_closure_args = (_sgd_blocks_0_0_weight_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref};

    // _sgd_blocks_0_0_weight_closure CLOSURE CALL
    _sgd_blocks_0_0_weight_closure(&DeeployOptNetwork__sgd_blocks_0_0_weight_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_output_0_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref;
} _sgd_blocks_3_0_bias_tiling_closure_args_t;

static void _sgd_blocks_3_0_bias_tiling_closure(void *_sgd_blocks_3_0_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_0_bias_tiling_closure_args_t *args = (_sgd_blocks_3_0_bias_tiling_closure_args_t *)_sgd_blocks_3_0_bias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_3_0_bias, Op: SGD)
  int8_t sgd_blocks_3_0_bias_core_id = pi_core_id();
  int8_t sgd_blocks_3_0_bias_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_3_0_bias_chunk = (32 >> sgd_blocks_3_0_bias_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_3_0_bias_chunk_start = MIN(sgd_blocks_3_0_bias_chunk * sgd_blocks_3_0_bias_core_id, 32);
  int32_t sgd_blocks_3_0_bias_chunk_stop = MIN(sgd_blocks_3_0_bias_chunk_start + sgd_blocks_3_0_bias_chunk, 32);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_3_0_bias_chunk_start;
  for (; i + 5 < sgd_blocks_3_0_bias_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_3_0_bias_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref;
} _sgd_blocks_3_0_bias_cluster_fork_args_t;

static void _sgd_blocks_3_0_bias_cluster_fork(void *_sgd_blocks_3_0_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_0_bias_cluster_fork_args_t *args = (_sgd_blocks_3_0_bias_cluster_fork_args_t *)_sgd_blocks_3_0_bias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_3_0_bias_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_3_0_bias_tiling_closure_args = (_sgd_blocks_3_0_bias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref};

  // _sgd_blocks_3_0_bias_tiling_closure CLOSURE CALL
  _sgd_blocks_3_0_bias_tiling_closure(&DeeployOptNetwork__sgd_blocks_3_0_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref;
} _sgd_blocks_3_0_bias_closure_args_t;

static void _sgd_blocks_3_0_bias_closure(void *_sgd_blocks_3_0_bias_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_0_bias_closure_args_t *args = (_sgd_blocks_3_0_bias_closure_args_t *)_sgd_blocks_3_0_bias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_input_26_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_input_27_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_output_13_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_input_26_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_input_27_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_3_0_bias_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_3_0_bias_cluster_fork_args = (_sgd_blocks_3_0_bias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_3_0_bias_cluster_fork, &DeeployOptNetwork__sgd_blocks_3_0_bias_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310848,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_output_13_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_26_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_27_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_output_13_ref;
} _sgd_blocks_3_0_bias_closure_L3_args_t;

static void _sgd_blocks_3_0_bias_closure_L3(void *_sgd_blocks_3_0_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_0_bias_closure_L3_args_t *args = (_sgd_blocks_3_0_bias_closure_L3_args_t *)_sgd_blocks_3_0_bias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_26_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_26_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_27_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_27_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_output_13_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_output_13_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_26_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_27_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_3_0_bias_closure_args_t DeeployOptNetwork__sgd_blocks_3_0_bias_closure_args = (_sgd_blocks_3_0_bias_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref};

    // _sgd_blocks_3_0_bias_closure CLOSURE CALL
    _sgd_blocks_3_0_bias_closure(&DeeployOptNetwork__sgd_blocks_3_0_bias_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_output_13_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref;
} _sgd_blocks_3_1_weight_tiling_closure_args_t;

static void _sgd_blocks_3_1_weight_tiling_closure(void *_sgd_blocks_3_1_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_1_weight_tiling_closure_args_t *args = (_sgd_blocks_3_1_weight_tiling_closure_args_t *)_sgd_blocks_3_1_weight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_3_1_weight, Op: SGD)
  int8_t sgd_blocks_3_1_weight_core_id = pi_core_id();
  int8_t sgd_blocks_3_1_weight_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_3_1_weight_chunk = (32 >> sgd_blocks_3_1_weight_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_3_1_weight_chunk_start = MIN(sgd_blocks_3_1_weight_chunk * sgd_blocks_3_1_weight_core_id, 32);
  int32_t sgd_blocks_3_1_weight_chunk_stop = MIN(sgd_blocks_3_1_weight_chunk_start + sgd_blocks_3_1_weight_chunk, 32);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_3_1_weight_chunk_start;
  for (; i + 5 < sgd_blocks_3_1_weight_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_3_1_weight_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref;
} _sgd_blocks_3_1_weight_cluster_fork_args_t;

static void _sgd_blocks_3_1_weight_cluster_fork(void *_sgd_blocks_3_1_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_1_weight_cluster_fork_args_t *args = (_sgd_blocks_3_1_weight_cluster_fork_args_t *)_sgd_blocks_3_1_weight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_3_1_weight_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_3_1_weight_tiling_closure_args = (_sgd_blocks_3_1_weight_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref =
          DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref};

  // _sgd_blocks_3_1_weight_tiling_closure CLOSURE CALL
  _sgd_blocks_3_1_weight_tiling_closure(&DeeployOptNetwork__sgd_blocks_3_1_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref;
} _sgd_blocks_3_1_weight_closure_args_t;

static void _sgd_blocks_3_1_weight_closure(void *_sgd_blocks_3_1_weight_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_1_weight_closure_args_t *args = (_sgd_blocks_3_1_weight_closure_args_t *)_sgd_blocks_3_1_weight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_input_28_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_input_29_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_output_14_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_input_28_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_input_29_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_3_1_weight_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_3_1_weight_cluster_fork_args = (_sgd_blocks_3_1_weight_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_3_1_weight_cluster_fork, &DeeployOptNetwork__sgd_blocks_3_1_weight_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310848,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_output_14_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_28_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_29_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_output_14_ref;
} _sgd_blocks_3_1_weight_closure_L3_args_t;

static void _sgd_blocks_3_1_weight_closure_L3(void *_sgd_blocks_3_1_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_1_weight_closure_L3_args_t *args = (_sgd_blocks_3_1_weight_closure_L3_args_t *)_sgd_blocks_3_1_weight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_28_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_28_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_29_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_29_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_output_14_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_output_14_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_28_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_29_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_3_1_weight_closure_args_t DeeployOptNetwork__sgd_blocks_3_1_weight_closure_args = (_sgd_blocks_3_1_weight_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref};

    // _sgd_blocks_3_1_weight_closure CLOSURE CALL
    _sgd_blocks_3_1_weight_closure(&DeeployOptNetwork__sgd_blocks_3_1_weight_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_output_14_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref;
} _sgd_blocks_3_1_bias_tiling_closure_args_t;

static void _sgd_blocks_3_1_bias_tiling_closure(void *_sgd_blocks_3_1_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_1_bias_tiling_closure_args_t *args = (_sgd_blocks_3_1_bias_tiling_closure_args_t *)_sgd_blocks_3_1_bias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_3_1_bias, Op: SGD)
  int8_t sgd_blocks_3_1_bias_core_id = pi_core_id();
  int8_t sgd_blocks_3_1_bias_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_3_1_bias_chunk = (32 >> sgd_blocks_3_1_bias_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_3_1_bias_chunk_start = MIN(sgd_blocks_3_1_bias_chunk * sgd_blocks_3_1_bias_core_id, 32);
  int32_t sgd_blocks_3_1_bias_chunk_stop = MIN(sgd_blocks_3_1_bias_chunk_start + sgd_blocks_3_1_bias_chunk, 32);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_3_1_bias_chunk_start;
  for (; i + 5 < sgd_blocks_3_1_bias_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_3_1_bias_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref;
} _sgd_blocks_3_1_bias_cluster_fork_args_t;

static void _sgd_blocks_3_1_bias_cluster_fork(void *_sgd_blocks_3_1_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_1_bias_cluster_fork_args_t *args = (_sgd_blocks_3_1_bias_cluster_fork_args_t *)_sgd_blocks_3_1_bias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_3_1_bias_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_3_1_bias_tiling_closure_args = (_sgd_blocks_3_1_bias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref};

  // _sgd_blocks_3_1_bias_tiling_closure CLOSURE CALL
  _sgd_blocks_3_1_bias_tiling_closure(&DeeployOptNetwork__sgd_blocks_3_1_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref;
} _sgd_blocks_3_1_bias_closure_args_t;

static void _sgd_blocks_3_1_bias_closure(void *_sgd_blocks_3_1_bias_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_1_bias_closure_args_t *args = (_sgd_blocks_3_1_bias_closure_args_t *)_sgd_blocks_3_1_bias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_input_30_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_input_31_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_output_15_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_input_30_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_input_31_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_3_1_bias_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_3_1_bias_cluster_fork_args = (_sgd_blocks_3_1_bias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_3_1_bias_cluster_fork, &DeeployOptNetwork__sgd_blocks_3_1_bias_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310848,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_output_15_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_30_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_31_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_output_15_ref;
} _sgd_blocks_3_1_bias_closure_L3_args_t;

static void _sgd_blocks_3_1_bias_closure_L3(void *_sgd_blocks_3_1_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_1_bias_closure_L3_args_t *args = (_sgd_blocks_3_1_bias_closure_L3_args_t *)_sgd_blocks_3_1_bias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_30_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_30_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_31_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_31_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_output_15_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_output_15_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_30_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_31_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_3_1_bias_closure_args_t DeeployOptNetwork__sgd_blocks_3_1_bias_closure_args = (_sgd_blocks_3_1_bias_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref};

    // _sgd_blocks_3_1_bias_closure CLOSURE CALL
    _sgd_blocks_3_1_bias_closure(&DeeployOptNetwork__sgd_blocks_3_1_bias_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_output_15_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref;
} _sgd_blocks_4_0_bias_tiling_closure_args_t;

static void _sgd_blocks_4_0_bias_tiling_closure(void *_sgd_blocks_4_0_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_0_bias_tiling_closure_args_t *args = (_sgd_blocks_4_0_bias_tiling_closure_args_t *)_sgd_blocks_4_0_bias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_4_0_bias, Op: SGD)
  int8_t sgd_blocks_4_0_bias_core_id = pi_core_id();
  int8_t sgd_blocks_4_0_bias_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_4_0_bias_chunk = (32 >> sgd_blocks_4_0_bias_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_4_0_bias_chunk_start = MIN(sgd_blocks_4_0_bias_chunk * sgd_blocks_4_0_bias_core_id, 32);
  int32_t sgd_blocks_4_0_bias_chunk_stop = MIN(sgd_blocks_4_0_bias_chunk_start + sgd_blocks_4_0_bias_chunk, 32);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_4_0_bias_chunk_start;
  for (; i + 5 < sgd_blocks_4_0_bias_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_4_0_bias_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref;
} _sgd_blocks_4_0_bias_cluster_fork_args_t;

static void _sgd_blocks_4_0_bias_cluster_fork(void *_sgd_blocks_4_0_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_0_bias_cluster_fork_args_t *args = (_sgd_blocks_4_0_bias_cluster_fork_args_t *)_sgd_blocks_4_0_bias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_4_0_bias_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_4_0_bias_tiling_closure_args = (_sgd_blocks_4_0_bias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref};

  // _sgd_blocks_4_0_bias_tiling_closure CLOSURE CALL
  _sgd_blocks_4_0_bias_tiling_closure(&DeeployOptNetwork__sgd_blocks_4_0_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref;
} _sgd_blocks_4_0_bias_closure_args_t;

static void _sgd_blocks_4_0_bias_closure(void *_sgd_blocks_4_0_bias_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_0_bias_closure_args_t *args = (_sgd_blocks_4_0_bias_closure_args_t *)_sgd_blocks_4_0_bias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_input_34_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_input_35_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_output_17_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_input_34_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_input_35_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_4_0_bias_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_4_0_bias_cluster_fork_args = (_sgd_blocks_4_0_bias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_4_0_bias_cluster_fork, &DeeployOptNetwork__sgd_blocks_4_0_bias_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310848,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_output_17_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_34_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_35_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_output_17_ref;
} _sgd_blocks_4_0_bias_closure_L3_args_t;

static void _sgd_blocks_4_0_bias_closure_L3(void *_sgd_blocks_4_0_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_0_bias_closure_L3_args_t *args = (_sgd_blocks_4_0_bias_closure_L3_args_t *)_sgd_blocks_4_0_bias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_34_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_34_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_35_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_35_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_output_17_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_output_17_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_34_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_35_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_4_0_bias_closure_args_t DeeployOptNetwork__sgd_blocks_4_0_bias_closure_args = (_sgd_blocks_4_0_bias_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref};

    // _sgd_blocks_4_0_bias_closure CLOSURE CALL
    _sgd_blocks_4_0_bias_closure(&DeeployOptNetwork__sgd_blocks_4_0_bias_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_output_17_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref;
} _sgd_blocks_4_1_weight_tiling_closure_args_t;

static void _sgd_blocks_4_1_weight_tiling_closure(void *_sgd_blocks_4_1_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_1_weight_tiling_closure_args_t *args = (_sgd_blocks_4_1_weight_tiling_closure_args_t *)_sgd_blocks_4_1_weight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_4_1_weight, Op: SGD)
  int8_t sgd_blocks_4_1_weight_core_id = pi_core_id();
  int8_t sgd_blocks_4_1_weight_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_4_1_weight_chunk = (32 >> sgd_blocks_4_1_weight_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_4_1_weight_chunk_start = MIN(sgd_blocks_4_1_weight_chunk * sgd_blocks_4_1_weight_core_id, 32);
  int32_t sgd_blocks_4_1_weight_chunk_stop = MIN(sgd_blocks_4_1_weight_chunk_start + sgd_blocks_4_1_weight_chunk, 32);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_4_1_weight_chunk_start;
  for (; i + 5 < sgd_blocks_4_1_weight_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_4_1_weight_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref;
} _sgd_blocks_4_1_weight_cluster_fork_args_t;

static void _sgd_blocks_4_1_weight_cluster_fork(void *_sgd_blocks_4_1_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_1_weight_cluster_fork_args_t *args = (_sgd_blocks_4_1_weight_cluster_fork_args_t *)_sgd_blocks_4_1_weight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_4_1_weight_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_4_1_weight_tiling_closure_args = (_sgd_blocks_4_1_weight_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref =
          DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref};

  // _sgd_blocks_4_1_weight_tiling_closure CLOSURE CALL
  _sgd_blocks_4_1_weight_tiling_closure(&DeeployOptNetwork__sgd_blocks_4_1_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref;
} _sgd_blocks_4_1_weight_closure_args_t;

static void _sgd_blocks_4_1_weight_closure(void *_sgd_blocks_4_1_weight_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_1_weight_closure_args_t *args = (_sgd_blocks_4_1_weight_closure_args_t *)_sgd_blocks_4_1_weight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_input_36_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_input_37_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_output_18_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_input_36_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_input_37_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_4_1_weight_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_4_1_weight_cluster_fork_args = (_sgd_blocks_4_1_weight_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_4_1_weight_cluster_fork, &DeeployOptNetwork__sgd_blocks_4_1_weight_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310848,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_output_18_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_36_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_37_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_output_18_ref;
} _sgd_blocks_4_1_weight_closure_L3_args_t;

static void _sgd_blocks_4_1_weight_closure_L3(void *_sgd_blocks_4_1_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_1_weight_closure_L3_args_t *args = (_sgd_blocks_4_1_weight_closure_L3_args_t *)_sgd_blocks_4_1_weight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_36_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_36_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_37_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_37_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_output_18_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_output_18_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_36_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_37_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_4_1_weight_closure_args_t DeeployOptNetwork__sgd_blocks_4_1_weight_closure_args = (_sgd_blocks_4_1_weight_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref};

    // _sgd_blocks_4_1_weight_closure CLOSURE CALL
    _sgd_blocks_4_1_weight_closure(&DeeployOptNetwork__sgd_blocks_4_1_weight_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_output_18_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref;
} _sgd_blocks_4_1_bias_tiling_closure_args_t;

static void _sgd_blocks_4_1_bias_tiling_closure(void *_sgd_blocks_4_1_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_1_bias_tiling_closure_args_t *args = (_sgd_blocks_4_1_bias_tiling_closure_args_t *)_sgd_blocks_4_1_bias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_4_1_bias, Op: SGD)
  int8_t sgd_blocks_4_1_bias_core_id = pi_core_id();
  int8_t sgd_blocks_4_1_bias_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_4_1_bias_chunk = (32 >> sgd_blocks_4_1_bias_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_4_1_bias_chunk_start = MIN(sgd_blocks_4_1_bias_chunk * sgd_blocks_4_1_bias_core_id, 32);
  int32_t sgd_blocks_4_1_bias_chunk_stop = MIN(sgd_blocks_4_1_bias_chunk_start + sgd_blocks_4_1_bias_chunk, 32);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_4_1_bias_chunk_start;
  for (; i + 5 < sgd_blocks_4_1_bias_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_4_1_bias_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref;
} _sgd_blocks_4_1_bias_cluster_fork_args_t;

static void _sgd_blocks_4_1_bias_cluster_fork(void *_sgd_blocks_4_1_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_1_bias_cluster_fork_args_t *args = (_sgd_blocks_4_1_bias_cluster_fork_args_t *)_sgd_blocks_4_1_bias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_4_1_bias_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_4_1_bias_tiling_closure_args = (_sgd_blocks_4_1_bias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref};

  // _sgd_blocks_4_1_bias_tiling_closure CLOSURE CALL
  _sgd_blocks_4_1_bias_tiling_closure(&DeeployOptNetwork__sgd_blocks_4_1_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref;
} _sgd_blocks_4_1_bias_closure_args_t;

static void _sgd_blocks_4_1_bias_closure(void *_sgd_blocks_4_1_bias_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_1_bias_closure_args_t *args = (_sgd_blocks_4_1_bias_closure_args_t *)_sgd_blocks_4_1_bias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_input_38_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_input_39_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_output_19_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref + 0);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_input_38_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_input_39_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_4_1_bias_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_4_1_bias_cluster_fork_args = (_sgd_blocks_4_1_bias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_4_1_bias_cluster_fork, &DeeployOptNetwork__sgd_blocks_4_1_bias_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310848,
                                      .size = 128,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_output_19_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_38_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_39_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_output_19_ref;
} _sgd_blocks_4_1_bias_closure_L3_args_t;

static void _sgd_blocks_4_1_bias_closure_L3(void *_sgd_blocks_4_1_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_1_bias_closure_L3_args_t *args = (_sgd_blocks_4_1_bias_closure_L3_args_t *)_sgd_blocks_4_1_bias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_38_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_38_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_39_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_39_ref;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_output_19_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_output_19_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};
  pi_cl_ram_req_t weight_updated_LocalToExternal = {0};
  pi_cl_ram_req_t weight_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_38_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &weight_ExternalToLocal);

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_39_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }

    if (weight_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&weight_ExternalToLocal);
      weight_ExternalToLocal.size = 0;
    }
    _sgd_blocks_4_1_bias_closure_args_t DeeployOptNetwork__sgd_blocks_4_1_bias_closure_args = (_sgd_blocks_4_1_bias_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref};

    // _sgd_blocks_4_1_bias_closure CLOSURE CALL
    _sgd_blocks_4_1_bias_closure(&DeeployOptNetwork__sgd_blocks_4_1_bias_closure_args);

    // Transfer output tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_output_19_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref, (uint32_t)128, (uint32_t)128, (uint32_t)128, 0,
                     &weight_updated_LocalToExternal);

    // Wait for output tiles

    if (weight_updated_LocalToExternal.size != 0) {
      pi_cl_ram_copy_wait(&weight_updated_LocalToExternal);
      weight_updated_LocalToExternal.size = 0;
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref;
} _sgd_fc_weight_tiling_closure_args_t;

static void _sgd_fc_weight_tiling_closure(void *_sgd_fc_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_fc_weight_tiling_closure_args_t *args = (_sgd_fc_weight_tiling_closure_args_t *)_sgd_fc_weight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_fc_weight, Op: SGD)
  int8_t sgd_fc_weight_core_id = pi_core_id();
  int8_t sgd_fc_weight_log2Core = log2(NUM_CORES);
  int32_t sgd_fc_weight_chunk = (288 >> sgd_fc_weight_log2Core) + ((288 & (NUM_CORES - 1)) != 0);
  int32_t sgd_fc_weight_chunk_start = MIN(sgd_fc_weight_chunk * sgd_fc_weight_core_id, 288);
  int32_t sgd_fc_weight_chunk_stop = MIN(sgd_fc_weight_chunk_start + sgd_fc_weight_chunk, 288);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_fc_weight_chunk_start;
  for (; i + 5 < sgd_fc_weight_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_fc_weight_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref;
} _sgd_fc_weight_cluster_fork_args_t;

static void _sgd_fc_weight_cluster_fork(void *_sgd_fc_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_fc_weight_cluster_fork_args_t *args = (_sgd_fc_weight_cluster_fork_args_t *)_sgd_fc_weight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_fc_weight_tiling_closure_args_t DeeployOptNetwork__sgd_fc_weight_tiling_closure_args = (_sgd_fc_weight_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref};

  // _sgd_fc_weight_tiling_closure CLOSURE CALL
  _sgd_fc_weight_tiling_closure(&DeeployOptNetwork__sgd_fc_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref;
} _sgd_fc_weight_closure_args_t;

static void _sgd_fc_weight_closure(void *_sgd_fc_weight_closure_args) {
  // CLOSURE ARG CAST
  _sgd_fc_weight_closure_args_t *args = (_sgd_fc_weight_closure_args_t *)_sgd_fc_weight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 1152);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_input_40_ref =
      (void *)((char *)DeeployOptNetwork_input_40 +
               DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_cumByteOffset[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_input_41_ref = (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_output_20_ref =
      (void *)((char *)DeeployOptNetwork_output_20 +
               DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_cumByteOffset[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr]);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442944,
                                      .size = 1152,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_input_40_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442944,
                                      .size = 1152,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_input_41_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_fc_weight_cluster_fork_args_t DeeployOptNetwork__sgd_fc_weight_cluster_fork_args = (_sgd_fc_weight_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_fc_weight_cluster_fork, &DeeployOptNetwork__sgd_fc_weight_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1311872,
                                      .size = 1152,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_output_20_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_input_41_ref;
} _sgd_fc_weight_closure_L3_args_t;

static void _sgd_fc_weight_closure_L3(void *_sgd_fc_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_fc_weight_closure_L3_args_t *args = (_sgd_fc_weight_closure_L3_args_t *)_sgd_fc_weight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_input_41_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_input_41_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_input_41_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref, (uint32_t)1152, (uint32_t)1152, (uint32_t)1152, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }
    _sgd_fc_weight_closure_args_t DeeployOptNetwork__sgd_fc_weight_closure_args = (_sgd_fc_weight_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref};

    // _sgd_fc_weight_closure CLOSURE CALL
    _sgd_fc_weight_closure(&DeeployOptNetwork__sgd_fc_weight_closure_args);

    // Transfer output tiles

    // Wait for output tiles

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref;
} _sgd_blocks_1_0_weight_tiling_closure_args_t;

static void _sgd_blocks_1_0_weight_tiling_closure(void *_sgd_blocks_1_0_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_0_weight_tiling_closure_args_t *args = (_sgd_blocks_1_0_weight_tiling_closure_args_t *)_sgd_blocks_1_0_weight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_1_0_weight, Op: SGD)
  int8_t sgd_blocks_1_0_weight_core_id = pi_core_id();
  int8_t sgd_blocks_1_0_weight_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_1_0_weight_chunk = (2048 >> sgd_blocks_1_0_weight_log2Core) + ((2048 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_1_0_weight_chunk_start = MIN(sgd_blocks_1_0_weight_chunk * sgd_blocks_1_0_weight_core_id, 2048);
  int32_t sgd_blocks_1_0_weight_chunk_stop = MIN(sgd_blocks_1_0_weight_chunk_start + sgd_blocks_1_0_weight_chunk, 2048);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_1_0_weight_chunk_start;
  for (; i + 5 < sgd_blocks_1_0_weight_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_1_0_weight_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref;
} _sgd_blocks_1_0_weight_cluster_fork_args_t;

static void _sgd_blocks_1_0_weight_cluster_fork(void *_sgd_blocks_1_0_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_0_weight_cluster_fork_args_t *args = (_sgd_blocks_1_0_weight_cluster_fork_args_t *)_sgd_blocks_1_0_weight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_1_0_weight_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_1_0_weight_tiling_closure_args = (_sgd_blocks_1_0_weight_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref =
          DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref};

  // _sgd_blocks_1_0_weight_tiling_closure CLOSURE CALL
  _sgd_blocks_1_0_weight_tiling_closure(&DeeployOptNetwork__sgd_blocks_1_0_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref;
} _sgd_blocks_1_0_weight_closure_args_t;

static void _sgd_blocks_1_0_weight_closure(void *_sgd_blocks_1_0_weight_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_0_weight_closure_args_t *args = (_sgd_blocks_1_0_weight_closure_args_t *)_sgd_blocks_1_0_weight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 8192);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 8192);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_input_8_ref =
      (void *)((char *)DeeployOptNetwork_input_8 + DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_input_9_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_output_4_ref =
      (void *)((char *)DeeployOptNetwork_output_4 + DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr]);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1449984,
                                      .size = 8192,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_input_8_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1449984,
                                      .size = 8192,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_input_9_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_1_0_weight_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_1_0_weight_cluster_fork_args = (_sgd_blocks_1_0_weight_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_1_0_weight_cluster_fork, &DeeployOptNetwork__sgd_blocks_1_0_weight_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1318912,
                                      .size = 8192,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_output_4_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_input_9_ref;
} _sgd_blocks_1_0_weight_closure_L3_args_t;

static void _sgd_blocks_1_0_weight_closure_L3(void *_sgd_blocks_1_0_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_1_0_weight_closure_L3_args_t *args = (_sgd_blocks_1_0_weight_closure_L3_args_t *)_sgd_blocks_1_0_weight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_input_9_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_input_9_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_input_9_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref, (uint32_t)8192, (uint32_t)8192, (uint32_t)8192, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }
    _sgd_blocks_1_0_weight_closure_args_t DeeployOptNetwork__sgd_blocks_1_0_weight_closure_args = (_sgd_blocks_1_0_weight_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref};

    // _sgd_blocks_1_0_weight_closure CLOSURE CALL
    _sgd_blocks_1_0_weight_closure(&DeeployOptNetwork__sgd_blocks_1_0_weight_closure_args);

    // Transfer output tiles

    // Wait for output tiles

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref;
} _sgd_blocks_2_0_weight_tiling_closure_args_t;

static void _sgd_blocks_2_0_weight_tiling_closure(void *_sgd_blocks_2_0_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_0_weight_tiling_closure_args_t *args = (_sgd_blocks_2_0_weight_tiling_closure_args_t *)_sgd_blocks_2_0_weight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_2_0_weight, Op: SGD)
  int8_t sgd_blocks_2_0_weight_core_id = pi_core_id();
  int8_t sgd_blocks_2_0_weight_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_2_0_weight_chunk = (2048 >> sgd_blocks_2_0_weight_log2Core) + ((2048 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_2_0_weight_chunk_start = MIN(sgd_blocks_2_0_weight_chunk * sgd_blocks_2_0_weight_core_id, 2048);
  int32_t sgd_blocks_2_0_weight_chunk_stop = MIN(sgd_blocks_2_0_weight_chunk_start + sgd_blocks_2_0_weight_chunk, 2048);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_2_0_weight_chunk_start;
  for (; i + 5 < sgd_blocks_2_0_weight_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_2_0_weight_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref;
} _sgd_blocks_2_0_weight_cluster_fork_args_t;

static void _sgd_blocks_2_0_weight_cluster_fork(void *_sgd_blocks_2_0_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_0_weight_cluster_fork_args_t *args = (_sgd_blocks_2_0_weight_cluster_fork_args_t *)_sgd_blocks_2_0_weight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_2_0_weight_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_2_0_weight_tiling_closure_args = (_sgd_blocks_2_0_weight_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref =
          DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref};

  // _sgd_blocks_2_0_weight_tiling_closure CLOSURE CALL
  _sgd_blocks_2_0_weight_tiling_closure(&DeeployOptNetwork__sgd_blocks_2_0_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref;
} _sgd_blocks_2_0_weight_closure_args_t;

static void _sgd_blocks_2_0_weight_closure(void *_sgd_blocks_2_0_weight_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_0_weight_closure_args_t *args = (_sgd_blocks_2_0_weight_closure_args_t *)_sgd_blocks_2_0_weight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 8192);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 8192);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_input_16_ref =
      (void *)((char *)DeeployOptNetwork_input_16 + DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_input_17_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_output_8_ref =
      (void *)((char *)DeeployOptNetwork_output_8 + DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr]);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1449984,
                                      .size = 8192,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_input_16_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1449984,
                                      .size = 8192,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_input_17_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_2_0_weight_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_2_0_weight_cluster_fork_args = (_sgd_blocks_2_0_weight_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_2_0_weight_cluster_fork, &DeeployOptNetwork__sgd_blocks_2_0_weight_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1318912,
                                      .size = 8192,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_output_8_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_input_17_ref;
} _sgd_blocks_2_0_weight_closure_L3_args_t;

static void _sgd_blocks_2_0_weight_closure_L3(void *_sgd_blocks_2_0_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_2_0_weight_closure_L3_args_t *args = (_sgd_blocks_2_0_weight_closure_L3_args_t *)_sgd_blocks_2_0_weight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_input_17_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_input_17_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_input_17_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref, (uint32_t)8192, (uint32_t)8192, (uint32_t)8192, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }
    _sgd_blocks_2_0_weight_closure_args_t DeeployOptNetwork__sgd_blocks_2_0_weight_closure_args = (_sgd_blocks_2_0_weight_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref};

    // _sgd_blocks_2_0_weight_closure CLOSURE CALL
    _sgd_blocks_2_0_weight_closure(&DeeployOptNetwork__sgd_blocks_2_0_weight_closure_args);

    // Transfer output tiles

    // Wait for output tiles

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref;
} _sgd_blocks_3_0_weight_tiling_closure_args_t;

static void _sgd_blocks_3_0_weight_tiling_closure(void *_sgd_blocks_3_0_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_0_weight_tiling_closure_args_t *args = (_sgd_blocks_3_0_weight_tiling_closure_args_t *)_sgd_blocks_3_0_weight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_3_0_weight, Op: SGD)
  int8_t sgd_blocks_3_0_weight_core_id = pi_core_id();
  int8_t sgd_blocks_3_0_weight_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_3_0_weight_chunk = (3584 >> sgd_blocks_3_0_weight_log2Core) + ((3584 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_3_0_weight_chunk_start = MIN(sgd_blocks_3_0_weight_chunk * sgd_blocks_3_0_weight_core_id, 3584);
  int32_t sgd_blocks_3_0_weight_chunk_stop = MIN(sgd_blocks_3_0_weight_chunk_start + sgd_blocks_3_0_weight_chunk, 3584);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_3_0_weight_chunk_start;
  for (; i + 5 < sgd_blocks_3_0_weight_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_3_0_weight_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref;
} _sgd_blocks_3_0_weight_cluster_fork_args_t;

static void _sgd_blocks_3_0_weight_cluster_fork(void *_sgd_blocks_3_0_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_0_weight_cluster_fork_args_t *args = (_sgd_blocks_3_0_weight_cluster_fork_args_t *)_sgd_blocks_3_0_weight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_3_0_weight_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_3_0_weight_tiling_closure_args = (_sgd_blocks_3_0_weight_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref =
          DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref};

  // _sgd_blocks_3_0_weight_tiling_closure CLOSURE CALL
  _sgd_blocks_3_0_weight_tiling_closure(&DeeployOptNetwork__sgd_blocks_3_0_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref;
} _sgd_blocks_3_0_weight_closure_args_t;

static void _sgd_blocks_3_0_weight_closure(void *_sgd_blocks_3_0_weight_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_0_weight_closure_args_t *args = (_sgd_blocks_3_0_weight_closure_args_t *)_sgd_blocks_3_0_weight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 14336);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_input_24_ref =
      (void *)((char *)DeeployOptNetwork_input_24 + DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_input_25_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_output_12_ref =
      (void *)((char *)DeeployOptNetwork_output_12 + DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr]);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1456128,
                                      .size = 14336,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_input_24_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1456128,
                                      .size = 14336,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_input_25_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_3_0_weight_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_3_0_weight_cluster_fork_args = (_sgd_blocks_3_0_weight_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_3_0_weight_cluster_fork, &DeeployOptNetwork__sgd_blocks_3_0_weight_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1325056,
                                      .size = 14336,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_output_12_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_input_25_ref;
} _sgd_blocks_3_0_weight_closure_L3_args_t;

static void _sgd_blocks_3_0_weight_closure_L3(void *_sgd_blocks_3_0_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_3_0_weight_closure_L3_args_t *args = (_sgd_blocks_3_0_weight_closure_L3_args_t *)_sgd_blocks_3_0_weight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_input_25_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_input_25_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_input_25_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref, (uint32_t)14336, (uint32_t)14336, (uint32_t)14336, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }
    _sgd_blocks_3_0_weight_closure_args_t DeeployOptNetwork__sgd_blocks_3_0_weight_closure_args = (_sgd_blocks_3_0_weight_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref};

    // _sgd_blocks_3_0_weight_closure CLOSURE CALL
    _sgd_blocks_3_0_weight_closure(&DeeployOptNetwork__sgd_blocks_3_0_weight_closure_args);

    // Transfer output tiles

    // Wait for output tiles

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref;
} _sgd_blocks_4_0_weight_tiling_closure_args_t;

static void _sgd_blocks_4_0_weight_tiling_closure(void *_sgd_blocks_4_0_weight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_0_weight_tiling_closure_args_t *args = (_sgd_blocks_4_0_weight_tiling_closure_args_t *)_sgd_blocks_4_0_weight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL

  // SGD Weight Update with Separated Multiplication and Subtraction Unrolling
  // (Name: sgd_blocks_4_0_weight, Op: SGD)
  int8_t sgd_blocks_4_0_weight_core_id = pi_core_id();
  int8_t sgd_blocks_4_0_weight_log2Core = log2(NUM_CORES);
  int32_t sgd_blocks_4_0_weight_chunk = (7168 >> sgd_blocks_4_0_weight_log2Core) + ((7168 & (NUM_CORES - 1)) != 0);
  int32_t sgd_blocks_4_0_weight_chunk_start = MIN(sgd_blocks_4_0_weight_chunk * sgd_blocks_4_0_weight_core_id, 7168);
  int32_t sgd_blocks_4_0_weight_chunk_stop = MIN(sgd_blocks_4_0_weight_chunk_start + sgd_blocks_4_0_weight_chunk, 7168);

  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref;
  float32_t *ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref =
      DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref;

  float32_t learning_rate = 0.0003000000142492354;

  // Temporary buffer for multiplication results
  float32_t temp_mul[6];

  uint32_t i = sgd_blocks_4_0_weight_chunk_start;
  for (; i + 5 < sgd_blocks_4_0_weight_chunk_stop; i += 6) {
    // Unrolled multiplication operations
    temp_mul[0] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref[i];
    temp_mul[1] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref[i + 1];
    temp_mul[2] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref[i + 2];
    temp_mul[3] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref[i + 3];
    temp_mul[4] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref[i + 4];
    temp_mul[5] = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref[i + 5];

    // Unrolled subtraction operations
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref[i] - temp_mul[0];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref[i + 1] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref[i + 1] - temp_mul[1];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref[i + 2] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref[i + 2] - temp_mul[2];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref[i + 3] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref[i + 3] - temp_mul[3];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref[i + 4] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref[i + 4] - temp_mul[4];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref[i + 5] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref[i + 5] - temp_mul[5];
  }

  // Handle remaining elements
  for (; i < sgd_blocks_4_0_weight_chunk_stop; i++) {
    float32_t temp_grad = learning_rate * ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref[i];
    ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref[i] =
        ref_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref[i] - temp_grad;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref;
} _sgd_blocks_4_0_weight_cluster_fork_args_t;

static void _sgd_blocks_4_0_weight_cluster_fork(void *_sgd_blocks_4_0_weight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_0_weight_cluster_fork_args_t *args = (_sgd_blocks_4_0_weight_cluster_fork_args_t *)_sgd_blocks_4_0_weight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref;

  // CLOSURE FUNCTION CALL
  _sgd_blocks_4_0_weight_tiling_closure_args_t DeeployOptNetwork__sgd_blocks_4_0_weight_tiling_closure_args = (_sgd_blocks_4_0_weight_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref =
          DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref};

  // _sgd_blocks_4_0_weight_tiling_closure CLOSURE CALL
  _sgd_blocks_4_0_weight_tiling_closure(&DeeployOptNetwork__sgd_blocks_4_0_weight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref;
} _sgd_blocks_4_0_weight_closure_args_t;

static void _sgd_blocks_4_0_weight_closure(void *_sgd_blocks_4_0_weight_closure_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_0_weight_closure_args_t *args = (_sgd_blocks_4_0_weight_closure_args_t *)_sgd_blocks_4_0_weight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 28672);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 28672);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_input_32_ref =
      (void *)((char *)DeeployOptNetwork_input_32 + DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_input_33_ref =
      (void *)((char *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref + 0);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_output_16_ref =
      (void *)((char *)DeeployOptNetwork_output_16 + DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr]);

  // Initialize DMA futures
  int weight_ExternalToLocal = -1;
  int grad_ExternalToLocal = -1;
  int weight_updated_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1470464,
                                      .size = 28672,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_input_32_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    grad_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1470464,
                                      .size = 28672,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_input_33_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (grad_ExternalToLocal >= 0) {
      mchan_transfer_wait(grad_ExternalToLocal);
      mchan_transfer_free(grad_ExternalToLocal);
    }

    _sgd_blocks_4_0_weight_cluster_fork_args_t DeeployOptNetwork__sgd_blocks_4_0_weight_cluster_fork_args = (_sgd_blocks_4_0_weight_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_sgd_blocks_4_0_weight_cluster_fork, &DeeployOptNetwork__sgd_blocks_4_0_weight_cluster_fork_args);

    // Transfer output tiles
    weight_updated_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1339392,
                                      .size = 28672,
                                      .loc = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_weight_updated_ref,
                                      .ext = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_output_16_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (weight_updated_LocalToExternal >= 0) {
      mchan_transfer_wait(weight_updated_LocalToExternal);
      mchan_transfer_free(weight_updated_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_input_33_ref;
} _sgd_blocks_4_0_weight_closure_L3_args_t;

static void _sgd_blocks_4_0_weight_closure_L3(void *_sgd_blocks_4_0_weight_closure_L3_args) {
  // CLOSURE ARG CAST
  _sgd_blocks_4_0_weight_closure_L3_args_t *args = (_sgd_blocks_4_0_weight_closure_L3_args_t *)_sgd_blocks_4_0_weight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref;
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr;
  void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_input_33_ref = args->DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_input_33_ref;

  // CLOSURE FUNCTION CALL

  // Initialize DMA futures
  pi_cl_ram_req_t grad_ExternalToLocal = {0};

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles

    gap9_ram_copy_2d((uint32_t)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_input_33_ref,
                     (void *)DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref, (uint32_t)28672, (uint32_t)28672, (uint32_t)28672, 1,
                     &grad_ExternalToLocal);

    // Wait for input tiles

    if (grad_ExternalToLocal.size != 0) {
      pi_cl_ram_copy_wait(&grad_ExternalToLocal);
      grad_ExternalToLocal.size = 0;
    }
    _sgd_blocks_4_0_weight_closure_args_t DeeployOptNetwork__sgd_blocks_4_0_weight_closure_args = (_sgd_blocks_4_0_weight_closure_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref};

    // _sgd_blocks_4_0_weight_closure CLOSURE CALL
    _sgd_blocks_4_0_weight_closure(&DeeployOptNetwork__sgd_blocks_4_0_weight_closure_args);

    // Transfer output tiles

    // Wait for output tiles

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

void RunOptimizerNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 32);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_2_ref =
        (void *)((char *)DeeployOptNetwork_input_2 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_3_ref =
        (void *)((char *)DeeployOptNetwork_input_3 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_output_1_ref =
        (void *)((char *)DeeployOptNetwork_output_1 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr]);
    _sgd_blocks_0_0_bias_closure_L3_args_t DeeployOptNetwork__sgd_blocks_0_0_bias_closure_L3_args = (_sgd_blocks_0_0_bias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_2_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_2_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_3_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_input_3_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_output_1_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_bias_output_1_ref};

    // _sgd_blocks_0_0_bias_closure_L3 CLOSURE CALL
    _sgd_blocks_0_0_bias_closure_L3(&DeeployOptNetwork__sgd_blocks_0_0_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 32);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 32);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_4_ref =
        (void *)((char *)DeeployOptNetwork_input_4 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_5_ref =
        (void *)((char *)DeeployOptNetwork_input_5 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_output_2_ref =
        (void *)((char *)DeeployOptNetwork_output_2 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr]);
    _sgd_blocks_0_1_weight_closure_L3_args_t DeeployOptNetwork__sgd_blocks_0_1_weight_closure_L3_args = (_sgd_blocks_0_1_weight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_4_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_4_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_5_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_input_5_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_output_2_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_weight_output_2_ref};

    // _sgd_blocks_0_1_weight_closure_L3 CLOSURE CALL
    _sgd_blocks_0_1_weight_closure_L3(&DeeployOptNetwork__sgd_blocks_0_1_weight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 32);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_6_ref =
        (void *)((char *)DeeployOptNetwork_input_6 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_7_ref =
        (void *)((char *)DeeployOptNetwork_input_7 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_output_3_ref =
        (void *)((char *)DeeployOptNetwork_output_3 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr]);
    _sgd_blocks_0_1_bias_closure_L3_args_t DeeployOptNetwork__sgd_blocks_0_1_bias_closure_L3_args = (_sgd_blocks_0_1_bias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_6_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_6_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_7_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_input_7_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_output_3_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_1_bias_output_3_ref};

    // _sgd_blocks_0_1_bias_closure_L3 CLOSURE CALL
    _sgd_blocks_0_1_bias_closure_L3(&DeeployOptNetwork__sgd_blocks_0_1_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 36);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_42_ref =
        (void *)((char *)DeeployOptNetwork_input_42 +
                 DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_cumByteOffset[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_43_ref =
        (void *)((char *)DeeployOptNetwork_input_43 +
                 DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_cumByteOffset[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_output_21_ref =
        (void *)((char *)DeeployOptNetwork_output_21 +
                 DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_cumByteOffset[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr]);
    _sgd_fc_bias_closure_L3_args_t DeeployOptNetwork__sgd_fc_bias_closure_L3_args = (_sgd_fc_bias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_42_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_42_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_43_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_input_43_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_output_21_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_bias_output_21_ref};

    // _sgd_fc_bias_closure_L3 CLOSURE CALL
    _sgd_fc_bias_closure_L3(&DeeployOptNetwork__sgd_fc_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 64);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_10_ref =
        (void *)((char *)DeeployOptNetwork_input_10 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_11_ref =
        (void *)((char *)DeeployOptNetwork_input_11 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_output_5_ref =
        (void *)((char *)DeeployOptNetwork_output_5 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr]);
    _sgd_blocks_1_0_bias_closure_L3_args_t DeeployOptNetwork__sgd_blocks_1_0_bias_closure_L3_args = (_sgd_blocks_1_0_bias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_10_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_10_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_11_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_input_11_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_output_5_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_bias_output_5_ref};

    // _sgd_blocks_1_0_bias_closure_L3 CLOSURE CALL
    _sgd_blocks_1_0_bias_closure_L3(&DeeployOptNetwork__sgd_blocks_1_0_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 64);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 64);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_12_ref =
        (void *)((char *)DeeployOptNetwork_input_12 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_13_ref =
        (void *)((char *)DeeployOptNetwork_input_13 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_output_6_ref =
        (void *)((char *)DeeployOptNetwork_output_6 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr]);
    _sgd_blocks_1_1_weight_closure_L3_args_t DeeployOptNetwork__sgd_blocks_1_1_weight_closure_L3_args = (_sgd_blocks_1_1_weight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_12_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_12_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_13_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_input_13_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_output_6_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_weight_output_6_ref};

    // _sgd_blocks_1_1_weight_closure_L3 CLOSURE CALL
    _sgd_blocks_1_1_weight_closure_L3(&DeeployOptNetwork__sgd_blocks_1_1_weight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 64);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_14_ref =
        (void *)((char *)DeeployOptNetwork_input_14 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_15_ref =
        (void *)((char *)DeeployOptNetwork_input_15 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_output_7_ref =
        (void *)((char *)DeeployOptNetwork_output_7 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr]);
    _sgd_blocks_1_1_bias_closure_L3_args_t DeeployOptNetwork__sgd_blocks_1_1_bias_closure_L3_args = (_sgd_blocks_1_1_bias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_14_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_14_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_15_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_input_15_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_output_7_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_1_bias_output_7_ref};

    // _sgd_blocks_1_1_bias_closure_L3 CLOSURE CALL
    _sgd_blocks_1_1_bias_closure_L3(&DeeployOptNetwork__sgd_blocks_1_1_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 64);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 64);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_18_ref =
        (void *)((char *)DeeployOptNetwork_input_18 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_19_ref =
        (void *)((char *)DeeployOptNetwork_input_19 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_output_9_ref =
        (void *)((char *)DeeployOptNetwork_output_9 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr]);
    _sgd_blocks_2_0_bias_closure_L3_args_t DeeployOptNetwork__sgd_blocks_2_0_bias_closure_L3_args = (_sgd_blocks_2_0_bias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_18_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_18_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_19_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_input_19_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_output_9_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_bias_output_9_ref};

    // _sgd_blocks_2_0_bias_closure_L3 CLOSURE CALL
    _sgd_blocks_2_0_bias_closure_L3(&DeeployOptNetwork__sgd_blocks_2_0_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 64);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_20_ref =
        (void *)((char *)DeeployOptNetwork_input_20 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_21_ref =
        (void *)((char *)DeeployOptNetwork_input_21 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_output_10_ref =
        (void *)((char *)DeeployOptNetwork_output_10 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_cumByteOffset
                                                           [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr]);
    _sgd_blocks_2_1_weight_closure_L3_args_t DeeployOptNetwork__sgd_blocks_2_1_weight_closure_L3_args = (_sgd_blocks_2_1_weight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_20_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_20_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_21_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_input_21_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_output_10_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_weight_output_10_ref};

    // _sgd_blocks_2_1_weight_closure_L3 CLOSURE CALL
    _sgd_blocks_2_1_weight_closure_L3(&DeeployOptNetwork__sgd_blocks_2_1_weight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 64);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 64);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_22_ref =
        (void *)((char *)DeeployOptNetwork_input_22 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_23_ref =
        (void *)((char *)DeeployOptNetwork_input_23 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_output_11_ref =
        (void *)((char *)DeeployOptNetwork_output_11 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_cumByteOffset
                                                           [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr]);
    _sgd_blocks_2_1_bias_closure_L3_args_t DeeployOptNetwork__sgd_blocks_2_1_bias_closure_L3_args = (_sgd_blocks_2_1_bias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_22_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_22_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_23_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_input_23_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_output_11_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_1_bias_output_11_ref};

    // _sgd_blocks_2_1_bias_closure_L3 CLOSURE CALL
    _sgd_blocks_2_1_bias_closure_L3(&DeeployOptNetwork__sgd_blocks_2_1_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 128);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_0_ref =
        (void *)((char *)DeeployOptNetwork_input_0 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_1_ref =
        (void *)((char *)DeeployOptNetwork_input_1 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_output_0_ref =
        (void *)((char *)DeeployOptNetwork_output_0 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr]);
    _sgd_blocks_0_0_weight_closure_L3_args_t DeeployOptNetwork__sgd_blocks_0_0_weight_closure_L3_args = (_sgd_blocks_0_0_weight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_0_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_0_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_0_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_1_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_input_1_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_output_0_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_0_0_weight_output_0_ref};

    // _sgd_blocks_0_0_weight_closure_L3 CLOSURE CALL
    _sgd_blocks_0_0_weight_closure_L3(&DeeployOptNetwork__sgd_blocks_0_0_weight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 128);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 128);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_26_ref =
        (void *)((char *)DeeployOptNetwork_input_26 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_27_ref =
        (void *)((char *)DeeployOptNetwork_input_27 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_output_13_ref =
        (void *)((char *)DeeployOptNetwork_output_13 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_cumByteOffset
                                                           [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr]);
    _sgd_blocks_3_0_bias_closure_L3_args_t DeeployOptNetwork__sgd_blocks_3_0_bias_closure_L3_args = (_sgd_blocks_3_0_bias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_26_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_26_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_27_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_input_27_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_output_13_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_bias_output_13_ref};

    // _sgd_blocks_3_0_bias_closure_L3 CLOSURE CALL
    _sgd_blocks_3_0_bias_closure_L3(&DeeployOptNetwork__sgd_blocks_3_0_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 128);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 128);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_28_ref =
        (void *)((char *)DeeployOptNetwork_input_28 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_29_ref =
        (void *)((char *)DeeployOptNetwork_input_29 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_output_14_ref =
        (void *)((char *)DeeployOptNetwork_output_14 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_cumByteOffset
                                                           [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr]);
    _sgd_blocks_3_1_weight_closure_L3_args_t DeeployOptNetwork__sgd_blocks_3_1_weight_closure_L3_args = (_sgd_blocks_3_1_weight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_28_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_28_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_29_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_input_29_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_output_14_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_weight_output_14_ref};

    // _sgd_blocks_3_1_weight_closure_L3 CLOSURE CALL
    _sgd_blocks_3_1_weight_closure_L3(&DeeployOptNetwork__sgd_blocks_3_1_weight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 128);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_30_ref =
        (void *)((char *)DeeployOptNetwork_input_30 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_31_ref =
        (void *)((char *)DeeployOptNetwork_input_31 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_output_15_ref =
        (void *)((char *)DeeployOptNetwork_output_15 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_cumByteOffset
                                                           [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr]);
    _sgd_blocks_3_1_bias_closure_L3_args_t DeeployOptNetwork__sgd_blocks_3_1_bias_closure_L3_args = (_sgd_blocks_3_1_bias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_30_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_30_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_31_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_input_31_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_output_15_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_1_bias_output_15_ref};

    // _sgd_blocks_3_1_bias_closure_L3 CLOSURE CALL
    _sgd_blocks_3_1_bias_closure_L3(&DeeployOptNetwork__sgd_blocks_3_1_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 128);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 128);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_34_ref =
        (void *)((char *)DeeployOptNetwork_input_34 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_35_ref =
        (void *)((char *)DeeployOptNetwork_input_35 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_output_17_ref =
        (void *)((char *)DeeployOptNetwork_output_17 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_cumByteOffset
                                                           [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr]);
    _sgd_blocks_4_0_bias_closure_L3_args_t DeeployOptNetwork__sgd_blocks_4_0_bias_closure_L3_args = (_sgd_blocks_4_0_bias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_34_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_34_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_35_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_input_35_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_output_17_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_bias_output_17_ref};

    // _sgd_blocks_4_0_bias_closure_L3 CLOSURE CALL
    _sgd_blocks_4_0_bias_closure_L3(&DeeployOptNetwork__sgd_blocks_4_0_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 128);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_36_ref =
        (void *)((char *)DeeployOptNetwork_input_36 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_37_ref =
        (void *)((char *)DeeployOptNetwork_input_37 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_output_18_ref =
        (void *)((char *)DeeployOptNetwork_output_18 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_cumByteOffset
                                                           [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr]);
    _sgd_blocks_4_1_weight_closure_L3_args_t DeeployOptNetwork__sgd_blocks_4_1_weight_closure_L3_args = (_sgd_blocks_4_1_weight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_36_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_36_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_37_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_input_37_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_output_18_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_weight_output_18_ref};

    // _sgd_blocks_4_1_weight_closure_L3 CLOSURE CALL
    _sgd_blocks_4_1_weight_closure_L3(&DeeployOptNetwork__sgd_blocks_4_1_weight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 128);
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_38_ref =
        (void *)((char *)DeeployOptNetwork_input_38 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_39_ref =
        (void *)((char *)DeeployOptNetwork_input_39 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr]);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_output_19_ref =
        (void *)((char *)DeeployOptNetwork_output_19 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_cumByteOffset
                                                           [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr]);
    _sgd_blocks_4_1_bias_closure_L3_args_t DeeployOptNetwork__sgd_blocks_4_1_bias_closure_L3_args = (_sgd_blocks_4_1_bias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref =
            DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_weight_updated_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_38_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_38_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_39_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_input_39_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_output_19_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_1_bias_output_19_ref};

    // _sgd_blocks_4_1_bias_closure_L3 CLOSURE CALL
    _sgd_blocks_4_1_bias_closure_L3(&DeeployOptNetwork__sgd_blocks_4_1_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_input_41_ref =
        (void *)((char *)DeeployOptNetwork_input_41 +
                 DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_cumByteOffset[*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr]);
    _sgd_fc_weight_closure_L3_args_t DeeployOptNetwork__sgd_fc_weight_closure_L3_args = (_sgd_fc_weight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_fc_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_input_41_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_fc_weight_input_41_ref};

    // _sgd_fc_weight_closure_L3 CLOSURE CALL
    _sgd_fc_weight_closure_L3(&DeeployOptNetwork__sgd_fc_weight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_input_9_ref =
        (void *)((char *)DeeployOptNetwork_input_9 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr]);
    _sgd_blocks_1_0_weight_closure_L3_args_t DeeployOptNetwork__sgd_blocks_1_0_weight_closure_L3_args = (_sgd_blocks_1_0_weight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_1_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_input_9_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_1_0_weight_input_9_ref};

    // _sgd_blocks_1_0_weight_closure_L3 CLOSURE CALL
    _sgd_blocks_1_0_weight_closure_L3(&DeeployOptNetwork__sgd_blocks_1_0_weight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_input_17_ref =
        (void *)((char *)DeeployOptNetwork_input_17 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr]);
    _sgd_blocks_2_0_weight_closure_L3_args_t DeeployOptNetwork__sgd_blocks_2_0_weight_closure_L3_args = (_sgd_blocks_2_0_weight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_2_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_input_17_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_2_0_weight_input_17_ref};

    // _sgd_blocks_2_0_weight_closure_L3 CLOSURE CALL
    _sgd_blocks_2_0_weight_closure_L3(&DeeployOptNetwork__sgd_blocks_2_0_weight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_input_25_ref =
        (void *)((char *)DeeployOptNetwork_input_25 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr]);
    _sgd_blocks_3_0_weight_closure_L3_args_t DeeployOptNetwork__sgd_blocks_3_0_weight_closure_L3_args = (_sgd_blocks_3_0_weight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_3_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_input_25_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_3_0_weight_input_25_ref};

    // _sgd_blocks_3_0_weight_closure_L3 CLOSURE CALL
    _sgd_blocks_3_0_weight_closure_L3(&DeeployOptNetwork__sgd_blocks_3_0_weight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr;

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr;
    float32_t *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L2 + 0);
    void *DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_input_33_ref =
        (void *)((char *)DeeployOptNetwork_input_33 + DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_cumByteOffset
                                                          [*DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr]);
    _sgd_blocks_4_0_weight_closure_L3_args_t DeeployOptNetwork__sgd_blocks_4_0_weight_closure_L3_args = (_sgd_blocks_4_0_weight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_sgd_blocks_4_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_grad_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_tileIdxPtr,
        .DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_input_33_ref = DeeployOptNetwork_TILING_CODEGEN_L2_sgd_blocks_4_0_weight_input_33_ref};

    // _sgd_blocks_4_0_weight_closure_L3 CLOSURE CALL
    _sgd_blocks_4_0_weight_closure_L3(&DeeployOptNetwork__sgd_blocks_4_0_weight_closure_L3_args);
  }
}

void InitOptimizerNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  DeeployOptNetwork_MEMORYARENA_L1 = (int8_t *)DeeployNetwork_MEMORYARENA_L1; /* shared with TrainingNetwork */
  if (DeeployOptNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployOptNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 57344));
    return;
  }

  DeeployOptNetwork_MEMORYARENA_L2 = (int8_t *)DeeployNetwork_MEMORYARENA_L2; /* shared with TrainingNetwork */

  DeeployOptNetwork_input_0 = (float32_t *)DeeployNetwork_input_2;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_1 = (float32_t *)DeeployNetwork_input_24; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_2 = (float32_t *)DeeployNetwork_input_3;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_3 = (float32_t *)DeeployNetwork_input_25; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_4 = (float32_t *)DeeployNetwork_input_4;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_5 = (float32_t *)DeeployNetwork_input_26; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_6 = (float32_t *)DeeployNetwork_input_5;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_7 = (float32_t *)DeeployNetwork_input_27; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_8 = (float32_t *)DeeployNetwork_input_6;  /* shared with TrainingNetwork */

  DeeployOptNetwork_input_9 = (float32_t *)DeeployNetwork_input_28;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_10 = (float32_t *)DeeployNetwork_input_7;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_11 = (float32_t *)DeeployNetwork_input_29; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_12 = (float32_t *)DeeployNetwork_input_8;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_13 = (float32_t *)DeeployNetwork_input_30; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_14 = (float32_t *)DeeployNetwork_input_9;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_15 = (float32_t *)DeeployNetwork_input_31; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_16 = (float32_t *)DeeployNetwork_input_10; /* shared with TrainingNetwork */

  DeeployOptNetwork_input_17 = (float32_t *)DeeployNetwork_input_32; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_18 = (float32_t *)DeeployNetwork_input_11; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_19 = (float32_t *)DeeployNetwork_input_33; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_20 = (float32_t *)DeeployNetwork_input_12; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_21 = (float32_t *)DeeployNetwork_input_34; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_22 = (float32_t *)DeeployNetwork_input_13; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_23 = (float32_t *)DeeployNetwork_input_35; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_24 = (float32_t *)DeeployNetwork_input_14; /* shared with TrainingNetwork */

  DeeployOptNetwork_input_25 = (float32_t *)DeeployNetwork_input_36; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_26 = (float32_t *)DeeployNetwork_input_15; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_27 = (float32_t *)DeeployNetwork_input_37; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_28 = (float32_t *)DeeployNetwork_input_16; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_29 = (float32_t *)DeeployNetwork_input_38; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_30 = (float32_t *)DeeployNetwork_input_17; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_31 = (float32_t *)DeeployNetwork_input_39; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_32 = (float32_t *)DeeployNetwork_input_18; /* shared with TrainingNetwork */

  DeeployOptNetwork_input_33 = (float32_t *)DeeployNetwork_input_40; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_34 = (float32_t *)DeeployNetwork_input_19; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_35 = (float32_t *)DeeployNetwork_input_41; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_36 = (float32_t *)DeeployNetwork_input_20; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_37 = (float32_t *)DeeployNetwork_input_42; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_38 = (float32_t *)DeeployNetwork_input_21; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_39 = (float32_t *)DeeployNetwork_input_43; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_40 = (float32_t *)DeeployNetwork_input_22; /* shared with TrainingNetwork */

  DeeployOptNetwork_input_41 = (float32_t *)DeeployNetwork_input_44; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_42 = (float32_t *)DeeployNetwork_input_23; /* shared with TrainingNetwork */
  DeeployOptNetwork_input_43 = (float32_t *)DeeployNetwork_input_45; /* shared with TrainingNetwork */
  DeeployOptNetwork_output_0 = (float32_t *)DeeployNetwork_input_2;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_1 = (float32_t *)DeeployNetwork_input_3;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_2 = (float32_t *)DeeployNetwork_input_4;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_3 = (float32_t *)DeeployNetwork_input_5;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_4 = (float32_t *)DeeployOptNetwork_input_8;
  DeeployOptNetwork_output_5 = (float32_t *)DeeployNetwork_input_7; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_6 = (float32_t *)DeeployNetwork_input_8; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_7 = (float32_t *)DeeployNetwork_input_9; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_8 = (float32_t *)DeeployOptNetwork_input_16;
  DeeployOptNetwork_output_9 = (float32_t *)DeeployNetwork_input_11;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_10 = (float32_t *)DeeployNetwork_input_12; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_11 = (float32_t *)DeeployNetwork_input_13; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_12 = (float32_t *)DeeployOptNetwork_input_24;
  DeeployOptNetwork_output_13 = (float32_t *)DeeployNetwork_input_15; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_14 = (float32_t *)DeeployNetwork_input_16; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_15 = (float32_t *)DeeployNetwork_input_17; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_16 = (float32_t *)DeeployOptNetwork_input_32;
  DeeployOptNetwork_output_17 = (float32_t *)DeeployNetwork_input_19; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_18 = (float32_t *)DeeployNetwork_input_20; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_19 = (float32_t *)DeeployNetwork_input_21; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_20 = (float32_t *)DeeployOptNetwork_input_40;
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
  DeeployOptNetwork_inputs[22] = (void *)DeeployOptNetwork_input_22;
  DeeployOptNetwork_inputs[23] = (void *)DeeployOptNetwork_input_23;
  DeeployOptNetwork_inputs[24] = (void *)DeeployOptNetwork_input_24;
  DeeployOptNetwork_inputs[25] = (void *)DeeployOptNetwork_input_25;
  DeeployOptNetwork_inputs[26] = (void *)DeeployOptNetwork_input_26;
  DeeployOptNetwork_inputs[27] = (void *)DeeployOptNetwork_input_27;
  DeeployOptNetwork_inputs[28] = (void *)DeeployOptNetwork_input_28;
  DeeployOptNetwork_inputs[29] = (void *)DeeployOptNetwork_input_29;
  DeeployOptNetwork_inputs[30] = (void *)DeeployOptNetwork_input_30;
  DeeployOptNetwork_inputs[31] = (void *)DeeployOptNetwork_input_31;
  DeeployOptNetwork_inputs[32] = (void *)DeeployOptNetwork_input_32;
  DeeployOptNetwork_inputs[33] = (void *)DeeployOptNetwork_input_33;
  DeeployOptNetwork_inputs[34] = (void *)DeeployOptNetwork_input_34;
  DeeployOptNetwork_inputs[35] = (void *)DeeployOptNetwork_input_35;
  DeeployOptNetwork_inputs[36] = (void *)DeeployOptNetwork_input_36;
  DeeployOptNetwork_inputs[37] = (void *)DeeployOptNetwork_input_37;
  DeeployOptNetwork_inputs[38] = (void *)DeeployOptNetwork_input_38;
  DeeployOptNetwork_inputs[39] = (void *)DeeployOptNetwork_input_39;
  DeeployOptNetwork_inputs[40] = (void *)DeeployOptNetwork_input_40;
  DeeployOptNetwork_inputs[41] = (void *)DeeployOptNetwork_input_41;
  DeeployOptNetwork_inputs[42] = (void *)DeeployOptNetwork_input_42;
  DeeployOptNetwork_inputs[43] = (void *)DeeployOptNetwork_input_43;
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
