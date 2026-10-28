#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "ne16.h"
#include "ne16_pulp_bsp.h"
#include "ne16_task.h"
#include "perf_utils.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "pulp_nnx_ne16.h"
#include "pulp_nnx_util.h"
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

static PI_L2 int32_t DeeployNetwork_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_b0_dual_wenc[8] = {85469,  143432, 99234,  126554,
                                                                                                        105298, 133299, 112298, 80873};

static PI_L2 int32_t DeeployNetwork_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_b0_dual_bias[8] = {85469,  143432, 99234,  126554,
                                                                                                        105298, 133299, 112298, 80873};

static PI_L2 int32_t DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor[8] = {7666, 12866, 8901, 11352, 9445, 11957, 10073, 7254};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor[8] = {391, 233, 337, 264, 318, 251, 298, 413};

static PI_L2 int32_t DeeployNetwork_mul2_tensor[16] = {391, 233, 337, 264, 318, 251, 298, 413, 391, 233, 337, 264, 318, 251, 298, 413};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_cumByteOffset[1] = {0};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride[4] = {1524, 1524, 1280, 1280};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride[4] = {6096, 6096, 5120, 5120};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo[4] = {4, 1, 4, 1};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo[4] = {127, 127, 107, 107};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo[4] = {3, 2, 3, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo[4] = {3, 3, 2, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi[4] = {3, 2, 3, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi[4] = {3, 3, 2, 2};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 4};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[4] = {0, 33648, 1524, 35172};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[4] = {1984368, 1969128, 1981440, 1968640};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size[4] = {18288, 3048, 15360, 2560};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size_1d[4] = {1524, 1524, 1280, 1280};

static PI_L1 int32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_relativeOffset[4] = {33648, -32124, 33648, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[4] = {0, 134592, 6096, 140688};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[4] = {1908160, 1847200, 1896448, 1845248};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size[4] = {73152, 12192, 61440, 10240};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size_1d[4] = {6096, 6096, 5120, 5120};

static PI_L1 int32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_relativeOffset[4] = {134592, -128496, 134592, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_numTiles[2] = {0, 4};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_cumByteOffset[4] = {0, 4, 8, 12};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_cumByteOffset[4] = {0, 39256, 78512, 117768};

void *DeeployNetwork_inputs[3];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  void *DeeployNetwork_b0_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref;
} _b0_dual_wenc_tiling_closure_args_t;

static void _b0_dual_wenc_tiling_closure(void *_b0_dual_wenc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b0_dual_wenc_tiling_closure_args_t *args = (_b0_dual_wenc_tiling_closure_args_t *)_b0_dual_wenc_tiling_closure_args;

  void *DeeployNetwork_b0_dual_wenc_buffer = args->DeeployNetwork_b0_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualPerturbEncode (Name: b0_dual_wenc, Op: NE16DualPerturbEncode) -- W+/W- from one draw, encoded as one 2*cout 1x1 weight -- QW exp6
  {
    uint32_t b0_dual_wenc_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 0);
    uint32_t b0_dual_wenc_tile_xor = (0 * 0x9E3779B1u);
    float32_t b0_dual_wenc_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualPerturbEncode((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref,
                               (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref,
                               (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref, (int8_t *)DeeployNetwork_b0_dual_wenc_buffer,
                               b0_dual_wenc_seed_w, b0_dual_wenc_tile_xor, 15, b0_dual_wenc_eps_scale, NUM_CORES, 8, 4);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b0_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref;
} _b0_dual_wenc_cluster_fork_args_t;

static void _b0_dual_wenc_cluster_fork(void *_b0_dual_wenc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b0_dual_wenc_cluster_fork_args_t *args = (_b0_dual_wenc_cluster_fork_args_t *)_b0_dual_wenc_cluster_fork_args;

  void *DeeployNetwork_b0_dual_wenc_buffer = args->DeeployNetwork_b0_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b0_dual_wenc_tiling_closure_args_t DeeployNetwork__b0_dual_wenc_tiling_closure_args = (_b0_dual_wenc_tiling_closure_args_t){
      .DeeployNetwork_b0_dual_wenc_buffer = DeeployNetwork_b0_dual_wenc_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref};

  // _b0_dual_wenc_tiling_closure CLOSURE CALL
  _b0_dual_wenc_tiling_closure(&DeeployNetwork__b0_dual_wenc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr;
} _b0_dual_wenc_closure_args_t;

static void _b0_dual_wenc_closure(void *_b0_dual_wenc_closure_args) {
  // CLOSURE ARG CAST
  _b0_dual_wenc_closure_args_t *args = (_b0_dual_wenc_closure_args_t *)_b0_dual_wenc_closure_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 320);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 352);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_b0_dual_wenc_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_b0_dual_wenc +
               DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_enc_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_enc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_b0_dual_wenc_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 256);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_input_1_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_b0_dual_wenc_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b0_dual_wenc_cluster_fork_args_t DeeployNetwork__b0_dual_wenc_cluster_fork_args = (_b0_dual_wenc_cluster_fork_args_t){
        .DeeployNetwork_b0_dual_wenc_buffer = DeeployNetwork_b0_dual_wenc_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b0_dual_wenc_cluster_fork, &DeeployNetwork__b0_dual_wenc_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310976, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_weight_enc_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr;
} _b0_dual_wenc_closure_L3_args_t;

static void _b0_dual_wenc_closure_L3(void *_b0_dual_wenc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b0_dual_wenc_closure_L3_args_t *args = (_b0_dual_wenc_closure_L3_args_t *)_b0_dual_wenc_closure_L3_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b0_dual_wenc_closure_args_t DeeployNetwork__b0_dual_wenc_closure_args =
      (_b0_dual_wenc_closure_args_t){.DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr};

  // _b0_dual_wenc_closure CLOSURE CALL
  _b0_dual_wenc_closure(&DeeployNetwork__b0_dual_wenc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b0_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref;
} _b0_dual_bias_tiling_closure_args_t;

static void _b0_dual_bias_tiling_closure(void *_b0_dual_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b0_dual_bias_tiling_closure_args_t *args = (_b0_dual_bias_tiling_closure_args_t *)_b0_dual_bias_tiling_closure_args;

  void *DeeployNetwork_b0_dual_bias_buffer = args->DeeployNetwork_b0_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualBias (Name: b0_dual_bias, Op: NE16DualBias) -- add+/add- with the signed-input correction, stacked -- QW exp6
  {
    uint32_t b0_dual_bias_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 0);
    uint32_t b0_dual_bias_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 1);
    uint32_t b0_dual_bias_tile_xor = (0 * 0x9E3779B1u);
    float32_t b0_dual_bias_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualBias(
        (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref, (int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref,
        (int8_t *)DeeployNetwork_b0_dual_bias_buffer, b0_dual_bias_seed_w, b0_dual_bias_seed_b, b0_dual_bias_tile_xor, 15, 16, b0_dual_bias_eps_scale,
        NUM_CORES, 8, 4, 128);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b0_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref;
} _b0_dual_bias_cluster_fork_args_t;

static void _b0_dual_bias_cluster_fork(void *_b0_dual_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b0_dual_bias_cluster_fork_args_t *args = (_b0_dual_bias_cluster_fork_args_t *)_b0_dual_bias_cluster_fork_args;

  void *DeeployNetwork_b0_dual_bias_buffer = args->DeeployNetwork_b0_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b0_dual_bias_tiling_closure_args_t DeeployNetwork__b0_dual_bias_tiling_closure_args = (_b0_dual_bias_tiling_closure_args_t){
      .DeeployNetwork_b0_dual_bias_buffer = DeeployNetwork_b0_dual_bias_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref};

  // _b0_dual_bias_tiling_closure CLOSURE CALL
  _b0_dual_bias_tiling_closure(&DeeployNetwork__b0_dual_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr;
} _b0_dual_bias_closure_args_t;

static void _b0_dual_bias_closure(void *_b0_dual_bias_closure_args) {
  // CLOSURE ARG CAST
  _b0_dual_bias_closure_args_t *args = (_b0_dual_bias_closure_args_t *)_b0_dual_bias_closure_args;

  int32_t *DeeployNetwork_add2_tensor = args->DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 192);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 256);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 224);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 288);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 320);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_b0_dual_bias_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_b0_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 +
               DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_blocks0convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add2_tensor_ref =
      (void *)((char *)DeeployNetwork_add2_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_b0_dual_bias_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_input_1_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_blocks0convweight_int8_pmul_tensor_DUPLICATE_FOR_b0_dual_bias_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_input_2_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_blocks0convbias_rqsadd_pmul_tensor_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b0_dual_bias_cluster_fork_args_t DeeployNetwork__b0_dual_bias_cluster_fork_args = (_b0_dual_bias_cluster_fork_args_t){
        .DeeployNetwork_b0_dual_bias_buffer = DeeployNetwork_b0_dual_bias_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_bmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b0_dual_bias_cluster_fork, &DeeployNetwork__b0_dual_bias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_add2_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr;
} _b0_dual_bias_closure_L3_args_t;

static void _b0_dual_bias_closure_L3(void *_b0_dual_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _b0_dual_bias_closure_L3_args_t *args = (_b0_dual_bias_closure_L3_args_t *)_b0_dual_bias_closure_L3_args;

  int32_t *DeeployNetwork_add2_tensor = args->DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b0_dual_bias_closure_args_t DeeployNetwork__b0_dual_bias_closure_args =
      (_b0_dual_bias_closure_args_t){.DeeployNetwork_add2_tensor = DeeployNetwork_add2_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr};

  // _b0_dual_bias_closure CLOSURE CALL
  _b0_dual_bias_closure(&DeeployNetwork__b0_dual_bias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref;
} _b0_im2col_tiling_closure_args_t;

static void _b0_im2col_tiling_closure(void *_b0_im2col_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b0_im2col_tiling_closure_args_t *args = (_b0_im2col_tiling_closure_args_t *)_b0_im2col_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16Im2ColU8 (Name: b0_im2col, Op: NE16Im2ColU8) -- column buffer for the NE16 im2col route -- QW exp6
  PULP_Im2ColU8_NCHW_to_HWC((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref,
                            (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref, NUM_CORES, 14, 700, 1, 14, 701, 1, 4, 0, 0, 2, 2, 1, 1,
                            (uint8_t)128, (uint8_t)128);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref;
} _b0_im2col_cluster_fork_args_t;

static void _b0_im2col_cluster_fork(void *_b0_im2col_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b0_im2col_cluster_fork_args_t *args = (_b0_im2col_cluster_fork_args_t *)_b0_im2col_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b0_im2col_tiling_closure_args_t DeeployNetwork__b0_im2col_tiling_closure_args =
      (_b0_im2col_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref,
                                         .DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref};

  // _b0_im2col_tiling_closure CLOSURE CALL
  _b0_im2col_tiling_closure(&DeeployNetwork__b0_im2col_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_col_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr;
} _b0_im2col_closure_args_t;

static void _b0_im2col_closure(void *_b0_im2col_closure_args) {
  // CLOSURE ARG CAST
  _b0_im2col_closure_args_t *args = (_b0_im2col_closure_args_t *)_b0_im2col_closure_args;

  uint8_t *DeeployNetwork_col_tensor = args->DeeployNetwork_col_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39256);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_col_tensor_ref =
      (void *)((char *)DeeployNetwork_col_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1451592, DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b0_im2col_cluster_fork_args_t DeeployNetwork__b0_im2col_cluster_fork_args =
        (_b0_im2col_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_in_ref,
                                         .DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b0_im2col_cluster_fork, &DeeployNetwork__b0_im2col_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1349976, DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_col_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_col_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr;
} _b0_im2col_closure_L3_args_t;

static void _b0_im2col_closure_L3(void *_b0_im2col_closure_L3_args) {
  // CLOSURE ARG CAST
  _b0_im2col_closure_L3_args_t *args = (_b0_im2col_closure_L3_args_t *)_b0_im2col_closure_L3_args;

  uint8_t *DeeployNetwork_col_tensor = args->DeeployNetwork_col_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b0_im2col_closure_args_t DeeployNetwork__b0_im2col_closure_args =
      (_b0_im2col_closure_args_t){.DeeployNetwork_col_tensor = DeeployNetwork_col_tensor,
                                  .DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr};

  // _b0_im2col_closure CLOSURE CALL
  _b0_im2col_closure(&DeeployNetwork__b0_im2col_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} __MERGE_CONVRQ_PASS_0_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_tiling_closure(void *__MERGE_CONVRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_tiling_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_0_tiling_closure_args;

  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // N-EUREKA Task Init
  ne16_task_t task = {
      .data = (ne16_task_data_t){
          .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref - 0 + 0,
          .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref - 0,
          .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
          .scale_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
          .scale_shift_addr = (uint32_t)NULL,
          .scale_bias_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
          .cfg = (ne16_cfg_t){
              .input_stride = (ne16_stride_t){.d0 = 4, .d1 = *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref, .d2 = 0},
              .output_stride = (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES,
                                               .d1 = 16,
                                               .d2 = *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref},
              task.data.cfg.weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
              .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                     .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref,
                                                                             *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref)},
                                          .remainder = {.KoKi = nnx_concat_half(16, 4),
                                                        .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref,
                                                                                *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref),
                                                        .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref,
                                                                                *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref)}},
              .padding = (0 << 28) + (0 << 24) + (0 << 20) + (0 << 16),
              .weight_offset_factor = -128,
              .filter_mask = 0,
              .conf0 = 43032663,
          }}};
  // NE16 top-level task struct fields (required by HAL helpers and NE16 HW for
  // non-1x1 paths). Kept consistent with ne16_task_set_op_to_conv/_set_bits.
  task.weight_d0_stride = NE16_WEIGHT_D0_STRIDE_MODE8;
  task.qw = 8;
  task.subtile_output_channel = 32;
  task.kernel_shape = 1;
  task.depthwise = 0;

  // N-EUREKA Task Execution
  ne16_nnx_dispatch_wait(ne16_pulp_get_dev());
  ne16_nnx_dispatch(ne16_pulp_get_dev(), &task);
  ne16_nnx_resolve_wait(ne16_pulp_get_dev(), &task);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_closure(void *__MERGE_CONVRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_args_t *)__MERGE_CONVRQ_PASS_0_closure_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_add2_tensor = args->DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped =
      args->DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_tensor_Reshaped = args->DeeployNetwork_y_nhwc_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride + 0);
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 73152);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 91440);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 91760);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 91696);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped_ref =
      (void *)((char *)DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_enc_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_enc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul2_tensor_ref =
      (void *)((char *)DeeployNetwork_mul2_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add2_tensor_ref =
      (void *)((char *)DeeployNetwork_add2_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_tensor_Reshaped_ref =
      (void *)((char *)DeeployNetwork_y_nhwc_tensor_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_output = -1;
  int transfer_input = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[TILING_I],
          .size = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size[TILING_I],
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped_ref,
          .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size_1d[TILING_I],
          .ext_stride_1d = 2804};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped_ref) +
                 DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_relativeOffset[TILING_I]);

    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442048,
                                      .size = 256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_enc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul2_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add2_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref =
        &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref =
        &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride[TILING_I];

    __MERGE_CONVRQ_PASS_0_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_tiling_closure_args = (__MERGE_CONVRQ_PASS_0_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref =
            DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref =
            DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

    // __MERGE_CONVRQ_PASS_0_tiling_closure CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_tiling_closure_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_tensor_Reshaped_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size_1d[TILING_I],
                                      .ext_stride_1d = 11216};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_tensor_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_tensor_Reshaped_ref) +
                 DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_relativeOffset[TILING_I]);

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_closure_L3(void *__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)__MERGE_CONVRQ_PASS_0_closure_L3_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_add2_tensor = args->DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped =
      args->DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_tensor_Reshaped = args->DeeployNetwork_y_nhwc_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args = (__MERGE_CONVRQ_PASS_0_closure_args_t){
      .DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
      .DeeployNetwork_add2_tensor = DeeployNetwork_add2_tensor,
      .DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped =
          DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped,
      .DeeployNetwork_y_nhwc_tensor_Reshaped = DeeployNetwork_y_nhwc_tensor_Reshaped,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref;
} _b0_out_nchw_tiling_closure_args_t;

static void _b0_out_nchw_tiling_closure(void *_b0_out_nchw_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b0_out_nchw_tiling_closure_args_t *args = (_b0_out_nchw_tiling_closure_args_t *)_b0_out_nchw_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 14, 701, 16] -> [1, 16, 14, 701] (Name: b0_out_nchw, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 14;

  uint16_t dimLen_2 = 701;

  uint16_t dimLen_3 = 4;

  // RW: GCC Segmentation fault
  int8_t (*src)[dimLen_1][dimLen_2][dimLen_3] = (int8_t (*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref;
  int8_t (*dst)[dimLen_3][dimLen_1][dimLen_2] = (int8_t (*)[dimLen_3][dimLen_1][dimLen_2])DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref;

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref;
} _b0_out_nchw_cluster_fork_args_t;

static void _b0_out_nchw_cluster_fork(void *_b0_out_nchw_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b0_out_nchw_cluster_fork_args_t *args = (_b0_out_nchw_cluster_fork_args_t *)_b0_out_nchw_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b0_out_nchw_tiling_closure_args_t DeeployNetwork__b0_out_nchw_tiling_closure_args = (_b0_out_nchw_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref};

  // _b0_out_nchw_tiling_closure CLOSURE CALL
  _b0_out_nchw_tiling_closure(&DeeployNetwork__b0_out_nchw_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_nhwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr;
} _b0_out_nchw_closure_args_t;

static void _b0_out_nchw_closure(void *_b0_out_nchw_closure_args) {
  // CLOSURE ARG CAST
  _b0_out_nchw_closure_args_t *args = (_b0_out_nchw_closure_args_t *)_b0_out_nchw_closure_args;

  int8_t *DeeployNetwork_y_nhwc_tensor = args->DeeployNetwork_y_nhwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39256);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_y_nhwc_tensor_ref =
      (void *)((char *)DeeployNetwork_y_nhwc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_output = -1;
  int transfer_input = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 2005336,
                                      .size = 39256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_y_nhwc_tensor_ref,
                                      .ext_size_1d = 4,
                                      .ext_stride_1d = 16};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_y_nhwc_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_y_nhwc_tensor_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_y_nhwc_tensor_ref) + 4);

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _b0_out_nchw_cluster_fork_args_t DeeployNetwork__b0_out_nchw_cluster_fork_args = (_b0_out_nchw_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b0_out_nchw_cluster_fork, &DeeployNetwork__b0_out_nchw_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1349976,
                                      .size = 39256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_output_0_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_output_0_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_output_0_ref) + 39256);

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_nhwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr;
} _b0_out_nchw_closure_L3_args_t;

static void _b0_out_nchw_closure_L3(void *_b0_out_nchw_closure_L3_args) {
  // CLOSURE ARG CAST
  _b0_out_nchw_closure_L3_args_t *args = (_b0_out_nchw_closure_L3_args_t *)_b0_out_nchw_closure_L3_args;

  int8_t *DeeployNetwork_y_nhwc_tensor = args->DeeployNetwork_y_nhwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b0_out_nchw_closure_args_t DeeployNetwork__b0_out_nchw_closure_args =
      (_b0_out_nchw_closure_args_t){.DeeployNetwork_y_nhwc_tensor = DeeployNetwork_y_nhwc_tensor,
                                    .DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr};

  // _b0_out_nchw_closure CLOSURE CALL
  _b0_out_nchw_closure(&DeeployNetwork__b0_out_nchw_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork_col_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_tensor;
  void *DeeployNetwork_b0_dual_wenc_buffer;
  void *DeeployNetwork_b0_dual_bias_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr;
    DeeployNetwork_weight_enc_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _b0_dual_wenc_closure_L3_args_t DeeployNetwork__b0_dual_wenc_closure_L3_args =
        (_b0_dual_wenc_closure_L3_args_t){.DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_wenc_tileIdxPtr};

    // _b0_dual_wenc_closure_L3 CLOSURE CALL
    _b0_dual_wenc_closure_L3(&DeeployNetwork__b0_dual_wenc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr;
    DeeployNetwork_add2_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 256);
    _b0_dual_bias_closure_L3_args_t DeeployNetwork__b0_dual_bias_closure_L3_args =
        (_b0_dual_bias_closure_L3_args_t){.DeeployNetwork_add2_tensor = DeeployNetwork_add2_tensor,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_dual_bias_tileIdxPtr};

    // _b0_dual_bias_closure_L3 CLOSURE CALL
    _b0_dual_bias_closure_L3(&DeeployNetwork__b0_dual_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr;
    DeeployNetwork_col_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 320);
    _b0_im2col_closure_L3_args_t DeeployNetwork__b0_im2col_closure_L3_args =
        (_b0_im2col_closure_L3_args_t){.DeeployNetwork_col_tensor = DeeployNetwork_col_tensor,
                                       .DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_im2col_tileIdxPtr};

    // _b0_im2col_closure_L3 CLOSURE CALL
    _b0_im2col_closure_L3(&DeeployNetwork__b0_im2col_closure_L3_args);
  }
  {

    // Reshape (Name: _NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshape, Op: Reshape)
    DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped = DeeployNetwork_col_tensor;
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
    DeeployNetwork_y_nhwc_tensor_Reshaped = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 157024);
    __MERGE_CONVRQ_PASS_0_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
        .DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
        .DeeployNetwork_add2_tensor = DeeployNetwork_add2_tensor,
        .DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped =
            DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_tensor_Reshaped,
        .DeeployNetwork_y_nhwc_tensor_Reshaped = DeeployNetwork_y_nhwc_tensor_Reshaped,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args);
  }
  {

    // Reshape (Name: _NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0y_nhwc_tensor_Reshaped_Reshape, Op: Reshape)
    DeeployNetwork_y_nhwc_tensor = DeeployNetwork_y_nhwc_tensor_Reshaped;
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr;
    _b0_out_nchw_closure_L3_args_t DeeployNetwork__b0_out_nchw_closure_L3_args =
        (_b0_out_nchw_closure_L3_args_t){.DeeployNetwork_y_nhwc_tensor = DeeployNetwork_y_nhwc_tensor,
                                         .DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_out_nchw_tileIdxPtr};

    // _b0_out_nchw_closure_L3 CLOSURE CALL
    _b0_out_nchw_closure_L3(&DeeployNetwork__b0_out_nchw_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 91824);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 314048);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 39576);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 320);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 352);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
