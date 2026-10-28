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
int8_t *DeeployNetwork_input_3;
int8_t *DeeployNetwork_output_0;
int8_t *DeeployNetwork_output_1;

static PI_L2 int32_t DeeployNetwork_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4p_dual_wenc[32] = {
    185997, 226785, 200920, 202444, 202823, 213378, 226168, 241806, 277784, 190200, 245924, 237923, 220822, 226733, 136034, 261893,
    225823, 238894, 213576, 255468, 216926, 210590, 245405, 206957, 286127, 233156, 236698, 206920, 278716, 190344, 195460, 232972};

static PI_L2 int32_t DeeployNetwork_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4p_dual_bias[32] = {
    185997, 226785, 200920, 202444, 202823, 213378, 226168, 241806, 277784, 190200, 245924, 237923, 220822, 226733, 136034, 261893,
    225823, 238894, 213576, 255468, 216926, 210590, 245405, 206957, 286127, 233156, 236698, 206920, 278716, 190344, 195460, 232972};

static PI_L2 int32_t DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b4p_dual_bias[32] = {
    4465028, 5444165, 4823268, 4859837, 4868948, 5122322, 5429352, 5804773, 6668456, 4565921, 5903610, 5711538, 5301021, 5442922, 3265610, 6286967,
    5421086, 5734863, 5127071, 6132738, 5207501, 5055397, 5891161, 4968182, 6868725, 5597122, 5682131, 4967301, 6690810, 4569368, 4692189, 5592699};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4p_dual_bias[32] = {
    178, 146, 165, 164, 163, 155, 146, 137, 119, 174, 135, 139, 150, 146, 243, 126,
    147, 139, 155, 130, 153, 157, 135, 160, 116, 142, 140, 160, 119, 174, 169, 142};

static PI_L2 int32_t DeeployNetwork_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4m_dual_wenc[32] = {
    185997, 226785, 200920, 202444, 202823, 213378, 226168, 241806, 277784, 190200, 245924, 237923, 220822, 226733, 136034, 261893,
    225823, 238894, 213576, 255468, 216926, 210590, 245405, 206957, 286127, 233156, 236698, 206920, 278716, 190344, 195460, 232972};

static PI_L2 int32_t DeeployNetwork_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4m_dual_bias[32] = {
    185997, 226785, 200920, 202444, 202823, 213378, 226168, 241806, 277784, 190200, 245924, 237923, 220822, 226733, 136034, 261893,
    225823, 238894, 213576, 255468, 216926, 210590, 245405, 206957, 286127, 233156, 236698, 206920, 278716, 190344, 195460, 232972};

static PI_L2 int32_t DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b4m_dual_bias[32] = {
    4465028, 5444165, 4823268, 4859837, 4868948, 5122322, 5429352, 5804773, 6668456, 4565921, 5903610, 5711538, 5301021, 5442922, 3265610, 6286967,
    5421086, 5734863, 5127071, 6132738, 5207501, 5055397, 5891161, 4968182, 6868725, 5597122, 5682131, 4967301, 6690810, 4569368, 4692189, 5592699};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4m_dual_bias[32] = {
    178, 146, 165, 164, 163, 155, 146, 137, 119, 174, 135, 139, 150, 146, 243, 126,
    147, 139, 155, 130, 153, 157, 135, 160, 116, 142, 140, 160, 119, 174, 169, 142};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4p_rqs[32] = {
    178, 146, 165, 164, 163, 155, 146, 137, 119, 174, 135, 139, 150, 146, 243, 126,
    147, 139, 155, 130, 153, 157, 135, 160, 116, 142, 140, 160, 119, 174, 169, 142};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4m_rqs[32] = {
    178, 146, 165, 164, 163, 155, 146, 137, 119, 174, 135, 139, 150, 146, 243, 126,
    147, 139, 155, 130, 153, 157, 135, 160, 116, 142, 140, 160, 119, 174, 169, 142};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_cumByteOffset[1] = {0};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo[2] = {3, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi[2] = {3, 2};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[2] = {0, 1344};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[2] = {1443136, 1442688};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size[2] = {1344, 896};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[2] = {0, 192};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[2] = {1310912, 1310848};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size[2] = {192, 128};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo[2] = {3, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi[2] = {3, 2};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[2] = {0, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_cumByteOffset[2] = {0, 1344};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_cmd[2] = {1443136, 1442688};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_size[2] = {1344, 896};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_cumByteOffset[2] = {0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_cumByteOffset[2] = {0, 192};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_cmd[2] = {1310912, 1310848};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_size[2] = {192, 128};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[4];
void *DeeployNetwork_outputs[2];
extern struct pi_device cluster_dev;
typedef struct {
  void *DeeployNetwork_b4p_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref;
} _b4p_dual_wenc_tiling_closure_args_t;

static void _b4p_dual_wenc_tiling_closure(void *_b4p_dual_wenc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4p_dual_wenc_tiling_closure_args_t *args = (_b4p_dual_wenc_tiling_closure_args_t *)_b4p_dual_wenc_tiling_closure_args;

  void *DeeployNetwork_b4p_dual_wenc_buffer = args->DeeployNetwork_b4p_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualPerturbEncode (Name: b4p_dual_wenc, Op: NE16DualPerturbEncode) -- W+/W- from one draw, encoded as one 2*cout 1x1 weight -- QW exp6
  {
    uint32_t b4p_dual_wenc_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 8);
    uint32_t b4p_dual_wenc_tile_xor = (0 * 0x9E3779B1u);
    float32_t b4p_dual_wenc_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualPerturbEncode((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref,
                               (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref,
                               (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref, (int8_t *)DeeployNetwork_b4p_dual_wenc_buffer,
                               b4p_dual_wenc_seed_w, b4p_dual_wenc_tile_xor, 15, b4p_dual_wenc_eps_scale, NUM_CORES, 32, 224, 1, 0, 32, 7, 1);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b4p_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref;
} _b4p_dual_wenc_cluster_fork_args_t;

static void _b4p_dual_wenc_cluster_fork(void *_b4p_dual_wenc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b4p_dual_wenc_cluster_fork_args_t *args = (_b4p_dual_wenc_cluster_fork_args_t *)_b4p_dual_wenc_cluster_fork_args;

  void *DeeployNetwork_b4p_dual_wenc_buffer = args->DeeployNetwork_b4p_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b4p_dual_wenc_tiling_closure_args_t DeeployNetwork__b4p_dual_wenc_tiling_closure_args = (_b4p_dual_wenc_tiling_closure_args_t){
      .DeeployNetwork_b4p_dual_wenc_buffer = DeeployNetwork_b4p_dual_wenc_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref};

  // _b4p_dual_wenc_tiling_closure CLOSURE CALL
  _b4p_dual_wenc_tiling_closure(&DeeployNetwork__b4p_dual_wenc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr;
} _b4p_dual_wenc_closure_args_t;

static void _b4p_dual_wenc_closure(void *_b4p_dual_wenc_closure_args) {
  // CLOSURE ARG CAST
  _b4p_dual_wenc_closure_args_t *args = (_b4p_dual_wenc_closure_args_t *)_b4p_dual_wenc_closure_args;

  uint8_t *DeeployNetwork_wenc_p_tensor = args->DeeployNetwork_wenc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14336);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 28672);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 21504);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4p_dual_wenc_ref =
      (void *)((char *)DeeployNetwork_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4p_dual_wenc +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wenc_p_tensor_ref =
      (void *)((char *)DeeployNetwork_wenc_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_b4p_dual_wenc_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1448960, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4p_dual_wenc_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b4p_dual_wenc_cluster_fork_args_t DeeployNetwork__b4p_dual_wenc_cluster_fork_args = (_b4p_dual_wenc_cluster_fork_args_t){
        .DeeployNetwork_b4p_dual_wenc_buffer = DeeployNetwork_b4p_dual_wenc_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b4p_dual_wenc_cluster_fork, &DeeployNetwork__b4p_dual_wenc_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1317888, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_wenc_p_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr;
} _b4p_dual_wenc_closure_L3_args_t;

static void _b4p_dual_wenc_closure_L3(void *_b4p_dual_wenc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4p_dual_wenc_closure_L3_args_t *args = (_b4p_dual_wenc_closure_L3_args_t *)_b4p_dual_wenc_closure_L3_args;

  uint8_t *DeeployNetwork_wenc_p_tensor = args->DeeployNetwork_wenc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4p_dual_wenc_closure_args_t DeeployNetwork__b4p_dual_wenc_closure_args =
      (_b4p_dual_wenc_closure_args_t){.DeeployNetwork_wenc_p_tensor = DeeployNetwork_wenc_p_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr};

  // _b4p_dual_wenc_closure CLOSURE CALL
  _b4p_dual_wenc_closure(&DeeployNetwork__b4p_dual_wenc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b4p_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref;
} _b4p_dual_bias_tiling_closure_args_t;

static void _b4p_dual_bias_tiling_closure(void *_b4p_dual_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4p_dual_bias_tiling_closure_args_t *args = (_b4p_dual_bias_tiling_closure_args_t *)_b4p_dual_bias_tiling_closure_args;

  void *DeeployNetwork_b4p_dual_bias_buffer = args->DeeployNetwork_b4p_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualBias (Name: b4p_dual_bias, Op: NE16DualBias) -- add+/add- with the signed-input correction, stacked -- QW exp6
  {
    uint32_t b4p_dual_bias_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 8);
    uint32_t b4p_dual_bias_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 9);
    uint32_t b4p_dual_bias_tile_xor = (0 * 0x9E3779B1u);
    float32_t b4p_dual_bias_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualBias(
        (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref, (int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref,
        (int8_t *)DeeployNetwork_b4p_dual_bias_buffer, b4p_dual_bias_seed_w, b4p_dual_bias_seed_b, b4p_dual_bias_tile_xor, 15, 16, b4p_dual_bias_eps_scale,
        NUM_CORES, 32, 224, 128, 1);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b4p_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref;
} _b4p_dual_bias_cluster_fork_args_t;

static void _b4p_dual_bias_cluster_fork(void *_b4p_dual_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b4p_dual_bias_cluster_fork_args_t *args = (_b4p_dual_bias_cluster_fork_args_t *)_b4p_dual_bias_cluster_fork_args;

  void *DeeployNetwork_b4p_dual_bias_buffer = args->DeeployNetwork_b4p_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b4p_dual_bias_tiling_closure_args_t DeeployNetwork__b4p_dual_bias_tiling_closure_args = (_b4p_dual_bias_tiling_closure_args_t){
      .DeeployNetwork_b4p_dual_bias_buffer = DeeployNetwork_b4p_dual_bias_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref};

  // _b4p_dual_bias_tiling_closure CLOSURE CALL
  _b4p_dual_bias_tiling_closure(&DeeployNetwork__b4p_dual_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr;
} _b4p_dual_bias_closure_args_t;

static void _b4p_dual_bias_closure(void *_b4p_dual_bias_closure_args) {
  // CLOSURE ARG CAST
  _b4p_dual_bias_closure_args_t *args = (_b4p_dual_bias_closure_args_t *)_b4p_dual_bias_closure_args;

  int32_t *DeeployNetwork_add_p_tensor = args->DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14592);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 21888);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 21760);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 22016);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 22144);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 22272);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4p_dual_bias_ref =
      (void *)((char *)DeeployNetwork_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4p_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_blocks4convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b4p_dual_bias_ref =
      (void *)((char *)DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b4p_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4p_dual_bias_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4p_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_p_tensor_ref =
      (void *)((char *)DeeployNetwork_add_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_b4p_dual_bias_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1448960, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4p_dual_bias_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_input_2_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_blocks4convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b4p_dual_bias_ref);
    mchan_transfer_1d(
        1441920, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4p_dual_bias_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b4p_dual_bias_cluster_fork_args_t DeeployNetwork__b4p_dual_bias_cluster_fork_args = (_b4p_dual_bias_cluster_fork_args_t){
        .DeeployNetwork_b4p_dual_bias_buffer = DeeployNetwork_b4p_dual_bias_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_bmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b4p_dual_bias_cluster_fork, &DeeployNetwork__b4p_dual_bias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_add_p_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr;
} _b4p_dual_bias_closure_L3_args_t;

static void _b4p_dual_bias_closure_L3(void *_b4p_dual_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4p_dual_bias_closure_L3_args_t *args = (_b4p_dual_bias_closure_L3_args_t *)_b4p_dual_bias_closure_L3_args;

  int32_t *DeeployNetwork_add_p_tensor = args->DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4p_dual_bias_closure_args_t DeeployNetwork__b4p_dual_bias_closure_args =
      (_b4p_dual_bias_closure_args_t){.DeeployNetwork_add_p_tensor = DeeployNetwork_add_p_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr};

  // _b4p_dual_bias_closure CLOSURE CALL
  _b4p_dual_bias_closure(&DeeployNetwork__b4p_dual_bias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref;
} _b4p_im2col_tiling_closure_args_t;

static void _b4p_im2col_tiling_closure(void *_b4p_im2col_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4p_im2col_tiling_closure_args_t *args = (_b4p_im2col_tiling_closure_args_t *)_b4p_im2col_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16Im2ColU8 (Name: b4p_im2col, Op: NE16Im2ColU8) -- column buffer for the NE16 im2col route -- QW exp6
  PULP_Im2ColU8_NCHW_to_HWC((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref,
                            (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref, NUM_CORES, 8, 5, 32, 2, 5, 7, 1, 0, 0, 0, 0, 1, 1,
                            (uint8_t)128, (uint8_t)128);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref;
} _b4p_im2col_cluster_fork_args_t;

static void _b4p_im2col_cluster_fork(void *_b4p_im2col_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b4p_im2col_cluster_fork_args_t *args = (_b4p_im2col_cluster_fork_args_t *)_b4p_im2col_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b4p_im2col_tiling_closure_args_t DeeployNetwork__b4p_im2col_tiling_closure_args =
      (_b4p_im2col_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref};

  // _b4p_im2col_tiling_closure CLOSURE CALL
  _b4p_im2col_tiling_closure(&DeeployNetwork__b4p_im2col_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_col_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr;
} _b4p_im2col_closure_args_t;

static void _b4p_im2col_closure(void *_b4p_im2col_closure_args) {
  // CLOSURE ARG CAST
  _b4p_im2col_closure_args_t *args = (_b4p_im2col_closure_args_t *)_b4p_im2col_closure_args;

  uint8_t *DeeployNetwork_col_p_tensor = args->DeeployNetwork_col_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2240);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_col_p_tensor_ref =
      (void *)((char *)DeeployNetwork_col_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443072, DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b4p_im2col_cluster_fork_args_t DeeployNetwork__b4p_im2col_cluster_fork_args =
        (_b4p_im2col_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_in_ref,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b4p_im2col_cluster_fork, &DeeployNetwork__b4p_im2col_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312960, DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_col_p_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_col_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr;
} _b4p_im2col_closure_L3_args_t;

static void _b4p_im2col_closure_L3(void *_b4p_im2col_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4p_im2col_closure_L3_args_t *args = (_b4p_im2col_closure_L3_args_t *)_b4p_im2col_closure_L3_args;

  uint8_t *DeeployNetwork_col_p_tensor = args->DeeployNetwork_col_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4p_im2col_closure_args_t DeeployNetwork__b4p_im2col_closure_args =
      (_b4p_im2col_closure_args_t){.DeeployNetwork_col_p_tensor = DeeployNetwork_col_p_tensor,
                                   .DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr};

  // _b4p_im2col_closure CLOSURE CALL
  _b4p_im2col_closure(&DeeployNetwork__b4p_im2col_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b4m_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref;
} _b4m_dual_wenc_tiling_closure_args_t;

static void _b4m_dual_wenc_tiling_closure(void *_b4m_dual_wenc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4m_dual_wenc_tiling_closure_args_t *args = (_b4m_dual_wenc_tiling_closure_args_t *)_b4m_dual_wenc_tiling_closure_args;

  void *DeeployNetwork_b4m_dual_wenc_buffer = args->DeeployNetwork_b4m_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualPerturbEncode (Name: b4m_dual_wenc, Op: NE16DualPerturbEncode) -- W+/W- from one draw, encoded as one 2*cout 1x1 weight -- QW exp6
  {
    uint32_t b4m_dual_wenc_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 8);
    uint32_t b4m_dual_wenc_tile_xor = (0 * 0x9E3779B1u);
    float32_t b4m_dual_wenc_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualPerturbEncode((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref,
                               (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref,
                               (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref, (int8_t *)DeeployNetwork_b4m_dual_wenc_buffer,
                               b4m_dual_wenc_seed_w, b4m_dual_wenc_tile_xor, 15, b4m_dual_wenc_eps_scale, NUM_CORES, 32, 224, 2, 0, 32, 7, 1);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b4m_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref;
} _b4m_dual_wenc_cluster_fork_args_t;

static void _b4m_dual_wenc_cluster_fork(void *_b4m_dual_wenc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b4m_dual_wenc_cluster_fork_args_t *args = (_b4m_dual_wenc_cluster_fork_args_t *)_b4m_dual_wenc_cluster_fork_args;

  void *DeeployNetwork_b4m_dual_wenc_buffer = args->DeeployNetwork_b4m_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b4m_dual_wenc_tiling_closure_args_t DeeployNetwork__b4m_dual_wenc_tiling_closure_args = (_b4m_dual_wenc_tiling_closure_args_t){
      .DeeployNetwork_b4m_dual_wenc_buffer = DeeployNetwork_b4m_dual_wenc_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref};

  // _b4m_dual_wenc_tiling_closure CLOSURE CALL
  _b4m_dual_wenc_tiling_closure(&DeeployNetwork__b4m_dual_wenc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr;
} _b4m_dual_wenc_closure_args_t;

static void _b4m_dual_wenc_closure(void *_b4m_dual_wenc_closure_args) {
  // CLOSURE ARG CAST
  _b4m_dual_wenc_closure_args_t *args = (_b4m_dual_wenc_closure_args_t *)_b4m_dual_wenc_closure_args;

  uint8_t *DeeployNetwork_wenc_m_tensor = args->DeeployNetwork_wenc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14336);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 28672);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 21504);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4m_dual_wenc_ref =
      (void *)((char *)DeeployNetwork_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4m_dual_wenc +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wenc_m_tensor_ref =
      (void *)((char *)DeeployNetwork_wenc_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_b4m_dual_wenc_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1448960, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4m_dual_wenc_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b4m_dual_wenc_cluster_fork_args_t DeeployNetwork__b4m_dual_wenc_cluster_fork_args = (_b4m_dual_wenc_cluster_fork_args_t){
        .DeeployNetwork_b4m_dual_wenc_buffer = DeeployNetwork_b4m_dual_wenc_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b4m_dual_wenc_cluster_fork, &DeeployNetwork__b4m_dual_wenc_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1317888, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_wenc_m_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr;
} _b4m_dual_wenc_closure_L3_args_t;

static void _b4m_dual_wenc_closure_L3(void *_b4m_dual_wenc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4m_dual_wenc_closure_L3_args_t *args = (_b4m_dual_wenc_closure_L3_args_t *)_b4m_dual_wenc_closure_L3_args;

  uint8_t *DeeployNetwork_wenc_m_tensor = args->DeeployNetwork_wenc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4m_dual_wenc_closure_args_t DeeployNetwork__b4m_dual_wenc_closure_args =
      (_b4m_dual_wenc_closure_args_t){.DeeployNetwork_wenc_m_tensor = DeeployNetwork_wenc_m_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr};

  // _b4m_dual_wenc_closure CLOSURE CALL
  _b4m_dual_wenc_closure(&DeeployNetwork__b4m_dual_wenc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b4m_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref;
} _b4m_dual_bias_tiling_closure_args_t;

static void _b4m_dual_bias_tiling_closure(void *_b4m_dual_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4m_dual_bias_tiling_closure_args_t *args = (_b4m_dual_bias_tiling_closure_args_t *)_b4m_dual_bias_tiling_closure_args;

  void *DeeployNetwork_b4m_dual_bias_buffer = args->DeeployNetwork_b4m_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualBias (Name: b4m_dual_bias, Op: NE16DualBias) -- add+/add- with the signed-input correction, stacked -- QW exp6
  {
    uint32_t b4m_dual_bias_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 8);
    uint32_t b4m_dual_bias_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 9);
    uint32_t b4m_dual_bias_tile_xor = (0 * 0x9E3779B1u);
    float32_t b4m_dual_bias_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualBias(
        (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref, (int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref,
        (int8_t *)DeeployNetwork_b4m_dual_bias_buffer, b4m_dual_bias_seed_w, b4m_dual_bias_seed_b, b4m_dual_bias_tile_xor, 15, 16, b4m_dual_bias_eps_scale,
        NUM_CORES, 32, 224, 128, 2);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b4m_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref;
} _b4m_dual_bias_cluster_fork_args_t;

static void _b4m_dual_bias_cluster_fork(void *_b4m_dual_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b4m_dual_bias_cluster_fork_args_t *args = (_b4m_dual_bias_cluster_fork_args_t *)_b4m_dual_bias_cluster_fork_args;

  void *DeeployNetwork_b4m_dual_bias_buffer = args->DeeployNetwork_b4m_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b4m_dual_bias_tiling_closure_args_t DeeployNetwork__b4m_dual_bias_tiling_closure_args = (_b4m_dual_bias_tiling_closure_args_t){
      .DeeployNetwork_b4m_dual_bias_buffer = DeeployNetwork_b4m_dual_bias_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref};

  // _b4m_dual_bias_tiling_closure CLOSURE CALL
  _b4m_dual_bias_tiling_closure(&DeeployNetwork__b4m_dual_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr;
} _b4m_dual_bias_closure_args_t;

static void _b4m_dual_bias_closure(void *_b4m_dual_bias_closure_args) {
  // CLOSURE ARG CAST
  _b4m_dual_bias_closure_args_t *args = (_b4m_dual_bias_closure_args_t *)_b4m_dual_bias_closure_args;

  int32_t *DeeployNetwork_add_m_tensor = args->DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14592);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 21888);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 21760);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 22016);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 22144);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 22272);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4m_dual_bias_ref =
      (void *)((char *)DeeployNetwork_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4m_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_blocks4convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b4m_dual_bias_ref =
      (void *)((char *)DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b4m_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4m_dual_bias_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4m_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_m_tensor_ref =
      (void *)((char *)DeeployNetwork_add_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_b4m_dual_bias_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1448960, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_blocks4convweight_int8_pmul_tensor_DUPLICATE_FOR_b4m_dual_bias_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_input_2_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_blocks4convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b4m_dual_bias_ref);
    mchan_transfer_1d(
        1441920, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4m_dual_bias_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b4m_dual_bias_cluster_fork_args_t DeeployNetwork__b4m_dual_bias_cluster_fork_args = (_b4m_dual_bias_cluster_fork_args_t){
        .DeeployNetwork_b4m_dual_bias_buffer = DeeployNetwork_b4m_dual_bias_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_bmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b4m_dual_bias_cluster_fork, &DeeployNetwork__b4m_dual_bias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_add_m_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr;
} _b4m_dual_bias_closure_L3_args_t;

static void _b4m_dual_bias_closure_L3(void *_b4m_dual_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4m_dual_bias_closure_L3_args_t *args = (_b4m_dual_bias_closure_L3_args_t *)_b4m_dual_bias_closure_L3_args;

  int32_t *DeeployNetwork_add_m_tensor = args->DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4m_dual_bias_closure_args_t DeeployNetwork__b4m_dual_bias_closure_args =
      (_b4m_dual_bias_closure_args_t){.DeeployNetwork_add_m_tensor = DeeployNetwork_add_m_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr};

  // _b4m_dual_bias_closure CLOSURE CALL
  _b4m_dual_bias_closure(&DeeployNetwork__b4m_dual_bias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref;
} _b4m_im2col_tiling_closure_args_t;

static void _b4m_im2col_tiling_closure(void *_b4m_im2col_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4m_im2col_tiling_closure_args_t *args = (_b4m_im2col_tiling_closure_args_t *)_b4m_im2col_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16Im2ColU8 (Name: b4m_im2col, Op: NE16Im2ColU8) -- column buffer for the NE16 im2col route -- QW exp6
  PULP_Im2ColU8_NCHW_to_HWC((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref,
                            (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref, NUM_CORES, 8, 5, 32, 2, 5, 7, 1, 0, 0, 0, 0, 1, 1,
                            (uint8_t)128, (uint8_t)128);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref;
} _b4m_im2col_cluster_fork_args_t;

static void _b4m_im2col_cluster_fork(void *_b4m_im2col_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b4m_im2col_cluster_fork_args_t *args = (_b4m_im2col_cluster_fork_args_t *)_b4m_im2col_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b4m_im2col_tiling_closure_args_t DeeployNetwork__b4m_im2col_tiling_closure_args =
      (_b4m_im2col_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref};

  // _b4m_im2col_tiling_closure CLOSURE CALL
  _b4m_im2col_tiling_closure(&DeeployNetwork__b4m_im2col_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_col_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr;
} _b4m_im2col_closure_args_t;

static void _b4m_im2col_closure(void *_b4m_im2col_closure_args) {
  // CLOSURE ARG CAST
  _b4m_im2col_closure_args_t *args = (_b4m_im2col_closure_args_t *)_b4m_im2col_closure_args;

  uint8_t *DeeployNetwork_col_m_tensor = args->DeeployNetwork_col_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2240);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_input_3_ref =
      (void *)((char *)DeeployNetwork_input_3 +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_col_m_tensor_ref =
      (void *)((char *)DeeployNetwork_col_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443072, DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_input_3_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b4m_im2col_cluster_fork_args_t DeeployNetwork__b4m_im2col_cluster_fork_args =
        (_b4m_im2col_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_in_ref,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b4m_im2col_cluster_fork, &DeeployNetwork__b4m_im2col_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312960, DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_col_m_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_col_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr;
} _b4m_im2col_closure_L3_args_t;

static void _b4m_im2col_closure_L3(void *_b4m_im2col_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4m_im2col_closure_L3_args_t *args = (_b4m_im2col_closure_L3_args_t *)_b4m_im2col_closure_L3_args;

  uint8_t *DeeployNetwork_col_m_tensor = args->DeeployNetwork_col_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4m_im2col_closure_args_t DeeployNetwork__b4m_im2col_closure_args =
      (_b4m_im2col_closure_args_t){.DeeployNetwork_col_m_tensor = DeeployNetwork_col_m_tensor,
                                   .DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr};

  // _b4m_im2col_closure CLOSURE CALL
  _b4m_im2col_closure(&DeeployNetwork__b4m_im2col_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref;
} __MERGE_CONVRQ_PASS_0_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_tiling_closure(void *__MERGE_CONVRQ_PASS_0_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_tiling_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_0_tiling_closure_args;

  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref;
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
              .input_stride = (ne16_stride_t){.d0 = 224, .d1 = 448, .d2 = 0},
              .output_stride = (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 32, .d2 = 64},
              task.data.cfg.weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 224, .d2 = 0},
              .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 14), .HoWo = nnx_concat_half(1, 1)},
                                          .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                        .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref, 2),
                                                        .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref, 2)}},
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
  uint8_t *DeeployNetwork_wenc_p_tensor;
  int32_t *DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_p_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_closure(void *__MERGE_CONVRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_args_t *)__MERGE_CONVRQ_PASS_0_closure_args;

  uint8_t *DeeployNetwork_wenc_p_tensor = args->DeeployNetwork_wenc_p_tensor;
  int32_t *DeeployNetwork_add_p_tensor = args->DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped =
      args->DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_p_tensor_Reshaped = args->DeeployNetwork_y_nhwc_p_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8832);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8704);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8512);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped_ref =
      (void *)((char *)DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_wenc_p_tensor_ref =
      (void *)((char *)DeeployNetwork_wenc_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4p_rqs_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4p_rqs +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_p_tensor_ref =
      (void *)((char *)DeeployNetwork_add_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_p_tensor_Reshaped_ref =
      (void *)((char *)DeeployNetwork_y_nhwc_p_tensor_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int add_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int weight_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[TILING_I],
          .size = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size[TILING_I],
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped_ref) + 1344);

    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1448960,
                                      .size = 7168,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_wenc_p_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441920,
          .size = 128,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4p_rqs_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_p_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo[TILING_I];

    __MERGE_CONVRQ_PASS_0_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_tiling_closure_args = (__MERGE_CONVRQ_PASS_0_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref};

    // __MERGE_CONVRQ_PASS_0_tiling_closure CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_tiling_closure_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_p_tensor_Reshaped_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_p_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_p_tensor_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_p_tensor_Reshaped_ref) + 192);

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc_p_tensor;
  int32_t *DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_p_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_closure_L3(void *__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)__MERGE_CONVRQ_PASS_0_closure_L3_args;

  uint8_t *DeeployNetwork_wenc_p_tensor = args->DeeployNetwork_wenc_p_tensor;
  int32_t *DeeployNetwork_add_p_tensor = args->DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped =
      args->DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_p_tensor_Reshaped = args->DeeployNetwork_y_nhwc_p_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args = (__MERGE_CONVRQ_PASS_0_closure_args_t){
      .DeeployNetwork_wenc_p_tensor = DeeployNetwork_wenc_p_tensor,
      .DeeployNetwork_add_p_tensor = DeeployNetwork_add_p_tensor,
      .DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped =
          DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped,
      .DeeployNetwork_y_nhwc_p_tensor_Reshaped = DeeployNetwork_y_nhwc_p_tensor_Reshaped,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;
} __MERGE_CONVRQ_PASS_1_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_1_tiling_closure(void *__MERGE_CONVRQ_PASS_1_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_tiling_closure_args_t *args = (__MERGE_CONVRQ_PASS_1_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_1_tiling_closure_args;

  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;

  // CLOSURE FUNCTION CALL

  // N-EUREKA Task Init
  ne16_task_t task = {
      .data = (ne16_task_data_t){
          .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref - 0 + 0,
          .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref - 0,
          .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref,
          .scale_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
          .scale_shift_addr = (uint32_t)NULL,
          .scale_bias_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
          .cfg = (ne16_cfg_t){
              .input_stride = (ne16_stride_t){.d0 = 224, .d1 = 448, .d2 = 0},
              .output_stride = (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 32, .d2 = 64},
              task.data.cfg.weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 224, .d2 = 0},
              .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 14), .HoWo = nnx_concat_half(1, 1)},
                                          .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                        .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref, 2),
                                                        .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref, 2)}},
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
  uint8_t *DeeployNetwork_wenc_m_tensor;
  int32_t *DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_m_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
} __MERGE_CONVRQ_PASS_1_closure_args_t;

static void __MERGE_CONVRQ_PASS_1_closure(void *__MERGE_CONVRQ_PASS_1_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_closure_args_t *args = (__MERGE_CONVRQ_PASS_1_closure_args_t *)__MERGE_CONVRQ_PASS_1_closure_args;

  uint8_t *DeeployNetwork_wenc_m_tensor = args->DeeployNetwork_wenc_m_tensor;
  int32_t *DeeployNetwork_add_m_tensor = args->DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped =
      args->DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_m_tensor_Reshaped = args->DeeployNetwork_y_nhwc_m_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8832);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8704);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8512);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped_ref =
      (void *)((char *)DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_wenc_m_tensor_ref =
      (void *)((char *)DeeployNetwork_wenc_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4m_rqs_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4m_rqs +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_m_tensor_ref =
      (void *)((char *)DeeployNetwork_add_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_y_nhwc_m_tensor_Reshaped_ref =
      (void *)((char *)DeeployNetwork_y_nhwc_m_tensor_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int add_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int weight_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_cmd[TILING_I],
          .size = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_size[TILING_I],
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped_ref) + 1344);

    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1448960,
                                      .size = 7168,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_wenc_m_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441920,
          .size = 128,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b4m_rqs_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_m_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo[TILING_I];

    __MERGE_CONVRQ_PASS_1_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_1_tiling_closure_args = (__MERGE_CONVRQ_PASS_1_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref};

    // __MERGE_CONVRQ_PASS_1_tiling_closure CLOSURE CALL
    __MERGE_CONVRQ_PASS_1_tiling_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_1_tiling_closure_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_y_nhwc_m_tensor_Reshaped_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_y_nhwc_m_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_y_nhwc_m_tensor_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_y_nhwc_m_tensor_Reshaped_ref) + 192);

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc_m_tensor;
  int32_t *DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_m_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
} __MERGE_CONVRQ_PASS_1_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_1_closure_L3(void *__MERGE_CONVRQ_PASS_1_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_1_closure_L3_args_t *)__MERGE_CONVRQ_PASS_1_closure_L3_args;

  uint8_t *DeeployNetwork_wenc_m_tensor = args->DeeployNetwork_wenc_m_tensor;
  int32_t *DeeployNetwork_add_m_tensor = args->DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped =
      args->DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_m_tensor_Reshaped = args->DeeployNetwork_y_nhwc_m_tensor_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_1_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_1_closure_args = (__MERGE_CONVRQ_PASS_1_closure_args_t){
      .DeeployNetwork_wenc_m_tensor = DeeployNetwork_wenc_m_tensor,
      .DeeployNetwork_add_m_tensor = DeeployNetwork_add_m_tensor,
      .DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped =
          DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped,
      .DeeployNetwork_y_nhwc_m_tensor_Reshaped = DeeployNetwork_y_nhwc_m_tensor_Reshaped,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_1_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_1_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_1_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref;
} _b4p_out_nchw_tiling_closure_args_t;

static void _b4p_out_nchw_tiling_closure(void *_b4p_out_nchw_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4p_out_nchw_tiling_closure_args_t *args = (_b4p_out_nchw_tiling_closure_args_t *)_b4p_out_nchw_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 2, 5, 32] -> [1, 32, 2, 5] (Name: b4p_out_nchw, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 2;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 32;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_3][dimLen_1][dimLen_2])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref;
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

    const uint32_t baseChunk_1 = dimLen_3 / NUM_CORES;
    const uint32_t leftover_1 = dimLen_3 - baseChunk_1 * NUM_CORES;
    const uint32_t offset_1 = baseChunk_1 * coreId + (coreId < leftover_1 ? coreId : leftover_1);
    const uint32_t chunk_1 = coreId < leftover_1 ? baseChunk_1 + 1 : baseChunk_1;
    for (uint32_t i_3 = offset_1; i_3 < offset_1 + chunk_1; i_3++) {
      const uint32_t offS_1 = offS_0 + i_3 * sIn_3;
      const uint32_t offD_1 = offD_0 + i_3 * sOut_1;

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {
        const uint32_t offS_2 = offS_1 + i_1 * sIn_1;
        const uint32_t offD_2 = offD_1 + i_1 * sOut_2;

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref;
} _b4p_out_nchw_cluster_fork_args_t;

static void _b4p_out_nchw_cluster_fork(void *_b4p_out_nchw_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b4p_out_nchw_cluster_fork_args_t *args = (_b4p_out_nchw_cluster_fork_args_t *)_b4p_out_nchw_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b4p_out_nchw_tiling_closure_args_t DeeployNetwork__b4p_out_nchw_tiling_closure_args = (_b4p_out_nchw_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref};

  // _b4p_out_nchw_tiling_closure CLOSURE CALL
  _b4p_out_nchw_tiling_closure(&DeeployNetwork__b4p_out_nchw_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_nhwc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr;
} _b4p_out_nchw_closure_args_t;

static void _b4p_out_nchw_closure(void *_b4p_out_nchw_closure_args) {
  // CLOSURE ARG CAST
  _b4p_out_nchw_closure_args_t *args = (_b4p_out_nchw_closure_args_t *)_b4p_out_nchw_closure_args;

  int8_t *DeeployNetwork_y_nhwc_p_tensor = args->DeeployNetwork_y_nhwc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 320);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_y_nhwc_p_tensor_ref =
      (void *)((char *)DeeployNetwork_y_nhwc_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442112,
                                      .size = 320,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_y_nhwc_p_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _b4p_out_nchw_cluster_fork_args_t DeeployNetwork__b4p_out_nchw_cluster_fork_args = (_b4p_out_nchw_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b4p_out_nchw_cluster_fork, &DeeployNetwork__b4p_out_nchw_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1311040,
                                      .size = 320,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_nhwc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr;
} _b4p_out_nchw_closure_L3_args_t;

static void _b4p_out_nchw_closure_L3(void *_b4p_out_nchw_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4p_out_nchw_closure_L3_args_t *args = (_b4p_out_nchw_closure_L3_args_t *)_b4p_out_nchw_closure_L3_args;

  int8_t *DeeployNetwork_y_nhwc_p_tensor = args->DeeployNetwork_y_nhwc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4p_out_nchw_closure_args_t DeeployNetwork__b4p_out_nchw_closure_args =
      (_b4p_out_nchw_closure_args_t){.DeeployNetwork_y_nhwc_p_tensor = DeeployNetwork_y_nhwc_p_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr};

  // _b4p_out_nchw_closure CLOSURE CALL
  _b4p_out_nchw_closure(&DeeployNetwork__b4p_out_nchw_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref;
} _b4m_out_nchw_tiling_closure_args_t;

static void _b4m_out_nchw_tiling_closure(void *_b4m_out_nchw_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b4m_out_nchw_tiling_closure_args_t *args = (_b4m_out_nchw_tiling_closure_args_t *)_b4m_out_nchw_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 2, 5, 32] -> [1, 32, 2, 5] (Name: b4m_out_nchw, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 2;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 32;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_3][dimLen_1][dimLen_2])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref;
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

    const uint32_t baseChunk_1 = dimLen_3 / NUM_CORES;
    const uint32_t leftover_1 = dimLen_3 - baseChunk_1 * NUM_CORES;
    const uint32_t offset_1 = baseChunk_1 * coreId + (coreId < leftover_1 ? coreId : leftover_1);
    const uint32_t chunk_1 = coreId < leftover_1 ? baseChunk_1 + 1 : baseChunk_1;
    for (uint32_t i_3 = offset_1; i_3 < offset_1 + chunk_1; i_3++) {
      const uint32_t offS_1 = offS_0 + i_3 * sIn_3;
      const uint32_t offD_1 = offD_0 + i_3 * sOut_1;

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {
        const uint32_t offS_2 = offS_1 + i_1 * sIn_1;
        const uint32_t offD_2 = offD_1 + i_1 * sOut_2;

        for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref;
} _b4m_out_nchw_cluster_fork_args_t;

static void _b4m_out_nchw_cluster_fork(void *_b4m_out_nchw_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b4m_out_nchw_cluster_fork_args_t *args = (_b4m_out_nchw_cluster_fork_args_t *)_b4m_out_nchw_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b4m_out_nchw_tiling_closure_args_t DeeployNetwork__b4m_out_nchw_tiling_closure_args = (_b4m_out_nchw_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref};

  // _b4m_out_nchw_tiling_closure CLOSURE CALL
  _b4m_out_nchw_tiling_closure(&DeeployNetwork__b4m_out_nchw_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_nhwc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr;
} _b4m_out_nchw_closure_args_t;

static void _b4m_out_nchw_closure(void *_b4m_out_nchw_closure_args) {
  // CLOSURE ARG CAST
  _b4m_out_nchw_closure_args_t *args = (_b4m_out_nchw_closure_args_t *)_b4m_out_nchw_closure_args;

  int8_t *DeeployNetwork_y_nhwc_m_tensor = args->DeeployNetwork_y_nhwc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 320);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_y_nhwc_m_tensor_ref =
      (void *)((char *)DeeployNetwork_y_nhwc_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_output_1_ref =
      (void *)((char *)DeeployNetwork_output_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442112,
                                      .size = 320,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_y_nhwc_m_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _b4m_out_nchw_cluster_fork_args_t DeeployNetwork__b4m_out_nchw_cluster_fork_args = (_b4m_out_nchw_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b4m_out_nchw_cluster_fork, &DeeployNetwork__b4m_out_nchw_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1311040,
                                      .size = 320,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_output_1_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_nhwc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr;
} _b4m_out_nchw_closure_L3_args_t;

static void _b4m_out_nchw_closure_L3(void *_b4m_out_nchw_closure_L3_args) {
  // CLOSURE ARG CAST
  _b4m_out_nchw_closure_L3_args_t *args = (_b4m_out_nchw_closure_L3_args_t *)_b4m_out_nchw_closure_L3_args;

  int8_t *DeeployNetwork_y_nhwc_m_tensor = args->DeeployNetwork_y_nhwc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b4m_out_nchw_closure_args_t DeeployNetwork__b4m_out_nchw_closure_args =
      (_b4m_out_nchw_closure_args_t){.DeeployNetwork_y_nhwc_m_tensor = DeeployNetwork_y_nhwc_m_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr};

  // _b4m_out_nchw_closure CLOSURE CALL
  _b4m_out_nchw_closure(&DeeployNetwork__b4m_out_nchw_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  uint8_t *DeeployNetwork_wenc_p_tensor;
  int32_t *DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork_col_p_tensor;
  uint8_t *DeeployNetwork_wenc_m_tensor;
  int32_t *DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork_col_m_tensor;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_p_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_m_tensor_Reshaped;
  int8_t *DeeployNetwork_y_nhwc_p_tensor;
  int8_t *DeeployNetwork_y_nhwc_m_tensor;
  void *DeeployNetwork_b4p_dual_wenc_buffer;
  void *DeeployNetwork_b4p_dual_bias_buffer;
  void *DeeployNetwork_b4m_dual_wenc_buffer;
  void *DeeployNetwork_b4m_dual_bias_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr;
    DeeployNetwork_wenc_p_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _b4p_dual_wenc_closure_L3_args_t DeeployNetwork__b4p_dual_wenc_closure_L3_args = (_b4p_dual_wenc_closure_L3_args_t){
        .DeeployNetwork_wenc_p_tensor = DeeployNetwork_wenc_p_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_wenc_tileIdxPtr};

    // _b4p_dual_wenc_closure_L3 CLOSURE CALL
    _b4p_dual_wenc_closure_L3(&DeeployNetwork__b4p_dual_wenc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr;
    DeeployNetwork_add_p_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 7168);
    _b4p_dual_bias_closure_L3_args_t DeeployNetwork__b4p_dual_bias_closure_L3_args = (_b4p_dual_bias_closure_L3_args_t){
        .DeeployNetwork_add_p_tensor = DeeployNetwork_add_p_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4p_dual_bias_tileIdxPtr};

    // _b4p_dual_bias_closure_L3 CLOSURE CALL
    _b4p_dual_bias_closure_L3(&DeeployNetwork__b4p_dual_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr;
    DeeployNetwork_col_p_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 14464);
    _b4p_im2col_closure_L3_args_t DeeployNetwork__b4p_im2col_closure_L3_args =
        (_b4p_im2col_closure_L3_args_t){.DeeployNetwork_col_p_tensor = DeeployNetwork_col_p_tensor,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4p_im2col_tileIdxPtr};

    // _b4p_im2col_closure_L3 CLOSURE CALL
    _b4p_im2col_closure_L3(&DeeployNetwork__b4p_im2col_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr;
    DeeployNetwork_wenc_m_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 7296);
    _b4m_dual_wenc_closure_L3_args_t DeeployNetwork__b4m_dual_wenc_closure_L3_args = (_b4m_dual_wenc_closure_L3_args_t){
        .DeeployNetwork_wenc_m_tensor = DeeployNetwork_wenc_m_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_wenc_tileIdxPtr};

    // _b4m_dual_wenc_closure_L3 CLOSURE CALL
    _b4m_dual_wenc_closure_L3(&DeeployNetwork__b4m_dual_wenc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr;
    DeeployNetwork_add_m_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 17984);
    _b4m_dual_bias_closure_L3_args_t DeeployNetwork__b4m_dual_bias_closure_L3_args = (_b4m_dual_bias_closure_L3_args_t){
        .DeeployNetwork_add_m_tensor = DeeployNetwork_add_m_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4m_dual_bias_tileIdxPtr};

    // _b4m_dual_bias_closure_L3 CLOSURE CALL
    _b4m_dual_bias_closure_L3(&DeeployNetwork__b4m_dual_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr;
    DeeployNetwork_col_m_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 18112);
    _b4m_im2col_closure_L3_args_t DeeployNetwork__b4m_im2col_closure_L3_args =
        (_b4m_im2col_closure_L3_args_t){.DeeployNetwork_col_m_tensor = DeeployNetwork_col_m_tensor,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4m_im2col_tileIdxPtr};

    // _b4m_im2col_closure_L3 CLOSURE CALL
    _b4m_im2col_closure_L3(&DeeployNetwork__b4m_im2col_closure_L3_args);
  }
  {

    // Reshape (Name: _NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshape, Op: Reshape)
    DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped = DeeployNetwork_col_p_tensor;
  }
  {

    // Reshape (Name: _NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshape, Op: Reshape)
    DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped = DeeployNetwork_col_m_tensor;
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
    DeeployNetwork_y_nhwc_p_tensor_Reshaped = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 16704);
    __MERGE_CONVRQ_PASS_0_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
        .DeeployNetwork_wenc_p_tensor = DeeployNetwork_wenc_p_tensor,
        .DeeployNetwork_add_p_tensor = DeeployNetwork_add_p_tensor,
        .DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped =
            DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped,
        .DeeployNetwork_y_nhwc_p_tensor_Reshaped = DeeployNetwork_y_nhwc_p_tensor_Reshaped,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr;
    DeeployNetwork_y_nhwc_m_tensor_Reshaped = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    __MERGE_CONVRQ_PASS_1_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_1_closure_L3_args = (__MERGE_CONVRQ_PASS_1_closure_L3_args_t){
        .DeeployNetwork_wenc_m_tensor = DeeployNetwork_wenc_m_tensor,
        .DeeployNetwork_add_m_tensor = DeeployNetwork_add_m_tensor,
        .DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped =
            DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped,
        .DeeployNetwork_y_nhwc_m_tensor_Reshaped = DeeployNetwork_y_nhwc_m_tensor_Reshaped,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_1_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_1_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_1_closure_L3_args);
  }
  {

    // Reshape (Name: _NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0y_nhwc_p_tensor_Reshaped_Reshape, Op: Reshape)
    DeeployNetwork_y_nhwc_p_tensor = DeeployNetwork_y_nhwc_p_tensor_Reshaped;
  }
  {

    // Reshape (Name: _NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1y_nhwc_m_tensor_Reshaped_Reshape, Op: Reshape)
    DeeployNetwork_y_nhwc_m_tensor = DeeployNetwork_y_nhwc_m_tensor_Reshaped;
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr;
    _b4p_out_nchw_closure_L3_args_t DeeployNetwork__b4p_out_nchw_closure_L3_args =
        (_b4p_out_nchw_closure_L3_args_t){.DeeployNetwork_y_nhwc_p_tensor = DeeployNetwork_y_nhwc_p_tensor,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4p_out_nchw_tileIdxPtr};

    // _b4p_out_nchw_closure_L3 CLOSURE CALL
    _b4p_out_nchw_closure_L3(&DeeployNetwork__b4p_out_nchw_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr;
    _b4m_out_nchw_closure_L3_args_t DeeployNetwork__b4m_out_nchw_closure_L3_args =
        (_b4m_out_nchw_closure_L3_args_t){.DeeployNetwork_y_nhwc_m_tensor = DeeployNetwork_y_nhwc_m_tensor,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b4m_out_nchw_tileIdxPtr};

    // _b4m_out_nchw_closure_L3 CLOSURE CALL
    _b4m_out_nchw_closure_L3(&DeeployNetwork__b4m_out_nchw_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 28800);
  if (DeeployNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 28800));
    return;
  }

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 25408);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 7296);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 18112);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 25280);
  DeeployNetwork_input_3 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 16704);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 320);
  DeeployNetwork_output_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 640);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_inputs[3] = (void *)DeeployNetwork_input_3;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
  DeeployNetwork_outputs[1] = (void *)DeeployNetwork_output_1;
}
