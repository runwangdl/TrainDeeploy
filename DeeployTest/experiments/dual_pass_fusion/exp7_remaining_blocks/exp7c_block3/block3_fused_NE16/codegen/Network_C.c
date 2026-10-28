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

static PI_L2 int32_t DeeployNetwork_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3p_dual_wenc[32] = {
    243041, 237342, 192477, 173475, 204002, 252155, 169682, 109053, 183913, 177070, 189471, 279557, 176599, 192092, 170877, 253092,
    212962, 224469, 200686, 193446, 159675, 161818, 229571, 154010, 265026, 247421, 219725, 150261, 247547, 116124, 278892, 219721};

static PI_L2 int32_t DeeployNetwork_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3p_dual_bias[32] = {
    243041, 237342, 192477, 173475, 204002, 252155, 169682, 109053, 183913, 177070, 189471, 279557, 176599, 192092, 170877, 253092,
    212962, 224469, 200686, 193446, 159675, 161818, 229571, 154010, 265026, 247421, 219725, 150261, 247547, 116124, 278892, 219721};

static PI_L2 int32_t DeeployNetwork_blocks3convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b3p_dual_bias[32] = {
    4543569, 4437028, 3598282, 3243057, 3813753, 4713948, 3172148, 2038716, 3438188, 3310265, 3542102, 5226218, 3301465, 3591097, 3194483, 4731471,
    3981257, 4196366, 3751757, 3616413, 2985078, 3025126, 4291753, 2879164, 4954569, 4625459, 4107688, 2809070, 4627800, 2170897, 5213795, 4107601};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3p_dual_bias[32] = {
    57, 58, 72, 80, 68, 55, 81, 127, 75, 78, 73, 49, 78, 72, 81, 55, 65, 61, 69, 71, 86, 85, 60, 90, 52, 56, 63, 92, 56, 119, 49, 63};

static PI_L2 int32_t DeeployNetwork_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3m_dual_wenc[32] = {
    243041, 237342, 192477, 173475, 204002, 252155, 169682, 109053, 183913, 177070, 189471, 279557, 176599, 192092, 170877, 253092,
    212962, 224469, 200686, 193446, 159675, 161818, 229571, 154010, 265026, 247421, 219725, 150261, 247547, 116124, 278892, 219721};

static PI_L2 int32_t DeeployNetwork_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3m_dual_bias[32] = {
    243041, 237342, 192477, 173475, 204002, 252155, 169682, 109053, 183913, 177070, 189471, 279557, 176599, 192092, 170877, 253092,
    212962, 224469, 200686, 193446, 159675, 161818, 229571, 154010, 265026, 247421, 219725, 150261, 247547, 116124, 278892, 219721};

static PI_L2 int32_t DeeployNetwork_blocks3convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b3m_dual_bias[32] = {
    4543569, 4437028, 3598282, 3243057, 3813753, 4713948, 3172148, 2038716, 3438188, 3310265, 3542102, 5226218, 3301465, 3591097, 3194483, 4731471,
    3981257, 4196366, 3751757, 3616413, 2985078, 3025126, 4291753, 2879164, 4954569, 4625459, 4107688, 2809070, 4627800, 2170897, 5213795, 4107601};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3m_dual_bias[32] = {
    57, 58, 72, 80, 68, 55, 81, 127, 75, 78, 73, 49, 78, 72, 81, 55, 65, 61, 69, 71, 86, 85, 60, 90, 52, 56, 63, 92, 56, 119, 49, 63};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3p_rqs[32] = {
    57, 58, 72, 80, 68, 55, 81, 127, 75, 78, 73, 49, 78, 72, 81, 55, 65, 61, 69, 71, 86, 85, 60, 90, 52, 56, 63, 92, 56, 119, 49, 63};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3m_rqs[32] = {
    57, 58, 72, 80, 68, 55, 81, 127, 75, 78, 73, 49, 78, 72, 81, 55, 65, 61, 69, 71, 86, 85, 60, 90, 52, 56, 63, 92, 56, 119, 49, 63};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_cumByteOffset[1] = {0};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride[4] = {336, 336, 224, 224};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride[4] = {96, 96, 64, 64};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo[4] = {2, 1, 2, 1};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo[4] = {3, 2, 3, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo[4] = {3, 3, 2, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi[4] = {3, 2, 3, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi[4] = {3, 3, 2, 2};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 4};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[4] = {0, 3360, 336, 3696};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[4] = {1968096, 1966752, 1967424, 1966528};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size[4] = {2016, 672, 1344, 448};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size_1d[4] = {336, 336, 224, 224};

static PI_L2 int16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_relativeOffset[4] = {3360, -3024, 3360, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[4] = {0, 960, 96, 1056};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[4] = {1835584, 1835200, 1835392, 1835136};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size[4] = {576, 192, 384, 128};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size_1d[4] = {96, 96, 64, 64};

static PI_L2 int16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_relativeOffset[4] = {960, -864, 960, 0};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride[4] = {336, 336, 224, 224};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride[4] = {96, 96, 64, 64};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo[4] = {2, 1, 2, 1};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo[4] = {3, 2, 3, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo[4] = {3, 3, 2, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi[4] = {3, 2, 3, 2};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi[4] = {3, 3, 2, 2};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_numTiles[2] = {0, 4};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_cumByteOffset[4] = {0, 3360, 336, 3696};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_cmd[4] = {1968096, 1966752, 1967424, 1966528};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_size[4] = {2016, 672, 1344, 448};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_size_1d[4] = {336, 336, 224, 224};

static PI_L2 int16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_relativeOffset[4] = {3360, -3024, 3360, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_cumByteOffset[4] = {0, 960, 96, 1056};

static PI_L2 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_cmd[4] = {1835584, 1835200, 1835392, 1835136};

static PI_L2 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_size[4] = {576, 192, 384, 128};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_size_1d[4] = {96, 96, 64, 64};

static PI_L2 int16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_relativeOffset[4] = {960, -864, 960, 0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[4];
void *DeeployNetwork_outputs[2];
extern struct pi_device cluster_dev;
typedef struct {
  void *DeeployNetwork_b3p_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref;
} _b3p_dual_wenc_tiling_closure_args_t;

static void _b3p_dual_wenc_tiling_closure(void *_b3p_dual_wenc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3p_dual_wenc_tiling_closure_args_t *args = (_b3p_dual_wenc_tiling_closure_args_t *)_b3p_dual_wenc_tiling_closure_args;

  void *DeeployNetwork_b3p_dual_wenc_buffer = args->DeeployNetwork_b3p_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualPerturbEncode (Name: b3p_dual_wenc, Op: NE16DualPerturbEncode) -- W+/W- from one draw, encoded as one 2*cout 1x1 weight -- QW exp6
  {
    uint32_t b3p_dual_wenc_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 6);
    uint32_t b3p_dual_wenc_tile_xor = (0 * 0x9E3779B1u);
    float32_t b3p_dual_wenc_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualPerturbEncode((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref,
                               (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref,
                               (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref, (int8_t *)DeeployNetwork_b3p_dual_wenc_buffer,
                               b3p_dual_wenc_seed_w, b3p_dual_wenc_tile_xor, 15, b3p_dual_wenc_eps_scale, NUM_CORES, 32, 112, 1, 0, 16, 7, 1);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b3p_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref;
} _b3p_dual_wenc_cluster_fork_args_t;

static void _b3p_dual_wenc_cluster_fork(void *_b3p_dual_wenc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b3p_dual_wenc_cluster_fork_args_t *args = (_b3p_dual_wenc_cluster_fork_args_t *)_b3p_dual_wenc_cluster_fork_args;

  void *DeeployNetwork_b3p_dual_wenc_buffer = args->DeeployNetwork_b3p_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b3p_dual_wenc_tiling_closure_args_t DeeployNetwork__b3p_dual_wenc_tiling_closure_args = (_b3p_dual_wenc_tiling_closure_args_t){
      .DeeployNetwork_b3p_dual_wenc_buffer = DeeployNetwork_b3p_dual_wenc_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref};

  // _b3p_dual_wenc_tiling_closure CLOSURE CALL
  _b3p_dual_wenc_tiling_closure(&DeeployNetwork__b3p_dual_wenc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr;
} _b3p_dual_wenc_closure_args_t;

static void _b3p_dual_wenc_closure(void *_b3p_dual_wenc_closure_args) {
  // CLOSURE ARG CAST
  _b3p_dual_wenc_closure_args_t *args = (_b3p_dual_wenc_closure_args_t *)_b3p_dual_wenc_closure_args;

  uint8_t *DeeployNetwork_wenc_p_tensor = args->DeeployNetwork_wenc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14336);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10752);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3p_dual_wenc_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3p_dual_wenc +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wenc_p_tensor_ref =
      (void *)((char *)DeeployNetwork_wenc_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_b3p_dual_wenc_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445376, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3p_dual_wenc_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b3p_dual_wenc_cluster_fork_args_t DeeployNetwork__b3p_dual_wenc_cluster_fork_args = (_b3p_dual_wenc_cluster_fork_args_t){
        .DeeployNetwork_b3p_dual_wenc_buffer = DeeployNetwork_b3p_dual_wenc_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b3p_dual_wenc_cluster_fork, &DeeployNetwork__b3p_dual_wenc_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314304, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_wenc_p_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr;
} _b3p_dual_wenc_closure_L3_args_t;

static void _b3p_dual_wenc_closure_L3(void *_b3p_dual_wenc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3p_dual_wenc_closure_L3_args_t *args = (_b3p_dual_wenc_closure_L3_args_t *)_b3p_dual_wenc_closure_L3_args;

  uint8_t *DeeployNetwork_wenc_p_tensor = args->DeeployNetwork_wenc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3p_dual_wenc_closure_args_t DeeployNetwork__b3p_dual_wenc_closure_args =
      (_b3p_dual_wenc_closure_args_t){.DeeployNetwork_wenc_p_tensor = DeeployNetwork_wenc_p_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr};

  // _b3p_dual_wenc_closure CLOSURE CALL
  _b3p_dual_wenc_closure(&DeeployNetwork__b3p_dual_wenc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b3p_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref;
} _b3p_dual_bias_tiling_closure_args_t;

static void _b3p_dual_bias_tiling_closure(void *_b3p_dual_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3p_dual_bias_tiling_closure_args_t *args = (_b3p_dual_bias_tiling_closure_args_t *)_b3p_dual_bias_tiling_closure_args;

  void *DeeployNetwork_b3p_dual_bias_buffer = args->DeeployNetwork_b3p_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualBias (Name: b3p_dual_bias, Op: NE16DualBias) -- add+/add- with the signed-input correction, stacked -- QW exp6
  {
    uint32_t b3p_dual_bias_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 6);
    uint32_t b3p_dual_bias_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 7);
    uint32_t b3p_dual_bias_tile_xor = (0 * 0x9E3779B1u);
    float32_t b3p_dual_bias_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualBias(
        (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref, (int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref,
        (int8_t *)DeeployNetwork_b3p_dual_bias_buffer, b3p_dual_bias_seed_w, b3p_dual_bias_seed_b, b3p_dual_bias_tile_xor, 15, 16, b3p_dual_bias_eps_scale,
        NUM_CORES, 32, 112, 128, 1);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b3p_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref;
} _b3p_dual_bias_cluster_fork_args_t;

static void _b3p_dual_bias_cluster_fork(void *_b3p_dual_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b3p_dual_bias_cluster_fork_args_t *args = (_b3p_dual_bias_cluster_fork_args_t *)_b3p_dual_bias_cluster_fork_args;

  void *DeeployNetwork_b3p_dual_bias_buffer = args->DeeployNetwork_b3p_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b3p_dual_bias_tiling_closure_args_t DeeployNetwork__b3p_dual_bias_tiling_closure_args = (_b3p_dual_bias_tiling_closure_args_t){
      .DeeployNetwork_b3p_dual_bias_buffer = DeeployNetwork_b3p_dual_bias_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref};

  // _b3p_dual_bias_tiling_closure CLOSURE CALL
  _b3p_dual_bias_tiling_closure(&DeeployNetwork__b3p_dual_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr;
} _b3p_dual_bias_closure_args_t;

static void _b3p_dual_bias_closure(void *_b3p_dual_bias_closure_args) {
  // CLOSURE ARG CAST
  _b3p_dual_bias_closure_args_t *args = (_b3p_dual_bias_closure_args_t *)_b3p_dual_bias_closure_args;

  int32_t *DeeployNetwork_add_p_tensor = args->DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7424);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 11136);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 11008);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 11264);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 11392);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 11520);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3p_dual_bias_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3p_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_blocks3convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b3p_dual_bias_ref =
      (void *)((char *)DeeployNetwork_blocks3convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b3p_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3p_dual_bias_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3p_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_p_tensor_ref =
      (void *)((char *)DeeployNetwork_add_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_b3p_dual_bias_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445376, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3p_dual_bias_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_input_2_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_blocks3convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b3p_dual_bias_ref);
    mchan_transfer_1d(
        1441920, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3p_dual_bias_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b3p_dual_bias_cluster_fork_args_t DeeployNetwork__b3p_dual_bias_cluster_fork_args = (_b3p_dual_bias_cluster_fork_args_t){
        .DeeployNetwork_b3p_dual_bias_buffer = DeeployNetwork_b3p_dual_bias_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_bmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b3p_dual_bias_cluster_fork, &DeeployNetwork__b3p_dual_bias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_add_p_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr;
} _b3p_dual_bias_closure_L3_args_t;

static void _b3p_dual_bias_closure_L3(void *_b3p_dual_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3p_dual_bias_closure_L3_args_t *args = (_b3p_dual_bias_closure_L3_args_t *)_b3p_dual_bias_closure_L3_args;

  int32_t *DeeployNetwork_add_p_tensor = args->DeeployNetwork_add_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3p_dual_bias_closure_args_t DeeployNetwork__b3p_dual_bias_closure_args =
      (_b3p_dual_bias_closure_args_t){.DeeployNetwork_add_p_tensor = DeeployNetwork_add_p_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr};

  // _b3p_dual_bias_closure CLOSURE CALL
  _b3p_dual_bias_closure(&DeeployNetwork__b3p_dual_bias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref;
} _b3p_im2col_tiling_closure_args_t;

static void _b3p_im2col_tiling_closure(void *_b3p_im2col_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3p_im2col_tiling_closure_args_t *args = (_b3p_im2col_tiling_closure_args_t *)_b3p_im2col_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16Im2ColU8 (Name: b3p_im2col, Op: NE16Im2ColU8) -- column buffer for the NE16 im2col route -- QW exp6
  PULP_Im2ColU8_NCHW_to_HWC((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref,
                            (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref, NUM_CORES, 14, 5, 16, 8, 5, 7, 1, 0, 0, 0, 0, 1, 1,
                            (uint8_t)128, (uint8_t)128);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref;
} _b3p_im2col_cluster_fork_args_t;

static void _b3p_im2col_cluster_fork(void *_b3p_im2col_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b3p_im2col_cluster_fork_args_t *args = (_b3p_im2col_cluster_fork_args_t *)_b3p_im2col_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b3p_im2col_tiling_closure_args_t DeeployNetwork__b3p_im2col_tiling_closure_args =
      (_b3p_im2col_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref};

  // _b3p_im2col_tiling_closure CLOSURE CALL
  _b3p_im2col_tiling_closure(&DeeployNetwork__b3p_im2col_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_col_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr;
} _b3p_im2col_closure_args_t;

static void _b3p_im2col_closure(void *_b3p_im2col_closure_args) {
  // CLOSURE ARG CAST
  _b3p_im2col_closure_args_t *args = (_b3p_im2col_closure_args_t *)_b3p_im2col_closure_args;

  uint8_t *DeeployNetwork_col_p_tensor = args->DeeployNetwork_col_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4480);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_col_p_tensor_ref =
      (void *)((char *)DeeployNetwork_col_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442912, DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b3p_im2col_cluster_fork_args_t DeeployNetwork__b3p_im2col_cluster_fork_args =
        (_b3p_im2col_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_in_ref,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b3p_im2col_cluster_fork, &DeeployNetwork__b3p_im2col_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1315200, DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_col_p_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_col_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr;
} _b3p_im2col_closure_L3_args_t;

static void _b3p_im2col_closure_L3(void *_b3p_im2col_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3p_im2col_closure_L3_args_t *args = (_b3p_im2col_closure_L3_args_t *)_b3p_im2col_closure_L3_args;

  uint8_t *DeeployNetwork_col_p_tensor = args->DeeployNetwork_col_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3p_im2col_closure_args_t DeeployNetwork__b3p_im2col_closure_args =
      (_b3p_im2col_closure_args_t){.DeeployNetwork_col_p_tensor = DeeployNetwork_col_p_tensor,
                                   .DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr};

  // _b3p_im2col_closure CLOSURE CALL
  _b3p_im2col_closure(&DeeployNetwork__b3p_im2col_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b3m_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref;
} _b3m_dual_wenc_tiling_closure_args_t;

static void _b3m_dual_wenc_tiling_closure(void *_b3m_dual_wenc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3m_dual_wenc_tiling_closure_args_t *args = (_b3m_dual_wenc_tiling_closure_args_t *)_b3m_dual_wenc_tiling_closure_args;

  void *DeeployNetwork_b3m_dual_wenc_buffer = args->DeeployNetwork_b3m_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualPerturbEncode (Name: b3m_dual_wenc, Op: NE16DualPerturbEncode) -- W+/W- from one draw, encoded as one 2*cout 1x1 weight -- QW exp6
  {
    uint32_t b3m_dual_wenc_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 6);
    uint32_t b3m_dual_wenc_tile_xor = (0 * 0x9E3779B1u);
    float32_t b3m_dual_wenc_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualPerturbEncode((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref,
                               (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref,
                               (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref, (int8_t *)DeeployNetwork_b3m_dual_wenc_buffer,
                               b3m_dual_wenc_seed_w, b3m_dual_wenc_tile_xor, 15, b3m_dual_wenc_eps_scale, NUM_CORES, 32, 112, 2, 0, 16, 7, 1);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b3m_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref;
} _b3m_dual_wenc_cluster_fork_args_t;

static void _b3m_dual_wenc_cluster_fork(void *_b3m_dual_wenc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b3m_dual_wenc_cluster_fork_args_t *args = (_b3m_dual_wenc_cluster_fork_args_t *)_b3m_dual_wenc_cluster_fork_args;

  void *DeeployNetwork_b3m_dual_wenc_buffer = args->DeeployNetwork_b3m_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b3m_dual_wenc_tiling_closure_args_t DeeployNetwork__b3m_dual_wenc_tiling_closure_args = (_b3m_dual_wenc_tiling_closure_args_t){
      .DeeployNetwork_b3m_dual_wenc_buffer = DeeployNetwork_b3m_dual_wenc_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref};

  // _b3m_dual_wenc_tiling_closure CLOSURE CALL
  _b3m_dual_wenc_tiling_closure(&DeeployNetwork__b3m_dual_wenc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr;
} _b3m_dual_wenc_closure_args_t;

static void _b3m_dual_wenc_closure(void *_b3m_dual_wenc_closure_args) {
  // CLOSURE ARG CAST
  _b3m_dual_wenc_closure_args_t *args = (_b3m_dual_wenc_closure_args_t *)_b3m_dual_wenc_closure_args;

  uint8_t *DeeployNetwork_wenc_m_tensor = args->DeeployNetwork_wenc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 14336);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10752);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3m_dual_wenc_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3m_dual_wenc +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wenc_m_tensor_ref =
      (void *)((char *)DeeployNetwork_wenc_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_b3m_dual_wenc_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445376, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3m_dual_wenc_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b3m_dual_wenc_cluster_fork_args_t DeeployNetwork__b3m_dual_wenc_cluster_fork_args = (_b3m_dual_wenc_cluster_fork_args_t){
        .DeeployNetwork_b3m_dual_wenc_buffer = DeeployNetwork_b3m_dual_wenc_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b3m_dual_wenc_cluster_fork, &DeeployNetwork__b3m_dual_wenc_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314304, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_wenc_m_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr;
} _b3m_dual_wenc_closure_L3_args_t;

static void _b3m_dual_wenc_closure_L3(void *_b3m_dual_wenc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3m_dual_wenc_closure_L3_args_t *args = (_b3m_dual_wenc_closure_L3_args_t *)_b3m_dual_wenc_closure_L3_args;

  uint8_t *DeeployNetwork_wenc_m_tensor = args->DeeployNetwork_wenc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3m_dual_wenc_closure_args_t DeeployNetwork__b3m_dual_wenc_closure_args =
      (_b3m_dual_wenc_closure_args_t){.DeeployNetwork_wenc_m_tensor = DeeployNetwork_wenc_m_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr};

  // _b3m_dual_wenc_closure CLOSURE CALL
  _b3m_dual_wenc_closure(&DeeployNetwork__b3m_dual_wenc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b3m_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref;
} _b3m_dual_bias_tiling_closure_args_t;

static void _b3m_dual_bias_tiling_closure(void *_b3m_dual_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3m_dual_bias_tiling_closure_args_t *args = (_b3m_dual_bias_tiling_closure_args_t *)_b3m_dual_bias_tiling_closure_args;

  void *DeeployNetwork_b3m_dual_bias_buffer = args->DeeployNetwork_b3m_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualBias (Name: b3m_dual_bias, Op: NE16DualBias) -- add+/add- with the signed-input correction, stacked -- QW exp6
  {
    uint32_t b3m_dual_bias_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 6);
    uint32_t b3m_dual_bias_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 7);
    uint32_t b3m_dual_bias_tile_xor = (0 * 0x9E3779B1u);
    float32_t b3m_dual_bias_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualBias(
        (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref, (int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref,
        (int8_t *)DeeployNetwork_b3m_dual_bias_buffer, b3m_dual_bias_seed_w, b3m_dual_bias_seed_b, b3m_dual_bias_tile_xor, 15, 16, b3m_dual_bias_eps_scale,
        NUM_CORES, 32, 112, 128, 2);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b3m_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref;
} _b3m_dual_bias_cluster_fork_args_t;

static void _b3m_dual_bias_cluster_fork(void *_b3m_dual_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b3m_dual_bias_cluster_fork_args_t *args = (_b3m_dual_bias_cluster_fork_args_t *)_b3m_dual_bias_cluster_fork_args;

  void *DeeployNetwork_b3m_dual_bias_buffer = args->DeeployNetwork_b3m_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b3m_dual_bias_tiling_closure_args_t DeeployNetwork__b3m_dual_bias_tiling_closure_args = (_b3m_dual_bias_tiling_closure_args_t){
      .DeeployNetwork_b3m_dual_bias_buffer = DeeployNetwork_b3m_dual_bias_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref};

  // _b3m_dual_bias_tiling_closure CLOSURE CALL
  _b3m_dual_bias_tiling_closure(&DeeployNetwork__b3m_dual_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr;
} _b3m_dual_bias_closure_args_t;

static void _b3m_dual_bias_closure(void *_b3m_dual_bias_closure_args) {
  // CLOSURE ARG CAST
  _b3m_dual_bias_closure_args_t *args = (_b3m_dual_bias_closure_args_t *)_b3m_dual_bias_closure_args;

  int32_t *DeeployNetwork_add_m_tensor = args->DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7424);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 11136);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 11008);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 11264);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 11392);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 11520);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3m_dual_bias_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3m_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_blocks3convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b3m_dual_bias_ref =
      (void *)((char *)DeeployNetwork_blocks3convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b3m_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3m_dual_bias_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3m_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_m_tensor_ref =
      (void *)((char *)DeeployNetwork_add_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_b3m_dual_bias_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445376, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_blocks3convweight_int8_pmul_tensor_DUPLICATE_FOR_b3m_dual_bias_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_input_2_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_blocks3convbias_rqsadd_pmul_tensor_DUPLICATE_FOR_b3m_dual_bias_ref);
    mchan_transfer_1d(
        1441920, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3m_dual_bias_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b3m_dual_bias_cluster_fork_args_t DeeployNetwork__b3m_dual_bias_cluster_fork_args = (_b3m_dual_bias_cluster_fork_args_t){
        .DeeployNetwork_b3m_dual_bias_buffer = DeeployNetwork_b3m_dual_bias_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_bmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b3m_dual_bias_cluster_fork, &DeeployNetwork__b3m_dual_bias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_add_m_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr;
} _b3m_dual_bias_closure_L3_args_t;

static void _b3m_dual_bias_closure_L3(void *_b3m_dual_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3m_dual_bias_closure_L3_args_t *args = (_b3m_dual_bias_closure_L3_args_t *)_b3m_dual_bias_closure_L3_args;

  int32_t *DeeployNetwork_add_m_tensor = args->DeeployNetwork_add_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3m_dual_bias_closure_args_t DeeployNetwork__b3m_dual_bias_closure_args =
      (_b3m_dual_bias_closure_args_t){.DeeployNetwork_add_m_tensor = DeeployNetwork_add_m_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr};

  // _b3m_dual_bias_closure CLOSURE CALL
  _b3m_dual_bias_closure(&DeeployNetwork__b3m_dual_bias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref;
} _b3m_im2col_tiling_closure_args_t;

static void _b3m_im2col_tiling_closure(void *_b3m_im2col_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3m_im2col_tiling_closure_args_t *args = (_b3m_im2col_tiling_closure_args_t *)_b3m_im2col_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16Im2ColU8 (Name: b3m_im2col, Op: NE16Im2ColU8) -- column buffer for the NE16 im2col route -- QW exp6
  PULP_Im2ColU8_NCHW_to_HWC((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref,
                            (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref, NUM_CORES, 14, 5, 16, 8, 5, 7, 1, 0, 0, 0, 0, 1, 1,
                            (uint8_t)128, (uint8_t)128);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref;
} _b3m_im2col_cluster_fork_args_t;

static void _b3m_im2col_cluster_fork(void *_b3m_im2col_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b3m_im2col_cluster_fork_args_t *args = (_b3m_im2col_cluster_fork_args_t *)_b3m_im2col_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b3m_im2col_tiling_closure_args_t DeeployNetwork__b3m_im2col_tiling_closure_args =
      (_b3m_im2col_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref};

  // _b3m_im2col_tiling_closure CLOSURE CALL
  _b3m_im2col_tiling_closure(&DeeployNetwork__b3m_im2col_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_col_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr;
} _b3m_im2col_closure_args_t;

static void _b3m_im2col_closure(void *_b3m_im2col_closure_args) {
  // CLOSURE ARG CAST
  _b3m_im2col_closure_args_t *args = (_b3m_im2col_closure_args_t *)_b3m_im2col_closure_args;

  uint8_t *DeeployNetwork_col_m_tensor = args->DeeployNetwork_col_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4480);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_input_3_ref =
      (void *)((char *)DeeployNetwork_input_3 +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_col_m_tensor_ref =
      (void *)((char *)DeeployNetwork_col_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442912, DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_input_3_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b3m_im2col_cluster_fork_args_t DeeployNetwork__b3m_im2col_cluster_fork_args =
        (_b3m_im2col_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_in_ref,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b3m_im2col_cluster_fork, &DeeployNetwork__b3m_im2col_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1315200, DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_col_m_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_col_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr;
} _b3m_im2col_closure_L3_args_t;

static void _b3m_im2col_closure_L3(void *_b3m_im2col_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3m_im2col_closure_L3_args_t *args = (_b3m_im2col_closure_L3_args_t *)_b3m_im2col_closure_L3_args;

  uint8_t *DeeployNetwork_col_m_tensor = args->DeeployNetwork_col_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3m_im2col_closure_args_t DeeployNetwork__b3m_im2col_closure_args =
      (_b3m_im2col_closure_args_t){.DeeployNetwork_col_m_tensor = DeeployNetwork_col_m_tensor,
                                   .DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr};

  // _b3m_im2col_closure CLOSURE CALL
  _b3m_im2col_closure(&DeeployNetwork__b3m_im2col_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref;
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
              .input_stride = (ne16_stride_t){.d0 = 112, .d1 = *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref, .d2 = 0},
              .output_stride = (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES,
                                               .d1 = 32,
                                               .d2 = *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref},
              task.data.cfg.weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 112, .d2 = 0},
              .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 7),
                                                     .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref, 1)},
                                          .remainder = {.KoKi = nnx_concat_half(32, 16),
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
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_in_x_stride + 0);
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_dim_im_out_x_stride + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6304);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6176);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5600);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped_ref =
      (void *)((char *)DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_wenc_p_tensor_ref =
      (void *)((char *)DeeployNetwork_wenc_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3p_rqs_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3p_rqs +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_p_tensor_ref =
      (void *)((char *)DeeployNetwork_add_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_p_tensor_Reshaped_ref =
      (void *)((char *)DeeployNetwork_y_nhwc_p_tensor_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int add_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int weight_ExternalToLocal = -1;

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
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped_ref,
          .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size_1d[TILING_I],
          .ext_stride_1d = 560};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0col_p_tensor_Reshaped_ref) +
                 DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_relativeOffset[TILING_I]);

    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1445376,
                                      .size = 3584,
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
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3p_rqs_ref};
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

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo[TILING_I];

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
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_p_tensor_Reshaped_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size_1d[TILING_I],
                                      .ext_stride_1d = 160};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_p_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_p_tensor_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_y_nhwc_p_tensor_Reshaped_ref) +
                 DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_relativeOffset[TILING_I]);

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
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref;
} __MERGE_CONVRQ_PASS_1_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_1_tiling_closure(void *__MERGE_CONVRQ_PASS_1_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_1_tiling_closure_args_t *args = (__MERGE_CONVRQ_PASS_1_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_1_tiling_closure_args;

  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi_ref;
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
              .input_stride = (ne16_stride_t){.d0 = 112, .d1 = *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride_ref, .d2 = 0},
              .output_stride = (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES,
                                               .d1 = 32,
                                               .d2 = *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride_ref},
              task.data.cfg.weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 112, .d2 = 0},
              .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 7),
                                                     .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo_ref, 1)},
                                          .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                        .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref,
                                                                                *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo_ref),
                                                        .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref,
                                                                                *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi_ref)}},
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
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride + 0);
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6304);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6176);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5600);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped_ref =
      (void *)((char *)DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_wenc_m_tensor_ref =
      (void *)((char *)DeeployNetwork_wenc_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3m_rqs_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3m_rqs +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_m_tensor_ref =
      (void *)((char *)DeeployNetwork_add_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_y_nhwc_m_tensor_Reshaped_ref =
      (void *)((char *)DeeployNetwork_y_nhwc_m_tensor_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int add_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;
  int weight_ExternalToLocal = -1;

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
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped_ref,
          .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_size_1d[TILING_I],
          .ext_stride_1d = 560};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_1col_m_tensor_Reshaped_ref) +
                 DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_in_relativeOffset[TILING_I]);

    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1445376,
                                      .size = 3584,
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
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b3m_rqs_ref};
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

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride_ref =
        &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride_ref =
        &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride[TILING_I];

    __MERGE_CONVRQ_PASS_1_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_1_tiling_closure_args = (__MERGE_CONVRQ_PASS_1_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride_ref =
            DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_in_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride_ref =
            DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_dim_im_out_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_nHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bHi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_bWi_ref,
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
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_y_nhwc_m_tensor_Reshaped_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_size_1d[TILING_I],
                                      .ext_stride_1d = 160};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_y_nhwc_m_tensor_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_y_nhwc_m_tensor_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_y_nhwc_m_tensor_Reshaped_ref) +
                 DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_1_data_out_relativeOffset[TILING_I]);

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref;
} _b3p_out_nchw_tiling_closure_args_t;

static void _b3p_out_nchw_tiling_closure(void *_b3p_out_nchw_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3p_out_nchw_tiling_closure_args_t *args = (_b3p_out_nchw_tiling_closure_args_t *)_b3p_out_nchw_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 8, 5, 32] -> [1, 32, 8, 5] (Name: b3p_out_nchw, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 32;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_3][dimLen_1][dimLen_2])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref;
} _b3p_out_nchw_cluster_fork_args_t;

static void _b3p_out_nchw_cluster_fork(void *_b3p_out_nchw_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b3p_out_nchw_cluster_fork_args_t *args = (_b3p_out_nchw_cluster_fork_args_t *)_b3p_out_nchw_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b3p_out_nchw_tiling_closure_args_t DeeployNetwork__b3p_out_nchw_tiling_closure_args = (_b3p_out_nchw_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref};

  // _b3p_out_nchw_tiling_closure CLOSURE CALL
  _b3p_out_nchw_tiling_closure(&DeeployNetwork__b3p_out_nchw_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_nhwc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr;
} _b3p_out_nchw_closure_args_t;

static void _b3p_out_nchw_closure(void *_b3p_out_nchw_closure_args) {
  // CLOSURE ARG CAST
  _b3p_out_nchw_closure_args_t *args = (_b3p_out_nchw_closure_args_t *)_b3p_out_nchw_closure_args;

  int8_t *DeeployNetwork_y_nhwc_p_tensor = args->DeeployNetwork_y_nhwc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_y_nhwc_p_tensor_ref =
      (void *)((char *)DeeployNetwork_y_nhwc_p_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1443072,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_y_nhwc_p_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _b3p_out_nchw_cluster_fork_args_t DeeployNetwork__b3p_out_nchw_cluster_fork_args = (_b3p_out_nchw_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b3p_out_nchw_cluster_fork, &DeeployNetwork__b3p_out_nchw_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1312000,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_nhwc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr;
} _b3p_out_nchw_closure_L3_args_t;

static void _b3p_out_nchw_closure_L3(void *_b3p_out_nchw_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3p_out_nchw_closure_L3_args_t *args = (_b3p_out_nchw_closure_L3_args_t *)_b3p_out_nchw_closure_L3_args;

  int8_t *DeeployNetwork_y_nhwc_p_tensor = args->DeeployNetwork_y_nhwc_p_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3p_out_nchw_closure_args_t DeeployNetwork__b3p_out_nchw_closure_args =
      (_b3p_out_nchw_closure_args_t){.DeeployNetwork_y_nhwc_p_tensor = DeeployNetwork_y_nhwc_p_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr};

  // _b3p_out_nchw_closure CLOSURE CALL
  _b3p_out_nchw_closure(&DeeployNetwork__b3p_out_nchw_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref;
} _b3m_out_nchw_tiling_closure_args_t;

static void _b3m_out_nchw_tiling_closure(void *_b3m_out_nchw_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3m_out_nchw_tiling_closure_args_t *args = (_b3m_out_nchw_tiling_closure_args_t *)_b3m_out_nchw_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 8, 5, 32] -> [1, 32, 8, 5] (Name: b3m_out_nchw, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 32;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_3][dimLen_1][dimLen_2])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref;
} _b3m_out_nchw_cluster_fork_args_t;

static void _b3m_out_nchw_cluster_fork(void *_b3m_out_nchw_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b3m_out_nchw_cluster_fork_args_t *args = (_b3m_out_nchw_cluster_fork_args_t *)_b3m_out_nchw_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b3m_out_nchw_tiling_closure_args_t DeeployNetwork__b3m_out_nchw_tiling_closure_args = (_b3m_out_nchw_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref};

  // _b3m_out_nchw_tiling_closure CLOSURE CALL
  _b3m_out_nchw_tiling_closure(&DeeployNetwork__b3m_out_nchw_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_nhwc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr;
} _b3m_out_nchw_closure_args_t;

static void _b3m_out_nchw_closure(void *_b3m_out_nchw_closure_args) {
  // CLOSURE ARG CAST
  _b3m_out_nchw_closure_args_t *args = (_b3m_out_nchw_closure_args_t *)_b3m_out_nchw_closure_args;

  int8_t *DeeployNetwork_y_nhwc_m_tensor = args->DeeployNetwork_y_nhwc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_y_nhwc_m_tensor_ref =
      (void *)((char *)DeeployNetwork_y_nhwc_m_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_output_1_ref =
      (void *)((char *)DeeployNetwork_output_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1443072,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_y_nhwc_m_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _b3m_out_nchw_cluster_fork_args_t DeeployNetwork__b3m_out_nchw_cluster_fork_args = (_b3m_out_nchw_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b3m_out_nchw_cluster_fork, &DeeployNetwork__b3m_out_nchw_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1312000,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_output_1_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_nhwc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr;
} _b3m_out_nchw_closure_L3_args_t;

static void _b3m_out_nchw_closure_L3(void *_b3m_out_nchw_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3m_out_nchw_closure_L3_args_t *args = (_b3m_out_nchw_closure_L3_args_t *)_b3m_out_nchw_closure_L3_args;

  int8_t *DeeployNetwork_y_nhwc_m_tensor = args->DeeployNetwork_y_nhwc_m_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3m_out_nchw_closure_args_t DeeployNetwork__b3m_out_nchw_closure_args =
      (_b3m_out_nchw_closure_args_t){.DeeployNetwork_y_nhwc_m_tensor = DeeployNetwork_y_nhwc_m_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr};

  // _b3m_out_nchw_closure CLOSURE CALL
  _b3m_out_nchw_closure(&DeeployNetwork__b3m_out_nchw_closure_args);

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
  void *DeeployNetwork_b3p_dual_wenc_buffer;
  void *DeeployNetwork_b3p_dual_bias_buffer;
  void *DeeployNetwork_b3m_dual_wenc_buffer;
  void *DeeployNetwork_b3m_dual_bias_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr;
    DeeployNetwork_wenc_p_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _b3p_dual_wenc_closure_L3_args_t DeeployNetwork__b3p_dual_wenc_closure_L3_args = (_b3p_dual_wenc_closure_L3_args_t){
        .DeeployNetwork_wenc_p_tensor = DeeployNetwork_wenc_p_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_wenc_tileIdxPtr};

    // _b3p_dual_wenc_closure_L3 CLOSURE CALL
    _b3p_dual_wenc_closure_L3(&DeeployNetwork__b3p_dual_wenc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr;
    DeeployNetwork_add_p_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 3584);
    _b3p_dual_bias_closure_L3_args_t DeeployNetwork__b3p_dual_bias_closure_L3_args = (_b3p_dual_bias_closure_L3_args_t){
        .DeeployNetwork_add_p_tensor = DeeployNetwork_add_p_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3p_dual_bias_tileIdxPtr};

    // _b3p_dual_bias_closure_L3 CLOSURE CALL
    _b3p_dual_bias_closure_L3(&DeeployNetwork__b3p_dual_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr;
    DeeployNetwork_col_p_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 3712);
    _b3p_im2col_closure_L3_args_t DeeployNetwork__b3p_im2col_closure_L3_args =
        (_b3p_im2col_closure_L3_args_t){.DeeployNetwork_col_p_tensor = DeeployNetwork_col_p_tensor,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3p_im2col_tileIdxPtr};

    // _b3p_im2col_closure_L3 CLOSURE CALL
    _b3p_im2col_closure_L3(&DeeployNetwork__b3p_im2col_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr;
    DeeployNetwork_wenc_m_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8192);
    _b3m_dual_wenc_closure_L3_args_t DeeployNetwork__b3m_dual_wenc_closure_L3_args = (_b3m_dual_wenc_closure_L3_args_t){
        .DeeployNetwork_wenc_m_tensor = DeeployNetwork_wenc_m_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_wenc_tileIdxPtr};

    // _b3m_dual_wenc_closure_L3 CLOSURE CALL
    _b3m_dual_wenc_closure_L3(&DeeployNetwork__b3m_dual_wenc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr;
    DeeployNetwork_add_m_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 17536);
    _b3m_dual_bias_closure_L3_args_t DeeployNetwork__b3m_dual_bias_closure_L3_args = (_b3m_dual_bias_closure_L3_args_t){
        .DeeployNetwork_add_m_tensor = DeeployNetwork_add_m_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3m_dual_bias_tileIdxPtr};

    // _b3m_dual_bias_closure_L3 CLOSURE CALL
    _b3m_dual_bias_closure_L3(&DeeployNetwork__b3m_dual_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr;
    DeeployNetwork_col_m_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 13056);
    _b3m_im2col_closure_L3_args_t DeeployNetwork__b3m_im2col_closure_L3_args =
        (_b3m_im2col_closure_L3_args_t){.DeeployNetwork_col_m_tensor = DeeployNetwork_col_m_tensor,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3m_im2col_tileIdxPtr};

    // _b3m_im2col_closure_L3 CLOSURE CALL
    _b3m_im2col_closure_L3(&DeeployNetwork__b3m_im2col_closure_L3_args);
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
    DeeployNetwork_y_nhwc_p_tensor_Reshaped = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 11776);
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

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr;
    _b3p_out_nchw_closure_L3_args_t DeeployNetwork__b3p_out_nchw_closure_L3_args =
        (_b3p_out_nchw_closure_L3_args_t){.DeeployNetwork_y_nhwc_p_tensor = DeeployNetwork_y_nhwc_p_tensor,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3p_out_nchw_tileIdxPtr};

    // _b3p_out_nchw_closure_L3 CLOSURE CALL
    _b3p_out_nchw_closure_L3(&DeeployNetwork__b3p_out_nchw_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr;
    _b3m_out_nchw_closure_L3_args_t DeeployNetwork__b3m_out_nchw_closure_L3_args =
        (_b3m_out_nchw_closure_L3_args_t){.DeeployNetwork_y_nhwc_m_tensor = DeeployNetwork_y_nhwc_m_tensor,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3m_out_nchw_tileIdxPtr};

    // _b3m_out_nchw_closure_L3 CLOSURE CALL
    _b3m_out_nchw_closure_L3(&DeeployNetwork__b3m_out_nchw_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 14464);
  if (DeeployNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 14464));
    return;
  }

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 17664);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8192);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 12896);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 16480);
  DeeployNetwork_input_3 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 11776);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 1280);
  DeeployNetwork_output_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 2560);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_inputs[3] = (void *)DeeployNetwork_input_3;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
  DeeployNetwork_outputs[1] = (void *)DeeployNetwork_output_1;
}
