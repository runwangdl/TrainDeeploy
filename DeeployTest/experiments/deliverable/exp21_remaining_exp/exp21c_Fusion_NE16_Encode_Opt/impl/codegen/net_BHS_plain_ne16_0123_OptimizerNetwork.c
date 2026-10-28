#include "OptimizerNetwork.h"
#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "TrainingNetwork.h"
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

int8_t *DeeployOptNetwork_MEMORYARENA_L1;
int8_t *DeeployOptNetwork_MEMORYARENA_L2;
int8_t *DeeployOptNetwork_input_0;
int32_t *DeeployOptNetwork_input_1;
int8_t *DeeployOptNetwork_input_2;
int32_t *DeeployOptNetwork_input_3;
int8_t *DeeployOptNetwork_input_4;
int32_t *DeeployOptNetwork_input_5;
int8_t *DeeployOptNetwork_input_6;
int32_t *DeeployOptNetwork_input_7;
int8_t *DeeployOptNetwork_input_8;
int32_t *DeeployOptNetwork_input_9;
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
int8_t *DeeployOptNetwork_output_0;
int32_t *DeeployOptNetwork_output_1;
int8_t *DeeployOptNetwork_output_2;
int32_t *DeeployOptNetwork_output_3;
int8_t *DeeployOptNetwork_output_4;
int32_t *DeeployOptNetwork_output_5;
int8_t *DeeployOptNetwork_output_6;
int32_t *DeeployOptNetwork_output_7;
int8_t *DeeployOptNetwork_output_8;
int32_t *DeeployOptNetwork_output_9;
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

static PI_L2 int32_t DeeployOptNetwork_blocks0convbias_rqsadd_pmul_tensor[8] = {16715, 28051, 19407, 24750, 20593, 26069, 21962, 15816};

static PI_L2 int32_t DeeployOptNetwork_blocks1convbias_rqsadd_pmul_tensor[16] = {3887854, 4028793, 3998692, 4560022, 4527256, 4318277, 4195455, 4117846,
                                                                                 3644722, 4679690, 3915633, 4146882, 4516702, 3937397, 4507614, 4631827};

static PI_L2 int32_t DeeployOptNetwork_blocks2convbias_rqsadd_pmul_tensor[16] = {5779922, 6811814, 7569997, 7017406, 6391308, 6797053, 6348470, 6703570,
                                                                                 7695533, 4777151, 6888298, 6076617, 7800891, 6162310, 4713482, 8093983};

static PI_L2 int32_t DeeployOptNetwork_blocks0convweight_int8_pmul_tensor[8] = {85469, 143432, 99234, 126554, 105298, 133299, 112298, 80873};

static PI_L2 int32_t DeeployOptNetwork_blocks3convbias_rqsadd_pmul_tensor[32] = {
    11216862, 10953840, 8883199, 8006243, 9415140, 11637480, 7831187,  5033047, 8487970,  8172161,  8744505,  12902138, 8150436,  8865462, 7886326,  11680739,
    9828662,  10359709, 9262086, 8927959, 7369362, 7468230,  10595193, 7107887, 12231511, 11419024, 10140785, 6934845,  11424804, 5359367, 12871468, 10140571};

static PI_L2 int32_t DeeployOptNetwork_blocks4convbias_rqsadd_pmul_tensor[32] = {
    15255979, 18601466, 16480003, 16604952, 16636081, 17501801, 18550852, 19833580, 22784586, 15600707, 20171281,
    19515017, 18112376, 18597217, 11157841, 21481128, 18522609, 19594714, 17518029, 20954163, 17792839, 17273134,
    20128746, 16975140, 23468860, 19124083, 19414542, 16972130, 22860964, 15612485, 16032134, 19108972};

static PI_L2 int32_t DeeployOptNetwork_blocks1convweight_int8_pmul_tensor[16] = {234330, 242824, 241010, 274843, 272868, 260272, 252869, 248192,
                                                                                 219675, 282055, 236004, 249942, 272232, 237316, 271684, 279170};

static PI_L2 int32_t DeeployOptNetwork_blocks2convweight_int8_pmul_tensor[16] = {187678, 221184, 245803, 227860, 207530, 220705, 206139, 217670,
                                                                                 249879, 155117, 223668, 197312, 253300, 200095, 153050, 262817};

static PI_L2 int32_t DeeployOptNetwork_blocks3convweight_int8_pmul_tensor[32] = {
    243041, 237342, 192477, 173475, 204002, 252155, 169682, 109053, 183913, 177070, 189471, 279557, 176599, 192092, 170877, 253092,
    212962, 224469, 200686, 193446, 159675, 161818, 229571, 154010, 265026, 247421, 219725, 150261, 247547, 116124, 278892, 219721};

static PI_L2 int32_t DeeployOptNetwork_blocks4convweight_int8_pmul_tensor[32] = {
    185997, 226785, 200920, 202444, 202823, 213378, 226168, 241806, 277784, 190200, 245924, 237923, 220822, 226733, 136034, 261893,
    225823, 238894, 213576, 255468, 216926, 210590, 245405, 206957, 286127, 233156, 236698, 206920, 278716, 190344, 195460, 232972};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_cumByteOffset[1] = {0};

void *DeeployOptNetwork_inputs[22];
void *DeeployOptNetwork_outputs[22];
extern struct pi_device cluster_dev;
typedef struct {
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref;
} _rqsp_upd_blocks0convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_upd_blocks0convbias_rqsadd_tiling_closure(void *_rqsp_upd_blocks0convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks0convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_upd_blocks0convbias_rqsadd_tiling_closure_args_t *)_rqsp_upd_blocks0convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_upd_blocks0convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_upd_blocks0convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_upd_blocks0convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_upd_blocks0convbias_rqsadd_chunk = (8 >> rqsp_upd_blocks0convbias_rqsadd_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_upd_blocks0convbias_rqsadd_chunk_start =
      (uint32_t)MIN(rqsp_upd_blocks0convbias_rqsadd_chunk * rqsp_upd_blocks0convbias_rqsadd_core_id, (uint32_t)8);
  uint32_t rqsp_upd_blocks0convbias_rqsadd_chunk_stop =
      (uint32_t)MIN(rqsp_upd_blocks0convbias_rqsadd_chunk_start + rqsp_upd_blocks0convbias_rqsadd_chunk, (uint32_t)8);
  uint32_t rqsp_upd_blocks0convbias_rqsadd_local_size = rqsp_upd_blocks0convbias_rqsadd_chunk_stop - rqsp_upd_blocks0convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_upd_blocks0convbias_rqsadd_channel_start_offset = rqsp_upd_blocks0convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 1 + rqsp_upd_blocks0convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_upd_blocks0convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref[rqsp_upd_blocks0convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref[rqsp_upd_blocks0convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref, 16, 1, chunk_seed,
        rqsp_upd_blocks0convbias_rqsadd_local_size, rqsp_upd_blocks0convbias_rqsadd_chunk_start, perturbation_sign,
        zo_eco_mom[1] + rqsp_upd_blocks0convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref[rqsp_upd_blocks0convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref[rqsp_upd_blocks0convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref, 16, 1, chunk_seed,
        rqsp_upd_blocks0convbias_rqsadd_local_size, rqsp_upd_blocks0convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_upd_blocks0convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref;
} _rqsp_upd_blocks0convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_upd_blocks0convbias_rqsadd_cluster_fork(void *_rqsp_upd_blocks0convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks0convbias_rqsadd_cluster_fork_args_t *args =
      (_rqsp_upd_blocks0convbias_rqsadd_cluster_fork_args_t *)_rqsp_upd_blocks0convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks0convbias_rqsadd_tiling_closure_args_t DeeployOptNetwork__rqsp_upd_blocks0convbias_rqsadd_tiling_closure_args =
      (_rqsp_upd_blocks0convbias_rqsadd_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref};

  // _rqsp_upd_blocks0convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_upd_blocks0convbias_rqsadd_tiling_closure(&DeeployOptNetwork__rqsp_upd_blocks0convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr;
} _rqsp_upd_blocks0convbias_rqsadd_closure_args_t;

static void _rqsp_upd_blocks0convbias_rqsadd_closure(void *_rqsp_upd_blocks0convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks0convbias_rqsadd_closure_args_t *args = (_rqsp_upd_blocks0convbias_rqsadd_closure_args_t *)_rqsp_upd_blocks0convbias_rqsadd_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 32);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_input_1_ref =
      (void *)((char *)DeeployOptNetwork_input_1 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_blocks0convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployOptNetwork_blocks0convbias_rqsadd_pmul_tensor +
               DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_cumByteOffset
                   [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_output_1_ref =
      (void *)((char *)DeeployOptNetwork_output_1 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_input_1_ref);
    mchan_transfer_1d(1441824, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_blocks0convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_upd_blocks0convbias_rqsadd_cluster_fork_args_t DeeployOptNetwork__rqsp_upd_blocks0convbias_rqsadd_cluster_fork_args =
        (_rqsp_upd_blocks0convbias_rqsadd_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_upd_blocks0convbias_rqsadd_cluster_fork, &DeeployOptNetwork__rqsp_upd_blocks0convbias_rqsadd_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_output_1_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr;
} _rqsp_upd_blocks0convbias_rqsadd_closure_L3_args_t;

static void _rqsp_upd_blocks0convbias_rqsadd_closure_L3(void *_rqsp_upd_blocks0convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks0convbias_rqsadd_closure_L3_args_t *args =
      (_rqsp_upd_blocks0convbias_rqsadd_closure_L3_args_t *)_rqsp_upd_blocks0convbias_rqsadd_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks0convbias_rqsadd_closure_args_t DeeployOptNetwork__rqsp_upd_blocks0convbias_rqsadd_closure_args =
      (_rqsp_upd_blocks0convbias_rqsadd_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr};

  // _rqsp_upd_blocks0convbias_rqsadd_closure CLOSURE CALL
  _rqsp_upd_blocks0convbias_rqsadd_closure(&DeeployOptNetwork__rqsp_upd_blocks0convbias_rqsadd_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref;
} _pert_upd_blocks0bnweight_tiling_closure_args_t;

static void _pert_upd_blocks0bnweight_tiling_closure(void *_pert_upd_blocks0bnweight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks0bnweight_tiling_closure_args_t *args = (_pert_upd_blocks0bnweight_tiling_closure_args_t *)_pert_upd_blocks0bnweight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_blocks0bnweight, Op: PerturbRademacher)
  uint8_t pert_upd_blocks0bnweight_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_blocks0bnweight_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_blocks0bnweight_chunk = (8 >> pert_upd_blocks0bnweight_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_blocks0bnweight_chunk_start = (uint32_t)MIN(pert_upd_blocks0bnweight_chunk * pert_upd_blocks0bnweight_core_id, (uint32_t)8);
  uint32_t pert_upd_blocks0bnweight_chunk_stop = (uint32_t)MIN(pert_upd_blocks0bnweight_chunk_start + pert_upd_blocks0bnweight_chunk, (uint32_t)8);
  uint32_t pert_upd_blocks0bnweight_local_size = pert_upd_blocks0bnweight_chunk_stop - pert_upd_blocks0bnweight_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 12 + pert_upd_blocks0bnweight_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_blocks0bnweight_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref[pert_upd_blocks0bnweight_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref[pert_upd_blocks0bnweight_chunk_start],
                            chunk_seed, perturbation_sign, pert_upd_blocks0bnweight_local_size, zo_eco_mom[12] + pert_upd_blocks0bnweight_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref[pert_upd_blocks0bnweight_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref[pert_upd_blocks0bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks0bnweight_local_size, pert_upd_blocks0bnweight_eps);
#else
    ApplyRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref[pert_upd_blocks0bnweight_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref[pert_upd_blocks0bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks0bnweight_local_size, pert_upd_blocks0bnweight_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref;
} _pert_upd_blocks0bnweight_cluster_fork_args_t;

static void _pert_upd_blocks0bnweight_cluster_fork(void *_pert_upd_blocks0bnweight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks0bnweight_cluster_fork_args_t *args = (_pert_upd_blocks0bnweight_cluster_fork_args_t *)_pert_upd_blocks0bnweight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks0bnweight_tiling_closure_args_t DeeployOptNetwork__pert_upd_blocks0bnweight_tiling_closure_args =
      (_pert_upd_blocks0bnweight_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref,
                                                        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref};

  // _pert_upd_blocks0bnweight_tiling_closure CLOSURE CALL
  _pert_upd_blocks0bnweight_tiling_closure(&DeeployOptNetwork__pert_upd_blocks0bnweight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr;
} _pert_upd_blocks0bnweight_closure_args_t;

static void _pert_upd_blocks0bnweight_closure(void *_pert_upd_blocks0bnweight_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks0bnweight_closure_args_t *args = (_pert_upd_blocks0bnweight_closure_args_t *)_pert_upd_blocks0bnweight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 32);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_input_12_ref =
      (void *)((char *)DeeployOptNetwork_input_12 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_output_12_ref =
      (void *)((char *)DeeployOptNetwork_output_12 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_input_12_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_blocks0bnweight_cluster_fork_args_t DeeployOptNetwork__pert_upd_blocks0bnweight_cluster_fork_args =
        (_pert_upd_blocks0bnweight_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_in_ref,
                                                        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_blocks0bnweight_cluster_fork, &DeeployOptNetwork__pert_upd_blocks0bnweight_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_output_12_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr;
} _pert_upd_blocks0bnweight_closure_L3_args_t;

static void _pert_upd_blocks0bnweight_closure_L3(void *_pert_upd_blocks0bnweight_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks0bnweight_closure_L3_args_t *args = (_pert_upd_blocks0bnweight_closure_L3_args_t *)_pert_upd_blocks0bnweight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks0bnweight_closure_args_t DeeployOptNetwork__pert_upd_blocks0bnweight_closure_args = (_pert_upd_blocks0bnweight_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr};

  // _pert_upd_blocks0bnweight_closure CLOSURE CALL
  _pert_upd_blocks0bnweight_closure(&DeeployOptNetwork__pert_upd_blocks0bnweight_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref;
} _pert_upd_blocks0bnbias_tiling_closure_args_t;

static void _pert_upd_blocks0bnbias_tiling_closure(void *_pert_upd_blocks0bnbias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks0bnbias_tiling_closure_args_t *args = (_pert_upd_blocks0bnbias_tiling_closure_args_t *)_pert_upd_blocks0bnbias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_blocks0bnbias, Op: PerturbRademacher)
  uint8_t pert_upd_blocks0bnbias_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_blocks0bnbias_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_blocks0bnbias_chunk = (8 >> pert_upd_blocks0bnbias_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_blocks0bnbias_chunk_start = (uint32_t)MIN(pert_upd_blocks0bnbias_chunk * pert_upd_blocks0bnbias_core_id, (uint32_t)8);
  uint32_t pert_upd_blocks0bnbias_chunk_stop = (uint32_t)MIN(pert_upd_blocks0bnbias_chunk_start + pert_upd_blocks0bnbias_chunk, (uint32_t)8);
  uint32_t pert_upd_blocks0bnbias_local_size = pert_upd_blocks0bnbias_chunk_stop - pert_upd_blocks0bnbias_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 13 + pert_upd_blocks0bnbias_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_blocks0bnbias_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref[pert_upd_blocks0bnbias_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref[pert_upd_blocks0bnbias_chunk_start],
                            chunk_seed, perturbation_sign, pert_upd_blocks0bnbias_local_size, zo_eco_mom[13] + pert_upd_blocks0bnbias_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref[pert_upd_blocks0bnbias_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref[pert_upd_blocks0bnbias_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks0bnbias_local_size, pert_upd_blocks0bnbias_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref[pert_upd_blocks0bnbias_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref[pert_upd_blocks0bnbias_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_upd_blocks0bnbias_local_size, pert_upd_blocks0bnbias_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref;
} _pert_upd_blocks0bnbias_cluster_fork_args_t;

static void _pert_upd_blocks0bnbias_cluster_fork(void *_pert_upd_blocks0bnbias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks0bnbias_cluster_fork_args_t *args = (_pert_upd_blocks0bnbias_cluster_fork_args_t *)_pert_upd_blocks0bnbias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks0bnbias_tiling_closure_args_t DeeployOptNetwork__pert_upd_blocks0bnbias_tiling_closure_args = (_pert_upd_blocks0bnbias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref};

  // _pert_upd_blocks0bnbias_tiling_closure CLOSURE CALL
  _pert_upd_blocks0bnbias_tiling_closure(&DeeployOptNetwork__pert_upd_blocks0bnbias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr;
} _pert_upd_blocks0bnbias_closure_args_t;

static void _pert_upd_blocks0bnbias_closure(void *_pert_upd_blocks0bnbias_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks0bnbias_closure_args_t *args = (_pert_upd_blocks0bnbias_closure_args_t *)_pert_upd_blocks0bnbias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 32);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_input_13_ref =
      (void *)((char *)DeeployOptNetwork_input_13 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_output_13_ref =
      (void *)((char *)DeeployOptNetwork_output_13 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_input_13_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_blocks0bnbias_cluster_fork_args_t DeeployOptNetwork__pert_upd_blocks0bnbias_cluster_fork_args = (_pert_upd_blocks0bnbias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_in_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_blocks0bnbias_cluster_fork, &DeeployOptNetwork__pert_upd_blocks0bnbias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_output_13_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr;
} _pert_upd_blocks0bnbias_closure_L3_args_t;

static void _pert_upd_blocks0bnbias_closure_L3(void *_pert_upd_blocks0bnbias_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks0bnbias_closure_L3_args_t *args = (_pert_upd_blocks0bnbias_closure_L3_args_t *)_pert_upd_blocks0bnbias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks0bnbias_closure_args_t DeeployOptNetwork__pert_upd_blocks0bnbias_closure_args = (_pert_upd_blocks0bnbias_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr};

  // _pert_upd_blocks0bnbias_closure CLOSURE CALL
  _pert_upd_blocks0bnbias_closure(&DeeployOptNetwork__pert_upd_blocks0bnbias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref;
} _pert_upd_fcbias_fp32_tiling_closure_args_t;

static void _pert_upd_fcbias_fp32_tiling_closure(void *_pert_upd_fcbias_fp32_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_fcbias_fp32_tiling_closure_args_t *args = (_pert_upd_fcbias_fp32_tiling_closure_args_t *)_pert_upd_fcbias_fp32_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_fcbias_fp32, Op: PerturbRademacher)
  uint8_t pert_upd_fcbias_fp32_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_fcbias_fp32_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_fcbias_fp32_chunk = (9 >> pert_upd_fcbias_fp32_log2Core) + ((9 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_fcbias_fp32_chunk_start = (uint32_t)MIN(pert_upd_fcbias_fp32_chunk * pert_upd_fcbias_fp32_core_id, (uint32_t)9);
  uint32_t pert_upd_fcbias_fp32_chunk_stop = (uint32_t)MIN(pert_upd_fcbias_fp32_chunk_start + pert_upd_fcbias_fp32_chunk, (uint32_t)9);
  uint32_t pert_upd_fcbias_fp32_local_size = pert_upd_fcbias_fp32_chunk_stop - pert_upd_fcbias_fp32_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 11 + pert_upd_fcbias_fp32_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_fcbias_fp32_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref[pert_upd_fcbias_fp32_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref[pert_upd_fcbias_fp32_chunk_start], chunk_seed,
                            perturbation_sign, pert_upd_fcbias_fp32_local_size, zo_eco_mom[11] + pert_upd_fcbias_fp32_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref[pert_upd_fcbias_fp32_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref[pert_upd_fcbias_fp32_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_fcbias_fp32_local_size, pert_upd_fcbias_fp32_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref[pert_upd_fcbias_fp32_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref[pert_upd_fcbias_fp32_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_upd_fcbias_fp32_local_size, pert_upd_fcbias_fp32_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref;
} _pert_upd_fcbias_fp32_cluster_fork_args_t;

static void _pert_upd_fcbias_fp32_cluster_fork(void *_pert_upd_fcbias_fp32_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_fcbias_fp32_cluster_fork_args_t *args = (_pert_upd_fcbias_fp32_cluster_fork_args_t *)_pert_upd_fcbias_fp32_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_fcbias_fp32_tiling_closure_args_t DeeployOptNetwork__pert_upd_fcbias_fp32_tiling_closure_args = (_pert_upd_fcbias_fp32_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref};

  // _pert_upd_fcbias_fp32_tiling_closure CLOSURE CALL
  _pert_upd_fcbias_fp32_tiling_closure(&DeeployOptNetwork__pert_upd_fcbias_fp32_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr;
} _pert_upd_fcbias_fp32_closure_args_t;

static void _pert_upd_fcbias_fp32_closure(void *_pert_upd_fcbias_fp32_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_fcbias_fp32_closure_args_t *args = (_pert_upd_fcbias_fp32_closure_args_t *)_pert_upd_fcbias_fp32_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 36);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_input_11_ref =
      (void *)((char *)DeeployOptNetwork_input_11 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_output_11_ref =
      (void *)((char *)DeeployOptNetwork_output_11 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441828, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_input_11_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_fcbias_fp32_cluster_fork_args_t DeeployOptNetwork__pert_upd_fcbias_fp32_cluster_fork_args = (_pert_upd_fcbias_fp32_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_in_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_fcbias_fp32_cluster_fork, &DeeployOptNetwork__pert_upd_fcbias_fp32_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310756, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_output_11_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr;
} _pert_upd_fcbias_fp32_closure_L3_args_t;

static void _pert_upd_fcbias_fp32_closure_L3(void *_pert_upd_fcbias_fp32_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_fcbias_fp32_closure_L3_args_t *args = (_pert_upd_fcbias_fp32_closure_L3_args_t *)_pert_upd_fcbias_fp32_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_fcbias_fp32_closure_args_t DeeployOptNetwork__pert_upd_fcbias_fp32_closure_args = (_pert_upd_fcbias_fp32_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr};

  // _pert_upd_fcbias_fp32_closure CLOSURE CALL
  _pert_upd_fcbias_fp32_closure(&DeeployOptNetwork__pert_upd_fcbias_fp32_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref;
} _rqsp_upd_blocks1convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_upd_blocks1convbias_rqsadd_tiling_closure(void *_rqsp_upd_blocks1convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks1convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_upd_blocks1convbias_rqsadd_tiling_closure_args_t *)_rqsp_upd_blocks1convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_upd_blocks1convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_upd_blocks1convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_upd_blocks1convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_upd_blocks1convbias_rqsadd_chunk = (16 >> rqsp_upd_blocks1convbias_rqsadd_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_upd_blocks1convbias_rqsadd_chunk_start =
      (uint32_t)MIN(rqsp_upd_blocks1convbias_rqsadd_chunk * rqsp_upd_blocks1convbias_rqsadd_core_id, (uint32_t)16);
  uint32_t rqsp_upd_blocks1convbias_rqsadd_chunk_stop =
      (uint32_t)MIN(rqsp_upd_blocks1convbias_rqsadd_chunk_start + rqsp_upd_blocks1convbias_rqsadd_chunk, (uint32_t)16);
  uint32_t rqsp_upd_blocks1convbias_rqsadd_local_size = rqsp_upd_blocks1convbias_rqsadd_chunk_stop - rqsp_upd_blocks1convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_upd_blocks1convbias_rqsadd_channel_start_offset = rqsp_upd_blocks1convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 3 + rqsp_upd_blocks1convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_upd_blocks1convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref[rqsp_upd_blocks1convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref[rqsp_upd_blocks1convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref, 16, 1, chunk_seed,
        rqsp_upd_blocks1convbias_rqsadd_local_size, rqsp_upd_blocks1convbias_rqsadd_chunk_start, perturbation_sign,
        zo_eco_mom[3] + rqsp_upd_blocks1convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref[rqsp_upd_blocks1convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref[rqsp_upd_blocks1convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref, 16, 1, chunk_seed,
        rqsp_upd_blocks1convbias_rqsadd_local_size, rqsp_upd_blocks1convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_upd_blocks1convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref;
} _rqsp_upd_blocks1convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_upd_blocks1convbias_rqsadd_cluster_fork(void *_rqsp_upd_blocks1convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks1convbias_rqsadd_cluster_fork_args_t *args =
      (_rqsp_upd_blocks1convbias_rqsadd_cluster_fork_args_t *)_rqsp_upd_blocks1convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks1convbias_rqsadd_tiling_closure_args_t DeeployOptNetwork__rqsp_upd_blocks1convbias_rqsadd_tiling_closure_args =
      (_rqsp_upd_blocks1convbias_rqsadd_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref};

  // _rqsp_upd_blocks1convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_upd_blocks1convbias_rqsadd_tiling_closure(&DeeployOptNetwork__rqsp_upd_blocks1convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr;
} _rqsp_upd_blocks1convbias_rqsadd_closure_args_t;

static void _rqsp_upd_blocks1convbias_rqsadd_closure(void *_rqsp_upd_blocks1convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks1convbias_rqsadd_closure_args_t *args = (_rqsp_upd_blocks1convbias_rqsadd_closure_args_t *)_rqsp_upd_blocks1convbias_rqsadd_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_input_3_ref =
      (void *)((char *)DeeployOptNetwork_input_3 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_blocks1convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployOptNetwork_blocks1convbias_rqsadd_pmul_tensor +
               DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_cumByteOffset
                   [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_output_3_ref =
      (void *)((char *)DeeployOptNetwork_output_3 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_input_3_ref);
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_blocks1convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_upd_blocks1convbias_rqsadd_cluster_fork_args_t DeeployOptNetwork__rqsp_upd_blocks1convbias_rqsadd_cluster_fork_args =
        (_rqsp_upd_blocks1convbias_rqsadd_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_upd_blocks1convbias_rqsadd_cluster_fork, &DeeployOptNetwork__rqsp_upd_blocks1convbias_rqsadd_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_output_3_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr;
} _rqsp_upd_blocks1convbias_rqsadd_closure_L3_args_t;

static void _rqsp_upd_blocks1convbias_rqsadd_closure_L3(void *_rqsp_upd_blocks1convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks1convbias_rqsadd_closure_L3_args_t *args =
      (_rqsp_upd_blocks1convbias_rqsadd_closure_L3_args_t *)_rqsp_upd_blocks1convbias_rqsadd_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks1convbias_rqsadd_closure_args_t DeeployOptNetwork__rqsp_upd_blocks1convbias_rqsadd_closure_args =
      (_rqsp_upd_blocks1convbias_rqsadd_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr};

  // _rqsp_upd_blocks1convbias_rqsadd_closure CLOSURE CALL
  _rqsp_upd_blocks1convbias_rqsadd_closure(&DeeployOptNetwork__rqsp_upd_blocks1convbias_rqsadd_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref;
} _rqsp_upd_blocks2convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_upd_blocks2convbias_rqsadd_tiling_closure(void *_rqsp_upd_blocks2convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks2convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_upd_blocks2convbias_rqsadd_tiling_closure_args_t *)_rqsp_upd_blocks2convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_upd_blocks2convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_upd_blocks2convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_upd_blocks2convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_upd_blocks2convbias_rqsadd_chunk = (16 >> rqsp_upd_blocks2convbias_rqsadd_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_upd_blocks2convbias_rqsadd_chunk_start =
      (uint32_t)MIN(rqsp_upd_blocks2convbias_rqsadd_chunk * rqsp_upd_blocks2convbias_rqsadd_core_id, (uint32_t)16);
  uint32_t rqsp_upd_blocks2convbias_rqsadd_chunk_stop =
      (uint32_t)MIN(rqsp_upd_blocks2convbias_rqsadd_chunk_start + rqsp_upd_blocks2convbias_rqsadd_chunk, (uint32_t)16);
  uint32_t rqsp_upd_blocks2convbias_rqsadd_local_size = rqsp_upd_blocks2convbias_rqsadd_chunk_stop - rqsp_upd_blocks2convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_upd_blocks2convbias_rqsadd_channel_start_offset = rqsp_upd_blocks2convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 5 + rqsp_upd_blocks2convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_upd_blocks2convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref[rqsp_upd_blocks2convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref[rqsp_upd_blocks2convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref, 16, 1, chunk_seed,
        rqsp_upd_blocks2convbias_rqsadd_local_size, rqsp_upd_blocks2convbias_rqsadd_chunk_start, perturbation_sign,
        zo_eco_mom[5] + rqsp_upd_blocks2convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref[rqsp_upd_blocks2convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref[rqsp_upd_blocks2convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref, 16, 1, chunk_seed,
        rqsp_upd_blocks2convbias_rqsadd_local_size, rqsp_upd_blocks2convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_upd_blocks2convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref;
} _rqsp_upd_blocks2convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_upd_blocks2convbias_rqsadd_cluster_fork(void *_rqsp_upd_blocks2convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks2convbias_rqsadd_cluster_fork_args_t *args =
      (_rqsp_upd_blocks2convbias_rqsadd_cluster_fork_args_t *)_rqsp_upd_blocks2convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks2convbias_rqsadd_tiling_closure_args_t DeeployOptNetwork__rqsp_upd_blocks2convbias_rqsadd_tiling_closure_args =
      (_rqsp_upd_blocks2convbias_rqsadd_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref};

  // _rqsp_upd_blocks2convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_upd_blocks2convbias_rqsadd_tiling_closure(&DeeployOptNetwork__rqsp_upd_blocks2convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr;
} _rqsp_upd_blocks2convbias_rqsadd_closure_args_t;

static void _rqsp_upd_blocks2convbias_rqsadd_closure(void *_rqsp_upd_blocks2convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks2convbias_rqsadd_closure_args_t *args = (_rqsp_upd_blocks2convbias_rqsadd_closure_args_t *)_rqsp_upd_blocks2convbias_rqsadd_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_input_5_ref =
      (void *)((char *)DeeployOptNetwork_input_5 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_blocks2convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployOptNetwork_blocks2convbias_rqsadd_pmul_tensor +
               DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_cumByteOffset
                   [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_output_5_ref =
      (void *)((char *)DeeployOptNetwork_output_5 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_input_5_ref);
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_blocks2convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_upd_blocks2convbias_rqsadd_cluster_fork_args_t DeeployOptNetwork__rqsp_upd_blocks2convbias_rqsadd_cluster_fork_args =
        (_rqsp_upd_blocks2convbias_rqsadd_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_upd_blocks2convbias_rqsadd_cluster_fork, &DeeployOptNetwork__rqsp_upd_blocks2convbias_rqsadd_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_output_5_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr;
} _rqsp_upd_blocks2convbias_rqsadd_closure_L3_args_t;

static void _rqsp_upd_blocks2convbias_rqsadd_closure_L3(void *_rqsp_upd_blocks2convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks2convbias_rqsadd_closure_L3_args_t *args =
      (_rqsp_upd_blocks2convbias_rqsadd_closure_L3_args_t *)_rqsp_upd_blocks2convbias_rqsadd_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks2convbias_rqsadd_closure_args_t DeeployOptNetwork__rqsp_upd_blocks2convbias_rqsadd_closure_args =
      (_rqsp_upd_blocks2convbias_rqsadd_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr};

  // _rqsp_upd_blocks2convbias_rqsadd_closure CLOSURE CALL
  _rqsp_upd_blocks2convbias_rqsadd_closure(&DeeployOptNetwork__rqsp_upd_blocks2convbias_rqsadd_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref;
} _pert_upd_blocks1bnweight_tiling_closure_args_t;

static void _pert_upd_blocks1bnweight_tiling_closure(void *_pert_upd_blocks1bnweight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks1bnweight_tiling_closure_args_t *args = (_pert_upd_blocks1bnweight_tiling_closure_args_t *)_pert_upd_blocks1bnweight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_blocks1bnweight, Op: PerturbRademacher)
  uint8_t pert_upd_blocks1bnweight_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_blocks1bnweight_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_blocks1bnweight_chunk = (16 >> pert_upd_blocks1bnweight_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_blocks1bnweight_chunk_start = (uint32_t)MIN(pert_upd_blocks1bnweight_chunk * pert_upd_blocks1bnweight_core_id, (uint32_t)16);
  uint32_t pert_upd_blocks1bnweight_chunk_stop = (uint32_t)MIN(pert_upd_blocks1bnweight_chunk_start + pert_upd_blocks1bnweight_chunk, (uint32_t)16);
  uint32_t pert_upd_blocks1bnweight_local_size = pert_upd_blocks1bnweight_chunk_stop - pert_upd_blocks1bnweight_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 14 + pert_upd_blocks1bnweight_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_blocks1bnweight_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref[pert_upd_blocks1bnweight_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref[pert_upd_blocks1bnweight_chunk_start],
                            chunk_seed, perturbation_sign, pert_upd_blocks1bnweight_local_size, zo_eco_mom[14] + pert_upd_blocks1bnweight_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref[pert_upd_blocks1bnweight_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref[pert_upd_blocks1bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks1bnweight_local_size, pert_upd_blocks1bnweight_eps);
#else
    ApplyRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref[pert_upd_blocks1bnweight_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref[pert_upd_blocks1bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks1bnweight_local_size, pert_upd_blocks1bnweight_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref;
} _pert_upd_blocks1bnweight_cluster_fork_args_t;

static void _pert_upd_blocks1bnweight_cluster_fork(void *_pert_upd_blocks1bnweight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks1bnweight_cluster_fork_args_t *args = (_pert_upd_blocks1bnweight_cluster_fork_args_t *)_pert_upd_blocks1bnweight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks1bnweight_tiling_closure_args_t DeeployOptNetwork__pert_upd_blocks1bnweight_tiling_closure_args =
      (_pert_upd_blocks1bnweight_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref,
                                                        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref};

  // _pert_upd_blocks1bnweight_tiling_closure CLOSURE CALL
  _pert_upd_blocks1bnweight_tiling_closure(&DeeployOptNetwork__pert_upd_blocks1bnweight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr;
} _pert_upd_blocks1bnweight_closure_args_t;

static void _pert_upd_blocks1bnweight_closure(void *_pert_upd_blocks1bnweight_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks1bnweight_closure_args_t *args = (_pert_upd_blocks1bnweight_closure_args_t *)_pert_upd_blocks1bnweight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_input_14_ref =
      (void *)((char *)DeeployOptNetwork_input_14 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_output_14_ref =
      (void *)((char *)DeeployOptNetwork_output_14 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_input_14_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_blocks1bnweight_cluster_fork_args_t DeeployOptNetwork__pert_upd_blocks1bnweight_cluster_fork_args =
        (_pert_upd_blocks1bnweight_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_in_ref,
                                                        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_blocks1bnweight_cluster_fork, &DeeployOptNetwork__pert_upd_blocks1bnweight_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_output_14_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr;
} _pert_upd_blocks1bnweight_closure_L3_args_t;

static void _pert_upd_blocks1bnweight_closure_L3(void *_pert_upd_blocks1bnweight_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks1bnweight_closure_L3_args_t *args = (_pert_upd_blocks1bnweight_closure_L3_args_t *)_pert_upd_blocks1bnweight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks1bnweight_closure_args_t DeeployOptNetwork__pert_upd_blocks1bnweight_closure_args = (_pert_upd_blocks1bnweight_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr};

  // _pert_upd_blocks1bnweight_closure CLOSURE CALL
  _pert_upd_blocks1bnweight_closure(&DeeployOptNetwork__pert_upd_blocks1bnweight_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref;
} _pert_upd_blocks1bnbias_tiling_closure_args_t;

static void _pert_upd_blocks1bnbias_tiling_closure(void *_pert_upd_blocks1bnbias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks1bnbias_tiling_closure_args_t *args = (_pert_upd_blocks1bnbias_tiling_closure_args_t *)_pert_upd_blocks1bnbias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_blocks1bnbias, Op: PerturbRademacher)
  uint8_t pert_upd_blocks1bnbias_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_blocks1bnbias_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_blocks1bnbias_chunk = (16 >> pert_upd_blocks1bnbias_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_blocks1bnbias_chunk_start = (uint32_t)MIN(pert_upd_blocks1bnbias_chunk * pert_upd_blocks1bnbias_core_id, (uint32_t)16);
  uint32_t pert_upd_blocks1bnbias_chunk_stop = (uint32_t)MIN(pert_upd_blocks1bnbias_chunk_start + pert_upd_blocks1bnbias_chunk, (uint32_t)16);
  uint32_t pert_upd_blocks1bnbias_local_size = pert_upd_blocks1bnbias_chunk_stop - pert_upd_blocks1bnbias_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 15 + pert_upd_blocks1bnbias_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_blocks1bnbias_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref[pert_upd_blocks1bnbias_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref[pert_upd_blocks1bnbias_chunk_start],
                            chunk_seed, perturbation_sign, pert_upd_blocks1bnbias_local_size, zo_eco_mom[15] + pert_upd_blocks1bnbias_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref[pert_upd_blocks1bnbias_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref[pert_upd_blocks1bnbias_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks1bnbias_local_size, pert_upd_blocks1bnbias_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref[pert_upd_blocks1bnbias_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref[pert_upd_blocks1bnbias_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_upd_blocks1bnbias_local_size, pert_upd_blocks1bnbias_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref;
} _pert_upd_blocks1bnbias_cluster_fork_args_t;

static void _pert_upd_blocks1bnbias_cluster_fork(void *_pert_upd_blocks1bnbias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks1bnbias_cluster_fork_args_t *args = (_pert_upd_blocks1bnbias_cluster_fork_args_t *)_pert_upd_blocks1bnbias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks1bnbias_tiling_closure_args_t DeeployOptNetwork__pert_upd_blocks1bnbias_tiling_closure_args = (_pert_upd_blocks1bnbias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref};

  // _pert_upd_blocks1bnbias_tiling_closure CLOSURE CALL
  _pert_upd_blocks1bnbias_tiling_closure(&DeeployOptNetwork__pert_upd_blocks1bnbias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr;
} _pert_upd_blocks1bnbias_closure_args_t;

static void _pert_upd_blocks1bnbias_closure(void *_pert_upd_blocks1bnbias_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks1bnbias_closure_args_t *args = (_pert_upd_blocks1bnbias_closure_args_t *)_pert_upd_blocks1bnbias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_input_15_ref =
      (void *)((char *)DeeployOptNetwork_input_15 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_output_15_ref =
      (void *)((char *)DeeployOptNetwork_output_15 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_input_15_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_blocks1bnbias_cluster_fork_args_t DeeployOptNetwork__pert_upd_blocks1bnbias_cluster_fork_args = (_pert_upd_blocks1bnbias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_in_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_blocks1bnbias_cluster_fork, &DeeployOptNetwork__pert_upd_blocks1bnbias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_output_15_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr;
} _pert_upd_blocks1bnbias_closure_L3_args_t;

static void _pert_upd_blocks1bnbias_closure_L3(void *_pert_upd_blocks1bnbias_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks1bnbias_closure_L3_args_t *args = (_pert_upd_blocks1bnbias_closure_L3_args_t *)_pert_upd_blocks1bnbias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks1bnbias_closure_args_t DeeployOptNetwork__pert_upd_blocks1bnbias_closure_args = (_pert_upd_blocks1bnbias_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr};

  // _pert_upd_blocks1bnbias_closure CLOSURE CALL
  _pert_upd_blocks1bnbias_closure(&DeeployOptNetwork__pert_upd_blocks1bnbias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref;
} _pert_upd_blocks2bnweight_tiling_closure_args_t;

static void _pert_upd_blocks2bnweight_tiling_closure(void *_pert_upd_blocks2bnweight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks2bnweight_tiling_closure_args_t *args = (_pert_upd_blocks2bnweight_tiling_closure_args_t *)_pert_upd_blocks2bnweight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_blocks2bnweight, Op: PerturbRademacher)
  uint8_t pert_upd_blocks2bnweight_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_blocks2bnweight_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_blocks2bnweight_chunk = (16 >> pert_upd_blocks2bnweight_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_blocks2bnweight_chunk_start = (uint32_t)MIN(pert_upd_blocks2bnweight_chunk * pert_upd_blocks2bnweight_core_id, (uint32_t)16);
  uint32_t pert_upd_blocks2bnweight_chunk_stop = (uint32_t)MIN(pert_upd_blocks2bnweight_chunk_start + pert_upd_blocks2bnweight_chunk, (uint32_t)16);
  uint32_t pert_upd_blocks2bnweight_local_size = pert_upd_blocks2bnweight_chunk_stop - pert_upd_blocks2bnweight_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 16 + pert_upd_blocks2bnweight_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_blocks2bnweight_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref[pert_upd_blocks2bnweight_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref[pert_upd_blocks2bnweight_chunk_start],
                            chunk_seed, perturbation_sign, pert_upd_blocks2bnweight_local_size, zo_eco_mom[16] + pert_upd_blocks2bnweight_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref[pert_upd_blocks2bnweight_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref[pert_upd_blocks2bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks2bnweight_local_size, pert_upd_blocks2bnweight_eps);
#else
    ApplyRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref[pert_upd_blocks2bnweight_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref[pert_upd_blocks2bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks2bnweight_local_size, pert_upd_blocks2bnweight_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref;
} _pert_upd_blocks2bnweight_cluster_fork_args_t;

static void _pert_upd_blocks2bnweight_cluster_fork(void *_pert_upd_blocks2bnweight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks2bnweight_cluster_fork_args_t *args = (_pert_upd_blocks2bnweight_cluster_fork_args_t *)_pert_upd_blocks2bnweight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks2bnweight_tiling_closure_args_t DeeployOptNetwork__pert_upd_blocks2bnweight_tiling_closure_args =
      (_pert_upd_blocks2bnweight_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref,
                                                        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref};

  // _pert_upd_blocks2bnweight_tiling_closure CLOSURE CALL
  _pert_upd_blocks2bnweight_tiling_closure(&DeeployOptNetwork__pert_upd_blocks2bnweight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr;
} _pert_upd_blocks2bnweight_closure_args_t;

static void _pert_upd_blocks2bnweight_closure(void *_pert_upd_blocks2bnweight_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks2bnweight_closure_args_t *args = (_pert_upd_blocks2bnweight_closure_args_t *)_pert_upd_blocks2bnweight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_input_16_ref =
      (void *)((char *)DeeployOptNetwork_input_16 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_output_16_ref =
      (void *)((char *)DeeployOptNetwork_output_16 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_input_16_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_blocks2bnweight_cluster_fork_args_t DeeployOptNetwork__pert_upd_blocks2bnweight_cluster_fork_args =
        (_pert_upd_blocks2bnweight_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_in_ref,
                                                        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_blocks2bnweight_cluster_fork, &DeeployOptNetwork__pert_upd_blocks2bnweight_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_output_16_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr;
} _pert_upd_blocks2bnweight_closure_L3_args_t;

static void _pert_upd_blocks2bnweight_closure_L3(void *_pert_upd_blocks2bnweight_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks2bnweight_closure_L3_args_t *args = (_pert_upd_blocks2bnweight_closure_L3_args_t *)_pert_upd_blocks2bnweight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks2bnweight_closure_args_t DeeployOptNetwork__pert_upd_blocks2bnweight_closure_args = (_pert_upd_blocks2bnweight_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr};

  // _pert_upd_blocks2bnweight_closure CLOSURE CALL
  _pert_upd_blocks2bnweight_closure(&DeeployOptNetwork__pert_upd_blocks2bnweight_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref;
} _pert_upd_blocks2bnbias_tiling_closure_args_t;

static void _pert_upd_blocks2bnbias_tiling_closure(void *_pert_upd_blocks2bnbias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks2bnbias_tiling_closure_args_t *args = (_pert_upd_blocks2bnbias_tiling_closure_args_t *)_pert_upd_blocks2bnbias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_blocks2bnbias, Op: PerturbRademacher)
  uint8_t pert_upd_blocks2bnbias_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_blocks2bnbias_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_blocks2bnbias_chunk = (16 >> pert_upd_blocks2bnbias_log2Core) + ((16 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_blocks2bnbias_chunk_start = (uint32_t)MIN(pert_upd_blocks2bnbias_chunk * pert_upd_blocks2bnbias_core_id, (uint32_t)16);
  uint32_t pert_upd_blocks2bnbias_chunk_stop = (uint32_t)MIN(pert_upd_blocks2bnbias_chunk_start + pert_upd_blocks2bnbias_chunk, (uint32_t)16);
  uint32_t pert_upd_blocks2bnbias_local_size = pert_upd_blocks2bnbias_chunk_stop - pert_upd_blocks2bnbias_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 17 + pert_upd_blocks2bnbias_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_blocks2bnbias_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref[pert_upd_blocks2bnbias_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref[pert_upd_blocks2bnbias_chunk_start],
                            chunk_seed, perturbation_sign, pert_upd_blocks2bnbias_local_size, zo_eco_mom[17] + pert_upd_blocks2bnbias_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref[pert_upd_blocks2bnbias_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref[pert_upd_blocks2bnbias_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks2bnbias_local_size, pert_upd_blocks2bnbias_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref[pert_upd_blocks2bnbias_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref[pert_upd_blocks2bnbias_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_upd_blocks2bnbias_local_size, pert_upd_blocks2bnbias_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref;
} _pert_upd_blocks2bnbias_cluster_fork_args_t;

static void _pert_upd_blocks2bnbias_cluster_fork(void *_pert_upd_blocks2bnbias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks2bnbias_cluster_fork_args_t *args = (_pert_upd_blocks2bnbias_cluster_fork_args_t *)_pert_upd_blocks2bnbias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks2bnbias_tiling_closure_args_t DeeployOptNetwork__pert_upd_blocks2bnbias_tiling_closure_args = (_pert_upd_blocks2bnbias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref};

  // _pert_upd_blocks2bnbias_tiling_closure CLOSURE CALL
  _pert_upd_blocks2bnbias_tiling_closure(&DeeployOptNetwork__pert_upd_blocks2bnbias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr;
} _pert_upd_blocks2bnbias_closure_args_t;

static void _pert_upd_blocks2bnbias_closure(void *_pert_upd_blocks2bnbias_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks2bnbias_closure_args_t *args = (_pert_upd_blocks2bnbias_closure_args_t *)_pert_upd_blocks2bnbias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_input_17_ref =
      (void *)((char *)DeeployOptNetwork_input_17 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_output_17_ref =
      (void *)((char *)DeeployOptNetwork_output_17 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_input_17_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_blocks2bnbias_cluster_fork_args_t DeeployOptNetwork__pert_upd_blocks2bnbias_cluster_fork_args = (_pert_upd_blocks2bnbias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_in_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_blocks2bnbias_cluster_fork, &DeeployOptNetwork__pert_upd_blocks2bnbias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310784, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_output_17_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr;
} _pert_upd_blocks2bnbias_closure_L3_args_t;

static void _pert_upd_blocks2bnbias_closure_L3(void *_pert_upd_blocks2bnbias_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks2bnbias_closure_L3_args_t *args = (_pert_upd_blocks2bnbias_closure_L3_args_t *)_pert_upd_blocks2bnbias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks2bnbias_closure_args_t DeeployOptNetwork__pert_upd_blocks2bnbias_closure_args = (_pert_upd_blocks2bnbias_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr};

  // _pert_upd_blocks2bnbias_closure CLOSURE CALL
  _pert_upd_blocks2bnbias_closure(&DeeployOptNetwork__pert_upd_blocks2bnbias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref;
} _rqsp_upd_blocks0convweight_int8_tiling_closure_args_t;

static void _rqsp_upd_blocks0convweight_int8_tiling_closure(void *_rqsp_upd_blocks0convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks0convweight_int8_tiling_closure_args_t *args =
      (_rqsp_upd_blocks0convweight_int8_tiling_closure_args_t *)_rqsp_upd_blocks0convweight_int8_tiling_closure_args;

  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_upd_blocks0convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_upd_blocks0convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_upd_blocks0convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_upd_blocks0convweight_int8_chunk = (32 >> rqsp_upd_blocks0convweight_int8_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_upd_blocks0convweight_int8_chunk_start =
      (uint32_t)MIN(rqsp_upd_blocks0convweight_int8_chunk * rqsp_upd_blocks0convweight_int8_core_id, (uint32_t)32);
  uint32_t rqsp_upd_blocks0convweight_int8_chunk_stop =
      (uint32_t)MIN(rqsp_upd_blocks0convweight_int8_chunk_start + rqsp_upd_blocks0convweight_int8_chunk, (uint32_t)32);
  uint32_t rqsp_upd_blocks0convweight_int8_local_size = rqsp_upd_blocks0convweight_int8_chunk_stop - rqsp_upd_blocks0convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_upd_blocks0convweight_int8_channel_start_offset = rqsp_upd_blocks0convweight_int8_chunk_start % 4;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 0 + rqsp_upd_blocks0convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_upd_blocks0convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref[rqsp_upd_blocks0convweight_int8_chunk_start],
        (int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref[rqsp_upd_blocks0convweight_int8_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref, 15, 4, chunk_seed,
        rqsp_upd_blocks0convweight_int8_local_size, rqsp_upd_blocks0convweight_int8_chunk_start, perturbation_sign,
        zo_eco_mom[0] + rqsp_upd_blocks0convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref[rqsp_upd_blocks0convweight_int8_chunk_start],
        (int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref[rqsp_upd_blocks0convweight_int8_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref, 15, 4, chunk_seed,
        rqsp_upd_blocks0convweight_int8_local_size, rqsp_upd_blocks0convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_upd_blocks0convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref;
} _rqsp_upd_blocks0convweight_int8_cluster_fork_args_t;

static void _rqsp_upd_blocks0convweight_int8_cluster_fork(void *_rqsp_upd_blocks0convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks0convweight_int8_cluster_fork_args_t *args =
      (_rqsp_upd_blocks0convweight_int8_cluster_fork_args_t *)_rqsp_upd_blocks0convweight_int8_cluster_fork_args;

  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks0convweight_int8_tiling_closure_args_t DeeployOptNetwork__rqsp_upd_blocks0convweight_int8_tiling_closure_args =
      (_rqsp_upd_blocks0convweight_int8_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref};

  // _rqsp_upd_blocks0convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_upd_blocks0convweight_int8_tiling_closure(&DeeployOptNetwork__rqsp_upd_blocks0convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr;
} _rqsp_upd_blocks0convweight_int8_closure_args_t;

static void _rqsp_upd_blocks0convweight_int8_closure(void *_rqsp_upd_blocks0convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks0convweight_int8_closure_args_t *args = (_rqsp_upd_blocks0convweight_int8_closure_args_t *)_rqsp_upd_blocks0convweight_int8_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref = (int8_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 64);
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref = (int8_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 32);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_input_0_ref =
      (void *)((char *)DeeployOptNetwork_input_0 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_blocks0convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployOptNetwork_blocks0convweight_int8_pmul_tensor +
               DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_cumByteOffset
                   [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_output_0_ref =
      (void *)((char *)DeeployOptNetwork_output_0 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_input_0_ref);
    mchan_transfer_1d(1441824, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_blocks0convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_upd_blocks0convweight_int8_cluster_fork_args_t DeeployOptNetwork__rqsp_upd_blocks0convweight_int8_cluster_fork_args =
        (_rqsp_upd_blocks0convweight_int8_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_upd_blocks0convweight_int8_cluster_fork, &DeeployOptNetwork__rqsp_upd_blocks0convweight_int8_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_output_0_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr;
} _rqsp_upd_blocks0convweight_int8_closure_L3_args_t;

static void _rqsp_upd_blocks0convweight_int8_closure_L3(void *_rqsp_upd_blocks0convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks0convweight_int8_closure_L3_args_t *args =
      (_rqsp_upd_blocks0convweight_int8_closure_L3_args_t *)_rqsp_upd_blocks0convweight_int8_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks0convweight_int8_closure_args_t DeeployOptNetwork__rqsp_upd_blocks0convweight_int8_closure_args =
      (_rqsp_upd_blocks0convweight_int8_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr};

  // _rqsp_upd_blocks0convweight_int8_closure CLOSURE CALL
  _rqsp_upd_blocks0convweight_int8_closure(&DeeployOptNetwork__rqsp_upd_blocks0convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref;
} _rqsp_upd_blocks3convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_upd_blocks3convbias_rqsadd_tiling_closure(void *_rqsp_upd_blocks3convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks3convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_upd_blocks3convbias_rqsadd_tiling_closure_args_t *)_rqsp_upd_blocks3convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_upd_blocks3convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_upd_blocks3convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_upd_blocks3convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_upd_blocks3convbias_rqsadd_chunk = (32 >> rqsp_upd_blocks3convbias_rqsadd_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_upd_blocks3convbias_rqsadd_chunk_start =
      (uint32_t)MIN(rqsp_upd_blocks3convbias_rqsadd_chunk * rqsp_upd_blocks3convbias_rqsadd_core_id, (uint32_t)32);
  uint32_t rqsp_upd_blocks3convbias_rqsadd_chunk_stop =
      (uint32_t)MIN(rqsp_upd_blocks3convbias_rqsadd_chunk_start + rqsp_upd_blocks3convbias_rqsadd_chunk, (uint32_t)32);
  uint32_t rqsp_upd_blocks3convbias_rqsadd_local_size = rqsp_upd_blocks3convbias_rqsadd_chunk_stop - rqsp_upd_blocks3convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_upd_blocks3convbias_rqsadd_channel_start_offset = rqsp_upd_blocks3convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 7 + rqsp_upd_blocks3convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_upd_blocks3convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref[rqsp_upd_blocks3convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref[rqsp_upd_blocks3convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref, 16, 1, chunk_seed,
        rqsp_upd_blocks3convbias_rqsadd_local_size, rqsp_upd_blocks3convbias_rqsadd_chunk_start, perturbation_sign,
        zo_eco_mom[7] + rqsp_upd_blocks3convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref[rqsp_upd_blocks3convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref[rqsp_upd_blocks3convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref, 16, 1, chunk_seed,
        rqsp_upd_blocks3convbias_rqsadd_local_size, rqsp_upd_blocks3convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_upd_blocks3convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref;
} _rqsp_upd_blocks3convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_upd_blocks3convbias_rqsadd_cluster_fork(void *_rqsp_upd_blocks3convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks3convbias_rqsadd_cluster_fork_args_t *args =
      (_rqsp_upd_blocks3convbias_rqsadd_cluster_fork_args_t *)_rqsp_upd_blocks3convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks3convbias_rqsadd_tiling_closure_args_t DeeployOptNetwork__rqsp_upd_blocks3convbias_rqsadd_tiling_closure_args =
      (_rqsp_upd_blocks3convbias_rqsadd_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref};

  // _rqsp_upd_blocks3convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_upd_blocks3convbias_rqsadd_tiling_closure(&DeeployOptNetwork__rqsp_upd_blocks3convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr;
} _rqsp_upd_blocks3convbias_rqsadd_closure_args_t;

static void _rqsp_upd_blocks3convbias_rqsadd_closure(void *_rqsp_upd_blocks3convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks3convbias_rqsadd_closure_args_t *args = (_rqsp_upd_blocks3convbias_rqsadd_closure_args_t *)_rqsp_upd_blocks3convbias_rqsadd_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 256);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_input_7_ref =
      (void *)((char *)DeeployOptNetwork_input_7 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_blocks3convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployOptNetwork_blocks3convbias_rqsadd_pmul_tensor +
               DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_cumByteOffset
                   [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_output_7_ref =
      (void *)((char *)DeeployOptNetwork_output_7 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_input_7_ref);
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_blocks3convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_upd_blocks3convbias_rqsadd_cluster_fork_args_t DeeployOptNetwork__rqsp_upd_blocks3convbias_rqsadd_cluster_fork_args =
        (_rqsp_upd_blocks3convbias_rqsadd_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_upd_blocks3convbias_rqsadd_cluster_fork, &DeeployOptNetwork__rqsp_upd_blocks3convbias_rqsadd_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_output_7_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr;
} _rqsp_upd_blocks3convbias_rqsadd_closure_L3_args_t;

static void _rqsp_upd_blocks3convbias_rqsadd_closure_L3(void *_rqsp_upd_blocks3convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks3convbias_rqsadd_closure_L3_args_t *args =
      (_rqsp_upd_blocks3convbias_rqsadd_closure_L3_args_t *)_rqsp_upd_blocks3convbias_rqsadd_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks3convbias_rqsadd_closure_args_t DeeployOptNetwork__rqsp_upd_blocks3convbias_rqsadd_closure_args =
      (_rqsp_upd_blocks3convbias_rqsadd_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr};

  // _rqsp_upd_blocks3convbias_rqsadd_closure CLOSURE CALL
  _rqsp_upd_blocks3convbias_rqsadd_closure(&DeeployOptNetwork__rqsp_upd_blocks3convbias_rqsadd_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref;
} _rqsp_upd_blocks4convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_upd_blocks4convbias_rqsadd_tiling_closure(void *_rqsp_upd_blocks4convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks4convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_upd_blocks4convbias_rqsadd_tiling_closure_args_t *)_rqsp_upd_blocks4convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_upd_blocks4convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_upd_blocks4convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_upd_blocks4convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_upd_blocks4convbias_rqsadd_chunk = (32 >> rqsp_upd_blocks4convbias_rqsadd_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_upd_blocks4convbias_rqsadd_chunk_start =
      (uint32_t)MIN(rqsp_upd_blocks4convbias_rqsadd_chunk * rqsp_upd_blocks4convbias_rqsadd_core_id, (uint32_t)32);
  uint32_t rqsp_upd_blocks4convbias_rqsadd_chunk_stop =
      (uint32_t)MIN(rqsp_upd_blocks4convbias_rqsadd_chunk_start + rqsp_upd_blocks4convbias_rqsadd_chunk, (uint32_t)32);
  uint32_t rqsp_upd_blocks4convbias_rqsadd_local_size = rqsp_upd_blocks4convbias_rqsadd_chunk_stop - rqsp_upd_blocks4convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_upd_blocks4convbias_rqsadd_channel_start_offset = rqsp_upd_blocks4convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 9 + rqsp_upd_blocks4convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_upd_blocks4convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref[rqsp_upd_blocks4convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref[rqsp_upd_blocks4convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref, 16, 1, chunk_seed,
        rqsp_upd_blocks4convbias_rqsadd_local_size, rqsp_upd_blocks4convbias_rqsadd_chunk_start, perturbation_sign,
        zo_eco_mom[9] + rqsp_upd_blocks4convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref[rqsp_upd_blocks4convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref[rqsp_upd_blocks4convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref, 16, 1, chunk_seed,
        rqsp_upd_blocks4convbias_rqsadd_local_size, rqsp_upd_blocks4convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_upd_blocks4convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref;
} _rqsp_upd_blocks4convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_upd_blocks4convbias_rqsadd_cluster_fork(void *_rqsp_upd_blocks4convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks4convbias_rqsadd_cluster_fork_args_t *args =
      (_rqsp_upd_blocks4convbias_rqsadd_cluster_fork_args_t *)_rqsp_upd_blocks4convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks4convbias_rqsadd_tiling_closure_args_t DeeployOptNetwork__rqsp_upd_blocks4convbias_rqsadd_tiling_closure_args =
      (_rqsp_upd_blocks4convbias_rqsadd_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref};

  // _rqsp_upd_blocks4convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_upd_blocks4convbias_rqsadd_tiling_closure(&DeeployOptNetwork__rqsp_upd_blocks4convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr;
} _rqsp_upd_blocks4convbias_rqsadd_closure_args_t;

static void _rqsp_upd_blocks4convbias_rqsadd_closure(void *_rqsp_upd_blocks4convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks4convbias_rqsadd_closure_args_t *args = (_rqsp_upd_blocks4convbias_rqsadd_closure_args_t *)_rqsp_upd_blocks4convbias_rqsadd_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 256);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_input_9_ref =
      (void *)((char *)DeeployOptNetwork_input_9 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_blocks4convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployOptNetwork_blocks4convbias_rqsadd_pmul_tensor +
               DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_cumByteOffset
                   [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_output_9_ref =
      (void *)((char *)DeeployOptNetwork_output_9 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_input_9_ref);
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_blocks4convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_upd_blocks4convbias_rqsadd_cluster_fork_args_t DeeployOptNetwork__rqsp_upd_blocks4convbias_rqsadd_cluster_fork_args =
        (_rqsp_upd_blocks4convbias_rqsadd_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_upd_blocks4convbias_rqsadd_cluster_fork, &DeeployOptNetwork__rqsp_upd_blocks4convbias_rqsadd_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_output_9_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr;
} _rqsp_upd_blocks4convbias_rqsadd_closure_L3_args_t;

static void _rqsp_upd_blocks4convbias_rqsadd_closure_L3(void *_rqsp_upd_blocks4convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks4convbias_rqsadd_closure_L3_args_t *args =
      (_rqsp_upd_blocks4convbias_rqsadd_closure_L3_args_t *)_rqsp_upd_blocks4convbias_rqsadd_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks4convbias_rqsadd_closure_args_t DeeployOptNetwork__rqsp_upd_blocks4convbias_rqsadd_closure_args =
      (_rqsp_upd_blocks4convbias_rqsadd_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr};

  // _rqsp_upd_blocks4convbias_rqsadd_closure CLOSURE CALL
  _rqsp_upd_blocks4convbias_rqsadd_closure(&DeeployOptNetwork__rqsp_upd_blocks4convbias_rqsadd_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref;
} _pert_upd_blocks3bnweight_tiling_closure_args_t;

static void _pert_upd_blocks3bnweight_tiling_closure(void *_pert_upd_blocks3bnweight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks3bnweight_tiling_closure_args_t *args = (_pert_upd_blocks3bnweight_tiling_closure_args_t *)_pert_upd_blocks3bnweight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_blocks3bnweight, Op: PerturbRademacher)
  uint8_t pert_upd_blocks3bnweight_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_blocks3bnweight_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_blocks3bnweight_chunk = (32 >> pert_upd_blocks3bnweight_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_blocks3bnweight_chunk_start = (uint32_t)MIN(pert_upd_blocks3bnweight_chunk * pert_upd_blocks3bnweight_core_id, (uint32_t)32);
  uint32_t pert_upd_blocks3bnweight_chunk_stop = (uint32_t)MIN(pert_upd_blocks3bnweight_chunk_start + pert_upd_blocks3bnweight_chunk, (uint32_t)32);
  uint32_t pert_upd_blocks3bnweight_local_size = pert_upd_blocks3bnweight_chunk_stop - pert_upd_blocks3bnweight_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 18 + pert_upd_blocks3bnweight_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_blocks3bnweight_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref[pert_upd_blocks3bnweight_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref[pert_upd_blocks3bnweight_chunk_start],
                            chunk_seed, perturbation_sign, pert_upd_blocks3bnweight_local_size, zo_eco_mom[18] + pert_upd_blocks3bnweight_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref[pert_upd_blocks3bnweight_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref[pert_upd_blocks3bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks3bnweight_local_size, pert_upd_blocks3bnweight_eps);
#else
    ApplyRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref[pert_upd_blocks3bnweight_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref[pert_upd_blocks3bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks3bnweight_local_size, pert_upd_blocks3bnweight_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref;
} _pert_upd_blocks3bnweight_cluster_fork_args_t;

static void _pert_upd_blocks3bnweight_cluster_fork(void *_pert_upd_blocks3bnweight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks3bnweight_cluster_fork_args_t *args = (_pert_upd_blocks3bnweight_cluster_fork_args_t *)_pert_upd_blocks3bnweight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks3bnweight_tiling_closure_args_t DeeployOptNetwork__pert_upd_blocks3bnweight_tiling_closure_args =
      (_pert_upd_blocks3bnweight_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref,
                                                        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref};

  // _pert_upd_blocks3bnweight_tiling_closure CLOSURE CALL
  _pert_upd_blocks3bnweight_tiling_closure(&DeeployOptNetwork__pert_upd_blocks3bnweight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr;
} _pert_upd_blocks3bnweight_closure_args_t;

static void _pert_upd_blocks3bnweight_closure(void *_pert_upd_blocks3bnweight_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks3bnweight_closure_args_t *args = (_pert_upd_blocks3bnweight_closure_args_t *)_pert_upd_blocks3bnweight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_input_18_ref =
      (void *)((char *)DeeployOptNetwork_input_18 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_output_18_ref =
      (void *)((char *)DeeployOptNetwork_output_18 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_input_18_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_blocks3bnweight_cluster_fork_args_t DeeployOptNetwork__pert_upd_blocks3bnweight_cluster_fork_args =
        (_pert_upd_blocks3bnweight_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_in_ref,
                                                        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_blocks3bnweight_cluster_fork, &DeeployOptNetwork__pert_upd_blocks3bnweight_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_output_18_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr;
} _pert_upd_blocks3bnweight_closure_L3_args_t;

static void _pert_upd_blocks3bnweight_closure_L3(void *_pert_upd_blocks3bnweight_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks3bnweight_closure_L3_args_t *args = (_pert_upd_blocks3bnweight_closure_L3_args_t *)_pert_upd_blocks3bnweight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks3bnweight_closure_args_t DeeployOptNetwork__pert_upd_blocks3bnweight_closure_args = (_pert_upd_blocks3bnweight_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr};

  // _pert_upd_blocks3bnweight_closure CLOSURE CALL
  _pert_upd_blocks3bnweight_closure(&DeeployOptNetwork__pert_upd_blocks3bnweight_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref;
} _pert_upd_blocks3bnbias_tiling_closure_args_t;

static void _pert_upd_blocks3bnbias_tiling_closure(void *_pert_upd_blocks3bnbias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks3bnbias_tiling_closure_args_t *args = (_pert_upd_blocks3bnbias_tiling_closure_args_t *)_pert_upd_blocks3bnbias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_blocks3bnbias, Op: PerturbRademacher)
  uint8_t pert_upd_blocks3bnbias_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_blocks3bnbias_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_blocks3bnbias_chunk = (32 >> pert_upd_blocks3bnbias_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_blocks3bnbias_chunk_start = (uint32_t)MIN(pert_upd_blocks3bnbias_chunk * pert_upd_blocks3bnbias_core_id, (uint32_t)32);
  uint32_t pert_upd_blocks3bnbias_chunk_stop = (uint32_t)MIN(pert_upd_blocks3bnbias_chunk_start + pert_upd_blocks3bnbias_chunk, (uint32_t)32);
  uint32_t pert_upd_blocks3bnbias_local_size = pert_upd_blocks3bnbias_chunk_stop - pert_upd_blocks3bnbias_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 19 + pert_upd_blocks3bnbias_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_blocks3bnbias_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref[pert_upd_blocks3bnbias_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref[pert_upd_blocks3bnbias_chunk_start],
                            chunk_seed, perturbation_sign, pert_upd_blocks3bnbias_local_size, zo_eco_mom[19] + pert_upd_blocks3bnbias_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref[pert_upd_blocks3bnbias_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref[pert_upd_blocks3bnbias_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks3bnbias_local_size, pert_upd_blocks3bnbias_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref[pert_upd_blocks3bnbias_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref[pert_upd_blocks3bnbias_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_upd_blocks3bnbias_local_size, pert_upd_blocks3bnbias_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref;
} _pert_upd_blocks3bnbias_cluster_fork_args_t;

static void _pert_upd_blocks3bnbias_cluster_fork(void *_pert_upd_blocks3bnbias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks3bnbias_cluster_fork_args_t *args = (_pert_upd_blocks3bnbias_cluster_fork_args_t *)_pert_upd_blocks3bnbias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks3bnbias_tiling_closure_args_t DeeployOptNetwork__pert_upd_blocks3bnbias_tiling_closure_args = (_pert_upd_blocks3bnbias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref};

  // _pert_upd_blocks3bnbias_tiling_closure CLOSURE CALL
  _pert_upd_blocks3bnbias_tiling_closure(&DeeployOptNetwork__pert_upd_blocks3bnbias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr;
} _pert_upd_blocks3bnbias_closure_args_t;

static void _pert_upd_blocks3bnbias_closure(void *_pert_upd_blocks3bnbias_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks3bnbias_closure_args_t *args = (_pert_upd_blocks3bnbias_closure_args_t *)_pert_upd_blocks3bnbias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_input_19_ref =
      (void *)((char *)DeeployOptNetwork_input_19 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_output_19_ref =
      (void *)((char *)DeeployOptNetwork_output_19 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_input_19_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_blocks3bnbias_cluster_fork_args_t DeeployOptNetwork__pert_upd_blocks3bnbias_cluster_fork_args = (_pert_upd_blocks3bnbias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_in_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_blocks3bnbias_cluster_fork, &DeeployOptNetwork__pert_upd_blocks3bnbias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_output_19_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr;
} _pert_upd_blocks3bnbias_closure_L3_args_t;

static void _pert_upd_blocks3bnbias_closure_L3(void *_pert_upd_blocks3bnbias_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks3bnbias_closure_L3_args_t *args = (_pert_upd_blocks3bnbias_closure_L3_args_t *)_pert_upd_blocks3bnbias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks3bnbias_closure_args_t DeeployOptNetwork__pert_upd_blocks3bnbias_closure_args = (_pert_upd_blocks3bnbias_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr};

  // _pert_upd_blocks3bnbias_closure CLOSURE CALL
  _pert_upd_blocks3bnbias_closure(&DeeployOptNetwork__pert_upd_blocks3bnbias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref;
} _pert_upd_blocks4bnweight_tiling_closure_args_t;

static void _pert_upd_blocks4bnweight_tiling_closure(void *_pert_upd_blocks4bnweight_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks4bnweight_tiling_closure_args_t *args = (_pert_upd_blocks4bnweight_tiling_closure_args_t *)_pert_upd_blocks4bnweight_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_blocks4bnweight, Op: PerturbRademacher)
  uint8_t pert_upd_blocks4bnweight_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_blocks4bnweight_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_blocks4bnweight_chunk = (32 >> pert_upd_blocks4bnweight_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_blocks4bnweight_chunk_start = (uint32_t)MIN(pert_upd_blocks4bnweight_chunk * pert_upd_blocks4bnweight_core_id, (uint32_t)32);
  uint32_t pert_upd_blocks4bnweight_chunk_stop = (uint32_t)MIN(pert_upd_blocks4bnweight_chunk_start + pert_upd_blocks4bnweight_chunk, (uint32_t)32);
  uint32_t pert_upd_blocks4bnweight_local_size = pert_upd_blocks4bnweight_chunk_stop - pert_upd_blocks4bnweight_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 20 + pert_upd_blocks4bnweight_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_blocks4bnweight_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref[pert_upd_blocks4bnweight_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref[pert_upd_blocks4bnweight_chunk_start],
                            chunk_seed, perturbation_sign, pert_upd_blocks4bnweight_local_size, zo_eco_mom[20] + pert_upd_blocks4bnweight_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref[pert_upd_blocks4bnweight_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref[pert_upd_blocks4bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks4bnweight_local_size, pert_upd_blocks4bnweight_eps);
#else
    ApplyRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref[pert_upd_blocks4bnweight_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref[pert_upd_blocks4bnweight_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks4bnweight_local_size, pert_upd_blocks4bnweight_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref;
} _pert_upd_blocks4bnweight_cluster_fork_args_t;

static void _pert_upd_blocks4bnweight_cluster_fork(void *_pert_upd_blocks4bnweight_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks4bnweight_cluster_fork_args_t *args = (_pert_upd_blocks4bnweight_cluster_fork_args_t *)_pert_upd_blocks4bnweight_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks4bnweight_tiling_closure_args_t DeeployOptNetwork__pert_upd_blocks4bnweight_tiling_closure_args =
      (_pert_upd_blocks4bnweight_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref,
                                                        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref};

  // _pert_upd_blocks4bnweight_tiling_closure CLOSURE CALL
  _pert_upd_blocks4bnweight_tiling_closure(&DeeployOptNetwork__pert_upd_blocks4bnweight_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr;
} _pert_upd_blocks4bnweight_closure_args_t;

static void _pert_upd_blocks4bnweight_closure(void *_pert_upd_blocks4bnweight_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks4bnweight_closure_args_t *args = (_pert_upd_blocks4bnweight_closure_args_t *)_pert_upd_blocks4bnweight_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_input_20_ref =
      (void *)((char *)DeeployOptNetwork_input_20 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_output_20_ref =
      (void *)((char *)DeeployOptNetwork_output_20 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_input_20_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_blocks4bnweight_cluster_fork_args_t DeeployOptNetwork__pert_upd_blocks4bnweight_cluster_fork_args =
        (_pert_upd_blocks4bnweight_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_in_ref,
                                                        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_blocks4bnweight_cluster_fork, &DeeployOptNetwork__pert_upd_blocks4bnweight_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_output_20_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr;
} _pert_upd_blocks4bnweight_closure_L3_args_t;

static void _pert_upd_blocks4bnweight_closure_L3(void *_pert_upd_blocks4bnweight_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks4bnweight_closure_L3_args_t *args = (_pert_upd_blocks4bnweight_closure_L3_args_t *)_pert_upd_blocks4bnweight_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks4bnweight_closure_args_t DeeployOptNetwork__pert_upd_blocks4bnweight_closure_args = (_pert_upd_blocks4bnweight_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr};

  // _pert_upd_blocks4bnweight_closure CLOSURE CALL
  _pert_upd_blocks4bnweight_closure(&DeeployOptNetwork__pert_upd_blocks4bnweight_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref;
} _pert_upd_blocks4bnbias_tiling_closure_args_t;

static void _pert_upd_blocks4bnbias_tiling_closure(void *_pert_upd_blocks4bnbias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks4bnbias_tiling_closure_args_t *args = (_pert_upd_blocks4bnbias_tiling_closure_args_t *)_pert_upd_blocks4bnbias_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_blocks4bnbias, Op: PerturbRademacher)
  uint8_t pert_upd_blocks4bnbias_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_blocks4bnbias_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_blocks4bnbias_chunk = (32 >> pert_upd_blocks4bnbias_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_blocks4bnbias_chunk_start = (uint32_t)MIN(pert_upd_blocks4bnbias_chunk * pert_upd_blocks4bnbias_core_id, (uint32_t)32);
  uint32_t pert_upd_blocks4bnbias_chunk_stop = (uint32_t)MIN(pert_upd_blocks4bnbias_chunk_start + pert_upd_blocks4bnbias_chunk, (uint32_t)32);
  uint32_t pert_upd_blocks4bnbias_local_size = pert_upd_blocks4bnbias_chunk_stop - pert_upd_blocks4bnbias_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 21 + pert_upd_blocks4bnbias_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_blocks4bnbias_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref[pert_upd_blocks4bnbias_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref[pert_upd_blocks4bnbias_chunk_start],
                            chunk_seed, perturbation_sign, pert_upd_blocks4bnbias_local_size, zo_eco_mom[21] + pert_upd_blocks4bnbias_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref[pert_upd_blocks4bnbias_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref[pert_upd_blocks4bnbias_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_blocks4bnbias_local_size, pert_upd_blocks4bnbias_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref[pert_upd_blocks4bnbias_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref[pert_upd_blocks4bnbias_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_upd_blocks4bnbias_local_size, pert_upd_blocks4bnbias_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref;
} _pert_upd_blocks4bnbias_cluster_fork_args_t;

static void _pert_upd_blocks4bnbias_cluster_fork(void *_pert_upd_blocks4bnbias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks4bnbias_cluster_fork_args_t *args = (_pert_upd_blocks4bnbias_cluster_fork_args_t *)_pert_upd_blocks4bnbias_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks4bnbias_tiling_closure_args_t DeeployOptNetwork__pert_upd_blocks4bnbias_tiling_closure_args = (_pert_upd_blocks4bnbias_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref};

  // _pert_upd_blocks4bnbias_tiling_closure CLOSURE CALL
  _pert_upd_blocks4bnbias_tiling_closure(&DeeployOptNetwork__pert_upd_blocks4bnbias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr;
} _pert_upd_blocks4bnbias_closure_args_t;

static void _pert_upd_blocks4bnbias_closure(void *_pert_upd_blocks4bnbias_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks4bnbias_closure_args_t *args = (_pert_upd_blocks4bnbias_closure_args_t *)_pert_upd_blocks4bnbias_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 128);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_input_21_ref =
      (void *)((char *)DeeployOptNetwork_input_21 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_output_21_ref =
      (void *)((char *)DeeployOptNetwork_output_21 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_input_21_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_blocks4bnbias_cluster_fork_args_t DeeployOptNetwork__pert_upd_blocks4bnbias_cluster_fork_args = (_pert_upd_blocks4bnbias_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_in_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_blocks4bnbias_cluster_fork, &DeeployOptNetwork__pert_upd_blocks4bnbias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_output_21_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr;
} _pert_upd_blocks4bnbias_closure_L3_args_t;

static void _pert_upd_blocks4bnbias_closure_L3(void *_pert_upd_blocks4bnbias_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_blocks4bnbias_closure_L3_args_t *args = (_pert_upd_blocks4bnbias_closure_L3_args_t *)_pert_upd_blocks4bnbias_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_blocks4bnbias_closure_args_t DeeployOptNetwork__pert_upd_blocks4bnbias_closure_args = (_pert_upd_blocks4bnbias_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr};

  // _pert_upd_blocks4bnbias_closure CLOSURE CALL
  _pert_upd_blocks4bnbias_closure(&DeeployOptNetwork__pert_upd_blocks4bnbias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref;
} _pert_upd_fcweight_fp32_tiling_closure_args_t;

static void _pert_upd_fcweight_fp32_tiling_closure(void *_pert_upd_fcweight_fp32_tiling_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_fcweight_fp32_tiling_closure_args_t *args = (_pert_upd_fcweight_fp32_tiling_closure_args_t *)_pert_upd_fcweight_fp32_tiling_closure_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: pert_upd_fcweight_fp32, Op: PerturbRademacher)
  uint8_t pert_upd_fcweight_fp32_core_id = (uint8_t)pi_core_id();
  uint8_t pert_upd_fcweight_fp32_log2Core = (uint8_t)log2(NUM_CORES);
  uint32_t pert_upd_fcweight_fp32_chunk = (288 >> pert_upd_fcweight_fp32_log2Core) + ((288 & (NUM_CORES - 1)) != 0);
  uint32_t pert_upd_fcweight_fp32_chunk_start = (uint32_t)MIN(pert_upd_fcweight_fp32_chunk * pert_upd_fcweight_fp32_core_id, (uint32_t)288);
  uint32_t pert_upd_fcweight_fp32_chunk_stop = (uint32_t)MIN(pert_upd_fcweight_fp32_chunk_start + pert_upd_fcweight_fp32_chunk, (uint32_t)288);
  uint32_t pert_upd_fcweight_fp32_local_size = pert_upd_fcweight_fp32_chunk_stop - pert_upd_fcweight_fp32_chunk_start;

  // ZO runtime controls (ZORuntime.h): perturb_seed_base / perturb_eps_(use_)override. Neutral defaults
  // (0 / off) reproduce the baked 42/0.009999999776482582 behavior; the ZO runner sets them per pass. -- QW
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 10 + pert_upd_fcweight_fp32_core_id) ^ (0 * 0x9E3779B1u);
  float32_t pert_upd_fcweight_fp32_eps = perturb_eps_use_override ? perturb_eps_override : 0.009999999776482582f;
  // RW 2026-10-13: ECO (SGD momentum for fp32 tensors) in the update pass, see ZORuntime.h
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECORademacher_fp32((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref[pert_upd_fcweight_fp32_chunk_start],
                            (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref[pert_upd_fcweight_fp32_chunk_start],
                            chunk_seed, perturbation_sign, pert_upd_fcweight_fp32_local_size, zo_eco_mom[10] + pert_upd_fcweight_fp32_chunk_start);
  } else {
#ifdef USE_SEQUENTIAL_RADEMACHER
    ApplySequentialRademacherPerturbation(
        (const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref[pert_upd_fcweight_fp32_chunk_start],
        (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref[pert_upd_fcweight_fp32_chunk_start], chunk_seed,
        perturbation_sign, // ZO runtime control (ZORuntime.h)
        pert_upd_fcweight_fp32_local_size, pert_upd_fcweight_fp32_eps);
#else
    ApplyRademacherPerturbation((const float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref[pert_upd_fcweight_fp32_chunk_start],
                                (float32_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref[pert_upd_fcweight_fp32_chunk_start],
                                chunk_seed,
                                perturbation_sign, // ZO runtime control (ZORuntime.h)
                                pert_upd_fcweight_fp32_local_size, pert_upd_fcweight_fp32_eps);
#endif
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref;
} _pert_upd_fcweight_fp32_cluster_fork_args_t;

static void _pert_upd_fcweight_fp32_cluster_fork(void *_pert_upd_fcweight_fp32_cluster_fork_args) {
  // CLOSURE ARG CAST
  _pert_upd_fcweight_fp32_cluster_fork_args_t *args = (_pert_upd_fcweight_fp32_cluster_fork_args_t *)_pert_upd_fcweight_fp32_cluster_fork_args;

  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref;
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref;

  // CLOSURE FUNCTION CALL
  _pert_upd_fcweight_fp32_tiling_closure_args_t DeeployOptNetwork__pert_upd_fcweight_fp32_tiling_closure_args = (_pert_upd_fcweight_fp32_tiling_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref,
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref};

  // _pert_upd_fcweight_fp32_tiling_closure CLOSURE CALL
  _pert_upd_fcweight_fp32_tiling_closure(&DeeployOptNetwork__pert_upd_fcweight_fp32_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr;
} _pert_upd_fcweight_fp32_closure_args_t;

static void _pert_upd_fcweight_fp32_closure(void *_pert_upd_fcweight_fp32_closure_args) {
  // CLOSURE ARG CAST
  _pert_upd_fcweight_fp32_closure_args_t *args = (_pert_upd_fcweight_fp32_closure_args_t *)_pert_upd_fcweight_fp32_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  float32_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref = (float32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 1152);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_input_10_ref =
      (void *)((char *)DeeployOptNetwork_input_10 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_output_10_ref =
      (void *)((char *)DeeployOptNetwork_output_10 + DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_cumByteOffset
                                                         [*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_numTiles[*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr];
       TILING_I <
       DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_numTiles[(*DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442944, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_input_10_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _pert_upd_fcweight_fp32_cluster_fork_args_t DeeployOptNetwork__pert_upd_fcweight_fp32_cluster_fork_args = (_pert_upd_fcweight_fp32_cluster_fork_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_in_ref,
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_pert_upd_fcweight_fp32_cluster_fork, &DeeployOptNetwork__pert_upd_fcweight_fp32_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311872, DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_output_10_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr;
} _pert_upd_fcweight_fp32_closure_L3_args_t;

static void _pert_upd_fcweight_fp32_closure_L3(void *_pert_upd_fcweight_fp32_closure_L3_args) {
  // CLOSURE ARG CAST
  _pert_upd_fcweight_fp32_closure_L3_args_t *args = (_pert_upd_fcweight_fp32_closure_L3_args_t *)_pert_upd_fcweight_fp32_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr = args->DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _pert_upd_fcweight_fp32_closure_args_t DeeployOptNetwork__pert_upd_fcweight_fp32_closure_args = (_pert_upd_fcweight_fp32_closure_args_t){
      .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr};

  // _pert_upd_fcweight_fp32_closure CLOSURE CALL
  _pert_upd_fcweight_fp32_closure(&DeeployOptNetwork__pert_upd_fcweight_fp32_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref;
} _rqsp_upd_blocks1convweight_int8_tiling_closure_args_t;

static void _rqsp_upd_blocks1convweight_int8_tiling_closure(void *_rqsp_upd_blocks1convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks1convweight_int8_tiling_closure_args_t *args =
      (_rqsp_upd_blocks1convweight_int8_tiling_closure_args_t *)_rqsp_upd_blocks1convweight_int8_tiling_closure_args;

  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_upd_blocks1convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_upd_blocks1convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_upd_blocks1convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_upd_blocks1convweight_int8_chunk = (2048 >> rqsp_upd_blocks1convweight_int8_log2Core) + ((2048 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_upd_blocks1convweight_int8_chunk_start =
      (uint32_t)MIN(rqsp_upd_blocks1convweight_int8_chunk * rqsp_upd_blocks1convweight_int8_core_id, (uint32_t)2048);
  uint32_t rqsp_upd_blocks1convweight_int8_chunk_stop =
      (uint32_t)MIN(rqsp_upd_blocks1convweight_int8_chunk_start + rqsp_upd_blocks1convweight_int8_chunk, (uint32_t)2048);
  uint32_t rqsp_upd_blocks1convweight_int8_local_size = rqsp_upd_blocks1convweight_int8_chunk_stop - rqsp_upd_blocks1convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_upd_blocks1convweight_int8_channel_start_offset = rqsp_upd_blocks1convweight_int8_chunk_start % 128;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 2 + rqsp_upd_blocks1convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_upd_blocks1convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref[rqsp_upd_blocks1convweight_int8_chunk_start],
        (int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref[rqsp_upd_blocks1convweight_int8_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref, 15, 128, chunk_seed,
        rqsp_upd_blocks1convweight_int8_local_size, rqsp_upd_blocks1convweight_int8_chunk_start, perturbation_sign,
        zo_eco_mom[2] + rqsp_upd_blocks1convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref[rqsp_upd_blocks1convweight_int8_chunk_start],
        (int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref[rqsp_upd_blocks1convweight_int8_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref, 15, 128, chunk_seed,
        rqsp_upd_blocks1convweight_int8_local_size, rqsp_upd_blocks1convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_upd_blocks1convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref;
} _rqsp_upd_blocks1convweight_int8_cluster_fork_args_t;

static void _rqsp_upd_blocks1convweight_int8_cluster_fork(void *_rqsp_upd_blocks1convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks1convweight_int8_cluster_fork_args_t *args =
      (_rqsp_upd_blocks1convweight_int8_cluster_fork_args_t *)_rqsp_upd_blocks1convweight_int8_cluster_fork_args;

  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks1convweight_int8_tiling_closure_args_t DeeployOptNetwork__rqsp_upd_blocks1convweight_int8_tiling_closure_args =
      (_rqsp_upd_blocks1convweight_int8_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref};

  // _rqsp_upd_blocks1convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_upd_blocks1convweight_int8_tiling_closure(&DeeployOptNetwork__rqsp_upd_blocks1convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr;
} _rqsp_upd_blocks1convweight_int8_closure_args_t;

static void _rqsp_upd_blocks1convweight_int8_closure(void *_rqsp_upd_blocks1convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks1convweight_int8_closure_args_t *args = (_rqsp_upd_blocks1convweight_int8_closure_args_t *)_rqsp_upd_blocks1convweight_int8_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref = (int8_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 4096);
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref = (int8_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 2048);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_input_2_ref =
      (void *)((char *)DeeployOptNetwork_input_2 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_blocks1convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployOptNetwork_blocks1convweight_int8_pmul_tensor +
               DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_cumByteOffset
                   [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_output_2_ref =
      (void *)((char *)DeeployOptNetwork_output_2 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_input_2_ref);
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_blocks1convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_upd_blocks1convweight_int8_cluster_fork_args_t DeeployOptNetwork__rqsp_upd_blocks1convweight_int8_cluster_fork_args =
        (_rqsp_upd_blocks1convweight_int8_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_upd_blocks1convweight_int8_cluster_fork, &DeeployOptNetwork__rqsp_upd_blocks1convweight_int8_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312768, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_output_2_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr;
} _rqsp_upd_blocks1convweight_int8_closure_L3_args_t;

static void _rqsp_upd_blocks1convweight_int8_closure_L3(void *_rqsp_upd_blocks1convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks1convweight_int8_closure_L3_args_t *args =
      (_rqsp_upd_blocks1convweight_int8_closure_L3_args_t *)_rqsp_upd_blocks1convweight_int8_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks1convweight_int8_closure_args_t DeeployOptNetwork__rqsp_upd_blocks1convweight_int8_closure_args =
      (_rqsp_upd_blocks1convweight_int8_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr};

  // _rqsp_upd_blocks1convweight_int8_closure CLOSURE CALL
  _rqsp_upd_blocks1convweight_int8_closure(&DeeployOptNetwork__rqsp_upd_blocks1convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref;
} _rqsp_upd_blocks2convweight_int8_tiling_closure_args_t;

static void _rqsp_upd_blocks2convweight_int8_tiling_closure(void *_rqsp_upd_blocks2convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks2convweight_int8_tiling_closure_args_t *args =
      (_rqsp_upd_blocks2convweight_int8_tiling_closure_args_t *)_rqsp_upd_blocks2convweight_int8_tiling_closure_args;

  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_upd_blocks2convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_upd_blocks2convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_upd_blocks2convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_upd_blocks2convweight_int8_chunk = (2048 >> rqsp_upd_blocks2convweight_int8_log2Core) + ((2048 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_upd_blocks2convweight_int8_chunk_start =
      (uint32_t)MIN(rqsp_upd_blocks2convweight_int8_chunk * rqsp_upd_blocks2convweight_int8_core_id, (uint32_t)2048);
  uint32_t rqsp_upd_blocks2convweight_int8_chunk_stop =
      (uint32_t)MIN(rqsp_upd_blocks2convweight_int8_chunk_start + rqsp_upd_blocks2convweight_int8_chunk, (uint32_t)2048);
  uint32_t rqsp_upd_blocks2convweight_int8_local_size = rqsp_upd_blocks2convweight_int8_chunk_stop - rqsp_upd_blocks2convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_upd_blocks2convweight_int8_channel_start_offset = rqsp_upd_blocks2convweight_int8_chunk_start % 128;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 4 + rqsp_upd_blocks2convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_upd_blocks2convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref[rqsp_upd_blocks2convweight_int8_chunk_start],
        (int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref[rqsp_upd_blocks2convweight_int8_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref, 15, 128, chunk_seed,
        rqsp_upd_blocks2convweight_int8_local_size, rqsp_upd_blocks2convweight_int8_chunk_start, perturbation_sign,
        zo_eco_mom[4] + rqsp_upd_blocks2convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref[rqsp_upd_blocks2convweight_int8_chunk_start],
        (int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref[rqsp_upd_blocks2convweight_int8_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref, 15, 128, chunk_seed,
        rqsp_upd_blocks2convweight_int8_local_size, rqsp_upd_blocks2convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_upd_blocks2convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref;
} _rqsp_upd_blocks2convweight_int8_cluster_fork_args_t;

static void _rqsp_upd_blocks2convweight_int8_cluster_fork(void *_rqsp_upd_blocks2convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks2convweight_int8_cluster_fork_args_t *args =
      (_rqsp_upd_blocks2convweight_int8_cluster_fork_args_t *)_rqsp_upd_blocks2convweight_int8_cluster_fork_args;

  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks2convweight_int8_tiling_closure_args_t DeeployOptNetwork__rqsp_upd_blocks2convweight_int8_tiling_closure_args =
      (_rqsp_upd_blocks2convweight_int8_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref};

  // _rqsp_upd_blocks2convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_upd_blocks2convweight_int8_tiling_closure(&DeeployOptNetwork__rqsp_upd_blocks2convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr;
} _rqsp_upd_blocks2convweight_int8_closure_args_t;

static void _rqsp_upd_blocks2convweight_int8_closure(void *_rqsp_upd_blocks2convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks2convweight_int8_closure_args_t *args = (_rqsp_upd_blocks2convweight_int8_closure_args_t *)_rqsp_upd_blocks2convweight_int8_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref = (int8_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 4096);
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref = (int8_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 2048);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_input_4_ref =
      (void *)((char *)DeeployOptNetwork_input_4 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_blocks2convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployOptNetwork_blocks2convweight_int8_pmul_tensor +
               DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_cumByteOffset
                   [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_output_4_ref =
      (void *)((char *)DeeployOptNetwork_output_4 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_input_4_ref);
    mchan_transfer_1d(1441856, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_blocks2convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_upd_blocks2convweight_int8_cluster_fork_args_t DeeployOptNetwork__rqsp_upd_blocks2convweight_int8_cluster_fork_args =
        (_rqsp_upd_blocks2convweight_int8_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_upd_blocks2convweight_int8_cluster_fork, &DeeployOptNetwork__rqsp_upd_blocks2convweight_int8_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312768, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_output_4_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr;
} _rqsp_upd_blocks2convweight_int8_closure_L3_args_t;

static void _rqsp_upd_blocks2convweight_int8_closure_L3(void *_rqsp_upd_blocks2convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks2convweight_int8_closure_L3_args_t *args =
      (_rqsp_upd_blocks2convweight_int8_closure_L3_args_t *)_rqsp_upd_blocks2convweight_int8_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks2convweight_int8_closure_args_t DeeployOptNetwork__rqsp_upd_blocks2convweight_int8_closure_args =
      (_rqsp_upd_blocks2convweight_int8_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr};

  // _rqsp_upd_blocks2convweight_int8_closure CLOSURE CALL
  _rqsp_upd_blocks2convweight_int8_closure(&DeeployOptNetwork__rqsp_upd_blocks2convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref;
} _rqsp_upd_blocks3convweight_int8_tiling_closure_args_t;

static void _rqsp_upd_blocks3convweight_int8_tiling_closure(void *_rqsp_upd_blocks3convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks3convweight_int8_tiling_closure_args_t *args =
      (_rqsp_upd_blocks3convweight_int8_tiling_closure_args_t *)_rqsp_upd_blocks3convweight_int8_tiling_closure_args;

  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_upd_blocks3convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_upd_blocks3convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_upd_blocks3convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_upd_blocks3convweight_int8_chunk = (3584 >> rqsp_upd_blocks3convweight_int8_log2Core) + ((3584 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_upd_blocks3convweight_int8_chunk_start =
      (uint32_t)MIN(rqsp_upd_blocks3convweight_int8_chunk * rqsp_upd_blocks3convweight_int8_core_id, (uint32_t)3584);
  uint32_t rqsp_upd_blocks3convweight_int8_chunk_stop =
      (uint32_t)MIN(rqsp_upd_blocks3convweight_int8_chunk_start + rqsp_upd_blocks3convweight_int8_chunk, (uint32_t)3584);
  uint32_t rqsp_upd_blocks3convweight_int8_local_size = rqsp_upd_blocks3convweight_int8_chunk_stop - rqsp_upd_blocks3convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_upd_blocks3convweight_int8_channel_start_offset = rqsp_upd_blocks3convweight_int8_chunk_start % 112;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 6 + rqsp_upd_blocks3convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_upd_blocks3convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref[rqsp_upd_blocks3convweight_int8_chunk_start],
        (int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref[rqsp_upd_blocks3convweight_int8_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref, 15, 112, chunk_seed,
        rqsp_upd_blocks3convweight_int8_local_size, rqsp_upd_blocks3convweight_int8_chunk_start, perturbation_sign,
        zo_eco_mom[6] + rqsp_upd_blocks3convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref[rqsp_upd_blocks3convweight_int8_chunk_start],
        (int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref[rqsp_upd_blocks3convweight_int8_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref, 15, 112, chunk_seed,
        rqsp_upd_blocks3convweight_int8_local_size, rqsp_upd_blocks3convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_upd_blocks3convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref;
} _rqsp_upd_blocks3convweight_int8_cluster_fork_args_t;

static void _rqsp_upd_blocks3convweight_int8_cluster_fork(void *_rqsp_upd_blocks3convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks3convweight_int8_cluster_fork_args_t *args =
      (_rqsp_upd_blocks3convweight_int8_cluster_fork_args_t *)_rqsp_upd_blocks3convweight_int8_cluster_fork_args;

  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks3convweight_int8_tiling_closure_args_t DeeployOptNetwork__rqsp_upd_blocks3convweight_int8_tiling_closure_args =
      (_rqsp_upd_blocks3convweight_int8_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref};

  // _rqsp_upd_blocks3convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_upd_blocks3convweight_int8_tiling_closure(&DeeployOptNetwork__rqsp_upd_blocks3convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr;
} _rqsp_upd_blocks3convweight_int8_closure_args_t;

static void _rqsp_upd_blocks3convweight_int8_closure(void *_rqsp_upd_blocks3convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks3convweight_int8_closure_args_t *args = (_rqsp_upd_blocks3convweight_int8_closure_args_t *)_rqsp_upd_blocks3convweight_int8_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref = (int8_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 7168);
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref = (int8_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 3584);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_input_6_ref =
      (void *)((char *)DeeployOptNetwork_input_6 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_blocks3convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployOptNetwork_blocks3convweight_int8_pmul_tensor +
               DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_cumByteOffset
                   [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_output_6_ref =
      (void *)((char *)DeeployOptNetwork_output_6 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445376, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_input_6_ref);
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_blocks3convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_upd_blocks3convweight_int8_cluster_fork_args_t DeeployOptNetwork__rqsp_upd_blocks3convweight_int8_cluster_fork_args =
        (_rqsp_upd_blocks3convweight_int8_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_upd_blocks3convweight_int8_cluster_fork, &DeeployOptNetwork__rqsp_upd_blocks3convweight_int8_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314304, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_output_6_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr;
} _rqsp_upd_blocks3convweight_int8_closure_L3_args_t;

static void _rqsp_upd_blocks3convweight_int8_closure_L3(void *_rqsp_upd_blocks3convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks3convweight_int8_closure_L3_args_t *args =
      (_rqsp_upd_blocks3convweight_int8_closure_L3_args_t *)_rqsp_upd_blocks3convweight_int8_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks3convweight_int8_closure_args_t DeeployOptNetwork__rqsp_upd_blocks3convweight_int8_closure_args =
      (_rqsp_upd_blocks3convweight_int8_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr};

  // _rqsp_upd_blocks3convweight_int8_closure CLOSURE CALL
  _rqsp_upd_blocks3convweight_int8_closure(&DeeployOptNetwork__rqsp_upd_blocks3convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref;
} _rqsp_upd_blocks4convweight_int8_tiling_closure_args_t;

static void _rqsp_upd_blocks4convweight_int8_tiling_closure(void *_rqsp_upd_blocks4convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks4convweight_int8_tiling_closure_args_t *args =
      (_rqsp_upd_blocks4convweight_int8_tiling_closure_args_t *)_rqsp_upd_blocks4convweight_int8_tiling_closure_args;

  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_upd_blocks4convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_upd_blocks4convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_upd_blocks4convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_upd_blocks4convweight_int8_chunk = (7168 >> rqsp_upd_blocks4convweight_int8_log2Core) + ((7168 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_upd_blocks4convweight_int8_chunk_start =
      (uint32_t)MIN(rqsp_upd_blocks4convweight_int8_chunk * rqsp_upd_blocks4convweight_int8_core_id, (uint32_t)7168);
  uint32_t rqsp_upd_blocks4convweight_int8_chunk_stop =
      (uint32_t)MIN(rqsp_upd_blocks4convweight_int8_chunk_start + rqsp_upd_blocks4convweight_int8_chunk, (uint32_t)7168);
  uint32_t rqsp_upd_blocks4convweight_int8_local_size = rqsp_upd_blocks4convweight_int8_chunk_stop - rqsp_upd_blocks4convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_upd_blocks4convweight_int8_channel_start_offset = rqsp_upd_blocks4convweight_int8_chunk_start % 224;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 8 + rqsp_upd_blocks4convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_upd_blocks4convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref[rqsp_upd_blocks4convweight_int8_chunk_start],
        (int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref[rqsp_upd_blocks4convweight_int8_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref, 15, 224, chunk_seed,
        rqsp_upd_blocks4convweight_int8_local_size, rqsp_upd_blocks4convweight_int8_chunk_start, perturbation_sign,
        zo_eco_mom[8] + rqsp_upd_blocks4convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref[rqsp_upd_blocks4convweight_int8_chunk_start],
        (int8_t *)&DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref[rqsp_upd_blocks4convweight_int8_chunk_start],
        (const int32_t *)DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref, 15, 224, chunk_seed,
        rqsp_upd_blocks4convweight_int8_local_size, rqsp_upd_blocks4convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_upd_blocks4convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref;
} _rqsp_upd_blocks4convweight_int8_cluster_fork_args_t;

static void _rqsp_upd_blocks4convweight_int8_cluster_fork(void *_rqsp_upd_blocks4convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks4convweight_int8_cluster_fork_args_t *args =
      (_rqsp_upd_blocks4convweight_int8_cluster_fork_args_t *)_rqsp_upd_blocks4convweight_int8_cluster_fork_args;

  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref;
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref;
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks4convweight_int8_tiling_closure_args_t DeeployOptNetwork__rqsp_upd_blocks4convweight_int8_tiling_closure_args =
      (_rqsp_upd_blocks4convweight_int8_tiling_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref};

  // _rqsp_upd_blocks4convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_upd_blocks4convweight_int8_tiling_closure(&DeeployOptNetwork__rqsp_upd_blocks4convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr;
} _rqsp_upd_blocks4convweight_int8_closure_args_t;

static void _rqsp_upd_blocks4convweight_int8_closure(void *_rqsp_upd_blocks4convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks4convweight_int8_closure_args_t *args = (_rqsp_upd_blocks4convweight_int8_closure_args_t *)_rqsp_upd_blocks4convweight_int8_closure_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref = (int8_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref = (int32_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 14336);
  int8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref = (int8_t *)((char *)DeeployOptNetwork_MEMORYARENA_L1 + 7168);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_input_8_ref =
      (void *)((char *)DeeployOptNetwork_input_8 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_cumByteOffset
                                                       [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_blocks4convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployOptNetwork_blocks4convweight_int8_pmul_tensor +
               DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_cumByteOffset
                   [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr]);
  void *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_output_8_ref =
      (void *)((char *)DeeployOptNetwork_output_8 + DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_cumByteOffset
                                                        [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_numTiles
           [*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr];
       TILING_I < DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_numTiles
                      [(*DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1448960, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_input_8_ref);
    mchan_transfer_1d(1441920, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_blocks4convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_upd_blocks4convweight_int8_cluster_fork_args_t DeeployOptNetwork__rqsp_upd_blocks4convweight_int8_cluster_fork_args =
        (_rqsp_upd_blocks4convweight_int8_cluster_fork_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_in_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_mul_ref,
                                                               .DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref =
                                                                   DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_upd_blocks4convweight_int8_cluster_fork, &DeeployOptNetwork__rqsp_upd_blocks4convweight_int8_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1317888, DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_data_out_ref,
                      DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_output_8_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr;
} _rqsp_upd_blocks4convweight_int8_closure_L3_args_t;

static void _rqsp_upd_blocks4convweight_int8_closure_L3(void *_rqsp_upd_blocks4convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_upd_blocks4convweight_int8_closure_L3_args_t *args =
      (_rqsp_upd_blocks4convweight_int8_closure_L3_args_t *)_rqsp_upd_blocks4convweight_int8_closure_L3_args;

  uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr =
      args->DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_upd_blocks4convweight_int8_closure_args_t DeeployOptNetwork__rqsp_upd_blocks4convweight_int8_closure_args =
      (_rqsp_upd_blocks4convweight_int8_closure_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr =
                                                            DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr};

  // _rqsp_upd_blocks4convweight_int8_closure CLOSURE CALL
  _rqsp_upd_blocks4convweight_int8_closure(&DeeployOptNetwork__rqsp_upd_blocks4convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunOptimizerNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr;
    _rqsp_upd_blocks0convbias_rqsadd_closure_L3_args_t DeeployOptNetwork__rqsp_upd_blocks0convbias_rqsadd_closure_L3_args =
        (_rqsp_upd_blocks0convbias_rqsadd_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr =
                                                                 DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convbias_rqsadd_tileIdxPtr};

    // _rqsp_upd_blocks0convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_upd_blocks0convbias_rqsadd_closure_L3(&DeeployOptNetwork__rqsp_upd_blocks0convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr;
    _pert_upd_blocks0bnweight_closure_L3_args_t DeeployOptNetwork__pert_upd_blocks0bnweight_closure_L3_args = (_pert_upd_blocks0bnweight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnweight_tileIdxPtr};

    // _pert_upd_blocks0bnweight_closure_L3 CLOSURE CALL
    _pert_upd_blocks0bnweight_closure_L3(&DeeployOptNetwork__pert_upd_blocks0bnweight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr;
    _pert_upd_blocks0bnbias_closure_L3_args_t DeeployOptNetwork__pert_upd_blocks0bnbias_closure_L3_args = (_pert_upd_blocks0bnbias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks0bnbias_tileIdxPtr};

    // _pert_upd_blocks0bnbias_closure_L3 CLOSURE CALL
    _pert_upd_blocks0bnbias_closure_L3(&DeeployOptNetwork__pert_upd_blocks0bnbias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr;
    _pert_upd_fcbias_fp32_closure_L3_args_t DeeployOptNetwork__pert_upd_fcbias_fp32_closure_L3_args = (_pert_upd_fcbias_fp32_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcbias_fp32_tileIdxPtr};

    // _pert_upd_fcbias_fp32_closure_L3 CLOSURE CALL
    _pert_upd_fcbias_fp32_closure_L3(&DeeployOptNetwork__pert_upd_fcbias_fp32_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr;
    _rqsp_upd_blocks1convbias_rqsadd_closure_L3_args_t DeeployOptNetwork__rqsp_upd_blocks1convbias_rqsadd_closure_L3_args =
        (_rqsp_upd_blocks1convbias_rqsadd_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr =
                                                                 DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convbias_rqsadd_tileIdxPtr};

    // _rqsp_upd_blocks1convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_upd_blocks1convbias_rqsadd_closure_L3(&DeeployOptNetwork__rqsp_upd_blocks1convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr;
    _rqsp_upd_blocks2convbias_rqsadd_closure_L3_args_t DeeployOptNetwork__rqsp_upd_blocks2convbias_rqsadd_closure_L3_args =
        (_rqsp_upd_blocks2convbias_rqsadd_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr =
                                                                 DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convbias_rqsadd_tileIdxPtr};

    // _rqsp_upd_blocks2convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_upd_blocks2convbias_rqsadd_closure_L3(&DeeployOptNetwork__rqsp_upd_blocks2convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr;
    _pert_upd_blocks1bnweight_closure_L3_args_t DeeployOptNetwork__pert_upd_blocks1bnweight_closure_L3_args = (_pert_upd_blocks1bnweight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnweight_tileIdxPtr};

    // _pert_upd_blocks1bnweight_closure_L3 CLOSURE CALL
    _pert_upd_blocks1bnweight_closure_L3(&DeeployOptNetwork__pert_upd_blocks1bnweight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr;
    _pert_upd_blocks1bnbias_closure_L3_args_t DeeployOptNetwork__pert_upd_blocks1bnbias_closure_L3_args = (_pert_upd_blocks1bnbias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks1bnbias_tileIdxPtr};

    // _pert_upd_blocks1bnbias_closure_L3 CLOSURE CALL
    _pert_upd_blocks1bnbias_closure_L3(&DeeployOptNetwork__pert_upd_blocks1bnbias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr;
    _pert_upd_blocks2bnweight_closure_L3_args_t DeeployOptNetwork__pert_upd_blocks2bnweight_closure_L3_args = (_pert_upd_blocks2bnweight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnweight_tileIdxPtr};

    // _pert_upd_blocks2bnweight_closure_L3 CLOSURE CALL
    _pert_upd_blocks2bnweight_closure_L3(&DeeployOptNetwork__pert_upd_blocks2bnweight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr;
    _pert_upd_blocks2bnbias_closure_L3_args_t DeeployOptNetwork__pert_upd_blocks2bnbias_closure_L3_args = (_pert_upd_blocks2bnbias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks2bnbias_tileIdxPtr};

    // _pert_upd_blocks2bnbias_closure_L3 CLOSURE CALL
    _pert_upd_blocks2bnbias_closure_L3(&DeeployOptNetwork__pert_upd_blocks2bnbias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr;
    _rqsp_upd_blocks0convweight_int8_closure_L3_args_t DeeployOptNetwork__rqsp_upd_blocks0convweight_int8_closure_L3_args =
        (_rqsp_upd_blocks0convweight_int8_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr =
                                                                 DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks0convweight_int8_tileIdxPtr};

    // _rqsp_upd_blocks0convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_upd_blocks0convweight_int8_closure_L3(&DeeployOptNetwork__rqsp_upd_blocks0convweight_int8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr;
    _rqsp_upd_blocks3convbias_rqsadd_closure_L3_args_t DeeployOptNetwork__rqsp_upd_blocks3convbias_rqsadd_closure_L3_args =
        (_rqsp_upd_blocks3convbias_rqsadd_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr =
                                                                 DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convbias_rqsadd_tileIdxPtr};

    // _rqsp_upd_blocks3convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_upd_blocks3convbias_rqsadd_closure_L3(&DeeployOptNetwork__rqsp_upd_blocks3convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr;
    _rqsp_upd_blocks4convbias_rqsadd_closure_L3_args_t DeeployOptNetwork__rqsp_upd_blocks4convbias_rqsadd_closure_L3_args =
        (_rqsp_upd_blocks4convbias_rqsadd_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr =
                                                                 DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convbias_rqsadd_tileIdxPtr};

    // _rqsp_upd_blocks4convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_upd_blocks4convbias_rqsadd_closure_L3(&DeeployOptNetwork__rqsp_upd_blocks4convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr;
    _pert_upd_blocks3bnweight_closure_L3_args_t DeeployOptNetwork__pert_upd_blocks3bnweight_closure_L3_args = (_pert_upd_blocks3bnweight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnweight_tileIdxPtr};

    // _pert_upd_blocks3bnweight_closure_L3 CLOSURE CALL
    _pert_upd_blocks3bnweight_closure_L3(&DeeployOptNetwork__pert_upd_blocks3bnweight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr;
    _pert_upd_blocks3bnbias_closure_L3_args_t DeeployOptNetwork__pert_upd_blocks3bnbias_closure_L3_args = (_pert_upd_blocks3bnbias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks3bnbias_tileIdxPtr};

    // _pert_upd_blocks3bnbias_closure_L3 CLOSURE CALL
    _pert_upd_blocks3bnbias_closure_L3(&DeeployOptNetwork__pert_upd_blocks3bnbias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr;
    _pert_upd_blocks4bnweight_closure_L3_args_t DeeployOptNetwork__pert_upd_blocks4bnweight_closure_L3_args = (_pert_upd_blocks4bnweight_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnweight_tileIdxPtr};

    // _pert_upd_blocks4bnweight_closure_L3 CLOSURE CALL
    _pert_upd_blocks4bnweight_closure_L3(&DeeployOptNetwork__pert_upd_blocks4bnweight_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr;
    _pert_upd_blocks4bnbias_closure_L3_args_t DeeployOptNetwork__pert_upd_blocks4bnbias_closure_L3_args = (_pert_upd_blocks4bnbias_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_blocks4bnbias_tileIdxPtr};

    // _pert_upd_blocks4bnbias_closure_L3 CLOSURE CALL
    _pert_upd_blocks4bnbias_closure_L3(&DeeployOptNetwork__pert_upd_blocks4bnbias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr = &bu_DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr;
    _pert_upd_fcweight_fp32_closure_L3_args_t DeeployOptNetwork__pert_upd_fcweight_fp32_closure_L3_args = (_pert_upd_fcweight_fp32_closure_L3_args_t){
        .DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr = DeeployOptNetwork_TILING_CODEGEN_L1_pert_upd_fcweight_fp32_tileIdxPtr};

    // _pert_upd_fcweight_fp32_closure_L3 CLOSURE CALL
    _pert_upd_fcweight_fp32_closure_L3(&DeeployOptNetwork__pert_upd_fcweight_fp32_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr;
    _rqsp_upd_blocks1convweight_int8_closure_L3_args_t DeeployOptNetwork__rqsp_upd_blocks1convweight_int8_closure_L3_args =
        (_rqsp_upd_blocks1convweight_int8_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr =
                                                                 DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks1convweight_int8_tileIdxPtr};

    // _rqsp_upd_blocks1convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_upd_blocks1convweight_int8_closure_L3(&DeeployOptNetwork__rqsp_upd_blocks1convweight_int8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr;
    _rqsp_upd_blocks2convweight_int8_closure_L3_args_t DeeployOptNetwork__rqsp_upd_blocks2convweight_int8_closure_L3_args =
        (_rqsp_upd_blocks2convweight_int8_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr =
                                                                 DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks2convweight_int8_tileIdxPtr};

    // _rqsp_upd_blocks2convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_upd_blocks2convweight_int8_closure_L3(&DeeployOptNetwork__rqsp_upd_blocks2convweight_int8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr;
    _rqsp_upd_blocks3convweight_int8_closure_L3_args_t DeeployOptNetwork__rqsp_upd_blocks3convweight_int8_closure_L3_args =
        (_rqsp_upd_blocks3convweight_int8_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr =
                                                                 DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks3convweight_int8_tileIdxPtr};

    // _rqsp_upd_blocks3convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_upd_blocks3convweight_int8_closure_L3(&DeeployOptNetwork__rqsp_upd_blocks3convweight_int8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr =
        &bu_DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr;
    _rqsp_upd_blocks4convweight_int8_closure_L3_args_t DeeployOptNetwork__rqsp_upd_blocks4convweight_int8_closure_L3_args =
        (_rqsp_upd_blocks4convweight_int8_closure_L3_args_t){.DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr =
                                                                 DeeployOptNetwork_TILING_CODEGEN_L1_rqsp_upd_blocks4convweight_int8_tileIdxPtr};

    // _rqsp_upd_blocks4convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_upd_blocks4convweight_int8_closure_L3(&DeeployOptNetwork__rqsp_upd_blocks4convweight_int8_closure_L3_args);
  }
}

void InitOptimizerNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployOptNetwork_MEMORYARENA_L1 = (int8_t *)DeeployNetwork_MEMORYARENA_L1; /* shared with TrainingNetwork */
  if (DeeployOptNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployOptNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 14464));
    return;
  }

  DeeployOptNetwork_input_0 = (float32_t *)DeeployNetwork_input_12;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_1 = (float32_t *)DeeployNetwork_input_17;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_2 = (float32_t *)DeeployNetwork_input_13;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_3 = (float32_t *)DeeployNetwork_input_18;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_4 = (float32_t *)DeeployNetwork_input_14;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_5 = (float32_t *)DeeployNetwork_input_19;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_6 = (float32_t *)DeeployNetwork_input_15;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_7 = (float32_t *)DeeployNetwork_input_20;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_8 = (float32_t *)DeeployNetwork_input_16;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_9 = (float32_t *)DeeployNetwork_input_21;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_10 = (float32_t *)DeeployNetwork_input_22;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_11 = (float32_t *)DeeployNetwork_input_23;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_12 = (float32_t *)DeeployNetwork_input_2;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_13 = (float32_t *)DeeployNetwork_input_3;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_14 = (float32_t *)DeeployNetwork_input_4;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_15 = (float32_t *)DeeployNetwork_input_5;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_16 = (float32_t *)DeeployNetwork_input_6;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_17 = (float32_t *)DeeployNetwork_input_7;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_18 = (float32_t *)DeeployNetwork_input_8;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_19 = (float32_t *)DeeployNetwork_input_9;   /* shared with TrainingNetwork */
  DeeployOptNetwork_input_20 = (float32_t *)DeeployNetwork_input_10;  /* shared with TrainingNetwork */
  DeeployOptNetwork_input_21 = (float32_t *)DeeployNetwork_input_11;  /* shared with TrainingNetwork */
  DeeployOptNetwork_output_0 = (float32_t *)DeeployNetwork_input_12;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_1 = (float32_t *)DeeployNetwork_input_17;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_2 = (float32_t *)DeeployNetwork_input_13;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_3 = (float32_t *)DeeployNetwork_input_18;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_4 = (float32_t *)DeeployNetwork_input_14;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_5 = (float32_t *)DeeployNetwork_input_19;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_6 = (float32_t *)DeeployNetwork_input_15;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_7 = (float32_t *)DeeployNetwork_input_20;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_8 = (float32_t *)DeeployNetwork_input_16;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_9 = (float32_t *)DeeployNetwork_input_21;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_10 = (float32_t *)DeeployNetwork_input_22; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_11 = (float32_t *)DeeployNetwork_input_23; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_12 = (float32_t *)DeeployNetwork_input_2;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_13 = (float32_t *)DeeployNetwork_input_3;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_14 = (float32_t *)DeeployNetwork_input_4;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_15 = (float32_t *)DeeployNetwork_input_5;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_16 = (float32_t *)DeeployNetwork_input_6;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_17 = (float32_t *)DeeployNetwork_input_7;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_18 = (float32_t *)DeeployNetwork_input_8;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_19 = (float32_t *)DeeployNetwork_input_9;  /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_20 = (float32_t *)DeeployNetwork_input_10; /* in-place, shared with TrainingNetwork */
  DeeployOptNetwork_output_21 = (float32_t *)DeeployNetwork_input_11; /* in-place, shared with TrainingNetwork */
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
