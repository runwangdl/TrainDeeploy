#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "perf_utils.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
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

static PI_L2 int32_t DeeployNetwork_blocks3convweight_int8_pmul_tensor[32] = {
    243041, 237342, 192477, 173475, 204002, 252155, 169682, 109053, 183913, 177070, 189471, 279557, 176599, 192092, 170877, 253092,
    212962, 224469, 200686, 193446, 159675, 161818, 229571, 154010, 265026, 247421, 219725, 150261, 247547, 116124, 278892, 219721};

static PI_L2 int32_t DeeployNetwork_blocks3convbias_rqsadd_pmul_tensor[32] = {
    4543569, 4437028, 3598282, 3243057, 3813753, 4713948, 3172148, 2038716, 3438188, 3310265, 3542102, 5226218, 3301465, 3591097, 3194483, 4731471,
    3981257, 4196366, 3751757, 3616413, 2985078, 3025126, 4291753, 2879164, 4954569, 4625459, 4107688, 2809070, 4627800, 2170897, 5213795, 4107601};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor[32] = {
    57, 58, 72, 80, 68, 55, 81, 127, 75, 78, 73, 49, 78, 72, 81, 55, 65, 61, 69, 71, 86, 85, 60, 90, 52, 56, 63, 92, 56, 119, 49, 63};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_cumByteOffset[1] = {0};

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref;
} _rqsp_blocks3convweight_int8_tiling_closure_args_t;

static void _rqsp_blocks3convweight_int8_tiling_closure(void *_rqsp_blocks3convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convweight_int8_tiling_closure_args_t *args =
      (_rqsp_blocks3convweight_int8_tiling_closure_args_t *)_rqsp_blocks3convweight_int8_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_blocks3convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks3convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks3convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks3convweight_int8_chunk = (3584 >> rqsp_blocks3convweight_int8_log2Core) + ((3584 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks3convweight_int8_chunk_start = (uint32_t)MIN(rqsp_blocks3convweight_int8_chunk * rqsp_blocks3convweight_int8_core_id, (uint32_t)3584);
  uint32_t rqsp_blocks3convweight_int8_chunk_stop = (uint32_t)MIN(rqsp_blocks3convweight_int8_chunk_start + rqsp_blocks3convweight_int8_chunk, (uint32_t)3584);
  uint32_t rqsp_blocks3convweight_int8_local_size = rqsp_blocks3convweight_int8_chunk_stop - rqsp_blocks3convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks3convweight_int8_channel_start_offset = rqsp_blocks3convweight_int8_chunk_start % 112;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 6 + rqsp_blocks3convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks3convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i8(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref[rqsp_blocks3convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref[rqsp_blocks3convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref, 15, 112, chunk_seed, rqsp_blocks3convweight_int8_local_size,
        rqsp_blocks3convweight_int8_chunk_start, perturbation_sign, zo_eco_mom[6] + rqsp_blocks3convweight_int8_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_CHW(
        (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref[rqsp_blocks3convweight_int8_chunk_start],
        (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref[rqsp_blocks3convweight_int8_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref, 15, 112, chunk_seed, rqsp_blocks3convweight_int8_local_size,
        rqsp_blocks3convweight_int8_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks3convweight_int8_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref;
} _rqsp_blocks3convweight_int8_cluster_fork_args_t;

static void _rqsp_blocks3convweight_int8_cluster_fork(void *_rqsp_blocks3convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convweight_int8_cluster_fork_args_t *args = (_rqsp_blocks3convweight_int8_cluster_fork_args_t *)_rqsp_blocks3convweight_int8_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks3convweight_int8_tiling_closure_args_t DeeployNetwork__rqsp_blocks3convweight_int8_tiling_closure_args =
      (_rqsp_blocks3convweight_int8_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref};

  // _rqsp_blocks3convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_blocks3convweight_int8_tiling_closure(&DeeployNetwork__rqsp_blocks3convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr;
} _rqsp_blocks3convweight_int8_closure_args_t;

static void _rqsp_blocks3convweight_int8_closure(void *_rqsp_blocks3convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convweight_int8_closure_args_t *args = (_rqsp_blocks3convweight_int8_closure_args_t *)_rqsp_blocks3convweight_int8_closure_args;

  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor = args->DeeployNetwork_blocks3convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_blocks3convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_blocks3convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445376, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_blocks3convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_blocks3convweight_int8_cluster_fork_args_t DeeployNetwork__rqsp_blocks3convweight_int8_cluster_fork_args =
        (_rqsp_blocks3convweight_int8_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks3convweight_int8_cluster_fork, &DeeployNetwork__rqsp_blocks3convweight_int8_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314304, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_blocks3convweight_int8_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr;
} _rqsp_blocks3convweight_int8_closure_L3_args_t;

static void _rqsp_blocks3convweight_int8_closure_L3(void *_rqsp_blocks3convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convweight_int8_closure_L3_args_t *args = (_rqsp_blocks3convweight_int8_closure_L3_args_t *)_rqsp_blocks3convweight_int8_closure_L3_args;

  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor = args->DeeployNetwork_blocks3convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks3convweight_int8_closure_args_t DeeployNetwork__rqsp_blocks3convweight_int8_closure_args = (_rqsp_blocks3convweight_int8_closure_args_t){
      .DeeployNetwork_blocks3convweight_int8_pert_tensor = DeeployNetwork_blocks3convweight_int8_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr};

  // _rqsp_blocks3convweight_int8_closure CLOSURE CALL
  _rqsp_blocks3convweight_int8_closure(&DeeployNetwork__rqsp_blocks3convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref;
} _rqsp_blocks3convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_blocks3convbias_rqsadd_tiling_closure(void *_rqsp_blocks3convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_blocks3convbias_rqsadd_tiling_closure_args_t *)_rqsp_blocks3convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_blocks3convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks3convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks3convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks3convbias_rqsadd_chunk = (32 >> rqsp_blocks3convbias_rqsadd_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks3convbias_rqsadd_chunk_start = (uint32_t)MIN(rqsp_blocks3convbias_rqsadd_chunk * rqsp_blocks3convbias_rqsadd_core_id, (uint32_t)32);
  uint32_t rqsp_blocks3convbias_rqsadd_chunk_stop = (uint32_t)MIN(rqsp_blocks3convbias_rqsadd_chunk_start + rqsp_blocks3convbias_rqsadd_chunk, (uint32_t)32);
  uint32_t rqsp_blocks3convbias_rqsadd_local_size = rqsp_blocks3convbias_rqsadd_chunk_stop - rqsp_blocks3convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks3convbias_rqsadd_channel_start_offset = rqsp_blocks3convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 7 + rqsp_blocks3convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks3convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  // RW 2026-10-13: ECO error feedback in the update pass (ZORuntime.h). zo_eco_mom[idx] covers the whole tensor;
  // the update tensors are single-tile, so chunk_start is the element offset into it.
  if (perturb_eps_use_override && zo_eco_on) {
    ApplyECOQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref[rqsp_blocks3convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref[rqsp_blocks3convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks3convbias_rqsadd_local_size,
        rqsp_blocks3convbias_rqsadd_chunk_start, perturbation_sign, zo_eco_mom[7] + rqsp_blocks3convbias_rqsadd_chunk_start);
  } else {
    ApplyPerturbQuantRademacher_i32(
        (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref[rqsp_blocks3convbias_rqsadd_chunk_start],
        (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref[rqsp_blocks3convbias_rqsadd_chunk_start],
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks3convbias_rqsadd_local_size,
        rqsp_blocks3convbias_rqsadd_chunk_start,
        perturbation_sign, // -- QW: +eps (L+) / -eps (L-)
        rqsp_blocks3convbias_rqsadd_eps_scale);
  } // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref;
} _rqsp_blocks3convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_blocks3convbias_rqsadd_cluster_fork(void *_rqsp_blocks3convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convbias_rqsadd_cluster_fork_args_t *args = (_rqsp_blocks3convbias_rqsadd_cluster_fork_args_t *)_rqsp_blocks3convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks3convbias_rqsadd_tiling_closure_args_t DeeployNetwork__rqsp_blocks3convbias_rqsadd_tiling_closure_args =
      (_rqsp_blocks3convbias_rqsadd_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref};

  // _rqsp_blocks3convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_blocks3convbias_rqsadd_tiling_closure(&DeeployNetwork__rqsp_blocks3convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks3convbias_rqsadd_closure_args_t;

static void _rqsp_blocks3convbias_rqsadd_closure(void *_rqsp_blocks3convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convbias_rqsadd_closure_args_t *args = (_rqsp_blocks3convbias_rqsadd_closure_args_t *)_rqsp_blocks3convbias_rqsadd_closure_args;

  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 128);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 256);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_blocks3convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convbias_rqsadd_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_blocks3convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convbias_rqsadd_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_input_2_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_blocks3convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_blocks3convbias_rqsadd_cluster_fork_args_t DeeployNetwork__rqsp_blocks3convbias_rqsadd_cluster_fork_args =
        (_rqsp_blocks3convbias_rqsadd_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks3convbias_rqsadd_cluster_fork, &DeeployNetwork__rqsp_blocks3convbias_rqsadd_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_blocks3convbias_rqsadd_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks3convbias_rqsadd_closure_L3_args_t;

static void _rqsp_blocks3convbias_rqsadd_closure_L3(void *_rqsp_blocks3convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks3convbias_rqsadd_closure_L3_args_t *args = (_rqsp_blocks3convbias_rqsadd_closure_L3_args_t *)_rqsp_blocks3convbias_rqsadd_closure_L3_args;

  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks3convbias_rqsadd_closure_args_t DeeployNetwork__rqsp_blocks3convbias_rqsadd_closure_args = (_rqsp_blocks3convbias_rqsadd_closure_args_t){
      .DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = DeeployNetwork_blocks3convbias_rqsadd_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr};

  // _rqsp_blocks3convbias_rqsadd_closure CLOSURE CALL
  _rqsp_blocks3convbias_rqsadd_closure(&DeeployNetwork__rqsp_blocks3convbias_rqsadd_closure_args);

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

  // Transpose [1, 16, 14, 5] -> [1, 14, 5, 16] (Name: _MERGE_CONVRQ_PASS_0_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 16;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 5;

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

      for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {
        const uint32_t offS_2 = offS_1 + i_3 * sIn_3;
        const uint32_t offD_2 = offD_1 + i_3 * sOut_2;

        const uint32_t baseChunk_3 = dimLen_1 / NUM_CORES;
        const uint32_t leftover_3 = dimLen_1 - baseChunk_3 * NUM_CORES;
        const uint32_t offset_3 = baseChunk_3 * coreId + (coreId < leftover_3 ? coreId : leftover_3);
        const uint32_t chunk_3 = coreId < leftover_3 ? baseChunk_3 + 1 : baseChunk_3;
        for (uint32_t i_1 = offset_3; i_1 < offset_3 + chunk_3; i_1++) {
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1120);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 + DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442912,
                                      .size = 1120,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    __MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args =
        (__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
                                                                          DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                                                                      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
                                                                          DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1311840,
                                      .size = 1120,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
                                      .ext =
                                          DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure(
    void *__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [32, 16, 7, 1] -> [32, 7, 1, 16] (Name: _MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 32;

  uint16_t dimLen_1 = 16;

  uint16_t dimLen_2 = 7;

  uint16_t dimLen_3 = 1;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_cluster_fork(
    void *__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args =
          (__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref};

  // __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure(
    void *__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_args;

  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor = args->DeeployNetwork_blocks3convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_blocks3convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convweight_int8_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1445376,
          .size = 3584,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_blocks3convweight_int8_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args =
            (__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1314304,
          .size = 3584,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3(
    void *__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args;

  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor = args->DeeployNetwork_blocks3convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_args =
          (__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_args_t){
              .DeeployNetwork_blocks3convweight_int8_pert_tensor = DeeployNetwork_blocks3convweight_int8_pert_tensor,
              .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed =
                  DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_args);

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

  // PULP NN CONV

  pulp_nn_conv_i8_i8_i8(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref, DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer, NULL,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref, 1, 16, 5,
                        14, 16, 5, 8, 32, 1, 7, 0, 0, 0, 0, 1, 1, 1, 1);

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
  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_closure(void *__MERGE_CONVRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_args_t *)__MERGE_CONVRQ_PASS_0_closure_args;

  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6656);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7904);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7776);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5376);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks3convbias_rqsadd_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int add_ExternalToLocal = -1;
  int weight_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442912,
                                      .size = 1120,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1445376,
          .size = 3584,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convbias_rqsadd_pert_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441920,
          .size = 128,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_3_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
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
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1312000,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

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
  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_closure_L3(void *__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)__MERGE_CONVRQ_PASS_0_closure_L3_args;

  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args = (__MERGE_CONVRQ_PASS_0_closure_args_t){
      .DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = DeeployNetwork_blocks3convbias_rqsadd_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed,
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

  // Transpose [1, 8, 5, 32] -> [1, 32, 8, 5] (Name: _MERGE_CONVRQ_PASS_0_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 32;

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 + DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1443072,
          .size = 1280,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
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
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1312000,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
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

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  int8_t *DeeployNetwork_blocks3convweight_int8_pert_tensor;
  int32_t *DeeployNetwork_blocks3convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr;
    DeeployNetwork_blocks3convweight_int8_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _rqsp_blocks3convweight_int8_closure_L3_args_t DeeployNetwork__rqsp_blocks3convweight_int8_closure_L3_args =
        (_rqsp_blocks3convweight_int8_closure_L3_args_t){.DeeployNetwork_blocks3convweight_int8_pert_tensor = DeeployNetwork_blocks3convweight_int8_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convweight_int8_tileIdxPtr};

    // _rqsp_blocks3convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_blocks3convweight_int8_closure_L3(&DeeployNetwork__rqsp_blocks3convweight_int8_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr;
    DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 3584);
    _rqsp_blocks3convbias_rqsadd_closure_L3_args_t DeeployNetwork__rqsp_blocks3convbias_rqsadd_closure_L3_args =
        (_rqsp_blocks3convbias_rqsadd_closure_L3_args_t){.DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = DeeployNetwork_blocks3convbias_rqsadd_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks3convbias_rqsadd_tileIdxPtr};

    // _rqsp_blocks3convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_blocks3convbias_rqsadd_closure_L3(&DeeployNetwork__rqsp_blocks3convbias_rqsadd_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 3712);
    __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args =
        (__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t){.DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed =
                                                                        DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                                                                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 4832);
    __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args =
            (__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args_t){
                .DeeployNetwork_blocks3convweight_int8_pert_tensor = DeeployNetwork_blocks3convweight_int8_pert_tensor,
                .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed =
                    DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3(
        &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    __MERGE_CONVRQ_PASS_0_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
        .DeeployNetwork_blocks3convbias_rqsadd_pert_tensor = DeeployNetwork_blocks3convbias_rqsadd_pert_tensor,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks3convweight_int8_pert_tensor_transposed,
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

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 8032);
  if (DeeployNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 8032));
    return;
  }

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 8416);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 7168);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 3584);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8288);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 1280);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
