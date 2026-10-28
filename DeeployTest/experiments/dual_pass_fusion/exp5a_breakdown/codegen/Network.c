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

static PI_L2 int32_t DeeployNetwork_blocks0convweight_int8_pmul_tensor[8] = {85469, 143432, 99234, 126554, 105298, 133299, 112298, 80873};

static PI_L2 int32_t DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor[8] = {7666, 12866, 8901, 11352, 9445, 11957, 10073, 7254};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor[8] = {391, 233, 337, 264, 318, 251, 298, 413};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[1] = {0};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3[2] = {6, 2};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles[2] = {0, 2};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_cumByteOffset[2] = {0, 6};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_cmd[2] = {2024964, 1985708};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_size_1d[2] = {6, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_cumByteOffset[2] = {0, 58884};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_cmd[2] = {1369604, 1330348};

void *DeeployNetwork_inputs[3];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref;
} _rqsp_blocks0convweight_int8_tiling_closure_args_t;

static void _rqsp_blocks0convweight_int8_tiling_closure(void *_rqsp_blocks0convweight_int8_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convweight_int8_tiling_closure_args_t *args =
      (_rqsp_blocks0convweight_int8_tiling_closure_args_t *)_rqsp_blocks0convweight_int8_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: rqsp_blocks0convweight_int8, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks0convweight_int8_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks0convweight_int8_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks0convweight_int8_chunk = (32 >> rqsp_blocks0convweight_int8_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks0convweight_int8_chunk_start = (uint32_t)MIN(rqsp_blocks0convweight_int8_chunk * rqsp_blocks0convweight_int8_core_id, (uint32_t)32);
  uint32_t rqsp_blocks0convweight_int8_chunk_stop = (uint32_t)MIN(rqsp_blocks0convweight_int8_chunk_start + rqsp_blocks0convweight_int8_chunk, (uint32_t)32);
  uint32_t rqsp_blocks0convweight_int8_local_size = rqsp_blocks0convweight_int8_chunk_stop - rqsp_blocks0convweight_int8_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks0convweight_int8_channel_start_offset = rqsp_blocks0convweight_int8_chunk_start % 4;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 0 + rqsp_blocks0convweight_int8_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks0convweight_int8_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  ApplyPerturbQuantRademacher_CHW(
      (const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref[rqsp_blocks0convweight_int8_chunk_start],
      (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref[rqsp_blocks0convweight_int8_chunk_start],
      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref, 15, 4, chunk_seed, rqsp_blocks0convweight_int8_local_size,
      rqsp_blocks0convweight_int8_chunk_start,
      perturbation_sign,                      // -- QW: +eps (L+) / -eps (L-)
      rqsp_blocks0convweight_int8_eps_scale); // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref;
} _rqsp_blocks0convweight_int8_cluster_fork_args_t;

static void _rqsp_blocks0convweight_int8_cluster_fork(void *_rqsp_blocks0convweight_int8_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convweight_int8_cluster_fork_args_t *args = (_rqsp_blocks0convweight_int8_cluster_fork_args_t *)_rqsp_blocks0convweight_int8_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks0convweight_int8_tiling_closure_args_t DeeployNetwork__rqsp_blocks0convweight_int8_tiling_closure_args =
      (_rqsp_blocks0convweight_int8_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref};

  // _rqsp_blocks0convweight_int8_tiling_closure CLOSURE CALL
  _rqsp_blocks0convweight_int8_tiling_closure(&DeeployNetwork__rqsp_blocks0convweight_int8_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr;
} _rqsp_blocks0convweight_int8_closure_args_t;

static void _rqsp_blocks0convweight_int8_closure(void *_rqsp_blocks0convweight_int8_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convweight_int8_closure_args_t *args = (_rqsp_blocks0convweight_int8_closure_args_t *)_rqsp_blocks0convweight_int8_closure_args;

  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor = args->DeeployNetwork_blocks0convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_blocks0convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_blocks0convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_input_1_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_blocks0convweight_int8_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_blocks0convweight_int8_cluster_fork_args_t DeeployNetwork__rqsp_blocks0convweight_int8_cluster_fork_args =
        (_rqsp_blocks0convweight_int8_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks0convweight_int8_cluster_fork, &DeeployNetwork__rqsp_blocks0convweight_int8_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_blocks0convweight_int8_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr;
} _rqsp_blocks0convweight_int8_closure_L3_args_t;

static void _rqsp_blocks0convweight_int8_closure_L3(void *_rqsp_blocks0convweight_int8_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convweight_int8_closure_L3_args_t *args = (_rqsp_blocks0convweight_int8_closure_L3_args_t *)_rqsp_blocks0convweight_int8_closure_L3_args;

  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor = args->DeeployNetwork_blocks0convweight_int8_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks0convweight_int8_closure_args_t DeeployNetwork__rqsp_blocks0convweight_int8_closure_args = (_rqsp_blocks0convweight_int8_closure_args_t){
      .DeeployNetwork_blocks0convweight_int8_pert_tensor = DeeployNetwork_blocks0convweight_int8_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr};

  // _rqsp_blocks0convweight_int8_closure CLOSURE CALL
  _rqsp_blocks0convweight_int8_closure(&DeeployNetwork__rqsp_blocks0convweight_int8_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref;
} _rqsp_blocks0convbias_rqsadd_tiling_closure_args_t;

static void _rqsp_blocks0convbias_rqsadd_tiling_closure(void *_rqsp_blocks0convbias_rqsadd_tiling_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convbias_rqsadd_tiling_closure_args_t *args =
      (_rqsp_blocks0convbias_rqsadd_tiling_closure_args_t *)_rqsp_blocks0convbias_rqsadd_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher_i32 (Name: rqsp_blocks0convbias_rqsadd, Op: RQSPerturbRademacher)
  uint8_t rqsp_blocks0convbias_rqsadd_core_id = (uint8_t)pi_core_id();
  uint8_t rqsp_blocks0convbias_rqsadd_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t rqsp_blocks0convbias_rqsadd_chunk = (8 >> rqsp_blocks0convbias_rqsadd_log2Core) + ((8 & (NUM_CORES - 1)) != 0);
  uint32_t rqsp_blocks0convbias_rqsadd_chunk_start = (uint32_t)MIN(rqsp_blocks0convbias_rqsadd_chunk * rqsp_blocks0convbias_rqsadd_core_id, (uint32_t)8);
  uint32_t rqsp_blocks0convbias_rqsadd_chunk_stop = (uint32_t)MIN(rqsp_blocks0convbias_rqsadd_chunk_start + rqsp_blocks0convbias_rqsadd_chunk, (uint32_t)8);
  uint32_t rqsp_blocks0convbias_rqsadd_local_size = rqsp_blocks0convbias_rqsadd_chunk_stop - rqsp_blocks0convbias_rqsadd_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t rqsp_blocks0convbias_rqsadd_channel_start_offset = rqsp_blocks0convbias_rqsadd_chunk_start % 1;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base + perturbation_sign (ZORuntime.h) for L+/L- flip.
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 1 + rqsp_blocks0convbias_rqsadd_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t rqsp_blocks0convbias_rqsadd_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  ApplyPerturbQuantRademacher_i32(
      (const int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref[rqsp_blocks0convbias_rqsadd_chunk_start],
      (int32_t *)&DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref[rqsp_blocks0convbias_rqsadd_chunk_start],
      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref, 16, 1, chunk_seed, rqsp_blocks0convbias_rqsadd_local_size,
      rqsp_blocks0convbias_rqsadd_chunk_start,
      perturbation_sign,                      // -- QW: +eps (L+) / -eps (L-)
      rqsp_blocks0convbias_rqsadd_eps_scale); // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref;
} _rqsp_blocks0convbias_rqsadd_cluster_fork_args_t;

static void _rqsp_blocks0convbias_rqsadd_cluster_fork(void *_rqsp_blocks0convbias_rqsadd_cluster_fork_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convbias_rqsadd_cluster_fork_args_t *args = (_rqsp_blocks0convbias_rqsadd_cluster_fork_args_t *)_rqsp_blocks0convbias_rqsadd_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks0convbias_rqsadd_tiling_closure_args_t DeeployNetwork__rqsp_blocks0convbias_rqsadd_tiling_closure_args =
      (_rqsp_blocks0convbias_rqsadd_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref};

  // _rqsp_blocks0convbias_rqsadd_tiling_closure CLOSURE CALL
  _rqsp_blocks0convbias_rqsadd_tiling_closure(&DeeployNetwork__rqsp_blocks0convbias_rqsadd_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks0convbias_rqsadd_closure_args_t;

static void _rqsp_blocks0convbias_rqsadd_closure(void *_rqsp_blocks0convbias_rqsadd_closure_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convbias_rqsadd_closure_args_t *args = (_rqsp_blocks0convbias_rqsadd_closure_args_t *)_rqsp_blocks0convbias_rqsadd_closure_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_blocks0convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_blocks0convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_pert_tensor + DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_cumByteOffset
                                                                               [*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_input_2_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_blocks0convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _rqsp_blocks0convbias_rqsadd_cluster_fork_args_t DeeployNetwork__rqsp_blocks0convbias_rqsadd_cluster_fork_args =
        (_rqsp_blocks0convbias_rqsadd_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_in_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_mul_ref,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref =
                                                               DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_rqsp_blocks0convbias_rqsadd_cluster_fork, &DeeployNetwork__rqsp_blocks0convbias_rqsadd_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_blocks0convbias_rqsadd_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr;
} _rqsp_blocks0convbias_rqsadd_closure_L3_args_t;

static void _rqsp_blocks0convbias_rqsadd_closure_L3(void *_rqsp_blocks0convbias_rqsadd_closure_L3_args) {
  // CLOSURE ARG CAST
  _rqsp_blocks0convbias_rqsadd_closure_L3_args_t *args = (_rqsp_blocks0convbias_rqsadd_closure_L3_args_t *)_rqsp_blocks0convbias_rqsadd_closure_L3_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _rqsp_blocks0convbias_rqsadd_closure_args_t DeeployNetwork__rqsp_blocks0convbias_rqsadd_closure_args = (_rqsp_blocks0convbias_rqsadd_closure_args_t){
      .DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = DeeployNetwork_blocks0convbias_rqsadd_pert_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr};

  // _rqsp_blocks0convbias_rqsadd_closure CLOSURE CALL
  _rqsp_blocks0convbias_rqsadd_closure(&DeeployNetwork__rqsp_blocks0convbias_rqsadd_closure_args);

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

  // Transpose [1, 1, 14, 700] -> [1, 14, 700, 1] (Name: _MERGE_CONVRQ_PASS_0_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 700;

  // RW: GCC Segmentation fault
  int8_t (*src)[dimLen_1][dimLen_2][dimLen_3] =
      (int8_t (*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  int8_t (*dst)[dimLen_2][dimLen_3][dimLen_1] =
      (int8_t (*)[dimLen_2][dimLen_3][dimLen_1])DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 9800);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 + DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1451592, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
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
    mchan_transfer_1d(1320520, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure(
    void *__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [8, 1, 1, 4] -> [8, 1, 4, 1] (Name: _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 8;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 1;

  uint16_t dimLen_3 = 4;

  // RW: GCC Segmentation fault
  int8_t (*src)[dimLen_1][dimLen_2][dimLen_3] =
      (int8_t (*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t (*dst)[dimLen_2][dimLen_3][dimLen_1] =
      (int8_t (*)[dimLen_2][dimLen_3][dimLen_1])DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref;

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork(
    void *__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args =
          (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref};

  // __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure(
    void *__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args;

  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor = args->DeeployNetwork_blocks0convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_blocks0convweight_int8_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed_ref =
          (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1441824, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_blocks0convweight_int8_pert_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args =
            (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1310752, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3(
    void *__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args_t *)
          __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args;

  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor = args->DeeployNetwork_blocks0convweight_int8_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args_t
      DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args =
          (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args_t){
              .DeeployNetwork_blocks0convweight_int8_pert_tensor = DeeployNetwork_blocks0convweight_int8_pert_tensor,
              .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
                  DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure(
      &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_args);

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

  // PULP FLAT CONV (single-pass control kernel for exp5a; Name: _MERGE_CONVRQ_PASS_0, Op: RequantizedConv)  -- QW
  PULP_FlatConv2d_i8_i8_i8_HWC(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref, (int8_t *)DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer,
                               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
                               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref, NUM_CORES, 16, 700, 14, 1, 701, 14, 8, 4, 1, 0, 0, 2, 2, 1, 1);

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
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_closure(void *__MERGE_CONVRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_args_t *)__MERGE_CONVRQ_PASS_0_closure_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 78512);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88312);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88376);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88344);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convbias_rqsadd_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 88408);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1451592, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_input_0_transposed_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convbias_rqsadd_pert_tensor_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);

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
    mchan_transfer_1d(1389232, DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
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
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_closure_L3(void *__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)__MERGE_CONVRQ_PASS_0_closure_L3_args;

  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = args->DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
      args->DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args = (__MERGE_CONVRQ_PASS_0_closure_args_t){
      .DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = DeeployNetwork_blocks0convbias_rqsadd_pert_tensor,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
          DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

  // __MERGE_CONVRQ_PASS_0_closure CLOSURE CALL
  __MERGE_CONVRQ_PASS_0_closure(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure(void *__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args;

  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 14, 701, 8] -> [1, 8, 14, 701] (Name: _MERGE_CONVRQ_PASS_0_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 14;

  uint16_t dimLen_2 = 701;

  uint16_t dimLen_3 = *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref;

  // RW: GCC Segmentation fault
  int8_t (*src)[dimLen_1][dimLen_2][dimLen_3] =
      (int8_t (*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t (*dst)[dimLen_3][dimLen_1][dimLen_2] =
      (int8_t (*)[dimLen_3][dimLen_1][dimLen_2])DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;

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
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork(void *__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t *)__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args;

  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args =
      (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref,
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
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref =
      (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 58884);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 + DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_cmd[TILING_I],
                                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
                                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref,
                                  DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_size_1d[TILING_I], 8);

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref) + 6);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref =
        &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3[TILING_I];

    __MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args =
        (__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref =
                DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_dimLen_3_ref,
            .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_0_output_0_pre_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_cmd[TILING_I],
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref);

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref) + 58884);

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
  int8_t *DeeployNetwork_blocks0convweight_int8_pert_tensor;
  int32_t *DeeployNetwork_blocks0convbias_rqsadd_pert_tensor;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  void *DeeployNetwork__MERGE_CONVRQ_PASS_0_buffer;
  {

    perf_stats_t rqsp_blocks0convweight_int8_perf_start, rqsp_blocks0convweight_int8_perf_end, rqsp_blocks0convweight_int8_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&rqsp_blocks0convweight_int8_perf_start);
    }

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr;
    DeeployNetwork_blocks0convweight_int8_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 32);
    _rqsp_blocks0convweight_int8_closure_L3_args_t DeeployNetwork__rqsp_blocks0convweight_int8_closure_L3_args =
        (_rqsp_blocks0convweight_int8_closure_L3_args_t){.DeeployNetwork_blocks0convweight_int8_pert_tensor = DeeployNetwork_blocks0convweight_int8_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convweight_int8_tileIdxPtr};

    // _rqsp_blocks0convweight_int8_closure_L3 CLOSURE CALL
    _rqsp_blocks0convweight_int8_closure_L3(&DeeployNetwork__rqsp_blocks0convweight_int8_closure_L3_args);

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&rqsp_blocks0convweight_int8_perf_end);
      perf_bench_diff(&rqsp_blocks0convweight_int8_perf_total, &rqsp_blocks0convweight_int8_perf_end, &rqsp_blocks0convweight_int8_perf_start);
      perf_bench_print("rqsp_blocks0convweight_int8", &rqsp_blocks0convweight_int8_perf_total);
    }
  }
  {

    perf_stats_t rqsp_blocks0convbias_rqsadd_perf_start, rqsp_blocks0convbias_rqsadd_perf_end, rqsp_blocks0convbias_rqsadd_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&rqsp_blocks0convbias_rqsadd_perf_start);
    }

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr;
    DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _rqsp_blocks0convbias_rqsadd_closure_L3_args_t DeeployNetwork__rqsp_blocks0convbias_rqsadd_closure_L3_args =
        (_rqsp_blocks0convbias_rqsadd_closure_L3_args_t){.DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = DeeployNetwork_blocks0convbias_rqsadd_pert_tensor,
                                                         .DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr =
                                                             DeeployNetwork_TILING_CODEGEN_L1_rqsp_blocks0convbias_rqsadd_tileIdxPtr};

    // _rqsp_blocks0convbias_rqsadd_closure_L3 CLOSURE CALL
    _rqsp_blocks0convbias_rqsadd_closure_L3(&DeeployNetwork__rqsp_blocks0convbias_rqsadd_closure_L3_args);

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&rqsp_blocks0convbias_rqsadd_perf_end);
      perf_bench_diff(&rqsp_blocks0convbias_rqsadd_perf_total, &rqsp_blocks0convbias_rqsadd_perf_end, &rqsp_blocks0convbias_rqsadd_perf_start);
      perf_bench_print("rqsp_blocks0convbias_rqsadd", &rqsp_blocks0convbias_rqsadd_perf_total);
    }
  }
  {

    perf_stats_t _MERGE_CONVRQ_PASS_0_input_0_transpose_perf_start, _MERGE_CONVRQ_PASS_0_input_0_transpose_perf_end,
        _MERGE_CONVRQ_PASS_0_input_0_transpose_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&_MERGE_CONVRQ_PASS_0_input_0_transpose_perf_start);
    }

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 64);
    __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args =
        (__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t){.DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed =
                                                                        DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                                                                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args);

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&_MERGE_CONVRQ_PASS_0_input_0_transpose_perf_end);
      perf_bench_diff(&_MERGE_CONVRQ_PASS_0_input_0_transpose_perf_total, &_MERGE_CONVRQ_PASS_0_input_0_transpose_perf_end,
                      &_MERGE_CONVRQ_PASS_0_input_0_transpose_perf_start);
      perf_bench_print("_MERGE_CONVRQ_PASS_0_input_0_transpose", &_MERGE_CONVRQ_PASS_0_input_0_transpose_perf_total);
    }
  }
  {

    perf_stats_t _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_perf_start,
        _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_perf_end,
        _MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&_MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_perf_start);
    }

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9864);
    __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args_t
        DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args =
            (__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args_t){
                .DeeployNetwork_blocks0convweight_int8_pert_tensor = DeeployNetwork_blocks0convweight_int8_pert_tensor,
                .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
                    DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3(
        &DeeployNetwork___MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_closure_L3_args);

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&_MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_perf_end);
      perf_bench_diff(&_MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_perf_total,
                      &_MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_perf_end,
                      &_MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_perf_start);
      perf_bench_print("_MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose",
                       &_MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transpose_perf_total);
    }
  }
  {

    perf_stats_t _MERGE_CONVRQ_PASS_0_perf_start, _MERGE_CONVRQ_PASS_0_perf_end, _MERGE_CONVRQ_PASS_0_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&_MERGE_CONVRQ_PASS_0_perf_start);
    }

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 78512);
    __MERGE_CONVRQ_PASS_0_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
        .DeeployNetwork_blocks0convbias_rqsadd_pert_tensor = DeeployNetwork_blocks0convbias_rqsadd_pert_tensor,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed =
            DeeployNetwork__MERGE_CONVRQ_PASS_0_blocks0convweight_int8_pert_tensor_transposed,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args);

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&_MERGE_CONVRQ_PASS_0_perf_end);
      perf_bench_diff(&_MERGE_CONVRQ_PASS_0_perf_total, &_MERGE_CONVRQ_PASS_0_perf_end, &_MERGE_CONVRQ_PASS_0_perf_start);
      perf_bench_print("_MERGE_CONVRQ_PASS_0", &_MERGE_CONVRQ_PASS_0_perf_total);
    }
  }
  {

    perf_stats_t _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_perf_start, _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_perf_end,
        _MERGE_CONVRQ_PASS_0_output_0_pre_transpose_perf_total;
    if (pi_core_id() == 0) {
      perf_bench_init();
      perf_bench_start();
      perf_bench_read(&_MERGE_CONVRQ_PASS_0_output_0_pre_transpose_perf_start);
    }

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

    if (pi_core_id() == 0) {
      perf_bench_stop();
      perf_bench_read(&_MERGE_CONVRQ_PASS_0_output_0_pre_transpose_perf_end);
      perf_bench_diff(&_MERGE_CONVRQ_PASS_0_output_0_pre_transpose_perf_total, &_MERGE_CONVRQ_PASS_0_output_0_pre_transpose_perf_end,
                      &_MERGE_CONVRQ_PASS_0_output_0_pre_transpose_perf_start);
      perf_bench_print("_MERGE_CONVRQ_PASS_0_output_0_pre_transpose", &_MERGE_CONVRQ_PASS_0_output_0_pre_transpose_perf_total);
    }
  }
}

void InitNetwork() {

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 117768);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 157024);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9864);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 64);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
