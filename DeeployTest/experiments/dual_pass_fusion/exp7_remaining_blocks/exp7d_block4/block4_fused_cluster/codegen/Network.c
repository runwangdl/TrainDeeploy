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
int8_t *DeeployNetwork_input_3;
int8_t *DeeployNetwork_output_0;

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor[32] = {
    178, 146, 165, 164, 163, 155, 146, 137, 119, 174, 135, 139, 150, 146, 243, 126,
    147, 139, 155, 130, 153, 157, 135, 160, 116, 142, 140, 160, 119, 174, 169, 142};

static PI_L2 int32_t DeeployNetwork_blocks4convweight_int8_pmul_tensor[32] = {
    185997, 226785, 200920, 202444, 202823, 213378, 226168, 241806, 277784, 190200, 245924, 237923, 220822, 226733, 136034, 261893,
    225823, 238894, 213576, 255468, 216926, 210590, 245405, 206957, 286127, 233156, 236698, 206920, 278716, 190344, 195460, 232972};

static PI_L2 int32_t DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor[32] = {
    4465028, 5444165, 4823268, 4859837, 4868948, 5122322, 5429352, 5804773, 6668456, 4565921, 5903610, 5711538, 5301021, 5442922, 3265610, 6286967,
    5421086, 5734863, 5127071, 6132738, 5207501, 5055397, 5891161, 4968182, 6868725, 5597122, 5682131, 4967301, 6690810, 4569368, 4692189, 5592699};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[4];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref;
} _dualconv_blocks4_input_0_transpose_tiling_closure_args_t;

static void _dualconv_blocks4_input_0_transpose_tiling_closure(void *_dualconv_blocks4_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_input_0_transpose_tiling_closure_args_t *args =
      (_dualconv_blocks4_input_0_transpose_tiling_closure_args_t *)_dualconv_blocks4_input_0_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 32, 8, 5] -> [1, 8, 5, 32] (Name: dualconv_blocks4_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 32;

  uint16_t dimLen_2 = 8;

  uint16_t dimLen_3 = 5;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref;
} _dualconv_blocks4_input_0_transpose_cluster_fork_args_t;

static void _dualconv_blocks4_input_0_transpose_cluster_fork(void *_dualconv_blocks4_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_input_0_transpose_cluster_fork_args_t *args =
      (_dualconv_blocks4_input_0_transpose_cluster_fork_args_t *)_dualconv_blocks4_input_0_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks4_input_0_transpose_tiling_closure_args_t DeeployNetwork__dualconv_blocks4_input_0_transpose_tiling_closure_args =
      (_dualconv_blocks4_input_0_transpose_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref};

  // _dualconv_blocks4_input_0_transpose_tiling_closure CLOSURE CALL
  _dualconv_blocks4_input_0_transpose_tiling_closure(&DeeployNetwork__dualconv_blocks4_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks4_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr;
} _dualconv_blocks4_input_0_transpose_closure_args_t;

static void _dualconv_blocks4_input_0_transpose_closure(void *_dualconv_blocks4_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_input_0_transpose_closure_args_t *args =
      (_dualconv_blocks4_input_0_transpose_closure_args_t *)_dualconv_blocks4_input_0_transpose_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks4_input_0_transposed = args->DeeployNetwork_dualconv_blocks4_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 + DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_dualconv_blocks4_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks4_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443072, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _dualconv_blocks4_input_0_transpose_cluster_fork_args_t DeeployNetwork__dualconv_blocks4_input_0_transpose_cluster_fork_args =
        (_dualconv_blocks4_input_0_transpose_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks4_input_0_transpose_cluster_fork, &DeeployNetwork__dualconv_blocks4_input_0_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312000, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_dualconv_blocks4_input_0_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks4_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr;
} _dualconv_blocks4_input_0_transpose_closure_L3_args_t;

static void _dualconv_blocks4_input_0_transpose_closure_L3(void *_dualconv_blocks4_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_input_0_transpose_closure_L3_args_t *args =
      (_dualconv_blocks4_input_0_transpose_closure_L3_args_t *)_dualconv_blocks4_input_0_transpose_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks4_input_0_transposed = args->DeeployNetwork_dualconv_blocks4_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks4_input_0_transpose_closure_args_t DeeployNetwork__dualconv_blocks4_input_0_transpose_closure_args =
      (_dualconv_blocks4_input_0_transpose_closure_args_t){.DeeployNetwork_dualconv_blocks4_input_0_transposed =
                                                               DeeployNetwork_dualconv_blocks4_input_0_transposed,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr};

  // _dualconv_blocks4_input_0_transpose_closure CLOSURE CALL
  _dualconv_blocks4_input_0_transpose_closure(&DeeployNetwork__dualconv_blocks4_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref;
} _dualconv_blocks4_input_3_transpose_tiling_closure_args_t;

static void _dualconv_blocks4_input_3_transpose_tiling_closure(void *_dualconv_blocks4_input_3_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_input_3_transpose_tiling_closure_args_t *args =
      (_dualconv_blocks4_input_3_transpose_tiling_closure_args_t *)_dualconv_blocks4_input_3_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 32, 8, 5] -> [1, 8, 5, 32] (Name: dualconv_blocks4_input_3_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 32;

  uint16_t dimLen_2 = 8;

  uint16_t dimLen_3 = 5;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref;
} _dualconv_blocks4_input_3_transpose_cluster_fork_args_t;

static void _dualconv_blocks4_input_3_transpose_cluster_fork(void *_dualconv_blocks4_input_3_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_input_3_transpose_cluster_fork_args_t *args =
      (_dualconv_blocks4_input_3_transpose_cluster_fork_args_t *)_dualconv_blocks4_input_3_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks4_input_3_transpose_tiling_closure_args_t DeeployNetwork__dualconv_blocks4_input_3_transpose_tiling_closure_args =
      (_dualconv_blocks4_input_3_transpose_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref};

  // _dualconv_blocks4_input_3_transpose_tiling_closure CLOSURE CALL
  _dualconv_blocks4_input_3_transpose_tiling_closure(&DeeployNetwork__dualconv_blocks4_input_3_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks4_input_3_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr;
} _dualconv_blocks4_input_3_transpose_closure_args_t;

static void _dualconv_blocks4_input_3_transpose_closure(void *_dualconv_blocks4_input_3_transpose_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_input_3_transpose_closure_args_t *args =
      (_dualconv_blocks4_input_3_transpose_closure_args_t *)_dualconv_blocks4_input_3_transpose_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks4_input_3_transposed = args->DeeployNetwork_dualconv_blocks4_input_3_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_input_3_ref =
      (void *)((char *)DeeployNetwork_input_3 + DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_dualconv_blocks4_input_3_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks4_input_3_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443072, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_input_3_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _dualconv_blocks4_input_3_transpose_cluster_fork_args_t DeeployNetwork__dualconv_blocks4_input_3_transpose_cluster_fork_args =
        (_dualconv_blocks4_input_3_transpose_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks4_input_3_transpose_cluster_fork, &DeeployNetwork__dualconv_blocks4_input_3_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312000, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_dualconv_blocks4_input_3_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks4_input_3_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr;
} _dualconv_blocks4_input_3_transpose_closure_L3_args_t;

static void _dualconv_blocks4_input_3_transpose_closure_L3(void *_dualconv_blocks4_input_3_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_input_3_transpose_closure_L3_args_t *args =
      (_dualconv_blocks4_input_3_transpose_closure_L3_args_t *)_dualconv_blocks4_input_3_transpose_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks4_input_3_transposed = args->DeeployNetwork_dualconv_blocks4_input_3_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks4_input_3_transpose_closure_args_t DeeployNetwork__dualconv_blocks4_input_3_transpose_closure_args =
      (_dualconv_blocks4_input_3_transpose_closure_args_t){.DeeployNetwork_dualconv_blocks4_input_3_transposed =
                                                               DeeployNetwork_dualconv_blocks4_input_3_transposed,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr};

  // _dualconv_blocks4_input_3_transpose_closure CLOSURE CALL
  _dualconv_blocks4_input_3_transpose_closure(&DeeployNetwork__dualconv_blocks4_input_3_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_dualconv_blocks4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref;
} _dualconv_blocks4_tiling_closure_args_t;

static void _dualconv_blocks4_tiling_closure(void *_dualconv_blocks4_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_tiling_closure_args_t *args = (_dualconv_blocks4_tiling_closure_args_t *)_dualconv_blocks4_tiling_closure_args;

  void *DeeployNetwork_dualconv_blocks4_buffer = args->DeeployNetwork_dualconv_blocks4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP DUAL-PASS QZO CONV, TWO INPUTS (Name: dualconv_blocks4, Op: RequantizedDualConv)  -- QW exp7
  // Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
  uint32_t dualconv_blocks4_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 8);
  uint32_t dualconv_blocks4_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 9);
  uint32_t dualconv_blocks4_tile_xor = (0 * 0x9E3779B1u);
  float32_t dualconv_blocks4_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_DualConv2d_2in_i8_i8_i8_HWC(
      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref,
      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref,
      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref,
      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref,
      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref + ((64) >> 1), (64), (int8_t *)DeeployNetwork_dualconv_blocks4_buffer,
      dualconv_blocks4_seed_w, dualconv_blocks4_seed_b, dualconv_blocks4_tile_xor, 15, 16, dualconv_blocks4_eps_scale, NUM_CORES, 16, 5, 8, 32, 5, 2,
      ((64) >> 1), 1, 7, 0, 0, 0, 0, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_dualconv_blocks4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref;
} _dualconv_blocks4_cluster_fork_args_t;

static void _dualconv_blocks4_cluster_fork(void *_dualconv_blocks4_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_cluster_fork_args_t *args = (_dualconv_blocks4_cluster_fork_args_t *)_dualconv_blocks4_cluster_fork_args;

  void *DeeployNetwork_dualconv_blocks4_buffer = args->DeeployNetwork_dualconv_blocks4_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks4_tiling_closure_args_t DeeployNetwork__dualconv_blocks4_tiling_closure_args = (_dualconv_blocks4_tiling_closure_args_t){
      .DeeployNetwork_dualconv_blocks4_buffer = DeeployNetwork_dualconv_blocks4_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref};

  // _dualconv_blocks4_tiling_closure CLOSURE CALL
  _dualconv_blocks4_tiling_closure(&DeeployNetwork__dualconv_blocks4_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks4_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks4_input_3_transposed;
  int8_t *DeeployNetwork_dualconv_blocks4_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr;
} _dualconv_blocks4_closure_args_t;

static void _dualconv_blocks4_closure(void *_dualconv_blocks4_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_closure_args_t *args = (_dualconv_blocks4_closure_args_t *)_dualconv_blocks4_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks4_input_0_transposed = args->DeeployNetwork_dualconv_blocks4_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks4_input_3_transposed = args->DeeployNetwork_dualconv_blocks4_input_3_transposed;
  int8_t *DeeployNetwork_dualconv_blocks4_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks4_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 28928);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 21760);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32256);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32128);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32384);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32512);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 30208);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 31488);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_dualconv_blocks4_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks4_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_dualconv_blocks4_input_3_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks4_input_3_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_blocks4convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_blocks4convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_dualconv_blocks4_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks4_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr]);
  void *DeeployNetwork_dualconv_blocks4_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443072, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_dualconv_blocks4_input_0_transposed_ref);
    mchan_transfer_1d(1443072, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_dualconv_blocks4_input_3_transposed_ref);
    mchan_transfer_1d(1448960, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_2_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_blocks4convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_blocks4convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _dualconv_blocks4_cluster_fork_args_t DeeployNetwork__dualconv_blocks4_cluster_fork_args = (_dualconv_blocks4_cluster_fork_args_t){
        .DeeployNetwork_dualconv_blocks4_buffer = DeeployNetwork_dualconv_blocks4_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_bmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_in_m_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks4_cluster_fork, &DeeployNetwork__dualconv_blocks4_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311360, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_dualconv_blocks4_output_0_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks4_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks4_input_3_transposed;
  int8_t *DeeployNetwork_dualconv_blocks4_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr;
} _dualconv_blocks4_closure_L3_args_t;

static void _dualconv_blocks4_closure_L3(void *_dualconv_blocks4_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_closure_L3_args_t *args = (_dualconv_blocks4_closure_L3_args_t *)_dualconv_blocks4_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks4_input_0_transposed = args->DeeployNetwork_dualconv_blocks4_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks4_input_3_transposed = args->DeeployNetwork_dualconv_blocks4_input_3_transposed;
  int8_t *DeeployNetwork_dualconv_blocks4_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks4_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks4_closure_args_t DeeployNetwork__dualconv_blocks4_closure_args = (_dualconv_blocks4_closure_args_t){
      .DeeployNetwork_dualconv_blocks4_input_0_transposed = DeeployNetwork_dualconv_blocks4_input_0_transposed,
      .DeeployNetwork_dualconv_blocks4_input_3_transposed = DeeployNetwork_dualconv_blocks4_input_3_transposed,
      .DeeployNetwork_dualconv_blocks4_output_0_pre_transposed = DeeployNetwork_dualconv_blocks4_output_0_pre_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr};

  // _dualconv_blocks4_closure CLOSURE CALL
  _dualconv_blocks4_closure(&DeeployNetwork__dualconv_blocks4_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref;
} _dualconv_blocks4_output_0_pre_transpose_tiling_closure_args_t;

static void _dualconv_blocks4_output_0_pre_transpose_tiling_closure(void *_dualconv_blocks4_output_0_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_output_0_pre_transpose_tiling_closure_args_t *args =
      (_dualconv_blocks4_output_0_pre_transpose_tiling_closure_args_t *)_dualconv_blocks4_output_0_pre_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 2, 5, 64] -> [1, 64, 2, 5] (Name: dualconv_blocks4_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 2;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 64;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_3][dimLen_1][dimLen_2])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref;
} _dualconv_blocks4_output_0_pre_transpose_cluster_fork_args_t;

static void _dualconv_blocks4_output_0_pre_transpose_cluster_fork(void *_dualconv_blocks4_output_0_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_output_0_pre_transpose_cluster_fork_args_t *args =
      (_dualconv_blocks4_output_0_pre_transpose_cluster_fork_args_t *)_dualconv_blocks4_output_0_pre_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks4_output_0_pre_transpose_tiling_closure_args_t DeeployNetwork__dualconv_blocks4_output_0_pre_transpose_tiling_closure_args =
      (_dualconv_blocks4_output_0_pre_transpose_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref};

  // _dualconv_blocks4_output_0_pre_transpose_tiling_closure CLOSURE CALL
  _dualconv_blocks4_output_0_pre_transpose_tiling_closure(&DeeployNetwork__dualconv_blocks4_output_0_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks4_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr;
} _dualconv_blocks4_output_0_pre_transpose_closure_args_t;

static void _dualconv_blocks4_output_0_pre_transpose_closure(void *_dualconv_blocks4_output_0_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_output_0_pre_transpose_closure_args_t *args =
      (_dualconv_blocks4_output_0_pre_transpose_closure_args_t *)_dualconv_blocks4_output_0_pre_transpose_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks4_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks4_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 640);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_dualconv_blocks4_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks4_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 + DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1442432, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_dualconv_blocks4_output_0_pre_transposed_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _dualconv_blocks4_output_0_pre_transpose_cluster_fork_args_t DeeployNetwork__dualconv_blocks4_output_0_pre_transpose_cluster_fork_args =
        (_dualconv_blocks4_output_0_pre_transpose_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks4_output_0_pre_transpose_cluster_fork,
                    &DeeployNetwork__dualconv_blocks4_output_0_pre_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311360, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_output_0_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks4_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr;
} _dualconv_blocks4_output_0_pre_transpose_closure_L3_args_t;

static void _dualconv_blocks4_output_0_pre_transpose_closure_L3(void *_dualconv_blocks4_output_0_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks4_output_0_pre_transpose_closure_L3_args_t *args =
      (_dualconv_blocks4_output_0_pre_transpose_closure_L3_args_t *)_dualconv_blocks4_output_0_pre_transpose_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks4_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks4_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks4_output_0_pre_transpose_closure_args_t DeeployNetwork__dualconv_blocks4_output_0_pre_transpose_closure_args =
      (_dualconv_blocks4_output_0_pre_transpose_closure_args_t){.DeeployNetwork_dualconv_blocks4_output_0_pre_transposed =
                                                                    DeeployNetwork_dualconv_blocks4_output_0_pre_transposed,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr};

  // _dualconv_blocks4_output_0_pre_transpose_closure CLOSURE CALL
  _dualconv_blocks4_output_0_pre_transpose_closure(&DeeployNetwork__dualconv_blocks4_output_0_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork() {
  int8_t *DeeployNetwork_dualconv_blocks4_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks4_input_3_transposed;
  int8_t *DeeployNetwork_dualconv_blocks4_output_0_pre_transposed;
  void *DeeployNetwork_dualconv_blocks4_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr;
    DeeployNetwork_dualconv_blocks4_input_0_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 7168);
    _dualconv_blocks4_input_0_transpose_closure_L3_args_t DeeployNetwork__dualconv_blocks4_input_0_transpose_closure_L3_args =
        (_dualconv_blocks4_input_0_transpose_closure_L3_args_t){.DeeployNetwork_dualconv_blocks4_input_0_transposed =
                                                                    DeeployNetwork_dualconv_blocks4_input_0_transposed,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_0_transpose_tileIdxPtr};

    // _dualconv_blocks4_input_0_transpose_closure_L3 CLOSURE CALL
    _dualconv_blocks4_input_0_transpose_closure_L3(&DeeployNetwork__dualconv_blocks4_input_0_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr;
    DeeployNetwork_dualconv_blocks4_input_3_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9856);
    _dualconv_blocks4_input_3_transpose_closure_L3_args_t DeeployNetwork__dualconv_blocks4_input_3_transpose_closure_L3_args =
        (_dualconv_blocks4_input_3_transpose_closure_L3_args_t){.DeeployNetwork_dualconv_blocks4_input_3_transposed =
                                                                    DeeployNetwork_dualconv_blocks4_input_3_transposed,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_input_3_transpose_tileIdxPtr};

    // _dualconv_blocks4_input_3_transpose_closure_L3 CLOSURE CALL
    _dualconv_blocks4_input_3_transpose_closure_L3(&DeeployNetwork__dualconv_blocks4_input_3_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr;
    DeeployNetwork_dualconv_blocks4_output_0_pre_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8576);
    _dualconv_blocks4_closure_L3_args_t DeeployNetwork__dualconv_blocks4_closure_L3_args = (_dualconv_blocks4_closure_L3_args_t){
        .DeeployNetwork_dualconv_blocks4_input_0_transposed = DeeployNetwork_dualconv_blocks4_input_0_transposed,
        .DeeployNetwork_dualconv_blocks4_input_3_transposed = DeeployNetwork_dualconv_blocks4_input_3_transposed,
        .DeeployNetwork_dualconv_blocks4_output_0_pre_transposed = DeeployNetwork_dualconv_blocks4_output_0_pre_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_tileIdxPtr};

    // _dualconv_blocks4_closure_L3 CLOSURE CALL
    _dualconv_blocks4_closure_L3(&DeeployNetwork__dualconv_blocks4_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr;
    _dualconv_blocks4_output_0_pre_transpose_closure_L3_args_t DeeployNetwork__dualconv_blocks4_output_0_pre_transpose_closure_L3_args =
        (_dualconv_blocks4_output_0_pre_transpose_closure_L3_args_t){.DeeployNetwork_dualconv_blocks4_output_0_pre_transposed =
                                                                         DeeployNetwork_dualconv_blocks4_output_0_pre_transposed,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks4_output_0_pre_transpose_tileIdxPtr};

    // _dualconv_blocks4_output_0_pre_transpose_closure_L3 CLOSURE CALL
    _dualconv_blocks4_output_0_pre_transpose_closure_L3(&DeeployNetwork__dualconv_blocks4_output_0_pre_transpose_closure_L3_args);
  }
}

void InitNetwork() {

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 32640);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 11136);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9856);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8448);
  DeeployNetwork_input_3 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8576);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_inputs[3] = (void *)DeeployNetwork_input_3;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
