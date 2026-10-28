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

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor[8] = {391, 233, 337, 264, 318, 251, 298, 413};

static PI_L2 int32_t DeeployNetwork_blocks0convweight_int8_pmul_tensor[8] = {85469, 143432, 99234, 126554, 105298, 133299, 112298, 80873};

static PI_L2 int32_t DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor[8] = {7666, 12866, 8901, 11352, 9445, 11957, 10073, 7254};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_cumByteOffset[1] = {0};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y[2] = {386, 317};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y[2] = {385, 316};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left[2] = {2, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right[2] = {0, 2};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_numTiles[2] = {0, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_cumByteOffset[2] = {0, 383};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_cmd[2] = {1971484, 1970518};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_size_1d[2] = {386, 317};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_cumByteOffset[2] = {0, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_cumByteOffset[2] = {0, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_cumByteOffset[2] = {0, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_cumByteOffset[2] = {0, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_cumByteOffset[2] = {0, 0};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_cumByteOffset[2] = {0, 6160};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_cmd[2] = {1921248, 1905792};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_size_1d[2] = {6160, 5056};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_numTiles[2] = {0, 4};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_cumByteOffset[4] = {0, 4, 8, 12};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_cumByteOffset[4] = {0, 39256, 78512, 117768};

void *DeeployNetwork_inputs[3];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref;
} _dualconv_blocks0_input_0_transpose_tiling_closure_args_t;

static void _dualconv_blocks0_input_0_transpose_tiling_closure(void *_dualconv_blocks0_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_input_0_transpose_tiling_closure_args_t *args =
      (_dualconv_blocks0_input_0_transpose_tiling_closure_args_t *)_dualconv_blocks0_input_0_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 1, 14, 700] -> [1, 14, 700, 1] (Name: dualconv_blocks0_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 700;

  // RW: GCC Segmentation fault
  int8_t (*src)[dimLen_1][dimLen_2][dimLen_3] =
      (int8_t (*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref;
  int8_t (*dst)[dimLen_2][dimLen_3][dimLen_1] =
      (int8_t (*)[dimLen_2][dimLen_3][dimLen_1])DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref;

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref;
} _dualconv_blocks0_input_0_transpose_cluster_fork_args_t;

static void _dualconv_blocks0_input_0_transpose_cluster_fork(void *_dualconv_blocks0_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_input_0_transpose_cluster_fork_args_t *args =
      (_dualconv_blocks0_input_0_transpose_cluster_fork_args_t *)_dualconv_blocks0_input_0_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks0_input_0_transpose_tiling_closure_args_t DeeployNetwork__dualconv_blocks0_input_0_transpose_tiling_closure_args =
      (_dualconv_blocks0_input_0_transpose_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref};

  // _dualconv_blocks0_input_0_transpose_tiling_closure CLOSURE CALL
  _dualconv_blocks0_input_0_transpose_tiling_closure(&DeeployNetwork__dualconv_blocks0_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks0_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr;
} _dualconv_blocks0_input_0_transpose_closure_args_t;

static void _dualconv_blocks0_input_0_transpose_closure(void *_dualconv_blocks0_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_input_0_transpose_closure_args_t *args =
      (_dualconv_blocks0_input_0_transpose_closure_args_t *)_dualconv_blocks0_input_0_transpose_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks0_input_0_transposed = args->DeeployNetwork_dualconv_blocks0_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 9800);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 + DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_dualconv_blocks0_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks0_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1451592,
                                      .size = 9800,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_input_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _dualconv_blocks0_input_0_transpose_cluster_fork_args_t DeeployNetwork__dualconv_blocks0_input_0_transpose_cluster_fork_args =
        (_dualconv_blocks0_input_0_transpose_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks0_input_0_transpose_cluster_fork, &DeeployNetwork__dualconv_blocks0_input_0_transpose_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1320520,
                                      .size = 9800,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_dualconv_blocks0_input_0_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks0_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr;
} _dualconv_blocks0_input_0_transpose_closure_L3_args_t;

static void _dualconv_blocks0_input_0_transpose_closure_L3(void *_dualconv_blocks0_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_input_0_transpose_closure_L3_args_t *args =
      (_dualconv_blocks0_input_0_transpose_closure_L3_args_t *)_dualconv_blocks0_input_0_transpose_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks0_input_0_transposed = args->DeeployNetwork_dualconv_blocks0_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks0_input_0_transpose_closure_args_t DeeployNetwork__dualconv_blocks0_input_0_transpose_closure_args =
      (_dualconv_blocks0_input_0_transpose_closure_args_t){.DeeployNetwork_dualconv_blocks0_input_0_transposed =
                                                               DeeployNetwork_dualconv_blocks0_input_0_transposed,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr};

  // _dualconv_blocks0_input_0_transpose_closure CLOSURE CALL
  _dualconv_blocks0_input_0_transpose_closure(&DeeployNetwork__dualconv_blocks0_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref;
} _dualconv_blocks0_input_1_transpose_tiling_closure_args_t;

static void _dualconv_blocks0_input_1_transpose_tiling_closure(void *_dualconv_blocks0_input_1_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_input_1_transpose_tiling_closure_args_t *args =
      (_dualconv_blocks0_input_1_transpose_tiling_closure_args_t *)_dualconv_blocks0_input_1_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [8, 1, 1, 4] -> [8, 1, 4, 1] (Name: dualconv_blocks0_input_1_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 8;

  uint16_t dimLen_1 = 1;

  uint16_t dimLen_2 = 1;

  uint16_t dimLen_3 = 4;

  // RW: GCC Segmentation fault
  int8_t (*src)[dimLen_1][dimLen_2][dimLen_3] =
      (int8_t (*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref;
  int8_t (*dst)[dimLen_2][dimLen_3][dimLen_1] =
      (int8_t (*)[dimLen_2][dimLen_3][dimLen_1])DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref;

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref;
} _dualconv_blocks0_input_1_transpose_cluster_fork_args_t;

static void _dualconv_blocks0_input_1_transpose_cluster_fork(void *_dualconv_blocks0_input_1_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_input_1_transpose_cluster_fork_args_t *args =
      (_dualconv_blocks0_input_1_transpose_cluster_fork_args_t *)_dualconv_blocks0_input_1_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks0_input_1_transpose_tiling_closure_args_t DeeployNetwork__dualconv_blocks0_input_1_transpose_tiling_closure_args =
      (_dualconv_blocks0_input_1_transpose_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref};

  // _dualconv_blocks0_input_1_transpose_tiling_closure CLOSURE CALL
  _dualconv_blocks0_input_1_transpose_tiling_closure(&DeeployNetwork__dualconv_blocks0_input_1_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks0_input_1_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr;
} _dualconv_blocks0_input_1_transpose_closure_args_t;

static void _dualconv_blocks0_input_1_transpose_closure(void *_dualconv_blocks0_input_1_transpose_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_input_1_transpose_closure_args_t *args =
      (_dualconv_blocks0_input_1_transpose_closure_args_t *)_dualconv_blocks0_input_1_transpose_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks0_input_1_transposed = args->DeeployNetwork_dualconv_blocks0_input_1_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 + DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_dualconv_blocks0_input_1_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks0_input_1_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441824,
                                      .size = 32,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_input_1_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _dualconv_blocks0_input_1_transpose_cluster_fork_args_t DeeployNetwork__dualconv_blocks0_input_1_transpose_cluster_fork_args =
        (_dualconv_blocks0_input_1_transpose_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks0_input_1_transpose_cluster_fork, &DeeployNetwork__dualconv_blocks0_input_1_transpose_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1310752,
                                      .size = 32,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_dualconv_blocks0_input_1_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks0_input_1_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr;
} _dualconv_blocks0_input_1_transpose_closure_L3_args_t;

static void _dualconv_blocks0_input_1_transpose_closure_L3(void *_dualconv_blocks0_input_1_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_input_1_transpose_closure_L3_args_t *args =
      (_dualconv_blocks0_input_1_transpose_closure_L3_args_t *)_dualconv_blocks0_input_1_transpose_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks0_input_1_transposed = args->DeeployNetwork_dualconv_blocks0_input_1_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks0_input_1_transpose_closure_args_t DeeployNetwork__dualconv_blocks0_input_1_transpose_closure_args =
      (_dualconv_blocks0_input_1_transpose_closure_args_t){.DeeployNetwork_dualconv_blocks0_input_1_transposed =
                                                               DeeployNetwork_dualconv_blocks0_input_1_transposed,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr};

  // _dualconv_blocks0_input_1_transpose_closure CLOSURE CALL
  _dualconv_blocks0_input_1_transpose_closure(&DeeployNetwork__dualconv_blocks0_input_1_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_dualconv_blocks0_buffer;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref;
} _dualconv_blocks0_tiling_closure_args_t;

static void _dualconv_blocks0_tiling_closure(void *_dualconv_blocks0_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_tiling_closure_args_t *args = (_dualconv_blocks0_tiling_closure_args_t *)_dualconv_blocks0_tiling_closure_args;

  void *DeeployNetwork_dualconv_blocks0_buffer = args->DeeployNetwork_dualconv_blocks0_buffer;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP DUAL-PASS QZO CONV (Name: dualconv_blocks0, Op: RequantizedDualConv)  -- QW exp5
  // Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
  uint32_t dualconv_blocks0_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 0);
  uint32_t dualconv_blocks0_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 1);
  uint32_t dualconv_blocks0_tile_xor = (0 * 0x9E3779B1u);
  float32_t dualconv_blocks0_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_DualConv2d_i8_i8_i8_HWC(DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref,
                               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref,
                               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref,
                               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref, (int8_t *)DeeployNetwork_dualconv_blocks0_buffer,
                               dualconv_blocks0_seed_w, dualconv_blocks0_seed_b, dualconv_blocks0_tile_xor, 15, 16, dualconv_blocks0_eps_scale, NUM_CORES, 16,
                               *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref, 14, 1,
                               *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref, 14, ((16) >> 1), 4, 1, 0, 0,
                               *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref,
                               *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_dualconv_blocks0_buffer;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref;
} _dualconv_blocks0_cluster_fork_args_t;

static void _dualconv_blocks0_cluster_fork(void *_dualconv_blocks0_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_cluster_fork_args_t *args = (_dualconv_blocks0_cluster_fork_args_t *)_dualconv_blocks0_cluster_fork_args;

  void *DeeployNetwork_dualconv_blocks0_buffer = args->DeeployNetwork_dualconv_blocks0_buffer;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks0_tiling_closure_args_t DeeployNetwork__dualconv_blocks0_tiling_closure_args = (_dualconv_blocks0_tiling_closure_args_t){
      .DeeployNetwork_dualconv_blocks0_buffer = DeeployNetwork_dualconv_blocks0_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref};

  // _dualconv_blocks0_tiling_closure CLOSURE CALL
  _dualconv_blocks0_tiling_closure(&DeeployNetwork__dualconv_blocks0_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks0_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks0_input_1_transposed;
  int8_t *DeeployNetwork_dualconv_blocks0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr;
} _dualconv_blocks0_closure_args_t;

static void _dualconv_blocks0_closure(void *_dualconv_blocks0_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_closure_args_t *args = (_dualconv_blocks0_closure_args_t *)_dualconv_blocks0_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks0_input_0_transposed = args->DeeployNetwork_dualconv_blocks0_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks0_input_1_transposed = args->DeeployNetwork_dualconv_blocks0_input_1_transposed;
  int8_t *DeeployNetwork_dualconv_blocks0_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref =
      (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref =
      (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref =
      (uint8_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref =
      (uint8_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 86240);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 91864);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 91896);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 91832);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 91928);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 91960);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks0_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_input_1_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks0_input_1_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_blocks0convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_blocks0convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks0convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks0_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr]);
  void *DeeployNetwork_dualconv_blocks0_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 91672);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_cmd[TILING_I],
                                  DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref,
                                  DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_input_0_transposed_ref,
                                  DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_size_1d[TILING_I], 700);

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_input_0_transposed_ref
    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_input_0_transposed_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_input_0_transposed_ref) + 383);

    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_input_1_transposed_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_QSTANDALONE_QCDQ_blocks_0_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_2_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_blocks0convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_blocks0convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref
    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref = &DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref
    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref = &DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref
    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref = &DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref
    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref = &DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y[TILING_I];

    _dualconv_blocks0_cluster_fork_args_t DeeployNetwork__dualconv_blocks0_cluster_fork_args = (_dualconv_blocks0_cluster_fork_args_t){
        .DeeployNetwork_dualconv_blocks0_buffer = DeeployNetwork_dualconv_blocks0_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_in_y_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dim_im_out_y_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_left_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_padding_x_right_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_bmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks0_cluster_fork, &DeeployNetwork__dualconv_blocks0_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_2d_ext_strided(DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_cmd[TILING_I],
                                  DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_ref,
                                  DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_output_0_pre_transposed_ref,
                                  DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_data_out_size_1d[TILING_I], 11216);

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_output_0_pre_transposed_ref
    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_output_0_pre_transposed_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_dualconv_blocks0_output_0_pre_transposed_ref) + 6160);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks0_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks0_input_1_transposed;
  int8_t *DeeployNetwork_dualconv_blocks0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr;
} _dualconv_blocks0_closure_L3_args_t;

static void _dualconv_blocks0_closure_L3(void *_dualconv_blocks0_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_closure_L3_args_t *args = (_dualconv_blocks0_closure_L3_args_t *)_dualconv_blocks0_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks0_input_0_transposed = args->DeeployNetwork_dualconv_blocks0_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks0_input_1_transposed = args->DeeployNetwork_dualconv_blocks0_input_1_transposed;
  int8_t *DeeployNetwork_dualconv_blocks0_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks0_closure_args_t DeeployNetwork__dualconv_blocks0_closure_args = (_dualconv_blocks0_closure_args_t){
      .DeeployNetwork_dualconv_blocks0_input_0_transposed = DeeployNetwork_dualconv_blocks0_input_0_transposed,
      .DeeployNetwork_dualconv_blocks0_input_1_transposed = DeeployNetwork_dualconv_blocks0_input_1_transposed,
      .DeeployNetwork_dualconv_blocks0_output_0_pre_transposed = DeeployNetwork_dualconv_blocks0_output_0_pre_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr};

  // _dualconv_blocks0_closure CLOSURE CALL
  _dualconv_blocks0_closure(&DeeployNetwork__dualconv_blocks0_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref;
} _dualconv_blocks0_output_0_pre_transpose_tiling_closure_args_t;

static void _dualconv_blocks0_output_0_pre_transpose_tiling_closure(void *_dualconv_blocks0_output_0_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_output_0_pre_transpose_tiling_closure_args_t *args =
      (_dualconv_blocks0_output_0_pre_transpose_tiling_closure_args_t *)_dualconv_blocks0_output_0_pre_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 14, 701, 16] -> [1, 16, 14, 701] (Name: dualconv_blocks0_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 14;

  uint16_t dimLen_2 = 701;

  uint16_t dimLen_3 = 4;

  // RW: GCC Segmentation fault
  int8_t (*src)[dimLen_1][dimLen_2][dimLen_3] =
      (int8_t (*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref;
  int8_t (*dst)[dimLen_3][dimLen_1][dimLen_2] =
      (int8_t (*)[dimLen_3][dimLen_1][dimLen_2])DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref;

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref;
} _dualconv_blocks0_output_0_pre_transpose_cluster_fork_args_t;

static void _dualconv_blocks0_output_0_pre_transpose_cluster_fork(void *_dualconv_blocks0_output_0_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_output_0_pre_transpose_cluster_fork_args_t *args =
      (_dualconv_blocks0_output_0_pre_transpose_cluster_fork_args_t *)_dualconv_blocks0_output_0_pre_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks0_output_0_pre_transpose_tiling_closure_args_t DeeployNetwork__dualconv_blocks0_output_0_pre_transpose_tiling_closure_args =
      (_dualconv_blocks0_output_0_pre_transpose_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref};

  // _dualconv_blocks0_output_0_pre_transpose_tiling_closure CLOSURE CALL
  _dualconv_blocks0_output_0_pre_transpose_tiling_closure(&DeeployNetwork__dualconv_blocks0_output_0_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr;
} _dualconv_blocks0_output_0_pre_transpose_closure_args_t;

static void _dualconv_blocks0_output_0_pre_transpose_closure(void *_dualconv_blocks0_output_0_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_output_0_pre_transpose_closure_args_t *args =
      (_dualconv_blocks0_output_0_pre_transpose_closure_args_t *)_dualconv_blocks0_output_0_pre_transpose_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks0_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39256);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_dualconv_blocks0_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks0_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 + DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 2005336,
                                      .size = 39256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref,
                                      .ext =
                                          DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_dualconv_blocks0_output_0_pre_transposed_ref,
                                      .ext_size_1d = 4,
                                      .ext_stride_1d = 16};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_dualconv_blocks0_output_0_pre_transposed_ref
    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_dualconv_blocks0_output_0_pre_transposed_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_dualconv_blocks0_output_0_pre_transposed_ref) + 4);

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _dualconv_blocks0_output_0_pre_transpose_cluster_fork_args_t DeeployNetwork__dualconv_blocks0_output_0_pre_transpose_cluster_fork_args =
        (_dualconv_blocks0_output_0_pre_transpose_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks0_output_0_pre_transpose_cluster_fork,
                    &DeeployNetwork__dualconv_blocks0_output_0_pre_transpose_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1349976,
                                      .size = 39256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_output_0_ref
    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_output_0_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_output_0_ref) + 39256);

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr;
} _dualconv_blocks0_output_0_pre_transpose_closure_L3_args_t;

static void _dualconv_blocks0_output_0_pre_transpose_closure_L3(void *_dualconv_blocks0_output_0_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks0_output_0_pre_transpose_closure_L3_args_t *args =
      (_dualconv_blocks0_output_0_pre_transpose_closure_L3_args_t *)_dualconv_blocks0_output_0_pre_transpose_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks0_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks0_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks0_output_0_pre_transpose_closure_args_t DeeployNetwork__dualconv_blocks0_output_0_pre_transpose_closure_args =
      (_dualconv_blocks0_output_0_pre_transpose_closure_args_t){.DeeployNetwork_dualconv_blocks0_output_0_pre_transposed =
                                                                    DeeployNetwork_dualconv_blocks0_output_0_pre_transposed,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr};

  // _dualconv_blocks0_output_0_pre_transpose_closure CLOSURE CALL
  _dualconv_blocks0_output_0_pre_transpose_closure(&DeeployNetwork__dualconv_blocks0_output_0_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  int8_t *DeeployNetwork_dualconv_blocks0_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks0_input_1_transposed;
  int8_t *DeeployNetwork_dualconv_blocks0_output_0_pre_transposed;
  void *DeeployNetwork_dualconv_blocks0_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr;
    DeeployNetwork_dualconv_blocks0_input_0_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _dualconv_blocks0_input_0_transpose_closure_L3_args_t DeeployNetwork__dualconv_blocks0_input_0_transpose_closure_L3_args =
        (_dualconv_blocks0_input_0_transpose_closure_L3_args_t){.DeeployNetwork_dualconv_blocks0_input_0_transposed =
                                                                    DeeployNetwork_dualconv_blocks0_input_0_transposed,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_0_transpose_tileIdxPtr};

    // _dualconv_blocks0_input_0_transpose_closure_L3 CLOSURE CALL
    _dualconv_blocks0_input_0_transpose_closure_L3(&DeeployNetwork__dualconv_blocks0_input_0_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr;
    DeeployNetwork_dualconv_blocks0_input_1_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9832);
    _dualconv_blocks0_input_1_transpose_closure_L3_args_t DeeployNetwork__dualconv_blocks0_input_1_transpose_closure_L3_args =
        (_dualconv_blocks0_input_1_transpose_closure_L3_args_t){.DeeployNetwork_dualconv_blocks0_input_1_transposed =
                                                                    DeeployNetwork_dualconv_blocks0_input_1_transposed,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_input_1_transpose_tileIdxPtr};

    // _dualconv_blocks0_input_1_transpose_closure_L3 CLOSURE CALL
    _dualconv_blocks0_input_1_transpose_closure_L3(&DeeployNetwork__dualconv_blocks0_input_1_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr;
    DeeployNetwork_dualconv_blocks0_output_0_pre_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 157024);
    _dualconv_blocks0_closure_L3_args_t DeeployNetwork__dualconv_blocks0_closure_L3_args = (_dualconv_blocks0_closure_L3_args_t){
        .DeeployNetwork_dualconv_blocks0_input_0_transposed = DeeployNetwork_dualconv_blocks0_input_0_transposed,
        .DeeployNetwork_dualconv_blocks0_input_1_transposed = DeeployNetwork_dualconv_blocks0_input_1_transposed,
        .DeeployNetwork_dualconv_blocks0_output_0_pre_transposed = DeeployNetwork_dualconv_blocks0_output_0_pre_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_tileIdxPtr};

    // _dualconv_blocks0_closure_L3 CLOSURE CALL
    _dualconv_blocks0_closure_L3(&DeeployNetwork__dualconv_blocks0_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr;
    _dualconv_blocks0_output_0_pre_transpose_closure_L3_args_t DeeployNetwork__dualconv_blocks0_output_0_pre_transpose_closure_L3_args =
        (_dualconv_blocks0_output_0_pre_transpose_closure_L3_args_t){.DeeployNetwork_dualconv_blocks0_output_0_pre_transposed =
                                                                         DeeployNetwork_dualconv_blocks0_output_0_pre_transposed,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks0_output_0_pre_transpose_tileIdxPtr};

    // _dualconv_blocks0_output_0_pre_transpose_closure_L3 CLOSURE CALL
    _dualconv_blocks0_output_0_pre_transpose_closure_L3(&DeeployNetwork__dualconv_blocks0_output_0_pre_transpose_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 91992);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 314048);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9832);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 19632);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9800);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
