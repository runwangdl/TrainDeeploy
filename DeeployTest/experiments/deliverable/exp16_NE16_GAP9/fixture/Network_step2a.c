#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "ne16.h"
#include "ne16_pulp_bsp.h"
#include "ne16_task.h"
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
uint8_t *DeeployNetwork_input_0;
int8_t *DeeployNetwork_output_0;

static PI_L2 uint8_t DeeployNetwork__NE16_ADJUST_WEIGHT_MEMORY_LAYOUT_PASS_0_b1_weight_int8_tensor[256] = {
    5,   0, 91,  0, 62,  0, 185, 0, 168, 0, 0,   0, 31,  0, 251, 0, 194, 0, 247, 0, 246, 0, 242, 0, 141, 0, 195, 0, 34,  0, 125, 0, 88,  0, 19,  0, 59,  0,
    124, 0, 176, 0, 55,  0, 232, 0, 71,  0, 80,  0, 230, 0, 224, 0, 64,  0, 255, 0, 96,  0, 19,  0, 4,   0, 167, 0, 228, 0, 156, 0, 230, 0, 254, 0, 146, 0,
    53,  0, 75,  0, 163, 0, 218, 0, 3,   0, 73,  0, 228, 0, 22,  0, 67,  0, 188, 0, 170, 0, 51,  0, 140, 0, 239, 0, 133, 0, 16,  0, 47,  0, 243, 0, 25,  0,
    102, 0, 62,  0, 93,  0, 70,  0, 70,  0, 89,  0, 243, 0, 62,  0, 60,  0, 165, 0, 59,  0, 16,  0, 169, 0, 142, 0, 0,   0, 39,  0, 225, 0, 247, 0, 19,  0,
    123, 0, 195, 0, 202, 0, 16,  0, 65,  0, 119, 0, 201, 0, 82,  0, 219, 0, 31,  0, 210, 0, 32,  0, 74,  0, 172, 0, 226, 0, 138, 0, 47,  0, 172, 0, 33,  0,
    250, 0, 254, 0, 180, 0, 171, 0, 117, 0, 5,   0, 242, 0, 29,  0, 107, 0, 150, 0, 144, 0, 68,  0, 39,  0, 114, 0, 209, 0, 192, 0, 14,  0, 69,  0, 177, 0,
    227, 0, 77,  0, 31,  0, 188, 0, 70,  0, 47,  0, 101, 0, 158, 0, 109, 0, 184, 0, 126, 0, 94,  0, 19,  0, 140, 0};

static PI_L2 int32_t DeeployNetwork_b1_rqs_mul_tensor[16] = {60, 76, 74, 89, 87, 85, 68, 68, 71, 74, 66, 66, 81, 69, 61, 88};

static PI_L2 int32_t DeeployNetwork_b1_rqs_add_tensor[16] = {32788, 32655, 32775, 32747, 32796, 32779, 32833, 32761,
                                                             32708, 32809, 32757, 32856, 32803, 32779, 32840, 32737};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_cumByteOffset[1] = {0};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo[2] = {135, 1};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo[2] = {3, 1};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi[2] = {3, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_numTiles[2] = {0, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[2] = {0, 9720};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cmd[2] = {1451512, 1441816};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_size[2] = {9720, 24};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[2] = {0, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[2] = {0, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[2] = {0, 0};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cumByteOffset[2] = {0, 19440};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[2] = {1330160, 1310768};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size[2] = {19440, 48};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[1];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure(void *__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args_t *)__MERGE_CONVRQ_PASS_0_input_0_transpose_tiling_closure_args;

  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 8, 14, 87] -> [1, 14, 87, 8] (Name: _MERGE_CONVRQ_PASS_0_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 87;

  // RW: GCC Segmentation fault
  uint8_t (*src)[dimLen_1][dimLen_2][dimLen_3] =
      (uint8_t (*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  uint8_t (*dst)[dimLen_2][dimLen_3][dimLen_1] =
      (uint8_t (*)[dimLen_2][dimLen_3][dimLen_1])DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;

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
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref;
} __MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t;

static void __MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork(void *__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *args =
      (__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t *)__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args;

  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
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
  uint8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_input_0_transpose_closure(void *__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *args =
      (__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args_t *)__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_args;

  uint8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 9744);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 + DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_output = -1;
  int transfer_input = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1451536,
                                      .size = 9744,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_input_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    __MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args =
        (__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref =
                                                                          DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_in_ref,
                                                                      .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref =
                                                                          DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)__MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork,
                    &DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1320464,
                                      .size = 9744,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_data_out_ref,
                                      .ext =
                                          DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose__MERGE_CONVRQ_PASS_0_input_0_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(void *__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *args =
      (__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t *)__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args;

  uint8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
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
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref;
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

  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref;
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
              .input_stride = (ne16_stride_t){.d0 = 8, .d1 = 24, .d2 = 0},
              .output_stride = (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 16, .d2 = 48},
              task.data.cfg.weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
              .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                     .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref, 1)},
                                          .remainder = {.KoKi = nnx_concat_half(16, 8),
                                                        .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref, 3),
                                                        .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref, 3)}},
              .padding = (0 << 28) + (0 << 24) + (0 << 20) + (0 << 16),
              .weight_offset_factor = -123,
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
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_args_t;

static void __MERGE_CONVRQ_PASS_0_closure(void *__MERGE_CONVRQ_PASS_0_closure_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_args_t *)__MERGE_CONVRQ_PASS_0_closure_args;

  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped =
      args->DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19440);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 29160);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 29416);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 29480);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped_ref =
      (void *)((char *)DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_ADJUST_WEIGHT_MEMORY_LAYOUT_PASS_0_b1_weight_int8_tensor_ref =
      (void *)((char *)DeeployNetwork__NE16_ADJUST_WEIGHT_MEMORY_LAYOUT_PASS_0_b1_weight_int8_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_b1_rqs_mul_tensor_ref =
      (void *)((char *)DeeployNetwork_b1_rqs_mul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_b1_rqs_add_tensor_ref =
      (void *)((char *)DeeployNetwork_b1_rqs_add_tensor +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped +
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
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE
    // DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped_ref =
        (void
             *)((char
                     *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped_ref) +
                9720);

    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1442048,
          .size = 256,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_weight_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__NE16_ADJUST_WEIGHT_MEMORY_LAYOUT_PASS_0_b1_weight_int8_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_mul_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_b1_rqs_mul_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_b1_rqs_add_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_bHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = &DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo[TILING_I];

    __MERGE_CONVRQ_PASS_0_tiling_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_tiling_closure_args = (__MERGE_CONVRQ_PASS_0_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_nHo_ref,
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
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped_ref
    DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped_ref) + 19440);

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
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
} __MERGE_CONVRQ_PASS_0_closure_L3_args_t;

static void __MERGE_CONVRQ_PASS_0_closure_L3(void *__MERGE_CONVRQ_PASS_0_closure_L3_args) {
  // CLOSURE ARG CAST
  __MERGE_CONVRQ_PASS_0_closure_L3_args_t *args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t *)__MERGE_CONVRQ_PASS_0_closure_L3_args;

  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped =
      args->DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped = args->DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  __MERGE_CONVRQ_PASS_0_closure_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_args = (__MERGE_CONVRQ_PASS_0_closure_args_t){
      .DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped =
          DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped,
      .DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped = DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped,
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

  // Transpose [1, 14, 87, 16] -> [1, 16, 14, 87] (Name: _MERGE_CONVRQ_PASS_0_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 14;

  uint16_t dimLen_2 = 87;

  uint16_t dimLen_3 = 16;

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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19488);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 + DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_output = -1;
  int transfer_input = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1461280,
          .size = 19488,
          .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
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
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1330208,
                                      .size = 19488,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_output_0_pre_transpose_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
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
  uint8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  uint8_t *DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped;
  int8_t *DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 19488);
    __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args =
        (__MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args_t){.DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed =
                                                                        DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed,
                                                                    .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr =
                                                                        DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_input_0_transpose_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_input_0_transpose_closure_L3_args);
  }
  {

    // Reshape (Name: _NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshape, Op: Reshape)
    DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped =
        DeeployNetwork__MERGE_CONVRQ_PASS_0_input_0_transposed;
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr;
    DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    __MERGE_CONVRQ_PASS_0_closure_L3_args_t DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args = (__MERGE_CONVRQ_PASS_0_closure_L3_args_t){
        .DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped =
            DeeployNetwork__NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_input_0_transposed_Reshaped,
        .DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped = DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped,
        .DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1__MERGE_CONVRQ_PASS_0_tileIdxPtr};

    // __MERGE_CONVRQ_PASS_0_closure_L3 CLOSURE CALL
    __MERGE_CONVRQ_PASS_0_closure_L3(&DeeployNetwork___MERGE_CONVRQ_PASS_0_closure_L3_args);
  }
  {

    // Reshape (Name: _NE16_RESHAPE_POINTWISE_CONVOLUTION_PASS_0_MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped_Reshape, Op: Reshape)
    DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed = DeeployNetwork__MERGE_CONVRQ_PASS_0_output_0_pre_transposed_Reshaped;
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

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 38976);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 38976);

  DeeployNetwork_input_0 = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 19488);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
