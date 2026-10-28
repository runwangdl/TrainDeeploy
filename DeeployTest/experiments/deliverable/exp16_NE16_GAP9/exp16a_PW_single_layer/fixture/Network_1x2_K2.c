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
uint8_t *DeeployNetwork_input_1;
int8_t *DeeployNetwork_output_0;

static PI_L2 int32_t DeeployNetwork_rqs_mul_tensor[16] = {60, 76, 74, 89, 87, 85, 68, 68, 71, 74, 66, 66, 81, 69, 61, 88};

static PI_L2 int32_t DeeployNetwork_rqs_add_tensor[16] = {32819, 32809, 32726, 32733, 32804, 32732, 32814, 32723,
                                                          32811, 32810, 32721, 32721, 32730, 32813, 32819, 32803};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[2];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref;
} _b1_conv1xk_input_0_transpose_tiling_closure_args_t;

static void _b1_conv1xk_input_0_transpose_tiling_closure(void *_b1_conv1xk_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_input_0_transpose_tiling_closure_args_t *args =
      (_b1_conv1xk_input_0_transpose_tiling_closure_args_t *)_b1_conv1xk_input_0_transpose_tiling_closure_args;

  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 8, 4, 26] -> [1, 4, 26, 8] (Name: b1_conv1xk_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 4;

  uint16_t dimLen_3 = 26;

  // RW: GCC Segmentation fault
  uint8_t (*src)[dimLen_1][dimLen_2][dimLen_3] =
      (uint8_t (*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref;
  uint8_t (*dst)[dimLen_2][dimLen_3][dimLen_1] =
      (uint8_t (*)[dimLen_2][dimLen_3][dimLen_1])DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref;

  for (uint32_t i_0 = 0; i_0 < dimLen_0; i_0++) {

    for (uint32_t i_2 = 0; i_2 < dimLen_2; i_2++) {

      const uint32_t baseChunk = dimLen_3 / NUM_CORES;
      const uint32_t leftover = dimLen_3 - baseChunk * NUM_CORES;
      const uint32_t offset = baseChunk * coreId + (coreId < leftover ? coreId : leftover);
      const uint32_t chunk = coreId < leftover ? baseChunk + 1 : baseChunk;
      for (uint32_t i_3 = offset; i_3 < offset + chunk; i_3++) {

        for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {

          dst[i_0][i_2][i_3][i_1] = src[i_0][i_1][i_2][i_3];
        }
      }
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref;
} _b1_conv1xk_input_0_transpose_cluster_fork_args_t;

static void _b1_conv1xk_input_0_transpose_cluster_fork(void *_b1_conv1xk_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_input_0_transpose_cluster_fork_args_t *args =
      (_b1_conv1xk_input_0_transpose_cluster_fork_args_t *)_b1_conv1xk_input_0_transpose_cluster_fork_args;

  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b1_conv1xk_input_0_transpose_tiling_closure_args_t DeeployNetwork__b1_conv1xk_input_0_transpose_tiling_closure_args =
      (_b1_conv1xk_input_0_transpose_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref =
                                                                DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref =
                                                                DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref};

  // _b1_conv1xk_input_0_transpose_tiling_closure CLOSURE CALL
  _b1_conv1xk_input_0_transpose_tiling_closure(&DeeployNetwork__b1_conv1xk_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_b1_conv1xk_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr;
} _b1_conv1xk_input_0_transpose_closure_args_t;

static void _b1_conv1xk_input_0_transpose_closure(void *_b1_conv1xk_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_input_0_transpose_closure_args_t *args = (_b1_conv1xk_input_0_transpose_closure_args_t *)_b1_conv1xk_input_0_transpose_closure_args;

  uint8_t *DeeployNetwork_b1_conv1xk_input_0_transposed = args->DeeployNetwork_b1_conv1xk_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 832);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 + DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_b1_conv1xk_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_b1_conv1xk_input_0_transposed + DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_cumByteOffset
                                                                          [*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I =
           DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr];
       TILING_I <
       DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442624,
                                      .size = 832,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_input_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _b1_conv1xk_input_0_transpose_cluster_fork_args_t DeeployNetwork__b1_conv1xk_input_0_transpose_cluster_fork_args =
        (_b1_conv1xk_input_0_transpose_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref =
                                                                DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_in_ref,
                                                            .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref =
                                                                DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b1_conv1xk_input_0_transpose_cluster_fork, &DeeployNetwork__b1_conv1xk_input_0_transpose_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1311552,
                                      .size = 832,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_b1_conv1xk_input_0_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_b1_conv1xk_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr;
} _b1_conv1xk_input_0_transpose_closure_L3_args_t;

static void _b1_conv1xk_input_0_transpose_closure_L3(void *_b1_conv1xk_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_input_0_transpose_closure_L3_args_t *args = (_b1_conv1xk_input_0_transpose_closure_L3_args_t *)_b1_conv1xk_input_0_transpose_closure_L3_args;

  uint8_t *DeeployNetwork_b1_conv1xk_input_0_transposed = args->DeeployNetwork_b1_conv1xk_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b1_conv1xk_input_0_transpose_closure_args_t DeeployNetwork__b1_conv1xk_input_0_transpose_closure_args = (_b1_conv1xk_input_0_transpose_closure_args_t){
      .DeeployNetwork_b1_conv1xk_input_0_transposed = DeeployNetwork_b1_conv1xk_input_0_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr};

  // _b1_conv1xk_input_0_transpose_closure CLOSURE CALL
  _b1_conv1xk_input_0_transpose_closure(&DeeployNetwork__b1_conv1xk_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref;
} _b1_conv1xk_tiling_closure_args_t;

static void _b1_conv1xk_tiling_closure(void *_b1_conv1xk_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_tiling_closure_args_t *args = (_b1_conv1xk_tiling_closure_args_t *)_b1_conv1xk_tiling_closure_args;

  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16 1x2 dense conv as 2 pointwise dispatches (streamin accumulation)
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref + 0 * 256,
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref + 0 * 8 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 8, .d1 = 208, .d2 = 0},
                .output_stride = (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = 1600},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1), .HoWo = nnx_concat_half(2, 9)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 8), .HoWo = nnx_concat_half(1, 1), .HiWi = nnx_concat_half(1, 1)}},
                .padding = 0,
                .weight_offset_factor = -128,
                .filter_mask = 0,
                // tap 0 zeroes the accumulator; taps 1..K-1 stream the running int32 partial
                // sums back in from outfeat_addr (CONFIG0[14]).
                .conf0 = 4227143,
            }}};
    task.weight_d0_stride = NE16_WEIGHT_D0_STRIDE_MODE8;
    task.qw = 8;
    task.subtile_output_channel = 32;
    task.kernel_shape = 1;
    task.depthwise = 0;

    ne16_nnx_dispatch_wait(ne16_pulp_get_dev());
    ne16_nnx_dispatch(ne16_pulp_get_dev(), &task);
    ne16_nnx_resolve_wait(ne16_pulp_get_dev(), &task);
  }
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref + 1 * 256,
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref + 1 * 8 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 8, .d1 = 208, .d2 = 0},
                .output_stride = (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = 1600},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1), .HoWo = nnx_concat_half(2, 9)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 8), .HoWo = nnx_concat_half(1, 1), .HiWi = nnx_concat_half(1, 1)}},
                .padding = 0,
                .weight_offset_factor = -128,
                .filter_mask = 0,
                // tap 0 zeroes the accumulator; taps 1..K-1 stream the running int32 partial
                // sums back in from outfeat_addr (CONFIG0[14]).
                .conf0 = 4243527,
            }}};
    task.weight_d0_stride = NE16_WEIGHT_D0_STRIDE_MODE8;
    task.qw = 8;
    task.subtile_output_channel = 32;
    task.kernel_shape = 1;
    task.depthwise = 0;

    ne16_nnx_dispatch_wait(ne16_pulp_get_dev());
    ne16_nnx_dispatch(ne16_pulp_get_dev(), &task);
    ne16_nnx_resolve_wait(ne16_pulp_get_dev(), &task);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_b1_conv1xk_input_0_transposed;
  int32_t *DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr;
} _b1_conv1xk_closure_args_t;

static void _b1_conv1xk_closure(void *_b1_conv1xk_closure_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_closure_args_t *args = (_b1_conv1xk_closure_args_t *)_b1_conv1xk_closure_args;

  uint8_t *DeeployNetwork_b1_conv1xk_input_0_transposed = args->DeeployNetwork_b1_conv1xk_input_0_transposed;
  int32_t *DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed = args->DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6400);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7232);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_b1_conv1xk_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_b1_conv1xk_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_b1_conv1xk_conv_out_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442624,
                                      .size = 832,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_b1_conv1xk_input_0_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442304,
                                      .size = 512,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_1_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _b1_conv1xk_tiling_closure_args_t DeeployNetwork__b1_conv1xk_tiling_closure_args = (_b1_conv1xk_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref};

    // _b1_conv1xk_tiling_closure CLOSURE CALL
    _b1_conv1xk_tiling_closure(&DeeployNetwork__b1_conv1xk_tiling_closure_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1317120,
                                      .size = 6400,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_b1_conv1xk_conv_out_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_b1_conv1xk_input_0_transposed;
  int32_t *DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr;
} _b1_conv1xk_closure_L3_args_t;

static void _b1_conv1xk_closure_L3(void *_b1_conv1xk_closure_L3_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_closure_L3_args_t *args = (_b1_conv1xk_closure_L3_args_t *)_b1_conv1xk_closure_L3_args;

  uint8_t *DeeployNetwork_b1_conv1xk_input_0_transposed = args->DeeployNetwork_b1_conv1xk_input_0_transposed;
  int32_t *DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed = args->DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b1_conv1xk_closure_args_t DeeployNetwork__b1_conv1xk_closure_args =
      (_b1_conv1xk_closure_args_t){.DeeployNetwork_b1_conv1xk_input_0_transposed = DeeployNetwork_b1_conv1xk_input_0_transposed,
                                   .DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed = DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed,
                                   .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr};

  // _b1_conv1xk_closure CLOSURE CALL
  _b1_conv1xk_closure(&DeeployNetwork__b1_conv1xk_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref;
} _b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure_args_t;

static void _b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure(void *_b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure_args_t *args =
      (_b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure_args_t *)_b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 4, 25, 16] -> [1, 16, 4, 25] (Name: b1_conv1xk_conv_out_tensor_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 4;

  uint16_t dimLen_2 = 25;

  uint16_t dimLen_3 = 16;

  // RW: GCC Segmentation fault
  int32_t (*src)[dimLen_1][dimLen_2][dimLen_3] =
      (int32_t (*)[dimLen_1][dimLen_2][dimLen_3])DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref;
  int32_t (*dst)[dimLen_3][dimLen_1][dimLen_2] =
      (int32_t (*)[dimLen_3][dimLen_1][dimLen_2])DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref;

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
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref;
} _b1_conv1xk_conv_out_tensor_pre_transpose_cluster_fork_args_t;

static void _b1_conv1xk_conv_out_tensor_pre_transpose_cluster_fork(void *_b1_conv1xk_conv_out_tensor_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_conv_out_tensor_pre_transpose_cluster_fork_args_t *args =
      (_b1_conv1xk_conv_out_tensor_pre_transpose_cluster_fork_args_t *)_b1_conv1xk_conv_out_tensor_pre_transpose_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure_args_t DeeployNetwork__b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure_args =
      (_b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref};

  // _b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure CLOSURE CALL
  _b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure(&DeeployNetwork__b1_conv1xk_conv_out_tensor_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed;
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr;
} _b1_conv1xk_conv_out_tensor_pre_transpose_closure_args_t;

static void _b1_conv1xk_conv_out_tensor_pre_transpose_closure(void *_b1_conv1xk_conv_out_tensor_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_conv_out_tensor_pre_transpose_closure_args_t *args =
      (_b1_conv1xk_conv_out_tensor_pre_transpose_closure_args_t *)_b1_conv1xk_conv_out_tensor_pre_transpose_closure_args;

  int32_t *DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed = args->DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6400);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_b1_conv1xk_conv_out_tensor_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_conv_out_tensor_ref =
      (void *)((char *)DeeployNetwork_conv_out_tensor + DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_cumByteOffset
                                                            [*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1448192,
          .size = 6400,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref,
          .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_b1_conv1xk_conv_out_tensor_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _b1_conv1xk_conv_out_tensor_pre_transpose_cluster_fork_args_t DeeployNetwork__b1_conv1xk_conv_out_tensor_pre_transpose_cluster_fork_args =
        (_b1_conv1xk_conv_out_tensor_pre_transpose_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b1_conv1xk_conv_out_tensor_pre_transpose_cluster_fork,
                    &DeeployNetwork__b1_conv1xk_conv_out_tensor_pre_transpose_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1317120,
                                      .size = 6400,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_conv_out_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed;
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr;
} _b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3_args_t;

static void _b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3(void *_b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3_args_t *args =
      (_b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3_args_t *)_b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3_args;

  int32_t *DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed = args->DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b1_conv1xk_conv_out_tensor_pre_transpose_closure_args_t DeeployNetwork__b1_conv1xk_conv_out_tensor_pre_transpose_closure_args =
      (_b1_conv1xk_conv_out_tensor_pre_transpose_closure_args_t){.DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed =
                                                                     DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed,
                                                                 .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                                                 .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr =
                                                                     DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr};

  // _b1_conv1xk_conv_out_tensor_pre_transpose_closure CLOSURE CALL
  _b1_conv1xk_conv_out_tensor_pre_transpose_closure(&DeeployNetwork__b1_conv1xk_conv_out_tensor_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref;
} _b1_rqs_tiling_closure_args_t;

static void _b1_rqs_tiling_closure(void *_b1_rqs_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_rqs_tiling_closure_args_t *args = (_b1_rqs_tiling_closure_args_t *)_b1_rqs_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL

  // RequantShift (Name: b1_rqs, Op: RequantShift)
  RequantShift_s32_s8_NCHW(DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref, 1600, DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref,
                           DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref, 16, 100, 0, 0, -128, 127, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref;
} _b1_rqs_cluster_fork_args_t;

static void _b1_rqs_cluster_fork(void *_b1_rqs_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b1_rqs_cluster_fork_args_t *args = (_b1_rqs_cluster_fork_args_t *)_b1_rqs_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b1_rqs_tiling_closure_args_t DeeployNetwork__b1_rqs_tiling_closure_args =
      (_b1_rqs_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref};

  // _b1_rqs_tiling_closure CLOSURE CALL
  _b1_rqs_tiling_closure(&DeeployNetwork__b1_rqs_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr;
} _b1_rqs_closure_args_t;

static void _b1_rqs_closure(void *_b1_rqs_closure_args) {
  // CLOSURE ARG CAST
  _b1_rqs_closure_args_t *args = (_b1_rqs_closure_args_t *)_b1_rqs_closure_args;

  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1600);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8000);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8064);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_conv_out_tensor_ref =
      (void *)((char *)DeeployNetwork_conv_out_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_add_tensor_ref =
      (void *)((char *)DeeployNetwork_rqs_add_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_mul_tensor_ref =
      (void *)((char *)DeeployNetwork_rqs_mul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1448192,
                                      .size = 6400,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_conv_out_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_add_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_mul_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _b1_rqs_cluster_fork_args_t DeeployNetwork__b1_rqs_cluster_fork_args =
        (_b1_rqs_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b1_rqs_cluster_fork, &DeeployNetwork__b1_rqs_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1312320,
                                      .size = 1600,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr;
} _b1_rqs_closure_L3_args_t;

static void _b1_rqs_closure_L3(void *_b1_rqs_closure_L3_args) {
  // CLOSURE ARG CAST
  _b1_rqs_closure_L3_args_t *args = (_b1_rqs_closure_L3_args_t *)_b1_rqs_closure_L3_args;

  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b1_rqs_closure_args_t DeeployNetwork__b1_rqs_closure_args =
      (_b1_rqs_closure_args_t){.DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                               .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr};

  // _b1_rqs_closure CLOSURE CALL
  _b1_rqs_closure(&DeeployNetwork__b1_rqs_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  uint8_t *DeeployNetwork_b1_conv1xk_input_0_transposed;
  int32_t *DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed;
  int32_t *DeeployNetwork_conv_out_tensor;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr;
    DeeployNetwork_b1_conv1xk_input_0_transposed = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 6400);
    _b1_conv1xk_input_0_transpose_closure_L3_args_t DeeployNetwork__b1_conv1xk_input_0_transpose_closure_L3_args =
        (_b1_conv1xk_input_0_transpose_closure_L3_args_t){.DeeployNetwork_b1_conv1xk_input_0_transposed = DeeployNetwork_b1_conv1xk_input_0_transposed,
                                                          .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr =
                                                              DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_transpose_tileIdxPtr};

    // _b1_conv1xk_input_0_transpose_closure_L3 CLOSURE CALL
    _b1_conv1xk_input_0_transpose_closure_L3(&DeeployNetwork__b1_conv1xk_input_0_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr;
    DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _b1_conv1xk_closure_L3_args_t DeeployNetwork__b1_conv1xk_closure_L3_args =
        (_b1_conv1xk_closure_L3_args_t){.DeeployNetwork_b1_conv1xk_input_0_transposed = DeeployNetwork_b1_conv1xk_input_0_transposed,
                                        .DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed = DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr};

    // _b1_conv1xk_closure_L3 CLOSURE CALL
    _b1_conv1xk_closure_L3(&DeeployNetwork__b1_conv1xk_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr;
    DeeployNetwork_conv_out_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 6400);
    _b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3_args_t DeeployNetwork__b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3_args =
        (_b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3_args_t){.DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed =
                                                                          DeeployNetwork_b1_conv1xk_conv_out_tensor_pre_transposed,
                                                                      .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr =
                                                                          DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_pre_transpose_tileIdxPtr};

    // _b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3 CLOSURE CALL
    _b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3(&DeeployNetwork__b1_conv1xk_conv_out_tensor_pre_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr;
    _b1_rqs_closure_L3_args_t DeeployNetwork__b1_rqs_closure_L3_args =
        (_b1_rqs_closure_L3_args_t){.DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                    .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_tileIdxPtr};

    // _b1_rqs_closure_L3 CLOSURE CALL
    _b1_rqs_closure_L3(&DeeployNetwork__b1_rqs_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 12800);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 12800);

  DeeployNetwork_input_0 = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_1 = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 7232);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
