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

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[2];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
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

  // NE16 1xK dense conv as 1 DENSE 3x3 dispatches (streamin accumulation, exp16b)
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref + 0 * 2304,
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref + 0 * 24 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 8, .d1 = 208, .d2 = 0},
                .output_stride = (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = 1536},
                .weights_stride = (ne16_stride_t){.d0 = 18, .d1 = 144, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1), .HoWo = nnx_concat_half(2, 8)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 8), .HoWo = nnx_concat_half(1, 3), .HiWi = nnx_concat_half(1, 3)}},
                // H padding 1/1 keeps Hout = Hin; legal because this is 3x3 mode, and harmless
                // because the kernel's top and bottom rows are zero.
                .padding = (1 << 28) + (0 << 24) + (1 << 20) + (0 << 16),
                .weight_offset_factor = -128,
                .filter_mask = 16777472,
                .conf0 = 4227079,
            }}};
    task.weight_d0_stride = NE16_WEIGHT_D0_STRIDE_MODE8;
    task.qw = 8;
    task.subtile_output_channel = 32;
    task.kernel_shape = 3;
    task.depthwise = 0;

    ne16_nnx_dispatch_wait(ne16_pulp_get_dev());
    ne16_nnx_dispatch(ne16_pulp_get_dev(), &task);
    ne16_nnx_resolve_wait(ne16_pulp_get_dev(), &task);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr;
} _b1_conv1xk_closure_args_t;

static void _b1_conv1xk_closure(void *_b1_conv1xk_closure_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_closure_args_t *args = (_b1_conv1xk_closure_args_t *)_b1_conv1xk_closure_args;

  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8448);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6144);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_ref =
      (void *)((char *)DeeployNetwork_conv_out_tensor +
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
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1444096,
                                      .size = 2304,
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
      mchan_transfer_t __mchan_tmp = {.cmd = 1316864,
                                      .size = 6144,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_ref};
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
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr;
} _b1_conv1xk_closure_L3_args_t;

static void _b1_conv1xk_closure_L3(void *_b1_conv1xk_closure_L3_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_closure_L3_args_t *args = (_b1_conv1xk_closure_L3_args_t *)_b1_conv1xk_closure_L3_args;

  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b1_conv1xk_closure_args_t DeeployNetwork__b1_conv1xk_closure_args =
      (_b1_conv1xk_closure_args_t){.DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                   .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr};

  // _b1_conv1xk_closure CLOSURE CALL
  _b1_conv1xk_closure(&DeeployNetwork__b1_conv1xk_closure_args);

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
  RequantShift_s32_s8_NHWC(DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref, 1536, DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref,
                           DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref, 16, 16, 0, 0, -128, 127, 1);

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
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1536);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7680);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7744);
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
      mchan_transfer_t __mchan_tmp = {.cmd = 1447936,
                                      .size = 6144,
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
      mchan_transfer_t __mchan_tmp = {.cmd = 1312256,
                                      .size = 1536,
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
  int32_t *DeeployNetwork_conv_out_tensor;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr;
    DeeployNetwork_conv_out_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _b1_conv1xk_closure_L3_args_t DeeployNetwork__b1_conv1xk_closure_L3_args =
        (_b1_conv1xk_closure_L3_args_t){.DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_tileIdxPtr};

    // _b1_conv1xk_closure_L3 CLOSURE CALL
    _b1_conv1xk_closure_L3(&DeeployNetwork__b1_conv1xk_closure_L3_args);
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

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 9280);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 9280);

  DeeployNetwork_input_0 = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8448);
  DeeployNetwork_input_1 = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 6144);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 6144);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
