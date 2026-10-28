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

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride[2] = {736, 240};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride[2] = {4800, 832};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset[2] = {736, 240};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo[2] = {25, 5};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo[2] = {3, 1};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi[2] = {5, 3};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_numTiles[2] = {0, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_cumByteOffset[2] = {0, 600};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_cmd[2] = {1976384, 1969440};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_size[2] = {10304, 3360};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_size_1d[2] = {736, 240};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_cumByteOffset[2] = {0, 0};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_cumByteOffset[2] = {0, 4800};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_cmd[2] = {1902208, 1846656};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_size[2] = {67200, 11648};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_size_1d[2] = {4800, 832};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size[2] = {17248, 2464};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels[2] = {14, 2};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_numTiles[2] = {0, 2};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_cumByteOffset[2] = {0, 56};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_cmd[2] = {2035072, 1975936};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_size[2] = {68992, 9856};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_size_1d[2] = {56, 8};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_cumByteOffset[2] = {0, 896};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_cmd[2] = {1441848, 1441800};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_size[2] = {56, 8};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_cumByteOffset[2] = {0, 896};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_cmd[2] = {1441848, 1441800};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_size[2] = {56, 8};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_cumByteOffset[2] = {0, 14};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_cmd[2] = {1852256, 1837472};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_size[2] = {17248, 2464};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_size_1d[2] = {14, 2};

void *DeeployNetwork_inputs[2];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref;
} _b1_conv1xk_tiling_closure_args_t;

static void _b1_conv1xk_tiling_closure(void *_b1_conv1xk_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_conv1xk_tiling_closure_args_t *args = (_b1_conv1xk_tiling_closure_args_t *)_b1_conv1xk_tiling_closure_args;

  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16 1xK dense conv as 6 DENSE 3x3 dispatches (streamin accumulation, exp16b)
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref + 0 * 2304,
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref + 0 * 24 - *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 8, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 18, .d1 = 144, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(5, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 8),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(3, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref)}},
                // H padding 1/1 keeps Hout = Hin; legal because this is 3x3 mode, and harmless
                // because the kernel's top and bottom rows are zero.
                .padding = (1 << 28) + (0 << 24) + (1 << 20) + (0 << 16),
                .weight_offset_factor = -128,
                .filter_mask = 0,
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
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref + 1 * 2304,
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref + 1 * 24 - *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 8, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 18, .d1 = 144, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(5, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 8),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(3, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref)}},
                // H padding 1/1 keeps Hout = Hin; legal because this is 3x3 mode, and harmless
                // because the kernel's top and bottom rows are zero.
                .padding = (1 << 28) + (0 << 24) + (1 << 20) + (0 << 16),
                .weight_offset_factor = -128,
                .filter_mask = 0,
                .conf0 = 4243463,
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
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref + 2 * 2304,
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref + 2 * 24 - *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 8, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 18, .d1 = 144, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(5, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 8),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(3, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref)}},
                // H padding 1/1 keeps Hout = Hin; legal because this is 3x3 mode, and harmless
                // because the kernel's top and bottom rows are zero.
                .padding = (1 << 28) + (0 << 24) + (1 << 20) + (0 << 16),
                .weight_offset_factor = -128,
                .filter_mask = 0,
                .conf0 = 4243463,
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
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref + 3 * 2304,
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref + 3 * 24 - *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 8, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 18, .d1 = 144, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(5, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 8),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(3, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref)}},
                // H padding 1/1 keeps Hout = Hin; legal because this is 3x3 mode, and harmless
                // because the kernel's top and bottom rows are zero.
                .padding = (1 << 28) + (0 << 24) + (1 << 20) + (0 << 16),
                .weight_offset_factor = -128,
                .filter_mask = 0,
                .conf0 = 4243463,
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
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref + 4 * 2304,
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref + 4 * 24 - *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 8, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 18, .d1 = 144, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(5, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 8),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(3, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref)}},
                // H padding 1/1 keeps Hout = Hin; legal because this is 3x3 mode, and harmless
                // because the kernel's top and bottom rows are zero.
                .padding = (1 << 28) + (0 << 24) + (1 << 20) + (0 << 16),
                .weight_offset_factor = -128,
                .filter_mask = 0,
                .conf0 = 4243463,
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
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref + 5 * 2304,
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref + 5 * 24 - *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 8, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 18, .d1 = 144, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(5, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 8),
                                                          .HoWo = nnx_concat_half(2, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(3, *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref)}},
                // H padding 1/1 keeps Hout = Hin; legal because this is 3x3 mode, and harmless
                // because the kernel's top and bottom rows are zero.
                .padding = (1 << 28) + (0 << 24) + (1 << 20) + (0 << 16),
                .weight_offset_factor = -128,
                .filter_mask = 0,
                .conf0 = 4243463,
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
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride + 0);
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride + 0);
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 81024);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 67200);
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
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_size_1d[TILING_I],
                                      .ext_stride_1d = 840};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_0_ref) + 600);

    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1455616,
                                      .size = 13824,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_1_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref = &DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref = &DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref = &DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref = &DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride[TILING_I];

    _b1_conv1xk_tiling_closure_args_t DeeployNetwork__b1_conv1xk_tiling_closure_args = (_b1_conv1xk_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_in_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_dim_im_out_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_input_addr_offset_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_nWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_bWi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref};

    // _b1_conv1xk_tiling_closure CLOSURE CALL
    _b1_conv1xk_tiling_closure(&DeeployNetwork__b1_conv1xk_tiling_closure_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_data_out_size_1d[TILING_I],
                                      .ext_stride_1d = 5632};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b1_conv1xk_conv_out_tensor_ref) + 4800);

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
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref;
} _b1_rqs_tiling_closure_args_t;

static void _b1_rqs_tiling_closure(void *_b1_rqs_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_rqs_tiling_closure_args_t *args = (_b1_rqs_tiling_closure_args_t *)_b1_rqs_tiling_closure_args;

  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL

  // RequantShift (Name: b1_rqs, Op: RequantShift)
  RequantShift_s32_s8_NHWC(DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref, *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref,
                           DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref, DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref,
                           DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref, 16, *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref, 0, 0, -128, 127, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref;
} _b1_rqs_cluster_fork_args_t;

static void _b1_rqs_cluster_fork(void *_b1_rqs_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b1_rqs_cluster_fork_args_t *args = (_b1_rqs_cluster_fork_args_t *)_b1_rqs_cluster_fork_args;

  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b1_rqs_tiling_closure_args_t DeeployNetwork__b1_rqs_tiling_closure_args =
      (_b1_rqs_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref,
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
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 17248);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 86240);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 86296);
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
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_conv_out_tensor_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_size_1d[TILING_I],
                                      .ext_stride_1d = 64};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_conv_out_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_conv_out_tensor_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_conv_out_tensor_ref) + 56);

    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_add_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_add_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_add_tensor_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_add_tensor_ref) + 56);

    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_mul_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_mul_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_mul_tensor_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_rqs_mul_tensor_ref) + 56);

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref = &DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref = &DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size[TILING_I];

    _b1_rqs_cluster_fork_args_t DeeployNetwork__b1_rqs_cluster_fork_args =
        (_b1_rqs_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_size_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_channels_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_in_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_mul_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_add_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b1_rqs_cluster_fork, &DeeployNetwork__b1_rqs_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_output_0_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_data_out_size_1d[TILING_I],
                                      .ext_stride_1d = 16};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_output_0_ref
    DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_output_0_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b1_rqs_output_0_ref) + 14);

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

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 91328);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 104432);

  DeeployNetwork_input_0 = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 92672);
  DeeployNetwork_input_1 = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 78848);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 78848);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
