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

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor[32] = {
    178, 146, 165, 164, 163, 155, 146, 137, 119, 174, 135, 139, 150, 146, 243, 126,
    147, 139, 155, 130, 153, 157, 135, 160, 116, 142, 140, 160, 119, 174, 169, 142};

static PI_L2 int32_t DeeployNetwork_blocks4convweight_int8_pmul_tensor[32] = {
    185997, 226785, 200920, 202444, 202823, 213378, 226168, 241806, 277784, 190200, 245924, 237923, 220822, 226733, 136034, 261893,
    225823, 238894, 213576, 255468, 216926, 210590, 245405, 206957, 286127, 233156, 236698, 206920, 278716, 190344, 195460, 232972};

static PI_L2 int32_t DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor[32] = {
    4465028, 5444165, 4823268, 4859837, 4868948, 5122322, 5429352, 5804773, 6668456, 4565921, 5903610, 5711538, 5301021, 5442922, 3265610, 6286967,
    5421086, 5734863, 5127071, 6132738, 5207501, 5055397, 5891161, 4968182, 6868725, 5597122, 5682131, 4967301, 6690810, 4569368, 4692189, 5592699};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[3];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref;
} _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure(
    void *_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure_args_t *)
          _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 32, 8, 5] -> [1, 8, 5, 32] (Name: wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 32;

  uint16_t dimLen_2 = 8;

  uint16_t dimLen_3 = 5;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref;
} _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_cluster_fork(
    void *_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_cluster_fork_args_t *)
          _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure_args_t
      DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure_args =
          (_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref};

  // _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure(
      &DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr;
} _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_args_t;

static void
_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure(void *_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_args_t *)_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1443072,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_input_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_cluster_fork_args_t
        DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_cluster_fork_args =
            (_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1312000,
          .size = 1280,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr;
} _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3_args_t;

static void
_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3(void *_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3_args_t *)
          _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_args_t
      DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_args =
          (_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_args_t){
              .DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed =
                  DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr};

  // _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure CLOSURE CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure(
      &DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref;
} _wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure(void *_wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure_args_t *)_wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP SINGLE-PASS QZO CONV, PERTURBATION IN THE KERNEL (Name: wrappedInnerForwardImpl_4Conv_perturbconv, Op: RequantizedDualConv)  -- QW exp11
  uint32_t wrappedInnerForwardImpl_4Conv_perturbconv_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 8);
  uint32_t wrappedInnerForwardImpl_4Conv_perturbconv_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 9);
  uint32_t wrappedInnerForwardImpl_4Conv_perturbconv_tile_xor = (0 * 0x9E3779B1u);
  float32_t wrappedInnerForwardImpl_4Conv_perturbconv_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_PerturbConv4x2_i8_i8_i8_HWC(DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref,
                                   (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer, perturbation_sign,
                                   wrappedInnerForwardImpl_4Conv_perturbconv_seed_w, wrappedInnerForwardImpl_4Conv_perturbconv_seed_b,
                                   wrappedInnerForwardImpl_4Conv_perturbconv_tile_xor, 15, 16, wrappedInnerForwardImpl_4Conv_perturbconv_eps_scale, NUM_CORES,
                                   16, 5, 8, 32, 5, 2, 32, 1, 7, 0, 0, 0, 0, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref;
} _wrappedInnerForwardImpl_4Conv_perturbconv_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_4Conv_perturbconv_cluster_fork(void *_wrappedInnerForwardImpl_4Conv_perturbconv_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_cluster_fork_args_t *)_wrappedInnerForwardImpl_4Conv_perturbconv_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure_args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer = DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref};

  // _wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr;
} _wrappedInnerForwardImpl_4Conv_perturbconv_closure_args_t;

static void _wrappedInnerForwardImpl_4Conv_perturbconv_closure(void *_wrappedInnerForwardImpl_4Conv_perturbconv_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_closure_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_closure_args_t *)_wrappedInnerForwardImpl_4Conv_perturbconv_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 18048);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10880);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19776);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19648);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19904);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 20032);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19328);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_blocks4convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_blocks4convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks4convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1443072, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed_ref);
    mchan_transfer_1d(1448960, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_1_ref);
    mchan_transfer_1d(
        1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_QSTANDALONE_QCDQ_blocks_4_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_2_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_blocks4convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_blocks4convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_4Conv_perturbconv_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_cluster_fork_args =
        (_wrappedInnerForwardImpl_4Conv_perturbconv_cluster_fork_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer = DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_weight_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_mul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_add_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_bmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_4Conv_perturbconv_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1311040, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr;
} _wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3_args_t;

static void _wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3(void *_wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3_args_t *)_wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_closure_args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_closure_args_t){.DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed =
                                                                      DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed,
                                                                  .DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed =
                                                                      DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr};

  // _wrappedInnerForwardImpl_4Conv_perturbconv_closure CLOSURE CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_closure(&DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref;
} _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure(
    void *_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure_args_t *)
          _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 2, 5, 32] -> [1, 32, 2, 5] (Name: wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 2;

  uint16_t dimLen_2 = 5;

  uint16_t dimLen_3 = 32;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_3][dimLen_1][dimLen_2])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref;
} _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_cluster_fork(
    void *_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_cluster_fork_args_t *)
          _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure_args_t
      DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure_args =
          (_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref};

  // _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure(
      &DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr;
} _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_args_t;

static void _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure(
    void *_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_args_t *)
          _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 320);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1442112,
          .size = 320,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_cluster_fork_args_t
        DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_cluster_fork_args =
            (_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1311040,
                                      .size = 320,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr;
} _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3_args_t;

static void _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3(
    void *_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3_args_t *)
          _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_args_t
      DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_args =
          (_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_args_t){
              .DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed =
                  DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr};

  // _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure CLOSURE CALL
  _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure(
      &DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed;
  void *DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 7168);
    _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3_args_t
        DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3_args =
            (_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3_args_t){
                .DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed =
                    DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_tileIdxPtr};

    // _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3(
        &DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8448);
    _wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3_args =
        (_wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_input_0_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_tileIdxPtr};

    // _wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr;
    _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3_args_t
        DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3_args =
            (_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3_args_t){
                .DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed =
                    DeeployNetwork_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_tileIdxPtr};

    // _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3(
        &DeeployNetwork__wrappedInnerForwardImpl_4Conv_perturbconv_output_0_pre_transpose_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 20160);
  if (DeeployNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 20160));
    return;
  }

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 9856);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8448);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9728);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
