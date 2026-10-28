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

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor[16] = {65, 63, 64, 56, 56, 59, 61, 62,
                                                                                                                 70, 54, 65, 61, 56, 65, 56, 55};

static PI_L2 int32_t DeeployNetwork_blocks1convweight_int8_pmul_tensor[16] = {234330, 242824, 241010, 274843, 272868, 260272, 252869, 248192,
                                                                              219675, 282055, 236004, 249942, 272232, 237316, 271684, 279170};

static PI_L2 int32_t DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor[16] = {1973910, 2045467, 2030184, 2315178, 2298543, 2192441, 2130083, 2090680,
                                                                              1850469, 2375935, 1988014, 2105422, 2293184, 1999064, 2288570, 2351634};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[3];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref;
} _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure(
    void *_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure_args_t *)
          _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 8, 14, 87] -> [1, 14, 87, 8] (Name: wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 87;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref;
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

      const uint32_t baseChunk_2 = dimLen_3 / NUM_CORES;
      const uint32_t leftover_2 = dimLen_3 - baseChunk_2 * NUM_CORES;
      const uint32_t offset_2 = baseChunk_2 * coreId + (coreId < leftover_2 ? coreId : leftover_2);
      const uint32_t chunk_2 = coreId < leftover_2 ? baseChunk_2 + 1 : baseChunk_2;
      for (uint32_t i_3 = offset_2; i_3 < offset_2 + chunk_2; i_3++) {
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref;
} _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_cluster_fork(
    void *_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_cluster_fork_args_t *)
          _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure_args_t
      DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure_args =
          (_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref};

  // _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure(
      &DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_args_t;

static void
_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure(void *_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_args_t *)_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 9744);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1451536,
                                      .size = 9744,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_input_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_cluster_fork_args_t
        DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_cluster_fork_args =
            (_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1320464,
          .size = 9744,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_data_out_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3_args_t;

static void
_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3(void *_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3_args_t *)
          _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_args_t
      DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_args =
          (_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_args_t){
              .DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed =
                  DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr};

  // _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure(
      &DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref;
} _wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure(void *_wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure_args_t *)_wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP SINGLE-PASS QZO CONV, PERTURBATION IN THE KERNEL (Name: wrappedInnerForwardImpl_1Conv_perturbconv, Op: RequantizedDualConv)  -- QW exp11
  uint32_t wrappedInnerForwardImpl_1Conv_perturbconv_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 2);
  uint32_t wrappedInnerForwardImpl_1Conv_perturbconv_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 3);
  uint32_t wrappedInnerForwardImpl_1Conv_perturbconv_tile_xor = (0 * 0x9E3779B1u);
  float32_t wrappedInnerForwardImpl_1Conv_perturbconv_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_PerturbConv4x2_i8_i8_i8_HWC(DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref,
                                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref,
                                   (int8_t *)DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer, perturbation_sign,
                                   wrappedInnerForwardImpl_1Conv_perturbconv_seed_w, wrappedInnerForwardImpl_1Conv_perturbconv_seed_b,
                                   wrappedInnerForwardImpl_1Conv_perturbconv_tile_xor, 15, 16, wrappedInnerForwardImpl_1Conv_perturbconv_eps_scale, NUM_CORES,
                                   16, 87, 14, 8, 88, 14, 16, 16, 1, 0, 0, 8, 8, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref;
} _wrappedInnerForwardImpl_1Conv_perturbconv_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_1Conv_perturbconv_cluster_fork(void *_wrappedInnerForwardImpl_1Conv_perturbconv_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_cluster_fork_args_t *)_wrappedInnerForwardImpl_1Conv_perturbconv_cluster_fork_args;

  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer = args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure_args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure_args_t){
          .DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer = DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref};

  // _wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_perturbconv_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv_perturbconv_closure(void *_wrappedInnerForwardImpl_1Conv_perturbconv_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_closure_args_t *)_wrappedInnerForwardImpl_1Conv_perturbconv_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19712);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 33616);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 35728);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 35664);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 35792);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 35856);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 + DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_blocks1convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_blocks1convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr]);
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 29456);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(
        1451536, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed_ref);
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_1_ref);
    mchan_transfer_1d(
        1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_2_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_blocks1convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_blocks1convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _wrappedInnerForwardImpl_1Conv_perturbconv_cluster_fork_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_cluster_fork_args =
        (_wrappedInnerForwardImpl_1Conv_perturbconv_cluster_fork_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer = DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_weight_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_mul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_add_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_bmul_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_1Conv_perturbconv_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(
        1330432, DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_data_out_ref,
        DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3_args_t;

static void _wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3(void *_wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3_args_t *)_wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_closure_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_closure_args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_closure_args_t){.DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed =
                                                                      DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed,
                                                                  .DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed =
                                                                      DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr};

  // _wrappedInnerForwardImpl_1Conv_perturbconv_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_closure(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref;
} _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure(
    void *_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure_args_t *)
          _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 14, 88, 16] -> [1, 16, 14, 88] (Name: wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 14;

  uint16_t dimLen_2 = 88;

  uint16_t dimLen_3 = 16;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_3][dimLen_1][dimLen_2])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref;
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

    for (uint32_t i_3 = 0; i_3 < dimLen_3; i_3++) {
      const uint32_t offS_1 = offS_0 + i_3 * sIn_3;
      const uint32_t offD_1 = offD_0 + i_3 * sOut_1;

      for (uint32_t i_1 = 0; i_1 < dimLen_1; i_1++) {
        const uint32_t offS_2 = offS_1 + i_1 * sIn_1;
        const uint32_t offD_2 = offD_1 + i_1 * sOut_2;

        const uint32_t baseChunk_3 = dimLen_2 / NUM_CORES;
        const uint32_t leftover_3 = dimLen_2 - baseChunk_3 * NUM_CORES;
        const uint32_t offset_3 = baseChunk_3 * coreId + (coreId < leftover_3 ? coreId : leftover_3);
        const uint32_t chunk_3 = coreId < leftover_3 ? baseChunk_3 + 1 : baseChunk_3;
        for (uint32_t i_2 = offset_3; i_2 < offset_3 + chunk_3; i_2++) {
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref;
} _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_cluster_fork_args_t;

static void _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_cluster_fork(
    void *_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_cluster_fork_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_cluster_fork_args_t *)
          _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure_args_t
      DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure_args =
          (_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure_args_t){
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref,
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref};

  // _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure(
      &DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_args_t;

static void _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure(
    void *_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_args_t *)
          _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 19712);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref =
      (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed_ref =
          (void *)((char *)DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed +
                   DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_cumByteOffset
                       [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  int data_out_LocalToExternal = -1;
  int data_in_ExternalToLocal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1461504,
          .size = 19712,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_cluster_fork_args_t
        DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_cluster_fork_args =
            (_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_cluster_fork_args_t){
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_in_ref,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_cluster_fork,
                    &DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1330432,
                                      .size = 19712,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr;
} _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3_args_t;

static void _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3(
    void *_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3_args_t *args =
      (_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3_args_t *)
          _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3_args;

  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed =
      args->DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_args_t
      DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_args =
          (_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_args_t){
              .DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed =
                  DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed,
              .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr =
                  DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr};

  // _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure CLOSURE CALL
  _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure(
      &DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed;
  int8_t *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed;
  void *DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 19712);
    _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3_args_t
        DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3_args =
            (_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3_args_t){
                .DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed =
                    DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_tileIdxPtr};

    // _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3(
        &DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr;
    DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3_args_t DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3_args =
        (_wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3_args_t){
            .DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_input_0_transposed,
            .DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed =
                DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed,
            .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr =
                DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_tileIdxPtr};

    // _wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3(&DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr;
    _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3_args_t
        DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3_args =
            (_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3_args_t){
                .DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed =
                    DeeployNetwork_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transposed,
                .DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr =
                    DeeployNetwork_TILING_CODEGEN_L1_wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_tileIdxPtr};

    // _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3 CLOSURE CALL
    _wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3(
        &DeeployNetwork__wrappedInnerForwardImpl_1Conv_perturbconv_output_0_pre_transpose_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 39424);
  if (DeeployNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 39424));
    return;
  }

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 39424);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 29456);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 31504);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 19712);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
