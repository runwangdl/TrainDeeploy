#include "DeeployGAP9Math.h"
#include "DeeployMchan.h"
#include "ne16.h"
#include "ne16_pulp_bsp.h"
#include "ne16_task.h"
#include "perf_utils.h"
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
int8_t *DeeployNetwork_input_0;
int8_t *DeeployNetwork_input_1;
int32_t *DeeployNetwork_input_2;
int8_t *DeeployNetwork_input_3;
int8_t *DeeployNetwork_output_0;

static PI_L2 int32_t DeeployNetwork_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_b1_dual_wenc[16] = {
    234330, 242824, 241010, 274843, 272868, 260272, 252869, 248192, 219675, 282055, 236004, 249942, 272232, 237316, 271684, 279170};

static PI_L2 int32_t DeeployNetwork_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_b1_dual_bias[16] = {
    234330, 242824, 241010, 274843, 272868, 260272, 252869, 248192, 219675, 282055, 236004, 249942, 272232, 237316, 271684, 279170};

static PI_L2 int32_t DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor[16] = {1973910, 2045467, 2030184, 2315178, 2298543, 2192441, 2130083, 2090680,
                                                                              1850469, 2375935, 1988014, 2105422, 2293184, 1999064, 2288570, 2351634};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b1_dual_bias[16] = {
    65, 63, 64, 56, 56, 59, 61, 62, 70, 54, 65, 61, 56, 65, 56, 55};

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b1_ne16_dualconv[16] = {
    65, 63, 64, 56, 56, 59, 61, 62, 70, 54, 65, 61, 56, 65, 56, 55};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_m_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[4];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref;
} _b1_xp_hwc_tiling_closure_args_t;

static void _b1_xp_hwc_tiling_closure(void *_b1_xp_hwc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_xp_hwc_tiling_closure_args_t *args = (_b1_xp_hwc_tiling_closure_args_t *)_b1_xp_hwc_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 8, 14, 87] -> [1, 14, 87, 8] (Name: b1_xp_hwc, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 87;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref;
} _b1_xp_hwc_cluster_fork_args_t;

static void _b1_xp_hwc_cluster_fork(void *_b1_xp_hwc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b1_xp_hwc_cluster_fork_args_t *args = (_b1_xp_hwc_cluster_fork_args_t *)_b1_xp_hwc_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b1_xp_hwc_tiling_closure_args_t DeeployNetwork__b1_xp_hwc_tiling_closure_args =
      (_b1_xp_hwc_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref,
                                         .DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref};

  // _b1_xp_hwc_tiling_closure CLOSURE CALL
  _b1_xp_hwc_tiling_closure(&DeeployNetwork__b1_xp_hwc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_xp_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr;
} _b1_xp_hwc_closure_args_t;

static void _b1_xp_hwc_closure(void *_b1_xp_hwc_closure_args) {
  // CLOSURE ARG CAST
  _b1_xp_hwc_closure_args_t *args = (_b1_xp_hwc_closure_args_t *)_b1_xp_hwc_closure_args;

  int8_t *DeeployNetwork_xp_hwc_tensor = args->DeeployNetwork_xp_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 9744);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_xp_hwc_tensor_ref =
      (void *)((char *)DeeployNetwork_xp_hwc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1451536,
                                      .size = 9744,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_input_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _b1_xp_hwc_cluster_fork_args_t DeeployNetwork__b1_xp_hwc_cluster_fork_args =
        (_b1_xp_hwc_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_in_ref,
                                         .DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b1_xp_hwc_cluster_fork, &DeeployNetwork__b1_xp_hwc_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1320464,
                                      .size = 9744,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_xp_hwc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_xp_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr;
} _b1_xp_hwc_closure_L3_args_t;

static void _b1_xp_hwc_closure_L3(void *_b1_xp_hwc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b1_xp_hwc_closure_L3_args_t *args = (_b1_xp_hwc_closure_L3_args_t *)_b1_xp_hwc_closure_L3_args;

  int8_t *DeeployNetwork_xp_hwc_tensor = args->DeeployNetwork_xp_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b1_xp_hwc_closure_args_t DeeployNetwork__b1_xp_hwc_closure_args =
      (_b1_xp_hwc_closure_args_t){.DeeployNetwork_xp_hwc_tensor = DeeployNetwork_xp_hwc_tensor,
                                  .DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr};

  // _b1_xp_hwc_closure CLOSURE CALL
  _b1_xp_hwc_closure(&DeeployNetwork__b1_xp_hwc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref;
} _b1_xm_hwc_tiling_closure_args_t;

static void _b1_xm_hwc_tiling_closure(void *_b1_xm_hwc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_xm_hwc_tiling_closure_args_t *args = (_b1_xm_hwc_tiling_closure_args_t *)_b1_xm_hwc_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 8, 14, 87] -> [1, 14, 87, 8] (Name: b1_xm_hwc, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 8;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 87;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref;
} _b1_xm_hwc_cluster_fork_args_t;

static void _b1_xm_hwc_cluster_fork(void *_b1_xm_hwc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b1_xm_hwc_cluster_fork_args_t *args = (_b1_xm_hwc_cluster_fork_args_t *)_b1_xm_hwc_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b1_xm_hwc_tiling_closure_args_t DeeployNetwork__b1_xm_hwc_tiling_closure_args =
      (_b1_xm_hwc_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref,
                                         .DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref};

  // _b1_xm_hwc_tiling_closure CLOSURE CALL
  _b1_xm_hwc_tiling_closure(&DeeployNetwork__b1_xm_hwc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_xm_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr;
} _b1_xm_hwc_closure_args_t;

static void _b1_xm_hwc_closure(void *_b1_xm_hwc_closure_args) {
  // CLOSURE ARG CAST
  _b1_xm_hwc_closure_args_t *args = (_b1_xm_hwc_closure_args_t *)_b1_xm_hwc_closure_args;

  int8_t *DeeployNetwork_xm_hwc_tensor = args->DeeployNetwork_xm_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 9744);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_input_3_ref =
      (void *)((char *)DeeployNetwork_input_3 +
               DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_xm_hwc_tensor_ref =
      (void *)((char *)DeeployNetwork_xm_hwc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1451536,
                                      .size = 9744,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_input_3_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _b1_xm_hwc_cluster_fork_args_t DeeployNetwork__b1_xm_hwc_cluster_fork_args =
        (_b1_xm_hwc_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_in_ref,
                                         .DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b1_xm_hwc_cluster_fork, &DeeployNetwork__b1_xm_hwc_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1320464,
                                      .size = 9744,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_xm_hwc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_xm_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr;
} _b1_xm_hwc_closure_L3_args_t;

static void _b1_xm_hwc_closure_L3(void *_b1_xm_hwc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b1_xm_hwc_closure_L3_args_t *args = (_b1_xm_hwc_closure_L3_args_t *)_b1_xm_hwc_closure_L3_args;

  int8_t *DeeployNetwork_xm_hwc_tensor = args->DeeployNetwork_xm_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b1_xm_hwc_closure_args_t DeeployNetwork__b1_xm_hwc_closure_args =
      (_b1_xm_hwc_closure_args_t){.DeeployNetwork_xm_hwc_tensor = DeeployNetwork_xm_hwc_tensor,
                                  .DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr};

  // _b1_xm_hwc_closure CLOSURE CALL
  _b1_xm_hwc_closure(&DeeployNetwork__b1_xm_hwc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b1_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref;
} _b1_dual_wenc_tiling_closure_args_t;

static void _b1_dual_wenc_tiling_closure(void *_b1_dual_wenc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_dual_wenc_tiling_closure_args_t *args = (_b1_dual_wenc_tiling_closure_args_t *)_b1_dual_wenc_tiling_closure_args;

  void *DeeployNetwork_b1_dual_wenc_buffer = args->DeeployNetwork_b1_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualPerturbEncode (Name: b1_dual_wenc, Op: NE16DualPerturbEncode) -- W+/W- from one draw, encoded as one 2*cout 1x1 weight -- QW exp6
  {
    uint32_t b1_dual_wenc_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 2);
    uint32_t b1_dual_wenc_tile_xor = (0 * 0x9E3779B1u);
    float32_t b1_dual_wenc_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualPerturbEncode((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref,
                               (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref,
                               (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref, (int8_t *)DeeployNetwork_b1_dual_wenc_buffer,
                               b1_dual_wenc_seed_w, b1_dual_wenc_tile_xor, 15, b1_dual_wenc_eps_scale, NUM_CORES, 16, 128, 0, 1, 8, 1, 16);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b1_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref;
} _b1_dual_wenc_cluster_fork_args_t;

static void _b1_dual_wenc_cluster_fork(void *_b1_dual_wenc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b1_dual_wenc_cluster_fork_args_t *args = (_b1_dual_wenc_cluster_fork_args_t *)_b1_dual_wenc_cluster_fork_args;

  void *DeeployNetwork_b1_dual_wenc_buffer = args->DeeployNetwork_b1_dual_wenc_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b1_dual_wenc_tiling_closure_args_t DeeployNetwork__b1_dual_wenc_tiling_closure_args = (_b1_dual_wenc_tiling_closure_args_t){
      .DeeployNetwork_b1_dual_wenc_buffer = DeeployNetwork_b1_dual_wenc_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref};

  // _b1_dual_wenc_tiling_closure CLOSURE CALL
  _b1_dual_wenc_tiling_closure(&DeeployNetwork__b1_dual_wenc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr;
} _b1_dual_wenc_closure_args_t;

static void _b1_dual_wenc_closure(void *_b1_dual_wenc_closure_args) {
  // CLOSURE ARG CAST
  _b1_dual_wenc_closure_args_t *args = (_b1_dual_wenc_closure_args_t *)_b1_dual_wenc_closure_args;

  uint8_t *DeeployNetwork_wenc2_tensor = args->DeeployNetwork_wenc2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 8192);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10240);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4096);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_b1_dual_wenc_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_b1_dual_wenc +
               DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wenc2_tensor_ref =
      (void *)((char *)DeeployNetwork_wenc2_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr]);
  void *DeeployNetwork_b1_dual_wenc_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_input_1_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_b1_dual_wenc_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b1_dual_wenc_cluster_fork_args_t DeeployNetwork__b1_dual_wenc_cluster_fork_args = (_b1_dual_wenc_cluster_fork_args_t){
        .DeeployNetwork_b1_dual_wenc_buffer = DeeployNetwork_b1_dual_wenc_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b1_dual_wenc_cluster_fork, &DeeployNetwork__b1_dual_wenc_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314816, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_wenc2_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_wenc2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr;
} _b1_dual_wenc_closure_L3_args_t;

static void _b1_dual_wenc_closure_L3(void *_b1_dual_wenc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b1_dual_wenc_closure_L3_args_t *args = (_b1_dual_wenc_closure_L3_args_t *)_b1_dual_wenc_closure_L3_args;

  uint8_t *DeeployNetwork_wenc2_tensor = args->DeeployNetwork_wenc2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b1_dual_wenc_closure_args_t DeeployNetwork__b1_dual_wenc_closure_args =
      (_b1_dual_wenc_closure_args_t){.DeeployNetwork_wenc2_tensor = DeeployNetwork_wenc2_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr};

  // _b1_dual_wenc_closure CLOSURE CALL
  _b1_dual_wenc_closure(&DeeployNetwork__b1_dual_wenc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b1_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref;
} _b1_dual_bias_tiling_closure_args_t;

static void _b1_dual_bias_tiling_closure(void *_b1_dual_bias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_dual_bias_tiling_closure_args_t *args = (_b1_dual_bias_tiling_closure_args_t *)_b1_dual_bias_tiling_closure_args;

  void *DeeployNetwork_b1_dual_bias_buffer = args->DeeployNetwork_b1_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualBias (Name: b1_dual_bias, Op: NE16DualBias) -- add+/add- with the signed-input correction, stacked -- QW exp6
  {
    uint32_t b1_dual_bias_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 2);
    uint32_t b1_dual_bias_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 3);
    uint32_t b1_dual_bias_tile_xor = (0 * 0x9E3779B1u);
    float32_t b1_dual_bias_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
    PULP_NE16DualBias(
        (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref,
        (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref, (int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref,
        (int8_t *)DeeployNetwork_b1_dual_bias_buffer, b1_dual_bias_seed_w, b1_dual_bias_seed_b, b1_dual_bias_tile_xor, 15, 16, b1_dual_bias_eps_scale,
        NUM_CORES, 16, 128, 128, 0);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b1_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref;
} _b1_dual_bias_cluster_fork_args_t;

static void _b1_dual_bias_cluster_fork(void *_b1_dual_bias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b1_dual_bias_cluster_fork_args_t *args = (_b1_dual_bias_cluster_fork_args_t *)_b1_dual_bias_cluster_fork_args;

  void *DeeployNetwork_b1_dual_bias_buffer = args->DeeployNetwork_b1_dual_bias_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b1_dual_bias_tiling_closure_args_t DeeployNetwork__b1_dual_bias_tiling_closure_args = (_b1_dual_bias_tiling_closure_args_t){
      .DeeployNetwork_b1_dual_bias_buffer = DeeployNetwork_b1_dual_bias_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref};

  // _b1_dual_bias_tiling_closure CLOSURE CALL
  _b1_dual_bias_tiling_closure(&DeeployNetwork__b1_dual_bias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr;
} _b1_dual_bias_closure_args_t;

static void _b1_dual_bias_closure(void *_b1_dual_bias_closure_args) {
  // CLOSURE ARG CAST
  _b1_dual_bias_closure_args_t *args = (_b1_dual_bias_closure_args_t *)_b1_dual_bias_closure_args;

  int32_t *DeeployNetwork_add2_tensor = args->DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4224);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6464);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6400);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6528);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6592);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6272);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_b1_dual_bias_ref =
      (void *)((char *)DeeployNetwork_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_b1_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 +
               DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_blocks1convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks1convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b1_dual_bias_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b1_dual_bias +
               DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add2_tensor_ref =
      (void *)((char *)DeeployNetwork_add2_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr]);
  void *DeeployNetwork_b1_dual_bias_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_input_1_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_blocks1convweight_int8_pmul_tensor_DUPLICATE_FOR_b1_dual_bias_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_input_2_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_blocks1convbias_rqsadd_pmul_tensor_ref);
    mchan_transfer_1d(
        1441856, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref,
        DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b1_dual_bias_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b1_dual_bias_cluster_fork_args_t DeeployNetwork__b1_dual_bias_cluster_fork_args = (_b1_dual_bias_cluster_fork_args_t){
        .DeeployNetwork_b1_dual_bias_buffer = DeeployNetwork_b1_dual_bias_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_bmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b1_dual_bias_cluster_fork, &DeeployNetwork__b1_dual_bias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310848, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_add2_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr;
} _b1_dual_bias_closure_L3_args_t;

static void _b1_dual_bias_closure_L3(void *_b1_dual_bias_closure_L3_args) {
  // CLOSURE ARG CAST
  _b1_dual_bias_closure_L3_args_t *args = (_b1_dual_bias_closure_L3_args_t *)_b1_dual_bias_closure_L3_args;

  int32_t *DeeployNetwork_add2_tensor = args->DeeployNetwork_add2_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b1_dual_bias_closure_args_t DeeployNetwork__b1_dual_bias_closure_args =
      (_b1_dual_bias_closure_args_t){.DeeployNetwork_add2_tensor = DeeployNetwork_add2_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr};

  // _b1_dual_bias_closure CLOSURE CALL
  _b1_dual_bias_closure(&DeeployNetwork__b1_dual_bias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_b1_ne16_dualconv_colbuf;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_m_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_out_ref;
} _b1_ne16_dualconv_tiling_closure_args_t;

static void _b1_ne16_dualconv_tiling_closure(void *_b1_ne16_dualconv_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_ne16_dualconv_tiling_closure_args_t *args = (_b1_ne16_dualconv_tiling_closure_args_t *)_b1_ne16_dualconv_tiling_closure_args;

  void *DeeployNetwork_b1_ne16_dualconv_colbuf = args->DeeployNetwork_b1_ne16_dualconv_colbuf;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_m_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_m_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16DualConv2in (Name: b1_ne16_dualconv, Op: NE16DualConv2in) -- both passes on NE16, word-wise column, pipeline=1 -- QW exp7
  PULP_NE16DualConv2in(
      (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_ref,
      (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_m_ref,
      (const uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_weight_ref, (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_mul_ref,
      (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add_ref, (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_out_ref,
      (uint8_t *)DeeployNetwork_b1_ne16_dualconv_colbuf, NUM_CORES, 43032663u, 14, 87, 8, 14, 88, 16, 1, 16, 0, 0, 8, 8, 1, 1, -128, 1, 29);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_xp_hwc_tensor;
  int8_t *DeeployNetwork_xm_hwc_tensor;
  uint8_t *DeeployNetwork_wenc2_tensor;
  int32_t *DeeployNetwork_add2_tensor;
  int8_t *DeeployNetwork_y_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr;
} _b1_ne16_dualconv_closure_args_t;

static void _b1_ne16_dualconv_closure(void *_b1_ne16_dualconv_closure_args) {
  // CLOSURE ARG CAST
  _b1_ne16_dualconv_closure_args_t *args = (_b1_ne16_dualconv_closure_args_t *)_b1_ne16_dualconv_closure_args;

  int8_t *DeeployNetwork_xp_hwc_tensor = args->DeeployNetwork_xp_hwc_tensor;
  int8_t *DeeployNetwork_xm_hwc_tensor = args->DeeployNetwork_xm_hwc_tensor;
  uint8_t *DeeployNetwork_wenc2_tensor = args->DeeployNetwork_wenc2_tensor;
  int32_t *DeeployNetwork_add2_tensor = args->DeeployNetwork_add2_tensor;
  int8_t *DeeployNetwork_y_hwc_tensor = args->DeeployNetwork_y_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 61696);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_m_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 71440);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 81184);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 85408);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 85280);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_xp_hwc_tensor_ref =
      (void *)((char *)DeeployNetwork_xp_hwc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_xm_hwc_tensor_ref =
      (void *)((char *)DeeployNetwork_xm_hwc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_m_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_wenc2_tensor_ref =
      (void *)((char *)DeeployNetwork_wenc2_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr]);
  void *
      DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b1_ne16_dualconv_ref =
          (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b1_ne16_dualconv +
                   DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add2_tensor_ref =
      (void *)((char *)DeeployNetwork_add2_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_y_hwc_tensor_ref =
      (void *)((char *)DeeployNetwork_y_hwc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr]);
  void *DeeployNetwork_b1_ne16_dualconv_colbuf = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39424);

  // Initialize DMA futures
  int data_in_m_ExternalToLocal = -1;
  int data_in_ExternalToLocal = -1;
  int mul_ExternalToLocal = -1;
  int add_ExternalToLocal = -1;
  int weight_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1451536,
                                      .size = 9744,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_xp_hwc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    data_in_m_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1451536,
                                      .size = 9744,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_m_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_xm_hwc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    weight_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1445888,
                                      .size = 4096,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_weight_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_wenc2_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    mul_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {
          .cmd = 1441856,
          .size = 64,
          .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_mul_ref,
          .ext =
              DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_QSTANDALONE_QCDQ_blocks_1_conv_output_quant_1Clip_Quant_mul_pc_tensor_DUPLICATE_FOR_b1_ne16_dualconv_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    add_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add2_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    if (mul_ExternalToLocal >= 0) {
      mchan_transfer_wait(mul_ExternalToLocal);
      mchan_transfer_free(mul_ExternalToLocal);
    }

    if (add_ExternalToLocal >= 0) {
      mchan_transfer_wait(add_ExternalToLocal);
      mchan_transfer_free(add_ExternalToLocal);
    }

    if (data_in_m_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_m_ExternalToLocal);
      mchan_transfer_free(data_in_m_ExternalToLocal);
    }

    if (weight_ExternalToLocal >= 0) {
      mchan_transfer_wait(weight_ExternalToLocal);
      mchan_transfer_free(weight_ExternalToLocal);
    }

    _b1_ne16_dualconv_tiling_closure_args_t DeeployNetwork__b1_ne16_dualconv_tiling_closure_args = (_b1_ne16_dualconv_tiling_closure_args_t){
        .DeeployNetwork_b1_ne16_dualconv_colbuf = DeeployNetwork_b1_ne16_dualconv_colbuf,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_m_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_in_m_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_out_ref};

    // _b1_ne16_dualconv_tiling_closure CLOSURE CALL
    _b1_ne16_dualconv_tiling_closure(&DeeployNetwork__b1_ne16_dualconv_tiling_closure_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1350144,
                                      .size = 39424,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_y_hwc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_xp_hwc_tensor;
  int8_t *DeeployNetwork_xm_hwc_tensor;
  uint8_t *DeeployNetwork_wenc2_tensor;
  int32_t *DeeployNetwork_add2_tensor;
  int8_t *DeeployNetwork_y_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr;
} _b1_ne16_dualconv_closure_L3_args_t;

static void _b1_ne16_dualconv_closure_L3(void *_b1_ne16_dualconv_closure_L3_args) {
  // CLOSURE ARG CAST
  _b1_ne16_dualconv_closure_L3_args_t *args = (_b1_ne16_dualconv_closure_L3_args_t *)_b1_ne16_dualconv_closure_L3_args;

  int8_t *DeeployNetwork_xp_hwc_tensor = args->DeeployNetwork_xp_hwc_tensor;
  int8_t *DeeployNetwork_xm_hwc_tensor = args->DeeployNetwork_xm_hwc_tensor;
  uint8_t *DeeployNetwork_wenc2_tensor = args->DeeployNetwork_wenc2_tensor;
  int32_t *DeeployNetwork_add2_tensor = args->DeeployNetwork_add2_tensor;
  int8_t *DeeployNetwork_y_hwc_tensor = args->DeeployNetwork_y_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b1_ne16_dualconv_closure_args_t DeeployNetwork__b1_ne16_dualconv_closure_args = (_b1_ne16_dualconv_closure_args_t){
      .DeeployNetwork_xp_hwc_tensor = DeeployNetwork_xp_hwc_tensor,
      .DeeployNetwork_xm_hwc_tensor = DeeployNetwork_xm_hwc_tensor,
      .DeeployNetwork_wenc2_tensor = DeeployNetwork_wenc2_tensor,
      .DeeployNetwork_add2_tensor = DeeployNetwork_add2_tensor,
      .DeeployNetwork_y_hwc_tensor = DeeployNetwork_y_hwc_tensor,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr};

  // _b1_ne16_dualconv_closure CLOSURE CALL
  _b1_ne16_dualconv_closure(&DeeployNetwork__b1_ne16_dualconv_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref;
} _b1_out_nchw_tiling_closure_args_t;

static void _b1_out_nchw_tiling_closure(void *_b1_out_nchw_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b1_out_nchw_tiling_closure_args_t *args = (_b1_out_nchw_tiling_closure_args_t *)_b1_out_nchw_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 14, 88, 32] -> [1, 32, 14, 88] (Name: b1_out_nchw, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 14;

  uint16_t dimLen_2 = 88;

  uint16_t dimLen_3 = 32;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_3][dimLen_1][dimLen_2])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref;
} _b1_out_nchw_cluster_fork_args_t;

static void _b1_out_nchw_cluster_fork(void *_b1_out_nchw_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b1_out_nchw_cluster_fork_args_t *args = (_b1_out_nchw_cluster_fork_args_t *)_b1_out_nchw_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b1_out_nchw_tiling_closure_args_t DeeployNetwork__b1_out_nchw_tiling_closure_args = (_b1_out_nchw_tiling_closure_args_t){
      .DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref};

  // _b1_out_nchw_tiling_closure CLOSURE CALL
  _b1_out_nchw_tiling_closure(&DeeployNetwork__b1_out_nchw_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr;
} _b1_out_nchw_closure_args_t;

static void _b1_out_nchw_closure(void *_b1_out_nchw_closure_args) {
  // CLOSURE ARG CAST
  _b1_out_nchw_closure_args_t *args = (_b1_out_nchw_closure_args_t *)_b1_out_nchw_closure_args;

  int8_t *DeeployNetwork_y_hwc_tensor = args->DeeployNetwork_y_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 39424);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_y_hwc_tensor_ref =
      (void *)((char *)DeeployNetwork_y_hwc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr]);

  // Initialize DMA futures
  int data_in_ExternalToLocal = -1;
  int data_out_LocalToExternal = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    data_in_ExternalToLocal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1481216,
                                      .size = 39424,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_y_hwc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (data_in_ExternalToLocal >= 0) {
      mchan_transfer_wait(data_in_ExternalToLocal);
      mchan_transfer_free(data_in_ExternalToLocal);
    }

    _b1_out_nchw_cluster_fork_args_t DeeployNetwork__b1_out_nchw_cluster_fork_args = (_b1_out_nchw_cluster_fork_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b1_out_nchw_cluster_fork, &DeeployNetwork__b1_out_nchw_cluster_fork_args);

    // Transfer output tiles
    data_out_LocalToExternal = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1350144,
                                      .size = 39424,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (data_out_LocalToExternal >= 0) {
      mchan_transfer_wait(data_out_LocalToExternal);
      mchan_transfer_free(data_out_LocalToExternal);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_y_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr;
} _b1_out_nchw_closure_L3_args_t;

static void _b1_out_nchw_closure_L3(void *_b1_out_nchw_closure_L3_args) {
  // CLOSURE ARG CAST
  _b1_out_nchw_closure_L3_args_t *args = (_b1_out_nchw_closure_L3_args_t *)_b1_out_nchw_closure_L3_args;

  int8_t *DeeployNetwork_y_hwc_tensor = args->DeeployNetwork_y_hwc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b1_out_nchw_closure_args_t DeeployNetwork__b1_out_nchw_closure_args =
      (_b1_out_nchw_closure_args_t){.DeeployNetwork_y_hwc_tensor = DeeployNetwork_y_hwc_tensor,
                                    .DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr};

  // _b1_out_nchw_closure CLOSURE CALL
  _b1_out_nchw_closure(&DeeployNetwork__b1_out_nchw_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  int8_t *DeeployNetwork_xp_hwc_tensor;
  int8_t *DeeployNetwork_xm_hwc_tensor;
  uint8_t *DeeployNetwork_wenc2_tensor;
  int32_t *DeeployNetwork_add2_tensor;
  int8_t *DeeployNetwork_y_hwc_tensor;
  void *DeeployNetwork_b1_dual_wenc_buffer;
  void *DeeployNetwork_b1_dual_bias_buffer;
  void *DeeployNetwork_b1_ne16_dualconv_colbuf;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr;
    DeeployNetwork_xp_hwc_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _b1_xp_hwc_closure_L3_args_t DeeployNetwork__b1_xp_hwc_closure_L3_args =
        (_b1_xp_hwc_closure_L3_args_t){.DeeployNetwork_xp_hwc_tensor = DeeployNetwork_xp_hwc_tensor,
                                       .DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_xp_hwc_tileIdxPtr};

    // _b1_xp_hwc_closure_L3 CLOSURE CALL
    _b1_xp_hwc_closure_L3(&DeeployNetwork__b1_xp_hwc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr;
    DeeployNetwork_xm_hwc_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9744);
    _b1_xm_hwc_closure_L3_args_t DeeployNetwork__b1_xm_hwc_closure_L3_args =
        (_b1_xm_hwc_closure_L3_args_t){.DeeployNetwork_xm_hwc_tensor = DeeployNetwork_xm_hwc_tensor,
                                       .DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_xm_hwc_tileIdxPtr};

    // _b1_xm_hwc_closure_L3 CLOSURE CALL
    _b1_xm_hwc_closure_L3(&DeeployNetwork__b1_xm_hwc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr;
    DeeployNetwork_wenc2_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 21600);
    _b1_dual_wenc_closure_L3_args_t DeeployNetwork__b1_dual_wenc_closure_L3_args =
        (_b1_dual_wenc_closure_L3_args_t){.DeeployNetwork_wenc2_tensor = DeeployNetwork_wenc2_tensor,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_wenc_tileIdxPtr};

    // _b1_dual_wenc_closure_L3 CLOSURE CALL
    _b1_dual_wenc_closure_L3(&DeeployNetwork__b1_dual_wenc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr;
    DeeployNetwork_add2_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 25696);
    _b1_dual_bias_closure_L3_args_t DeeployNetwork__b1_dual_bias_closure_L3_args =
        (_b1_dual_bias_closure_L3_args_t){.DeeployNetwork_add2_tensor = DeeployNetwork_add2_tensor,
                                          .DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_dual_bias_tileIdxPtr};

    // _b1_dual_bias_closure_L3 CLOSURE CALL
    _b1_dual_bias_closure_L3(&DeeployNetwork__b1_dual_bias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr;
    DeeployNetwork_y_hwc_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 39424);
    _b1_ne16_dualconv_closure_L3_args_t DeeployNetwork__b1_ne16_dualconv_closure_L3_args = (_b1_ne16_dualconv_closure_L3_args_t){
        .DeeployNetwork_xp_hwc_tensor = DeeployNetwork_xp_hwc_tensor,
        .DeeployNetwork_xm_hwc_tensor = DeeployNetwork_xm_hwc_tensor,
        .DeeployNetwork_wenc2_tensor = DeeployNetwork_wenc2_tensor,
        .DeeployNetwork_add2_tensor = DeeployNetwork_add2_tensor,
        .DeeployNetwork_y_hwc_tensor = DeeployNetwork_y_hwc_tensor,
        .DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_ne16_dualconv_tileIdxPtr};

    // _b1_ne16_dualconv_closure_L3 CLOSURE CALL
    _b1_ne16_dualconv_closure_L3(&DeeployNetwork__b1_ne16_dualconv_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr;
    _b1_out_nchw_closure_L3_args_t DeeployNetwork__b1_out_nchw_closure_L3_args =
        (_b1_out_nchw_closure_L3_args_t){.DeeployNetwork_y_hwc_tensor = DeeployNetwork_y_hwc_tensor,
                                         .DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b1_out_nchw_tileIdxPtr};

    // _b1_out_nchw_closure_L3 CLOSURE CALL
    _b1_out_nchw_closure_L3(&DeeployNetwork__b1_out_nchw_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 85472);
  if (DeeployNetwork_MEMORYARENA_L1 == NULL) {
    printf("FATAL: arena allocation failed for DeeployNetwork_MEMORYARENA_L1 (%u B) -- cluster L1 exhausted; lower --l1 or the cluster stacks\r\n",
           (unsigned)(sizeof(int8_t) * 85472));
    return;
  }

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 78848);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9744);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 19488);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 21536);
  DeeployNetwork_input_3 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 21600);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_inputs[3] = (void *)DeeployNetwork_input_3;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
