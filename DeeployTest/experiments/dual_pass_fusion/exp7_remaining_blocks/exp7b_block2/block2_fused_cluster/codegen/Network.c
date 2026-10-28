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

static PI_L2 int32_t DeeployNetwork_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor[16] = {99, 84,  76, 82, 90, 84, 90,  86,
                                                                                                                 75, 120, 83, 94, 74, 93, 122, 71};

static PI_L2 int32_t DeeployNetwork_blocks2convweight_int8_pmul_tensor[16] = {187678, 221184, 245803, 227860, 207530, 220705, 206139, 217670,
                                                                              249879, 155117, 223668, 197312, 253300, 200095, 153050, 262817};

static PI_L2 int32_t DeeployNetwork_blocks2convbias_rqsadd_pmul_tensor[16] = {2923038, 3444888, 3828319, 3548861, 3232229, 3437423, 3210565, 3390147,
                                                                              3891805, 2415913, 3483568, 3073083, 3945087, 3116419, 2383714, 4093310};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_numTiles[2] = {0, 1};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_cumByteOffset[1] = {0};

static PI_L2 uint8_t DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[4];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref;
} _dualconv_blocks2_input_0_transpose_tiling_closure_args_t;

static void _dualconv_blocks2_input_0_transpose_tiling_closure(void *_dualconv_blocks2_input_0_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_input_0_transpose_tiling_closure_args_t *args =
      (_dualconv_blocks2_input_0_transpose_tiling_closure_args_t *)_dualconv_blocks2_input_0_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 16, 14, 22] -> [1, 14, 22, 16] (Name: dualconv_blocks2_input_0_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 16;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 22;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref;
} _dualconv_blocks2_input_0_transpose_cluster_fork_args_t;

static void _dualconv_blocks2_input_0_transpose_cluster_fork(void *_dualconv_blocks2_input_0_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_input_0_transpose_cluster_fork_args_t *args =
      (_dualconv_blocks2_input_0_transpose_cluster_fork_args_t *)_dualconv_blocks2_input_0_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks2_input_0_transpose_tiling_closure_args_t DeeployNetwork__dualconv_blocks2_input_0_transpose_tiling_closure_args =
      (_dualconv_blocks2_input_0_transpose_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref};

  // _dualconv_blocks2_input_0_transpose_tiling_closure CLOSURE CALL
  _dualconv_blocks2_input_0_transpose_tiling_closure(&DeeployNetwork__dualconv_blocks2_input_0_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks2_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr;
} _dualconv_blocks2_input_0_transpose_closure_args_t;

static void _dualconv_blocks2_input_0_transpose_closure(void *_dualconv_blocks2_input_0_transpose_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_input_0_transpose_closure_args_t *args =
      (_dualconv_blocks2_input_0_transpose_closure_args_t *)_dualconv_blocks2_input_0_transpose_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks2_input_0_transposed = args->DeeployNetwork_dualconv_blocks2_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4928);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 + DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_dualconv_blocks2_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks2_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1446720, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_input_0_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _dualconv_blocks2_input_0_transpose_cluster_fork_args_t DeeployNetwork__dualconv_blocks2_input_0_transpose_cluster_fork_args =
        (_dualconv_blocks2_input_0_transpose_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks2_input_0_transpose_cluster_fork, &DeeployNetwork__dualconv_blocks2_input_0_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1315648, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_dualconv_blocks2_input_0_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks2_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr;
} _dualconv_blocks2_input_0_transpose_closure_L3_args_t;

static void _dualconv_blocks2_input_0_transpose_closure_L3(void *_dualconv_blocks2_input_0_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_input_0_transpose_closure_L3_args_t *args =
      (_dualconv_blocks2_input_0_transpose_closure_L3_args_t *)_dualconv_blocks2_input_0_transpose_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks2_input_0_transposed = args->DeeployNetwork_dualconv_blocks2_input_0_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks2_input_0_transpose_closure_args_t DeeployNetwork__dualconv_blocks2_input_0_transpose_closure_args =
      (_dualconv_blocks2_input_0_transpose_closure_args_t){.DeeployNetwork_dualconv_blocks2_input_0_transposed =
                                                               DeeployNetwork_dualconv_blocks2_input_0_transposed,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr};

  // _dualconv_blocks2_input_0_transpose_closure CLOSURE CALL
  _dualconv_blocks2_input_0_transpose_closure(&DeeployNetwork__dualconv_blocks2_input_0_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref;
} _dualconv_blocks2_input_3_transpose_tiling_closure_args_t;

static void _dualconv_blocks2_input_3_transpose_tiling_closure(void *_dualconv_blocks2_input_3_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_input_3_transpose_tiling_closure_args_t *args =
      (_dualconv_blocks2_input_3_transpose_tiling_closure_args_t *)_dualconv_blocks2_input_3_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 16, 14, 22] -> [1, 14, 22, 16] (Name: dualconv_blocks2_input_3_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 16;

  uint16_t dimLen_2 = 14;

  uint16_t dimLen_3 = 22;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_2][dimLen_3][dimLen_1])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref;
} _dualconv_blocks2_input_3_transpose_cluster_fork_args_t;

static void _dualconv_blocks2_input_3_transpose_cluster_fork(void *_dualconv_blocks2_input_3_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_input_3_transpose_cluster_fork_args_t *args =
      (_dualconv_blocks2_input_3_transpose_cluster_fork_args_t *)_dualconv_blocks2_input_3_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks2_input_3_transpose_tiling_closure_args_t DeeployNetwork__dualconv_blocks2_input_3_transpose_tiling_closure_args =
      (_dualconv_blocks2_input_3_transpose_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref};

  // _dualconv_blocks2_input_3_transpose_tiling_closure CLOSURE CALL
  _dualconv_blocks2_input_3_transpose_tiling_closure(&DeeployNetwork__dualconv_blocks2_input_3_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks2_input_3_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr;
} _dualconv_blocks2_input_3_transpose_closure_args_t;

static void _dualconv_blocks2_input_3_transpose_closure(void *_dualconv_blocks2_input_3_transpose_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_input_3_transpose_closure_args_t *args =
      (_dualconv_blocks2_input_3_transpose_closure_args_t *)_dualconv_blocks2_input_3_transpose_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks2_input_3_transposed = args->DeeployNetwork_dualconv_blocks2_input_3_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4928);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_input_3_ref =
      (void *)((char *)DeeployNetwork_input_3 + DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_cumByteOffset
                                                    [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_dualconv_blocks2_input_3_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks2_input_3_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1446720, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_input_3_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _dualconv_blocks2_input_3_transpose_cluster_fork_args_t DeeployNetwork__dualconv_blocks2_input_3_transpose_cluster_fork_args =
        (_dualconv_blocks2_input_3_transpose_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_in_ref,
                                                                  .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref =
                                                                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks2_input_3_transpose_cluster_fork, &DeeployNetwork__dualconv_blocks2_input_3_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1315648, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_dualconv_blocks2_input_3_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks2_input_3_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr;
} _dualconv_blocks2_input_3_transpose_closure_L3_args_t;

static void _dualconv_blocks2_input_3_transpose_closure_L3(void *_dualconv_blocks2_input_3_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_input_3_transpose_closure_L3_args_t *args =
      (_dualconv_blocks2_input_3_transpose_closure_L3_args_t *)_dualconv_blocks2_input_3_transpose_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks2_input_3_transposed = args->DeeployNetwork_dualconv_blocks2_input_3_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks2_input_3_transpose_closure_args_t DeeployNetwork__dualconv_blocks2_input_3_transpose_closure_args =
      (_dualconv_blocks2_input_3_transpose_closure_args_t){.DeeployNetwork_dualconv_blocks2_input_3_transposed =
                                                               DeeployNetwork_dualconv_blocks2_input_3_transposed,
                                                           .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr =
                                                               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr};

  // _dualconv_blocks2_input_3_transpose_closure CLOSURE CALL
  _dualconv_blocks2_input_3_transpose_closure(&DeeployNetwork__dualconv_blocks2_input_3_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_dualconv_blocks2_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref;
} _dualconv_blocks2_tiling_closure_args_t;

static void _dualconv_blocks2_tiling_closure(void *_dualconv_blocks2_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_tiling_closure_args_t *args = (_dualconv_blocks2_tiling_closure_args_t *)_dualconv_blocks2_tiling_closure_args;

  void *DeeployNetwork_dualconv_blocks2_buffer = args->DeeployNetwork_dualconv_blocks2_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PULP DUAL-PASS QZO CONV, TWO INPUTS (Name: dualconv_blocks2, Op: RequantizedDualConv)  -- QW exp7
  // Seeds reproduce the two RQSPerturbRademacher nodes this kernel replaces (core_id is added inside).
  uint32_t dualconv_blocks2_seed_w = ((42 + perturb_seed_base) + NUM_CORES * 4);
  uint32_t dualconv_blocks2_seed_b = ((42 + perturb_seed_base) + NUM_CORES * 5);
  uint32_t dualconv_blocks2_tile_xor = (0 * 0x9E3779B1u);
  float32_t dualconv_blocks2_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;
  PULP_DualConv2d_2in_i8_i8_i8_HWC(
      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref,
      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref,
      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref,
      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref,
      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref + ((32) >> 1), (32), (int8_t *)DeeployNetwork_dualconv_blocks2_buffer,
      dualconv_blocks2_seed_w, dualconv_blocks2_seed_b, dualconv_blocks2_tile_xor, 15, 16, dualconv_blocks2_eps_scale, NUM_CORES, 16, 22, 14, 16, 23, 14,
      ((32) >> 1), 8, 1, 0, 0, 4, 4, 1, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  void *DeeployNetwork_dualconv_blocks2_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref;
} _dualconv_blocks2_cluster_fork_args_t;

static void _dualconv_blocks2_cluster_fork(void *_dualconv_blocks2_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_cluster_fork_args_t *args = (_dualconv_blocks2_cluster_fork_args_t *)_dualconv_blocks2_cluster_fork_args;

  void *DeeployNetwork_dualconv_blocks2_buffer = args->DeeployNetwork_dualconv_blocks2_buffer;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks2_tiling_closure_args_t DeeployNetwork__dualconv_blocks2_tiling_closure_args = (_dualconv_blocks2_tiling_closure_args_t){
      .DeeployNetwork_dualconv_blocks2_buffer = DeeployNetwork_dualconv_blocks2_buffer,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref};

  // _dualconv_blocks2_tiling_closure CLOSURE CALL
  _dualconv_blocks2_tiling_closure(&DeeployNetwork__dualconv_blocks2_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks2_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks2_input_3_transposed;
  int8_t *DeeployNetwork_dualconv_blocks2_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr;
} _dualconv_blocks2_closure_args_t;

static void _dualconv_blocks2_closure(void *_dualconv_blocks2_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_closure_args_t *args = (_dualconv_blocks2_closure_args_t *)_dualconv_blocks2_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks2_input_0_transposed = args->DeeployNetwork_dualconv_blocks2_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks2_input_3_transposed = args->DeeployNetwork_dualconv_blocks2_input_3_transposed;
  int8_t *DeeployNetwork_dualconv_blocks2_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks2_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 18624);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 28480);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 30592);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 30528);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 30656);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 30720);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 23552);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_dualconv_blocks2_input_0_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks2_input_0_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_dualconv_blocks2_input_3_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks2_input_3_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref =
      (void *)((char *)DeeployNetwork_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_2_ref =
      (void *)((char *)DeeployNetwork_input_2 +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_blocks2convweight_int8_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convweight_int8_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_blocks2convbias_rqsadd_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_blocks2convbias_rqsadd_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_dualconv_blocks2_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks2_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr]);
  void *DeeployNetwork_dualconv_blocks2_buffer = (void *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10304);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1446720, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_dualconv_blocks2_input_0_transposed_ref);
    mchan_transfer_1d(1446720, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_dualconv_blocks2_input_3_transposed_ref);
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_1_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_QSTANDALONE_QCDQ_blocks_2_conv_output_quant_1Clip_Quant_mul_pc_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_2_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_blocks2convweight_int8_pmul_tensor_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_blocks2convbias_rqsadd_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _dualconv_blocks2_cluster_fork_args_t DeeployNetwork__dualconv_blocks2_cluster_fork_args = (_dualconv_blocks2_cluster_fork_args_t){
        .DeeployNetwork_dualconv_blocks2_buffer = DeeployNetwork_dualconv_blocks2_buffer,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_mul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_add_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_wmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_bmul_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_in_m_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks2_cluster_fork, &DeeployNetwork__dualconv_blocks2_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1321024, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_dualconv_blocks2_output_0_pre_transposed_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks2_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks2_input_3_transposed;
  int8_t *DeeployNetwork_dualconv_blocks2_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr;
} _dualconv_blocks2_closure_L3_args_t;

static void _dualconv_blocks2_closure_L3(void *_dualconv_blocks2_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_closure_L3_args_t *args = (_dualconv_blocks2_closure_L3_args_t *)_dualconv_blocks2_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks2_input_0_transposed = args->DeeployNetwork_dualconv_blocks2_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks2_input_3_transposed = args->DeeployNetwork_dualconv_blocks2_input_3_transposed;
  int8_t *DeeployNetwork_dualconv_blocks2_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks2_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks2_closure_args_t DeeployNetwork__dualconv_blocks2_closure_args = (_dualconv_blocks2_closure_args_t){
      .DeeployNetwork_dualconv_blocks2_input_0_transposed = DeeployNetwork_dualconv_blocks2_input_0_transposed,
      .DeeployNetwork_dualconv_blocks2_input_3_transposed = DeeployNetwork_dualconv_blocks2_input_3_transposed,
      .DeeployNetwork_dualconv_blocks2_output_0_pre_transposed = DeeployNetwork_dualconv_blocks2_output_0_pre_transposed,
      .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr};

  // _dualconv_blocks2_closure CLOSURE CALL
  _dualconv_blocks2_closure(&DeeployNetwork__dualconv_blocks2_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref;
} _dualconv_blocks2_output_0_pre_transpose_tiling_closure_args_t;

static void _dualconv_blocks2_output_0_pre_transpose_tiling_closure(void *_dualconv_blocks2_output_0_pre_transpose_tiling_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_output_0_pre_transpose_tiling_closure_args_t *args =
      (_dualconv_blocks2_output_0_pre_transpose_tiling_closure_args_t *)_dualconv_blocks2_output_0_pre_transpose_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL

  // Transpose [1, 14, 23, 32] -> [1, 32, 14, 23] (Name: dualconv_blocks2_output_0_pre_transpose, Op: Transpose)

  const uint32_t coreId = pi_core_id();

  uint16_t dimLen_0 = 1;

  uint16_t dimLen_1 = 14;

  uint16_t dimLen_2 = 23;

  uint16_t dimLen_3 = 32;

  // RW: GCC Segmentation fault
  // RW 2026-10-13: flat pointers + strides (was: VLA casts src[dimLen_1][dimLen_2][dimLen_3] / dst[dimLen_3][dimLen_1][dimLen_2])
  const int8_t *src = (const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref;
  int8_t *dst = (int8_t *)DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref;
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
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref;
} _dualconv_blocks2_output_0_pre_transpose_cluster_fork_args_t;

static void _dualconv_blocks2_output_0_pre_transpose_cluster_fork(void *_dualconv_blocks2_output_0_pre_transpose_cluster_fork_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_output_0_pre_transpose_cluster_fork_args_t *args =
      (_dualconv_blocks2_output_0_pre_transpose_cluster_fork_args_t *)_dualconv_blocks2_output_0_pre_transpose_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks2_output_0_pre_transpose_tiling_closure_args_t DeeployNetwork__dualconv_blocks2_output_0_pre_transpose_tiling_closure_args =
      (_dualconv_blocks2_output_0_pre_transpose_tiling_closure_args_t){
          .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref =
              DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref,
          .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref =
              DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref};

  // _dualconv_blocks2_output_0_pre_transpose_tiling_closure CLOSURE CALL
  _dualconv_blocks2_output_0_pre_transpose_tiling_closure(&DeeployNetwork__dualconv_blocks2_output_0_pre_transpose_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks2_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr;
} _dualconv_blocks2_output_0_pre_transpose_closure_args_t;

static void _dualconv_blocks2_output_0_pre_transpose_closure(void *_dualconv_blocks2_output_0_pre_transpose_closure_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_output_0_pre_transpose_closure_args_t *args =
      (_dualconv_blocks2_output_0_pre_transpose_closure_args_t *)_dualconv_blocks2_output_0_pre_transpose_closure_args;

  int8_t *DeeployNetwork_dualconv_blocks2_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks2_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 10304);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_dualconv_blocks2_output_0_pre_transposed_ref =
      (void *)((char *)DeeployNetwork_dualconv_blocks2_output_0_pre_transposed +
               DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_cumByteOffset
                   [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 + DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_cumByteOffset
                                                     [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_numTiles
           [*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_numTiles
                      [(*DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr) + 1];
       TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1452096, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_dualconv_blocks2_output_0_pre_transposed_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _dualconv_blocks2_output_0_pre_transpose_cluster_fork_args_t DeeployNetwork__dualconv_blocks2_output_0_pre_transpose_cluster_fork_args =
        (_dualconv_blocks2_output_0_pre_transpose_cluster_fork_args_t){
            .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref =
                DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_in_ref,
            .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref =
                DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_dualconv_blocks2_output_0_pre_transpose_cluster_fork,
                    &DeeployNetwork__dualconv_blocks2_output_0_pre_transpose_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1321024, DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_data_out_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_output_0_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_dualconv_blocks2_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr;
} _dualconv_blocks2_output_0_pre_transpose_closure_L3_args_t;

static void _dualconv_blocks2_output_0_pre_transpose_closure_L3(void *_dualconv_blocks2_output_0_pre_transpose_closure_L3_args) {
  // CLOSURE ARG CAST
  _dualconv_blocks2_output_0_pre_transpose_closure_L3_args_t *args =
      (_dualconv_blocks2_output_0_pre_transpose_closure_L3_args_t *)_dualconv_blocks2_output_0_pre_transpose_closure_L3_args;

  int8_t *DeeployNetwork_dualconv_blocks2_output_0_pre_transposed = args->DeeployNetwork_dualconv_blocks2_output_0_pre_transposed;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr =
      args->DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _dualconv_blocks2_output_0_pre_transpose_closure_args_t DeeployNetwork__dualconv_blocks2_output_0_pre_transpose_closure_args =
      (_dualconv_blocks2_output_0_pre_transpose_closure_args_t){.DeeployNetwork_dualconv_blocks2_output_0_pre_transposed =
                                                                    DeeployNetwork_dualconv_blocks2_output_0_pre_transposed,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr};

  // _dualconv_blocks2_output_0_pre_transpose_closure CLOSURE CALL
  _dualconv_blocks2_output_0_pre_transpose_closure(&DeeployNetwork__dualconv_blocks2_output_0_pre_transpose_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork() {
  int8_t *DeeployNetwork_dualconv_blocks2_input_0_transposed;
  int8_t *DeeployNetwork_dualconv_blocks2_input_3_transposed;
  int8_t *DeeployNetwork_dualconv_blocks2_output_0_pre_transposed;
  void *DeeployNetwork_dualconv_blocks2_buffer;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr;
    DeeployNetwork_dualconv_blocks2_input_0_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
    _dualconv_blocks2_input_0_transpose_closure_L3_args_t DeeployNetwork__dualconv_blocks2_input_0_transpose_closure_L3_args =
        (_dualconv_blocks2_input_0_transpose_closure_L3_args_t){.DeeployNetwork_dualconv_blocks2_input_0_transposed =
                                                                    DeeployNetwork_dualconv_blocks2_input_0_transposed,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_0_transpose_tileIdxPtr};

    // _dualconv_blocks2_input_0_transpose_closure_L3 CLOSURE CALL
    _dualconv_blocks2_input_0_transpose_closure_L3(&DeeployNetwork__dualconv_blocks2_input_0_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr;
    DeeployNetwork_dualconv_blocks2_input_3_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 7040);
    _dualconv_blocks2_input_3_transpose_closure_L3_args_t DeeployNetwork__dualconv_blocks2_input_3_transpose_closure_L3_args =
        (_dualconv_blocks2_input_3_transpose_closure_L3_args_t){.DeeployNetwork_dualconv_blocks2_input_3_transposed =
                                                                    DeeployNetwork_dualconv_blocks2_input_3_transposed,
                                                                .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr =
                                                                    DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_input_3_transpose_tileIdxPtr};

    // _dualconv_blocks2_input_3_transpose_closure_L3 CLOSURE CALL
    _dualconv_blocks2_input_3_transpose_closure_L3(&DeeployNetwork__dualconv_blocks2_input_3_transpose_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr;
    DeeployNetwork_dualconv_blocks2_output_0_pre_transposed = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 11968);
    _dualconv_blocks2_closure_L3_args_t DeeployNetwork__dualconv_blocks2_closure_L3_args = (_dualconv_blocks2_closure_L3_args_t){
        .DeeployNetwork_dualconv_blocks2_input_0_transposed = DeeployNetwork_dualconv_blocks2_input_0_transposed,
        .DeeployNetwork_dualconv_blocks2_input_3_transposed = DeeployNetwork_dualconv_blocks2_input_3_transposed,
        .DeeployNetwork_dualconv_blocks2_output_0_pre_transposed = DeeployNetwork_dualconv_blocks2_output_0_pre_transposed,
        .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_tileIdxPtr};

    // _dualconv_blocks2_closure_L3 CLOSURE CALL
    _dualconv_blocks2_closure_L3(&DeeployNetwork__dualconv_blocks2_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr =
        &bu_DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr;
    _dualconv_blocks2_output_0_pre_transpose_closure_L3_args_t DeeployNetwork__dualconv_blocks2_output_0_pre_transpose_closure_L3_args =
        (_dualconv_blocks2_output_0_pre_transpose_closure_L3_args_t){.DeeployNetwork_dualconv_blocks2_output_0_pre_transposed =
                                                                         DeeployNetwork_dualconv_blocks2_output_0_pre_transposed,
                                                                     .DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr =
                                                                         DeeployNetwork_TILING_CODEGEN_L1_dualconv_blocks2_output_0_pre_transpose_tileIdxPtr};

    // _dualconv_blocks2_output_0_pre_transpose_closure_L3 CLOSURE CALL
    _dualconv_blocks2_output_0_pre_transpose_closure_L3(&DeeployNetwork__dualconv_blocks2_output_0_pre_transpose_closure_L3_args);
  }
}

void InitNetwork() {

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pmsis_l1_malloc(sizeof(int8_t) * 30784);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 22272);

  DeeployNetwork_input_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 7040);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 4928);
  DeeployNetwork_input_2 = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 6976);
  DeeployNetwork_input_3 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 11968);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_inputs[3] = (void *)DeeployNetwork_input_3;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
