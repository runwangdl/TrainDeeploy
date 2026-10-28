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
int8_t *DeeployNetwork_input_1;
int8_t *DeeployNetwork_output_0;

static PI_L2 int32_t DeeployNetwork_w_pmul_tensor[16] = {191402, 166541, 187775, 163960, 142931, 257763, 260869, 182937,
                                                         220184, 169087, 205414, 158004, 145902, 204173, 143479, 279511};

static PI_L2 int32_t DeeployNetwork_rqs_mul_tensor[16] = {117, 134, 119, 136, 156, 87, 86, 122, 102, 132, 109, 142, 153, 110, 156, 80};

static PI_L2 int32_t DeeployNetwork_rqs_add_tensor[16] = {32831, 32713, 32706, 32822, 32815, 32853, 32854, 32708,
                                                          32841, 32712, 32700, 32820, 32816, 32835, 32815, 32860};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_cumByteOffset[1] = {0};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride[4] = {448, 448, 144, 144};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride[4] = {1344, 1344, 128, 128};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo[4] = {4, 1, 4, 1};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo[4] = {7, 7, 1, 1};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo[4] = {3, 2, 3, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo[4] = {3, 3, 2, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi[4] = {3, 2, 3, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi[4] = {3, 3, 2, 2};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_numTiles[2] = {0, 4};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_cumByteOffset[4] = {0, 5760, 336, 6096};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_cmd[4] = {1971456, 1966976, 1967808, 1966368};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_size[4] = {5376, 896, 1728, 288};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_size_1d[4] = {448, 448, 144, 144};

static PI_L1 int16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_relativeOffset[4] = {5760, -5424, 5760, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_cumByteOffset[4] = {0, 17664, 1344, 19008};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_cmd[4] = {1851136, 1837696, 1836544, 1835264};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_size[4] = {16128, 2688, 1536, 256};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_size_1d[4] = {1344, 1344, 128, 128};

static PI_L1 int16_t DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_relativeOffset[4] = {17664, -16320, 17664, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[2];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref;
} _b2_wpert_tiling_closure_args_t;

static void _b2_wpert_tiling_closure(void *_b2_wpert_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b2_wpert_tiling_closure_args_t *args = (_b2_wpert_tiling_closure_args_t *)_b2_wpert_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: b2_wpert, Op: RQSPerturbRademacher)
  uint8_t b2_wpert_core_id = (uint8_t)pi_core_id();
  uint8_t b2_wpert_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t b2_wpert_chunk = (2048 >> b2_wpert_log2Core) + ((2048 & (NUM_CORES - 1)) != 0);
  uint32_t b2_wpert_chunk_start = (uint32_t)MIN(b2_wpert_chunk * b2_wpert_core_id, (uint32_t)2048);
  uint32_t b2_wpert_chunk_stop = (uint32_t)MIN(b2_wpert_chunk_start + b2_wpert_chunk, (uint32_t)2048);
  uint32_t b2_wpert_local_size = b2_wpert_chunk_stop - b2_wpert_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t b2_wpert_channel_start_offset = b2_wpert_chunk_start % 128;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 4 + b2_wpert_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t b2_wpert_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  ApplyPerturbQuantRademacher_CHW((const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref[b2_wpert_chunk_start],
                                  (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref[b2_wpert_chunk_start],
                                  (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref, 15, 128, chunk_seed, b2_wpert_local_size,
                                  b2_wpert_chunk_start,
                                  perturbation_sign,   // -- QW: +eps (L+) / -eps (L-)
                                  b2_wpert_eps_scale); // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref;
} _b2_wpert_cluster_fork_args_t;

static void _b2_wpert_cluster_fork(void *_b2_wpert_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b2_wpert_cluster_fork_args_t *args = (_b2_wpert_cluster_fork_args_t *)_b2_wpert_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b2_wpert_tiling_closure_args_t DeeployNetwork__b2_wpert_tiling_closure_args =
      (_b2_wpert_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref};

  // _b2_wpert_tiling_closure CLOSURE CALL
  _b2_wpert_tiling_closure(&DeeployNetwork__b2_wpert_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr;
} _b2_wpert_closure_args_t;

static void _b2_wpert_closure(void *_b2_wpert_closure_args) {
  // CLOSURE ARG CAST
  _b2_wpert_closure_args_t *args = (_b2_wpert_closure_args_t *)_b2_wpert_closure_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 4096);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_w_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_w_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_weight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_input_1_ref);
    mchan_transfer_1d(1441856, DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref, DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_w_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b2_wpert_cluster_fork_args_t DeeployNetwork__b2_wpert_cluster_fork_args =
        (_b2_wpert_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_in_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_mul_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b2_wpert_cluster_fork, &DeeployNetwork__b2_wpert_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312768, DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_weight_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr;
} _b2_wpert_closure_L3_args_t;

static void _b2_wpert_closure_L3(void *_b2_wpert_closure_L3_args) {
  // CLOSURE ARG CAST
  _b2_wpert_closure_L3_args_t *args = (_b2_wpert_closure_L3_args_t *)_b2_wpert_closure_L3_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b2_wpert_closure_args_t DeeployNetwork__b2_wpert_closure_args =
      (_b2_wpert_closure_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                 .DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr};

  // _b2_wpert_closure CLOSURE CALL
  _b2_wpert_closure(&DeeployNetwork__b2_wpert_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref;
} _b2_wenc_tiling_closure_args_t;

static void _b2_wenc_tiling_closure(void *_b2_wenc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b2_wenc_tiling_closure_args_t *args = (_b2_wenc_tiling_closure_args_t *)_b2_wenc_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16WeightEncode (Name: b2_wenc, Op: NE16WeightEncode) -- bit-serial encode on the cluster -- QW
  {
    uint32_t b2_wenc_nrows = 128;
    uint32_t b2_wenc_chunk = (b2_wenc_nrows + NUM_CORES - 1) / NUM_CORES;
    uint32_t b2_wenc_start = MIN(b2_wenc_chunk * (uint32_t)pi_core_id(), b2_wenc_nrows);
    uint32_t b2_wenc_stop = MIN(b2_wenc_start + b2_wenc_chunk, b2_wenc_nrows);

    NE16WeightEncode_i8_u8((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref,
                           (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref, 16, 16, 8, b2_wenc_start, b2_wenc_stop - b2_wenc_start);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref;
} _b2_wenc_cluster_fork_args_t;

static void _b2_wenc_cluster_fork(void *_b2_wenc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b2_wenc_cluster_fork_args_t *args = (_b2_wenc_cluster_fork_args_t *)_b2_wenc_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b2_wenc_tiling_closure_args_t DeeployNetwork__b2_wenc_tiling_closure_args =
      (_b2_wenc_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref,
                                       .DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref};

  // _b2_wenc_tiling_closure CLOSURE CALL
  _b2_wenc_tiling_closure(&DeeployNetwork__b2_wenc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr;
} _b2_wenc_closure_args_t;

static void _b2_wenc_closure(void *_b2_wenc_closure_args) {
  // CLOSURE ARG CAST
  _b2_wenc_closure_args_t *args = (_b2_wenc_closure_args_t *)_b2_wenc_closure_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 2048);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_weight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_weight_enc_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_enc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1443840, DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_weight_pert_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b2_wenc_cluster_fork_args_t DeeployNetwork__b2_wenc_cluster_fork_args =
        (_b2_wenc_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_in_ref,
                                       .DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b2_wenc_cluster_fork, &DeeployNetwork__b2_wenc_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1312768, DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_weight_enc_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr;
} _b2_wenc_closure_L3_args_t;

static void _b2_wenc_closure_L3(void *_b2_wenc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b2_wenc_closure_L3_args_t *args = (_b2_wenc_closure_L3_args_t *)_b2_wenc_closure_L3_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b2_wenc_closure_args_t DeeployNetwork__b2_wenc_closure_args =
      (_b2_wenc_closure_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                .DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                .DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr};

  // _b2_wenc_closure CLOSURE CALL
  _b2_wenc_closure(&DeeployNetwork__b2_wenc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref;
} _b2_conv1xk_tiling_closure_args_t;

static void _b2_conv1xk_tiling_closure(void *_b2_conv1xk_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b2_conv1xk_tiling_closure_args_t *args = (_b2_conv1xk_tiling_closure_args_t *)_b2_conv1xk_tiling_closure_args;

  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16 1x8 dense conv as 8 pointwise dispatches (streamin accumulation)
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref + 0 * 256,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref + 0 * 16 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref + 1 * 256,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref + 1 * 16 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref)}},
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
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref + 2 * 256,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref + 2 * 16 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref)}},
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
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref + 3 * 256,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref + 3 * 16 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref)}},
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
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref + 4 * 256,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref + 4 * 16 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref)}},
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
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref + 5 * 256,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref + 5 * 16 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref)}},
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
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref + 6 * 256,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref + 6 * 16 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref)}},
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
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref + 7 * 256,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref + 7 * 16 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 64, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(16, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref)}},
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
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr;
} _b2_conv1xk_closure_args_t;

static void _b2_conv1xk_closure(void *_b2_conv1xk_closure_args) {
  // CLOSURE ARG CAST
  _b2_conv1xk_closure_args_t *args = (_b2_conv1xk_closure_args_t *)_b2_conv1xk_closure_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride + 0);
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 16128);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 21504);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_enc_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_enc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_conv_out_tensor_ref =
      (void *)((char *)DeeployNetwork_conv_out_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_input_0_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_size_1d[TILING_I],
                                      .ext_stride_1d = 480};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_input_0_ref
    DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_input_0_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_input_0_ref) +
                                                                       DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_relativeOffset[TILING_I]);

    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1443840,
                                      .size = 2048,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_enc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref
    DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref = &DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref
    DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref = &DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref = &DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref = &DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride[TILING_I];

    _b2_conv1xk_tiling_closure_args_t DeeployNetwork__b2_conv1xk_tiling_closure_args = (_b2_conv1xk_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_in_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_dim_im_out_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_nWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bHi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_bWi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref};

    // _b2_conv1xk_tiling_closure CLOSURE CALL
    _b2_conv1xk_tiling_closure(&DeeployNetwork__b2_conv1xk_tiling_closure_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_conv_out_tensor_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_size_1d[TILING_I],
                                      .ext_stride_1d = 1472};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_conv_out_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_conv_out_tensor_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_conv_out_tensor_ref) +
                                                                               DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_data_out_relativeOffset[TILING_I]);

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr;
} _b2_conv1xk_closure_L3_args_t;

static void _b2_conv1xk_closure_L3(void *_b2_conv1xk_closure_L3_args) {
  // CLOSURE ARG CAST
  _b2_conv1xk_closure_L3_args_t *args = (_b2_conv1xk_closure_L3_args_t *)_b2_conv1xk_closure_L3_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b2_conv1xk_closure_args_t DeeployNetwork__b2_conv1xk_closure_args =
      (_b2_conv1xk_closure_args_t){.DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                   .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                   .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr};

  // _b2_conv1xk_closure CLOSURE CALL
  _b2_conv1xk_closure(&DeeployNetwork__b2_conv1xk_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref;
} _b2_rqs_tiling_closure_args_t;

static void _b2_rqs_tiling_closure(void *_b2_rqs_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b2_rqs_tiling_closure_args_t *args = (_b2_rqs_tiling_closure_args_t *)_b2_rqs_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL

  // RequantShift (Name: b2_rqs, Op: RequantShift)
  RequantShift_s32_s8_NHWC(DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref, 5152, DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref,
                           DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref, 16, 16, 0, 0, -128, 127, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref;
} _b2_rqs_cluster_fork_args_t;

static void _b2_rqs_cluster_fork(void *_b2_rqs_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b2_rqs_cluster_fork_args_t *args = (_b2_rqs_cluster_fork_args_t *)_b2_rqs_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b2_rqs_tiling_closure_args_t DeeployNetwork__b2_rqs_tiling_closure_args =
      (_b2_rqs_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref};

  // _b2_rqs_tiling_closure CLOSURE CALL
  _b2_rqs_tiling_closure(&DeeployNetwork__b2_rqs_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr;
} _b2_rqs_closure_args_t;

static void _b2_rqs_closure(void *_b2_rqs_closure_args) {
  // CLOSURE ARG CAST
  _b2_rqs_closure_args_t *args = (_b2_rqs_closure_args_t *)_b2_rqs_closure_args;

  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5152);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 25760);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 25824);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_conv_out_tensor_ref =
      (void *)((char *)DeeployNetwork_conv_out_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_rqs_add_tensor_ref =
      (void *)((char *)DeeployNetwork_rqs_add_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_rqs_mul_tensor_ref =
      (void *)((char *)DeeployNetwork_rqs_mul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1462400,
                                      .size = 20608,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_conv_out_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_rqs_add_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441856,
                                      .size = 64,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_rqs_mul_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _b2_rqs_cluster_fork_args_t DeeployNetwork__b2_rqs_cluster_fork_args =
        (_b2_rqs_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_in_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_mul_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_add_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b2_rqs_cluster_fork, &DeeployNetwork__b2_rqs_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1315872,
                                      .size = 5152,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr;
} _b2_rqs_closure_L3_args_t;

static void _b2_rqs_closure_L3(void *_b2_rqs_closure_L3_args) {
  // CLOSURE ARG CAST
  _b2_rqs_closure_L3_args_t *args = (_b2_rqs_closure_L3_args_t *)_b2_rqs_closure_L3_args;

  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b2_rqs_closure_args_t DeeployNetwork__b2_rqs_closure_args =
      (_b2_rqs_closure_args_t){.DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                               .DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr};

  // _b2_rqs_closure CLOSURE CALL
  _b2_rqs_closure(&DeeployNetwork__b2_rqs_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr;
    DeeployNetwork_weight_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 6720);
    _b2_wpert_closure_L3_args_t DeeployNetwork__b2_wpert_closure_L3_args =
        (_b2_wpert_closure_L3_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b2_wpert_tileIdxPtr};

    // _b2_wpert_closure_L3 CLOSURE CALL
    _b2_wpert_closure_L3(&DeeployNetwork__b2_wpert_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr;
    DeeployNetwork_weight_enc_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 27328);
    _b2_wenc_closure_L3_args_t DeeployNetwork__b2_wenc_closure_L3_args =
        (_b2_wenc_closure_L3_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                     .DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b2_wenc_tileIdxPtr};

    // _b2_wenc_closure_L3 CLOSURE CALL
    _b2_wenc_closure_L3(&DeeployNetwork__b2_wenc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr;
    DeeployNetwork_conv_out_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 6720);
    _b2_conv1xk_closure_L3_args_t DeeployNetwork__b2_conv1xk_closure_L3_args =
        (_b2_conv1xk_closure_L3_args_t){.DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                        .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b2_conv1xk_tileIdxPtr};

    // _b2_conv1xk_closure_L3 CLOSURE CALL
    _b2_conv1xk_closure_L3(&DeeployNetwork__b2_conv1xk_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr;
    _b2_rqs_closure_L3_args_t DeeployNetwork__b2_rqs_closure_L3_args =
        (_b2_rqs_closure_L3_args_t){.DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                    .DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b2_rqs_tileIdxPtr};

    // _b2_rqs_closure_L3 CLOSURE CALL
    _b2_rqs_closure_L3(&DeeployNetwork__b2_rqs_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 25888);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 29376);

  DeeployNetwork_input_0 = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 8768);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
