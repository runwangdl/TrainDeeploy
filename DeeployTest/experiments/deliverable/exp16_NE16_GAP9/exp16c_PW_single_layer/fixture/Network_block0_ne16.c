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

static PI_L2 int32_t DeeployNetwork_w_pmul_tensor[8] = {90721, 183761, 94308, 79564, 99005, 103189, 238955, 94555};

static PI_L2 int32_t DeeployNetwork_rqs_mul_tensor_DUPLICATE_FOR_b0_sbias[8] = {398, 196, 383, 454, 365, 350, 151, 382};

static PI_L2 int32_t DeeployNetwork_rqs_add_tensor[8] = {32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768};

static PI_L2 int32_t DeeployNetwork_rqs_mul_tensor_DUPLICATE_FOR_b0_rqs[8] = {398, 196, 383, 454, 365, 350, 151, 382};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_cumByteOffset[1] = {0};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride[8] = {231, 231, 231, 231, 231, 231, 20, 20};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride[8] = {7296, 7296, 7296, 7296, 7296, 7296, 544, 544};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo[8] = {4, 1, 4, 1, 4, 1, 4, 1};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo[8] = {76, 76, 76, 76, 76, 76, 6, 6};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo[8] = {3, 2, 3, 2, 3, 2, 3, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo[8] = {3, 3, 3, 3, 3, 3, 2, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi[8] = {3, 2, 3, 2, 3, 2, 3, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi[8] = {3, 3, 3, 3, 3, 3, 2, 2};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_numTiles[2] = {0, 8};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_cumByteOffset[8] = {0, 8448, 228, 8676, 456, 8904, 684, 9132};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_cmd[8] = {1968852, 1966542, 1968852, 1966542, 1968852, 1966542, 1966320, 1966120};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_size[8] = {2772, 462, 2772, 462, 2772, 462, 240, 40};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_size_1d[8] = {231, 231, 231, 231, 231, 231, 20, 20};

static PI_L1 int16_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_relativeOffset[8] = {8448, -8220, 8448, -8220, 8448, -8220, 8448, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_cumByteOffset[8] = {0, 0, 0, 0, 0, 0, 0, 0};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_cumByteOffset[8] = {0, 269184, 7296, 276480, 14592, 283776, 21888, 291072};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_cmd[8] = {1922560, 1849600, 1922560, 1849600, 1922560, 1849600, 1841536, 1836096};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_size[8] = {87552, 14592, 87552, 14592, 87552, 14592, 6528, 1088};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_size_1d[8] = {7296, 7296, 7296, 7296, 7296, 7296, 544, 544};

static PI_L1 int32_t DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_relativeOffset[8] = {269184, -261888, 269184, -261888, 269184, -261888, 269184, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_numTiles[2] = {0, 8};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_cumByteOffset[8] = {0, 4, 8, 12, 16, 20, 24, 28};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_cumByteOffset[8] = {0, 32, 64, 96, 128, 160, 192, 224};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_cumByteOffset[8] = {0, 32, 64, 96, 128, 160, 192, 224};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_cumByteOffset[8] = {0, 1, 2, 3, 4, 5, 6, 7};

void *DeeployNetwork_inputs[2];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref;
} _b0_wpert_tiling_closure_args_t;

static void _b0_wpert_tiling_closure(void *_b0_wpert_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b0_wpert_tiling_closure_args_t *args = (_b0_wpert_tiling_closure_args_t *)_b0_wpert_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: b0_wpert, Op: RQSPerturbRademacher)
  uint8_t b0_wpert_core_id = (uint8_t)pi_core_id();
  uint8_t b0_wpert_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t b0_wpert_chunk = (32 >> b0_wpert_log2Core) + ((32 & (NUM_CORES - 1)) != 0);
  uint32_t b0_wpert_chunk_start = (uint32_t)MIN(b0_wpert_chunk * b0_wpert_core_id, (uint32_t)32);
  uint32_t b0_wpert_chunk_stop = (uint32_t)MIN(b0_wpert_chunk_start + b0_wpert_chunk, (uint32_t)32);
  uint32_t b0_wpert_local_size = b0_wpert_chunk_stop - b0_wpert_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t b0_wpert_channel_start_offset = b0_wpert_chunk_start % 4;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 0 + b0_wpert_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t b0_wpert_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  ApplyPerturbQuantRademacher_CHW((const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref[b0_wpert_chunk_start],
                                  (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref[b0_wpert_chunk_start],
                                  (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref, 15, 4, chunk_seed, b0_wpert_local_size,
                                  b0_wpert_chunk_start,
                                  perturbation_sign,   // -- QW: +eps (L+) / -eps (L-)
                                  b0_wpert_eps_scale); // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref;
} _b0_wpert_cluster_fork_args_t;

static void _b0_wpert_cluster_fork(void *_b0_wpert_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b0_wpert_cluster_fork_args_t *args = (_b0_wpert_cluster_fork_args_t *)_b0_wpert_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b0_wpert_tiling_closure_args_t DeeployNetwork__b0_wpert_tiling_closure_args =
      (_b0_wpert_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref};

  // _b0_wpert_tiling_closure CLOSURE CALL
  _b0_wpert_tiling_closure(&DeeployNetwork__b0_wpert_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr;
} _b0_wpert_closure_args_t;

static void _b0_wpert_closure(void *_b0_wpert_closure_args) {
  // CLOSURE ARG CAST
  _b0_wpert_closure_args_t *args = (_b0_wpert_closure_args_t *)_b0_wpert_closure_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_w_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_w_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_weight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_input_1_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_w_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b0_wpert_cluster_fork_args_t DeeployNetwork__b0_wpert_cluster_fork_args =
        (_b0_wpert_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_in_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_mul_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b0_wpert_cluster_fork, &DeeployNetwork__b0_wpert_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_weight_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr;
} _b0_wpert_closure_L3_args_t;

static void _b0_wpert_closure_L3(void *_b0_wpert_closure_L3_args) {
  // CLOSURE ARG CAST
  _b0_wpert_closure_L3_args_t *args = (_b0_wpert_closure_L3_args_t *)_b0_wpert_closure_L3_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b0_wpert_closure_args_t DeeployNetwork__b0_wpert_closure_args =
      (_b0_wpert_closure_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                 .DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr};

  // _b0_wpert_closure CLOSURE CALL
  _b0_wpert_closure(&DeeployNetwork__b0_wpert_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref;
} _b0_sbias_tiling_closure_args_t;

static void _b0_sbias_tiling_closure(void *_b0_sbias_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b0_sbias_tiling_closure_args_t *args = (_b0_sbias_tiling_closure_args_t *)_b0_sbias_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16SignedInputBias (Name: b0_sbias, Op: NE16SignedInputBias) -- BLOCKER 3 correction -- QW
  {
    uint32_t b0_sbias_ncout = 8;
    uint32_t b0_sbias_chunk = (b0_sbias_ncout + NUM_CORES - 1) / NUM_CORES;
    uint32_t b0_sbias_start = MIN(b0_sbias_chunk * (uint32_t)pi_core_id(), b0_sbias_ncout);
    uint32_t b0_sbias_stop = MIN(b0_sbias_start + b0_sbias_chunk, b0_sbias_ncout);

    NE16SignedInputBias_i32((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref,
                            (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref,
                            (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref,
                            (int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref, 8, 4, 128, b0_sbias_start, b0_sbias_stop - b0_sbias_start);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref;
} _b0_sbias_cluster_fork_args_t;

static void _b0_sbias_cluster_fork(void *_b0_sbias_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b0_sbias_cluster_fork_args_t *args = (_b0_sbias_cluster_fork_args_t *)_b0_sbias_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b0_sbias_tiling_closure_args_t DeeployNetwork__b0_sbias_tiling_closure_args =
      (_b0_sbias_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref};

  // _b0_sbias_tiling_closure CLOSURE CALL
  _b0_sbias_tiling_closure(&DeeployNetwork__b0_sbias_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  int32_t *DeeployNetwork_rqs_add_corr_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr;
} _b0_sbias_closure_args_t;

static void _b0_sbias_closure(void *_b0_sbias_closure_args) {
  // CLOSURE ARG CAST
  _b0_sbias_closure_args_t *args = (_b0_sbias_closure_args_t *)_b0_sbias_closure_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  int32_t *DeeployNetwork_rqs_add_corr_tensor = args->DeeployNetwork_rqs_add_corr_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 32);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 64);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 96);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_rqs_mul_tensor_DUPLICATE_FOR_b0_sbias_ref =
      (void *)((char *)DeeployNetwork_rqs_mul_tensor_DUPLICATE_FOR_b0_sbias +
               DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_rqs_add_tensor_ref =
      (void *)((char *)DeeployNetwork_rqs_add_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_rqs_add_corr_tensor_ref =
      (void *)((char *)DeeployNetwork_rqs_add_corr_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_pert_tensor_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref,
                      DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_rqs_mul_tensor_DUPLICATE_FOR_b0_sbias_ref);
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_rqs_add_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b0_sbias_cluster_fork_args_t DeeployNetwork__b0_sbias_cluster_fork_args =
        (_b0_sbias_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_weight_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_mul_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_add_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b0_sbias_cluster_fork, &DeeployNetwork__b0_sbias_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1310752, DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_rqs_add_corr_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  int32_t *DeeployNetwork_rqs_add_corr_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr;
} _b0_sbias_closure_L3_args_t;

static void _b0_sbias_closure_L3(void *_b0_sbias_closure_L3_args) {
  // CLOSURE ARG CAST
  _b0_sbias_closure_L3_args_t *args = (_b0_sbias_closure_L3_args_t *)_b0_sbias_closure_L3_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  int32_t *DeeployNetwork_rqs_add_corr_tensor = args->DeeployNetwork_rqs_add_corr_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b0_sbias_closure_args_t DeeployNetwork__b0_sbias_closure_args =
      (_b0_sbias_closure_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                 .DeeployNetwork_rqs_add_corr_tensor = DeeployNetwork_rqs_add_corr_tensor,
                                 .DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr};

  // _b0_sbias_closure CLOSURE CALL
  _b0_sbias_closure(&DeeployNetwork__b0_sbias_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref;
} _b0_wenc_tiling_closure_args_t;

static void _b0_wenc_tiling_closure(void *_b0_wenc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b0_wenc_tiling_closure_args_t *args = (_b0_wenc_tiling_closure_args_t *)_b0_wenc_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16WeightEncode (Name: b0_wenc, Op: NE16WeightEncode) -- bit-serial encode on the cluster -- QW
  {
    uint32_t b0_wenc_nrows = 32;
    uint32_t b0_wenc_chunk = (b0_wenc_nrows + NUM_CORES - 1) / NUM_CORES;
    uint32_t b0_wenc_start = MIN(b0_wenc_chunk * (uint32_t)pi_core_id(), b0_wenc_nrows);
    uint32_t b0_wenc_stop = MIN(b0_wenc_start + b0_wenc_chunk, b0_wenc_nrows);

    NE16WeightEncode_i8_u8((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref,
                           (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref, 8, 1, 4, b0_wenc_start, b0_wenc_stop - b0_wenc_start);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref;
} _b0_wenc_cluster_fork_args_t;

static void _b0_wenc_cluster_fork(void *_b0_wenc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b0_wenc_cluster_fork_args_t *args = (_b0_wenc_cluster_fork_args_t *)_b0_wenc_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b0_wenc_tiling_closure_args_t DeeployNetwork__b0_wenc_tiling_closure_args =
      (_b0_wenc_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref,
                                       .DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref};

  // _b0_wenc_tiling_closure CLOSURE CALL
  _b0_wenc_tiling_closure(&DeeployNetwork__b0_wenc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr;
} _b0_wenc_closure_args_t;

static void _b0_wenc_closure(void *_b0_wenc_closure_args) {
  // CLOSURE ARG CAST
  _b0_wenc_closure_args_t *args = (_b0_wenc_closure_args_t *)_b0_wenc_closure_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 512);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_weight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_weight_enc_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_enc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_input = (uint32_t)-1;
  uint32_t channel_output = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1441824, DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_weight_pert_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b0_wenc_cluster_fork_args_t DeeployNetwork__b0_wenc_cluster_fork_args =
        (_b0_wenc_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_in_ref,
                                       .DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b0_wenc_cluster_fork, &DeeployNetwork__b0_wenc_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1311232, DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_weight_enc_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr;
} _b0_wenc_closure_L3_args_t;

static void _b0_wenc_closure_L3(void *_b0_wenc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b0_wenc_closure_L3_args_t *args = (_b0_wenc_closure_L3_args_t *)_b0_wenc_closure_L3_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b0_wenc_closure_args_t DeeployNetwork__b0_wenc_closure_args =
      (_b0_wenc_closure_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                .DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                .DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr};

  // _b0_wenc_closure CLOSURE CALL
  _b0_wenc_closure(&DeeployNetwork__b0_wenc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_ref;
} _b0_conv1xk_tiling_closure_args_t;

static void _b0_conv1xk_tiling_closure(void *_b0_conv1xk_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b0_conv1xk_tiling_closure_args_t *args = (_b0_conv1xk_tiling_closure_args_t *)_b0_conv1xk_tiling_closure_args;

  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16 1x4 dense conv as 4 pointwise dispatches (streamin accumulation)
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_ref + 0 * 128,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_ref + 0 * 1 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 1, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 32, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(8, 1),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_ref + 1 * 128,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_ref + 1 * 1 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 1, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 32, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(8, 1),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_ref + 2 * 128,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_ref + 2 * 1 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 1, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 32, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(8, 1),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_ref + 3 * 128,
            // 1xK: one tap is one PIXEL along W -> step by ch_im_in elements
            .infeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_ref + 3 * 1 - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 1, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 32, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref,
                                                                               *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref)},
                                            .remainder = {.KoKi = nnx_concat_half(8, 1),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref)}},
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
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr;
} _b0_conv1xk_closure_args_t;

static void _b0_conv1xk_closure(void *_b0_conv1xk_closure_args) {
  // CLOSURE ARG CAST
  _b0_conv1xk_closure_args_t *args = (_b0_conv1xk_closure_args_t *)_b0_conv1xk_closure_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride + 0);
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 87552);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 90324);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_enc_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_enc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_conv_out_tensor_ref =
      (void *)((char *)DeeployNetwork_conv_out_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_output = -1;
  int transfer_input = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_input_0_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_size_1d[TILING_I],
                                      .ext_stride_1d = 704};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_input_0_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_input_0_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_input_0_ref) +
                                                                       DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_relativeOffset[TILING_I]);

    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1442304,
                                      .size = 512,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_enc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref = &DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref = &DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref = &DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref = &DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride[TILING_I];

    _b0_conv1xk_tiling_closure_args_t DeeployNetwork__b0_conv1xk_tiling_closure_args = (_b0_conv1xk_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_in_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_dim_im_out_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_nWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bHi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_bWi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_ref};

    // _b0_conv1xk_tiling_closure CLOSURE CALL
    _b0_conv1xk_tiling_closure(&DeeployNetwork__b0_conv1xk_tiling_closure_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_conv_out_tensor_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_size_1d[TILING_I],
                                      .ext_stride_1d = 22432};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_conv_out_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_conv_out_tensor_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_conv_out_tensor_ref) +
                                                                               DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_data_out_relativeOffset[TILING_I]);

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr;
} _b0_conv1xk_closure_L3_args_t;

static void _b0_conv1xk_closure_L3(void *_b0_conv1xk_closure_L3_args) {
  // CLOSURE ARG CAST
  _b0_conv1xk_closure_L3_args_t *args = (_b0_conv1xk_closure_L3_args_t *)_b0_conv1xk_closure_L3_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b0_conv1xk_closure_args_t DeeployNetwork__b0_conv1xk_closure_args =
      (_b0_conv1xk_closure_args_t){.DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                   .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                   .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr};

  // _b0_conv1xk_closure CLOSURE CALL
  _b0_conv1xk_closure(&DeeployNetwork__b0_conv1xk_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref;
} _b0_rqs_tiling_closure_args_t;

static void _b0_rqs_tiling_closure(void *_b0_rqs_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b0_rqs_tiling_closure_args_t *args = (_b0_rqs_tiling_closure_args_t *)_b0_rqs_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL

  // RequantShift (Name: b0_rqs, Op: RequantShift)
  RequantShift_s32_s8_NHWC(DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref, 9814, DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref,
                           DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref, 16, 1, 0, 0, -128, 127, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref;
} _b0_rqs_cluster_fork_args_t;

static void _b0_rqs_cluster_fork(void *_b0_rqs_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b0_rqs_cluster_fork_args_t *args = (_b0_rqs_cluster_fork_args_t *)_b0_rqs_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b0_rqs_tiling_closure_args_t DeeployNetwork__b0_rqs_tiling_closure_args =
      (_b0_rqs_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref};

  // _b0_rqs_tiling_closure CLOSURE CALL
  _b0_rqs_tiling_closure(&DeeployNetwork__b0_rqs_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_rqs_add_corr_tensor;
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr;
} _b0_rqs_closure_args_t;

static void _b0_rqs_closure(void *_b0_rqs_closure_args) {
  // CLOSURE ARG CAST
  _b0_rqs_closure_args_t *args = (_b0_rqs_closure_args_t *)_b0_rqs_closure_args;

  int32_t *DeeployNetwork_rqs_add_corr_tensor = args->DeeployNetwork_rqs_add_corr_tensor;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 9816);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 49076);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 49072);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_conv_out_tensor_ref =
      (void *)((char *)DeeployNetwork_conv_out_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_rqs_add_corr_tensor_ref =
      (void *)((char *)DeeployNetwork_rqs_add_corr_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_rqs_mul_tensor_DUPLICATE_FOR_b0_rqs_ref =
      (void *)((char *)DeeployNetwork_rqs_mul_tensor_DUPLICATE_FOR_b0_rqs +
               DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_output = -1;
  int transfer_input = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 2005336,
                                      .size = 39256,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_conv_out_tensor_ref,
                                      .ext_size_1d = 4,
                                      .ext_stride_1d = 32};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_conv_out_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_conv_out_tensor_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_conv_out_tensor_ref) + 4);

    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441796,
                                      .size = 4,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_rqs_add_corr_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_rqs_add_corr_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_rqs_add_corr_tensor_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_rqs_add_corr_tensor_ref) + 4);

    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441796,
                                      .size = 4,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_rqs_mul_tensor_DUPLICATE_FOR_b0_rqs_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_rqs_mul_tensor_DUPLICATE_FOR_b0_rqs_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_rqs_mul_tensor_DUPLICATE_FOR_b0_rqs_ref =
        (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_rqs_mul_tensor_DUPLICATE_FOR_b0_rqs_ref) + 4);

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _b0_rqs_cluster_fork_args_t DeeployNetwork__b0_rqs_cluster_fork_args =
        (_b0_rqs_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_in_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_mul_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_add_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b0_rqs_cluster_fork, &DeeployNetwork__b0_rqs_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1844822,
                                      .size = 9814,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_output_0_ref,
                                      .ext_size_1d = 1,
                                      .ext_stride_1d = 8};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_output_0_ref
    DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_output_0_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_output_0_ref) + 1);

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_rqs_add_corr_tensor;
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr;
} _b0_rqs_closure_L3_args_t;

static void _b0_rqs_closure_L3(void *_b0_rqs_closure_L3_args) {
  // CLOSURE ARG CAST
  _b0_rqs_closure_L3_args_t *args = (_b0_rqs_closure_L3_args_t *)_b0_rqs_closure_L3_args;

  int32_t *DeeployNetwork_rqs_add_corr_tensor = args->DeeployNetwork_rqs_add_corr_tensor;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b0_rqs_closure_args_t DeeployNetwork__b0_rqs_closure_args =
      (_b0_rqs_closure_args_t){.DeeployNetwork_rqs_add_corr_tensor = DeeployNetwork_rqs_add_corr_tensor,
                               .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                               .DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr};

  // _b0_rqs_closure CLOSURE CALL
  _b0_rqs_closure(&DeeployNetwork__b0_rqs_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  int8_t *DeeployNetwork_weight_pert_tensor;
  int32_t *DeeployNetwork_rqs_add_corr_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr;
    DeeployNetwork_weight_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9856);
    _b0_wpert_closure_L3_args_t DeeployNetwork__b0_wpert_closure_L3_args =
        (_b0_wpert_closure_L3_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_wpert_tileIdxPtr};

    // _b0_wpert_closure_L3 CLOSURE CALL
    _b0_wpert_closure_L3(&DeeployNetwork__b0_wpert_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr;
    DeeployNetwork_rqs_add_corr_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 78512);
    _b0_sbias_closure_L3_args_t DeeployNetwork__b0_sbias_closure_L3_args =
        (_b0_sbias_closure_L3_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                      .DeeployNetwork_rqs_add_corr_tensor = DeeployNetwork_rqs_add_corr_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_sbias_tileIdxPtr};

    // _b0_sbias_closure_L3 CLOSURE CALL
    _b0_sbias_closure_L3(&DeeployNetwork__b0_sbias_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr;
    DeeployNetwork_weight_enc_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9888);
    _b0_wenc_closure_L3_args_t DeeployNetwork__b0_wenc_closure_L3_args =
        (_b0_wenc_closure_L3_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                     .DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_wenc_tileIdxPtr};

    // _b0_wenc_closure_L3 CLOSURE CALL
    _b0_wenc_closure_L3(&DeeployNetwork__b0_wenc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr;
    DeeployNetwork_conv_out_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 78544);
    _b0_conv1xk_closure_L3_args_t DeeployNetwork__b0_conv1xk_closure_L3_args =
        (_b0_conv1xk_closure_L3_args_t){.DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                        .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_conv1xk_tileIdxPtr};

    // _b0_conv1xk_closure_L3 CLOSURE CALL
    _b0_conv1xk_closure_L3(&DeeployNetwork__b0_conv1xk_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr;
    _b0_rqs_closure_L3_args_t DeeployNetwork__b0_rqs_closure_L3_args =
        (_b0_rqs_closure_L3_args_t){.DeeployNetwork_rqs_add_corr_tensor = DeeployNetwork_rqs_add_corr_tensor,
                                    .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                    .DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b0_rqs_tileIdxPtr};

    // _b0_rqs_closure_L3 CLOSURE CALL
    _b0_rqs_closure_L3(&DeeployNetwork__b0_rqs_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 90836);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 392592);

  DeeployNetwork_input_0 = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 9888);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
