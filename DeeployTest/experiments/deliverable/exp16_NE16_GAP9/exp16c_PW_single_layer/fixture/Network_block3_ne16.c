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

static PI_L2 int32_t DeeployNetwork_w_pmul_tensor[32] = {248787, 216882, 201474, 174510, 219147, 259040, 184174, 160982, 185684, 195178, 241322,
                                                         256189, 130503, 232271, 255256, 221739, 206960, 231038, 159191, 222818, 198061, 246434,
                                                         234354, 249346, 274450, 263337, 184550, 208225, 235788, 121571, 194099, 212029};

static PI_L2 int32_t DeeployNetwork_rqs_mul_tensor[32] = {78, 90, 97,  112, 89, 75, 106, 121, 105, 100, 81,  76, 149, 84,  76,  88,
                                                          94, 84, 123, 88,  98, 79, 83,  78,  71,  74,  106, 94, 83,  160, 100, 92};

static PI_L2 int32_t DeeployNetwork_rqs_add_tensor[32] = {32700, 32708, 32713, 32816, 32708, 32697, 32717, 32724, 32819, 32714, 32702,
                                                          32839, 32732, 32832, 32698, 32707, 32825, 32832, 32724, 32707, 32713, 32700,
                                                          32703, 32699, 32844, 32696, 32819, 32825, 32833, 32801, 32715, 32710};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_cumByteOffset[1] = {0};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride[4] = {48, 48, 32, 32};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride[4] = {384, 384, 256, 256};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo[4] = {2, 1, 2, 1};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo[4] = {3, 2, 3, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo[4] = {3, 3, 2, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi[4] = {3, 2, 3, 2};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi[4] = {3, 3, 2, 2};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_numTiles[2] = {0, 4};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_cumByteOffset[4] = {0, 480, 48, 528};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_cmd[4] = {1966656, 1966464, 1966464, 1966336};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_size[4] = {576, 384, 384, 256};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_size_1d[4] = {48, 48, 32, 32};

static PI_L1 int16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_relativeOffset[4] = {480, -432, 480, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_cumByteOffset[4] = {0, 0, 0, 0};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_cumByteOffset[4] = {0, 3840, 384, 4224};

static PI_L1 uint32_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_cmd[4] = {1837312, 1835776, 1836544, 1835520};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_size[4] = {2304, 768, 1536, 512};

static PI_L1 uint16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_size_1d[4] = {384, 384, 256, 256};

static PI_L1 int16_t DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_relativeOffset[4] = {3840, -3456, 3840, 0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_numTiles[2] = {0, 1};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_cumByteOffset[1] = {0};

static PI_L1 uint8_t DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_cumByteOffset[1] = {0};

void *DeeployNetwork_inputs[2];
void *DeeployNetwork_outputs[1];
extern struct pi_device cluster_dev;
typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref;
} _b3_wpert_tiling_closure_args_t;

static void _b3_wpert_tiling_closure(void *_b3_wpert_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3_wpert_tiling_closure_args_t *args = (_b3_wpert_tiling_closure_args_t *)_b3_wpert_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref;

  // CLOSURE FUNCTION CALL

  // PerturbRademacher (Name: b3_wpert, Op: RQSPerturbRademacher)
  uint8_t b3_wpert_core_id = (uint8_t)pi_core_id();
  uint8_t b3_wpert_log2Core = (uint8_t)log2(NUM_CORES);

  // Parallelize over the total size of the tensor
  uint32_t b3_wpert_chunk = (3584 >> b3_wpert_log2Core) + ((3584 & (NUM_CORES - 1)) != 0);
  uint32_t b3_wpert_chunk_start = (uint32_t)MIN(b3_wpert_chunk * b3_wpert_core_id, (uint32_t)3584);
  uint32_t b3_wpert_chunk_stop = (uint32_t)MIN(b3_wpert_chunk_start + b3_wpert_chunk, (uint32_t)3584);
  uint32_t b3_wpert_local_size = b3_wpert_chunk_stop - b3_wpert_chunk_start;

  // Calculate the starting channel for this core's chunk of M
  uint32_t b3_wpert_channel_start_offset = b3_wpert_chunk_start % 112;

  // Pick large enough stride to minimize correlation between nodes.
  // -- QW: add perturb_seed_base (ZORuntime.h) so L+/L- share one RNG pattern and
  // the seed advances per update step (neutral 0 -> baked 42 behavior).
  uint32_t chunk_seed = ((42 + perturb_seed_base) + NUM_CORES * 6 + b3_wpert_core_id) ^ (0 * 0x9E3779B1u);
  // QW: zo_update runtime coefficient — scale the baked integer mul by (override / baked eps); 1.0f in
  // train passes (override off). Mirrors FloatPerturbRademacherTemplate's eps override. -- QW
  float32_t b3_wpert_eps_scale = perturb_eps_use_override ? (perturb_eps_override / perturb_eps_baked) : 1.0f;

  ApplyPerturbQuantRademacher_CHW((const int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref[b3_wpert_chunk_start],
                                  (int8_t *)&DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref[b3_wpert_chunk_start],
                                  (const int32_t *)DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref, 15, 112, chunk_seed, b3_wpert_local_size,
                                  b3_wpert_chunk_start,
                                  perturbation_sign,   // -- QW: +eps (L+) / -eps (L-)
                                  b3_wpert_eps_scale); // -- QW: zo_update coeff scaling (1.0f = train pass)

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref;
} _b3_wpert_cluster_fork_args_t;

static void _b3_wpert_cluster_fork(void *_b3_wpert_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b3_wpert_cluster_fork_args_t *args = (_b3_wpert_cluster_fork_args_t *)_b3_wpert_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b3_wpert_tiling_closure_args_t DeeployNetwork__b3_wpert_tiling_closure_args =
      (_b3_wpert_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref};

  // _b3_wpert_tiling_closure CLOSURE CALL
  _b3_wpert_tiling_closure(&DeeployNetwork__b3_wpert_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr;
} _b3_wpert_closure_args_t;

static void _b3_wpert_closure(void *_b3_wpert_closure_args) {
  // CLOSURE ARG CAST
  _b3_wpert_closure_args_t *args = (_b3_wpert_closure_args_t *)_b3_wpert_closure_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 7168);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_input_1_ref =
      (void *)((char *)DeeployNetwork_input_1 +
               DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_w_pmul_tensor_ref =
      (void *)((char *)DeeployNetwork_w_pmul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_weight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445376, DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_input_1_ref);
    mchan_transfer_1d(1441920, DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref, DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_w_pmul_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b3_wpert_cluster_fork_args_t DeeployNetwork__b3_wpert_cluster_fork_args =
        (_b3_wpert_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_in_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_mul_ref,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b3_wpert_cluster_fork, &DeeployNetwork__b3_wpert_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314304, DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_weight_pert_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr;
} _b3_wpert_closure_L3_args_t;

static void _b3_wpert_closure_L3(void *_b3_wpert_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3_wpert_closure_L3_args_t *args = (_b3_wpert_closure_L3_args_t *)_b3_wpert_closure_L3_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3_wpert_closure_args_t DeeployNetwork__b3_wpert_closure_args =
      (_b3_wpert_closure_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                 .DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr};

  // _b3_wpert_closure CLOSURE CALL
  _b3_wpert_closure(&DeeployNetwork__b3_wpert_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref;
} _b3_wenc_tiling_closure_args_t;

static void _b3_wenc_tiling_closure(void *_b3_wenc_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3_wenc_tiling_closure_args_t *args = (_b3_wenc_tiling_closure_args_t *)_b3_wenc_tiling_closure_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16WeightEncode (Name: b3_wenc, Op: NE16WeightEncode) -- bit-serial encode on the cluster -- QW
  {
    uint32_t b3_wenc_nrows = 224;
    uint32_t b3_wenc_chunk = (b3_wenc_nrows + NUM_CORES - 1) / NUM_CORES;
    uint32_t b3_wenc_start = MIN(b3_wenc_chunk * (uint32_t)pi_core_id(), b3_wenc_nrows);
    uint32_t b3_wenc_stop = MIN(b3_wenc_start + b3_wenc_chunk, b3_wenc_nrows);

    NE16WeightEncode_i8_u8((const int8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref,
                           (uint8_t *)DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref, 32, 16, 7, b3_wenc_start, b3_wenc_stop - b3_wenc_start);
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref;
} _b3_wenc_cluster_fork_args_t;

static void _b3_wenc_cluster_fork(void *_b3_wenc_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b3_wenc_cluster_fork_args_t *args = (_b3_wenc_cluster_fork_args_t *)_b3_wenc_cluster_fork_args;

  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b3_wenc_tiling_closure_args_t DeeployNetwork__b3_wenc_tiling_closure_args =
      (_b3_wenc_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref,
                                       .DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref};

  // _b3_wenc_tiling_closure CLOSURE CALL
  _b3_wenc_tiling_closure(&DeeployNetwork__b3_wenc_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr;
} _b3_wenc_closure_args_t;

static void _b3_wenc_closure(void *_b3_wenc_closure_args) {
  // CLOSURE ARG CAST
  _b3_wenc_closure_args_t *args = (_b3_wenc_closure_args_t *)_b3_wenc_closure_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_weight_pert_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_pert_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_weight_enc_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_enc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr]);

  // Initialize DMA futures
  uint32_t channel_output = (uint32_t)-1;
  uint32_t channel_input = (uint32_t)-1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    channel_input = mchan_channel_alloc();
    mchan_transfer_1d(1445376, DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref, DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_weight_pert_tensor_ref);

    // Wait for input tiles

    if (channel_input <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_input);
      mchan_channel_free(channel_input);
    }

    _b3_wenc_cluster_fork_args_t DeeployNetwork__b3_wenc_cluster_fork_args =
        (_b3_wenc_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_in_ref,
                                       .DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b3_wenc_cluster_fork, &DeeployNetwork__b3_wenc_cluster_fork_args);

    // Transfer output tiles
    channel_output = mchan_channel_alloc();
    mchan_transfer_1d(1314304, DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_data_out_ref, DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_weight_enc_tensor_ref);

    // Wait for output tiles

    if (channel_output <= MCHAN_CHANNEL_ID_MAX) {
      mchan_channel_wait(channel_output);
      mchan_channel_free(channel_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr;
} _b3_wenc_closure_L3_args_t;

static void _b3_wenc_closure_L3(void *_b3_wenc_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3_wenc_closure_L3_args_t *args = (_b3_wenc_closure_L3_args_t *)_b3_wenc_closure_L3_args;

  int8_t *DeeployNetwork_weight_pert_tensor = args->DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3_wenc_closure_args_t DeeployNetwork__b3_wenc_closure_args =
      (_b3_wenc_closure_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                .DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                .DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr};

  // _b3_wenc_closure CLOSURE CALL
  _b3_wenc_closure(&DeeployNetwork__b3_wenc_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref;
} _b3_conv1xk_tiling_closure_args_t;

static void _b3_conv1xk_tiling_closure(void *_b3_conv1xk_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3_conv1xk_tiling_closure_args_t *args = (_b3_conv1xk_tiling_closure_args_t *)_b3_conv1xk_tiling_closure_args;

  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref;
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref;
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref;

  // CLOSURE FUNCTION CALL

  // NE16 1x7 dense conv as 7 pointwise dispatches (streamin accumulation)
  {
    ne16_task_t task = {
        .data = (ne16_task_data_t){
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref + 0 * 512,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref + 0 * *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref + 1 * 512,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref + 1 * *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref + 2 * 512,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref + 2 * *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref + 3 * 512,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref + 3 * *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref + 4 * 512,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref + 4 * *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref + 5 * 512,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref + 5 * *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref)}},
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
            .weights_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref + 6 * 512,
            // Kx1: one tap is one ROW along H -> step by the input row stride
            .infeat_addr =
                (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref + 6 * *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref - 0,
            .outfeat_addr = (uint32_t)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref,
            .scale_addr = (uint32_t)NULL,
            .scale_shift_addr = (uint32_t)NULL,
            .scale_bias_addr = (uint32_t)NULL,
            .cfg = (ne16_cfg_t){
                .input_stride = (ne16_stride_t){.d0 = 16, .d1 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref, .d2 = 0},
                .output_stride =
                    (ne16_stride_t){.d0 = NE16_OUTPUT_BANDWIDTH_BYTES, .d1 = 128, .d2 = *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref},
                .weights_stride = (ne16_stride_t){.d0 = 16, .d1 = 16, .d2 = 0},
                .subtile = (ne16_subtile_t){.number = {.KoKi = nnx_concat_half(1, 1),
                                                       .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref, 1)},
                                            .remainder = {.KoKi = nnx_concat_half(32, 16),
                                                          .HoWo = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref),
                                                          .HiWi = nnx_concat_half(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref,
                                                                                  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref)}},
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
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr;
} _b3_conv1xk_closure_args_t;

static void _b3_conv1xk_closure(void *_b3_conv1xk_closure_args) {
  // CLOSURE ARG CAST
  _b3_conv1xk_closure_args_t *args = (_b3_conv1xk_closure_args_t *)_b3_conv1xk_closure_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride + 0);
  uint32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref =
      (uint32_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi + 0);
  uint16_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref = (uint16_t *)((char *)DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi + 0);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 5888);
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 3584);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_input_0_ref =
      (void *)((char *)DeeployNetwork_input_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_enc_tensor_ref =
      (void *)((char *)DeeployNetwork_weight_enc_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_conv_out_tensor_ref =
      (void *)((char *)DeeployNetwork_conv_out_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_input_0_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_size_1d[TILING_I],
                                      .ext_stride_1d = 80};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_input_0_ref
    DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_input_0_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_input_0_ref) +
                                                                       DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_relativeOffset[TILING_I]);

    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1445376,
                                      .size = 3584,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_enc_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref
    DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref = &DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref
    DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref = &DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref
    DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref = &DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref = &DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride[TILING_I];

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref
    DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref = &DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride[TILING_I];

    _b3_conv1xk_tiling_closure_args_t DeeployNetwork__b3_conv1xk_tiling_closure_args = (_b3_conv1xk_tiling_closure_args_t){
        .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_in_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_dim_im_out_x_stride_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_nHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWo_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bHi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_bWi_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_in_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_weight_ref,
        .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref};

    // _b3_conv1xk_tiling_closure CLOSURE CALL
    _b3_conv1xk_tiling_closure(&DeeployNetwork__b3_conv1xk_tiling_closure_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_cmd[TILING_I],
                                      .size = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_size[TILING_I],
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_conv_out_tensor_ref,
                                      .ext_size_1d = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_size_1d[TILING_I],
                                      .ext_stride_1d = 640};
      mchan_transfer_push_2d(__mchan_tmp);
    }

    // UPDATE VARIABLE DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_conv_out_tensor_ref
    DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_conv_out_tensor_ref = (void *)((char *)(DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_conv_out_tensor_ref) +
                                                                               DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_data_out_relativeOffset[TILING_I]);

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr;
} _b3_conv1xk_closure_L3_args_t;

static void _b3_conv1xk_closure_L3(void *_b3_conv1xk_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3_conv1xk_closure_L3_args_t *args = (_b3_conv1xk_closure_L3_args_t *)_b3_conv1xk_closure_L3_args;

  uint8_t *DeeployNetwork_weight_enc_tensor = args->DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3_conv1xk_closure_args_t DeeployNetwork__b3_conv1xk_closure_args =
      (_b3_conv1xk_closure_args_t){.DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                   .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                   .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr};

  // _b3_conv1xk_closure CLOSURE CALL
  _b3_conv1xk_closure(&DeeployNetwork__b3_conv1xk_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref;
} _b3_rqs_tiling_closure_args_t;

static void _b3_rqs_tiling_closure(void *_b3_rqs_tiling_closure_args) {
  // CLOSURE ARG CAST
  _b3_rqs_tiling_closure_args_t *args = (_b3_rqs_tiling_closure_args_t *)_b3_rqs_tiling_closure_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL

  // RequantShift (Name: b3_rqs, Op: RequantShift)
  RequantShift_s32_s8_NHWC(DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref, 1280, DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref,
                           DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref, DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref, 16, 32, 0, 0, -128, 127, 1);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref;
} _b3_rqs_cluster_fork_args_t;

static void _b3_rqs_cluster_fork(void *_b3_rqs_cluster_fork_args) {
  // CLOSURE ARG CAST
  _b3_rqs_cluster_fork_args_t *args = (_b3_rqs_cluster_fork_args_t *)_b3_rqs_cluster_fork_args;

  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref;
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref;
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref = args->DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref;

  // CLOSURE FUNCTION CALL
  _b3_rqs_tiling_closure_args_t DeeployNetwork__b3_rqs_tiling_closure_args =
      (_b3_rqs_tiling_closure_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref};

  // _b3_rqs_tiling_closure CLOSURE CALL
  _b3_rqs_tiling_closure(&DeeployNetwork__b3_rqs_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr;
} _b3_rqs_closure_args_t;

static void _b3_rqs_closure(void *_b3_rqs_closure_args) {
  // CLOSURE ARG CAST
  _b3_rqs_closure_args_t *args = (_b3_rqs_closure_args_t *)_b3_rqs_closure_args;

  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 1280);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6400);
  int32_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 6528);
  int8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L1 + 0);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_conv_out_tensor_ref =
      (void *)((char *)DeeployNetwork_conv_out_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_rqs_add_tensor_ref =
      (void *)((char *)DeeployNetwork_rqs_add_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_rqs_mul_tensor_ref =
      (void *)((char *)DeeployNetwork_rqs_mul_tensor +
               DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr]);
  void *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_output_0_ref =
      (void *)((char *)DeeployNetwork_output_0 +
               DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_cumByteOffset[*DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr]);

  // Initialize DMA futures
  int transfer_input = -1;
  int transfer_output = -1;

  // TILING LOOP
  for (int TILING_I = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_numTiles[*DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr];
       TILING_I < DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_numTiles[(*DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr) + 1]; TILING_I++) {

    // Transfer input tiles
    transfer_input = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1446912,
                                      .size = 5120,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_conv_out_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_rqs_add_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1441920,
                                      .size = 128,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_rqs_mul_tensor_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for input tiles

    if (transfer_input >= 0) {
      mchan_transfer_wait(transfer_input);
      mchan_transfer_free(transfer_input);
    }

    _b3_rqs_cluster_fork_args_t DeeployNetwork__b3_rqs_cluster_fork_args =
        (_b3_rqs_cluster_fork_args_t){.DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_in_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_mul_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_add_ref,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref};

    pi_cl_team_fork(NUM_CORES, (void *)_b3_rqs_cluster_fork, &DeeployNetwork__b3_rqs_cluster_fork_args);

    // Transfer output tiles
    transfer_output = mchan_transfer_get_id();
    {
      mchan_transfer_t __mchan_tmp = {.cmd = 1312000,
                                      .size = 1280,
                                      .loc = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_data_out_ref,
                                      .ext = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_output_0_ref};
      mchan_transfer_push_1d(__mchan_tmp);
    }

    // Wait for output tiles

    if (transfer_output >= 0) {
      mchan_transfer_wait(transfer_output);
      mchan_transfer_free(transfer_output);
    }

    // CLOSE TILING LOOP
  }
  *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr += 1;

  // Deinitialize DMA futures

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  int32_t *DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr;
} _b3_rqs_closure_L3_args_t;

static void _b3_rqs_closure_L3(void *_b3_rqs_closure_L3_args) {
  // CLOSURE ARG CAST
  _b3_rqs_closure_L3_args_t *args = (_b3_rqs_closure_L3_args_t *)_b3_rqs_closure_L3_args;

  int32_t *DeeployNetwork_conv_out_tensor = args->DeeployNetwork_conv_out_tensor;
  uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr = args->DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr;

  // CLOSURE FUNCTION CALL
  _b3_rqs_closure_args_t DeeployNetwork__b3_rqs_closure_args =
      (_b3_rqs_closure_args_t){.DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                               .DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr};

  // _b3_rqs_closure CLOSURE CALL
  _b3_rqs_closure(&DeeployNetwork__b3_rqs_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {
  int8_t *DeeployNetwork_weight_pert_tensor;
  uint8_t *DeeployNetwork_weight_enc_tensor;
  int32_t *DeeployNetwork_conv_out_tensor;
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr;
    DeeployNetwork_weight_pert_tensor = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 4704);
    _b3_wpert_closure_L3_args_t DeeployNetwork__b3_wpert_closure_L3_args =
        (_b3_wpert_closure_L3_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                      .DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3_wpert_tileIdxPtr};

    // _b3_wpert_closure_L3 CLOSURE CALL
    _b3_wpert_closure_L3(&DeeployNetwork__b3_wpert_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr;
    DeeployNetwork_weight_enc_tensor = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 1120);
    _b3_wenc_closure_L3_args_t DeeployNetwork__b3_wenc_closure_L3_args =
        (_b3_wenc_closure_L3_args_t){.DeeployNetwork_weight_pert_tensor = DeeployNetwork_weight_pert_tensor,
                                     .DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                     .DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3_wenc_tileIdxPtr};

    // _b3_wenc_closure_L3 CLOSURE CALL
    _b3_wenc_closure_L3(&DeeployNetwork__b3_wenc_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr;
    DeeployNetwork_conv_out_tensor = (int32_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 4704);
    _b3_conv1xk_closure_L3_args_t DeeployNetwork__b3_conv1xk_closure_L3_args =
        (_b3_conv1xk_closure_L3_args_t){.DeeployNetwork_weight_enc_tensor = DeeployNetwork_weight_enc_tensor,
                                        .DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                        .DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3_conv1xk_tileIdxPtr};

    // _b3_conv1xk_closure_L3 CLOSURE CALL
    _b3_conv1xk_closure_L3(&DeeployNetwork__b3_conv1xk_closure_L3_args);
  }
  {

    uint8_t bu_DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr = 0;
    uint8_t *DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr = &bu_DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr;
    _b3_rqs_closure_L3_args_t DeeployNetwork__b3_rqs_closure_L3_args =
        (_b3_rqs_closure_L3_args_t){.DeeployNetwork_conv_out_tensor = DeeployNetwork_conv_out_tensor,
                                    .DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr = DeeployNetwork_TILING_CODEGEN_L1_b3_rqs_tileIdxPtr};

    // _b3_rqs_closure_L3 CLOSURE CALL
    _b3_rqs_closure_L3(&DeeployNetwork__b3_rqs_closure_L3_args);
  }
}

void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads) {

  ne16_pulp_conf_t conf = {.max_stall = 8};
  ne16_nnx_init(ne16_pulp_get_dev(), &conf);

  DeeployNetwork_MEMORYARENA_L1 = (int8_t *)pi_l1_malloc((void *)0, sizeof(int8_t) * 7296);

  DeeployNetwork_MEMORYARENA_L2 = (int8_t *)pi_l2_malloc(sizeof(int8_t) * 9824);

  DeeployNetwork_input_0 = (uint8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_input_1 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 1120);
  DeeployNetwork_output_0 = (int8_t *)((char *)DeeployNetwork_MEMORYARENA_L2 + 0);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
